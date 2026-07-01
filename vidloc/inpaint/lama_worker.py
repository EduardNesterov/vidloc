#!/usr/bin/env python
"""In-memory, batched LaMa inpaint worker with *reuse-first* acceleration.
Runs in ``.venv-inpaint`` (torch + MPS + cv2); no vidloc imports.

Streams a cropped region out of ffmpeg as raw frames, erases the marked regions,
and pipes the patched crop into a second ffmpeg that overlays it onto the original
video and encodes. No PNGs on disk.

The ffmpeg I/O crop is the *union* bounding box of all regions (piping the full
frame was tried and reverted — it spiked ``kernel_task`` and lagged the UI). But
LaMa runs **per-region tiles** (each box + ``LAMA_CTX`` of context), not on the whole
union crop: with scattered regions the union rectangle is nearly the full frame,
yet the pixels that actually need inpainting are small. So GPU cost scales with the
regions' area, not their bounding rectangle.

Reuse-first (v1, per-region, no motion-comp)
--------------------------------------------
Profiling showed LaMa is ~94% of the cost, so the lever is *fewer LaMa calls*, not
a faster LaMa. Video is temporally redundant: the background behind a watermark is
almost always visible in nearby frames. Per region we keep:

* ``plate``   — a clean background fill for that region's box (from a frame where
                the box was *not* masked, or, failing that, from a LaMa result).
* ``ref_ring``— a snapshot of the unmasked pixels just *outside* the box, taken at
                the moment ``plate`` was captured.

On an active (masked) frame, if we already have a plate we compare the box's
current surrounding ring against ``ref_ring``. Small residual ⇒ the background has
not moved ⇒ fill the box straight from the plate (no LaMa). Large residual (camera
move / scene cut / lighting) ⇒ run LaMa and reseed the plate + ref. The ring check
is the artifact safeguard — there is deliberately **no motion compensation** in v1.

For time-gated subtitles the true background is usually visible during the gaps, so
the plate is real (not estimated) and the run can approach ~0 LaMa calls. For a
static corner logo the first frame seeds via LaMa and the rest reuse.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import vision_text   # sibling module (Apple Vision text detection); no vidloc imports

LAMA_CTX = 48   # px of surrounding context fed to LaMa around each region's box


def _ceil8(n: int) -> int:
    return ((int(n) + 7) // 8) * 8


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return bytes(buf)


def _region_mask(box, cw, ch, iw, ih, wp, hp, scale, dev):
    """Per-region masks: downscaled+padded inpaint mask [1,1,hp,wp] and the
    composite feather [1,1,ch,cw]."""
    mx, my, mw, mh = box
    mfull = np.zeros((ch, cw), np.float32)
    mfull[max(0, my):my + mh, max(0, mx):mx + mw] = 1.0
    m = torch.from_numpy(mfull)[None, None].to(dev)
    if scale > 1:                       # dilate so the downscaled mask covers the object
        d = scale * 4
        m = (F.max_pool2d(m, 2 * d + 1, stride=1, padding=d) > 0.5).float()
    mds = F.interpolate(m, size=(ih, iw), mode="bilinear", align_corners=False)
    mds = (mds > 0.5).float()
    mp = F.pad(mds, (0, wp - iw, 0, hp - ih))
    if scale > 1:                       # feather only when downscaling
        feather = m
        kf = scale * 4 + 1
        for _ in range(2):
            feather = F.avg_pool2d(feather, kf, stride=1, padding=kf // 2)
    else:
        feather = m
    return mp, feather


def _active(intervals, t) -> bool:
    return (not intervals) or any(t0 <= t <= t1 for (t0, t1) in intervals)


def _ring(hard: np.ndarray, gap: int, pad: int) -> np.ndarray:
    """A band ``gap..gap+pad`` px outside the box. Used to detect whether the
    background around the box has drifted since the plate was captured. The small
    ``gap`` keeps anti-aliased watermark edges out of the comparison."""
    inner = cv2.dilate(hard, np.ones((2 * gap + 1, 2 * gap + 1), np.uint8))
    outer = cv2.dilate(hard, np.ones((2 * (gap + pad) + 1, 2 * (gap + pad) + 1), np.uint8))
    return (outer > 0) & (inner == 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cx", type=int, required=True)
    ap.add_argument("--cy", type=int, required=True)
    ap.add_argument("--cw", type=int, required=True)
    ap.add_argument("--ch", type=int, required=True)
    ap.add_argument("--regions", required=True)  # json [{"box":[..],"intervals":[[t0,t1]]}]
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--total", type=int, default=0)
    ap.add_argument("--venc", required=True)    # json list of ffmpeg encoder args
    # reuse-first knobs
    ap.add_argument("--reuse", type=int, default=1)        # 0 disables -> LaMa every active frame
    ap.add_argument("--reuse-thresh", type=float, default=0.02)  # mean |Δ| on the ring (0..1)
    ap.add_argument("--reuse-pad", type=int, default=6)    # ring band width (px, full-res)
    ap.add_argument("--reuse-gap", type=int, default=2)    # ring inset from the box edge (px)
    ap.add_argument("--reuse-keyint", type=int, default=120)  # force LaMa every N reuses (0=never)
    # text mode: detect text inside each box (Apple Vision) and erase only the glyphs
    ap.add_argument("--text", type=int, default=0)         # 1 = detect+mask text per frame
    ap.add_argument("--text-dilate", type=int, default=4)  # px grown around detected text
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.jit.load(a.model, map_location=dev).eval()
    cw, ch = a.cw, a.ch
    regions_spec = json.loads(a.regions)   # [{"box":[mx,my,mw,mh],"intervals":[[t0,t1],..]}]
    venc = json.loads(a.venc)
    print(f"WORKER lama in-memory {cw}x{ch} scale={a.scale} batch={a.batch} "
          f"regions={len(regions_spec)} reuse={a.reuse} text={a.text} dev={dev}", flush=True)
    if a.text and not vision_text.available():
        print("ERROR: text mode needs pyobjc (Vision) in .venv-inpaint", flush=True)
        return 1

    iw = max(8, round(cw / a.scale))
    ih = max(8, round(ch / a.scale))
    wp, hp = _ceil8(iw), _ceil8(ih)

    # Per-region precompute + mutable reuse state. ``box`` is relative to the crop;
    # ``intervals`` are in seconds. Reuse-path work is confined to each region's
    # bounding box ``bb`` (box + feather + ring); LaMa runs on a slightly larger
    # ``tile`` (box + context), downscaled — cost scales with the box, not the crop.
    regions = []
    for spec in regions_spec:
        box = spec["box"]
        mx, my, mw, mh = box
        hard = np.zeros((ch, cw), np.uint8)
        hard[max(0, my):my + mh, max(0, mx):mx + mw] = 1
        _, fe_t = _region_mask(box, cw, ch, iw, ih, wp, hp, a.scale, dev)
        fe_np = fe_t[0, 0].cpu().numpy()          # [ch,cw] composite weight
        ring = _ring(hard, a.reuse_gap, a.reuse_pad)
        roi = (fe_np > 1e-6) | ring               # everything the reuse path touches
        ys, xs = np.where(roi)
        if ys.size:
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        else:
            y0 = y1 = x0 = x1 = 0                  # degenerate box -> nothing to erase

        # LaMa tile: box + context, and always ⊇ bb so the box prediction is covered.
        tx0 = max(0, min(mx - LAMA_CTX, x0))
        ty0 = max(0, min(my - LAMA_CTX, y0))
        tx1 = min(cw, max(mx + mw + LAMA_CTX, x1))
        ty1 = min(ch, max(my + mh + LAMA_CTX, y1))
        tw, th = tx1 - tx0, ty1 - ty0
        lama_ok = mw > 0 and mh > 0 and tw > 0 and th > 0
        if lama_ok:
            tiw, tih = max(8, round(tw / a.scale)), max(8, round(th / a.scale))
            twp, thp = _ceil8(tiw), _ceil8(tih)
            mp_tile, _ = _region_mask((mx - tx0, my - ty0, mw, mh),
                                      tw, th, tiw, tih, twp, thp, a.scale, dev)
        else:
            tiw = tih = twp = thp = 0
            mp_tile = None
        regions.append({
            "intervals": [tuple(iv) for iv in spec.get("intervals", [])],
            "box": (mx, my, mw, mh),               # raw user box (crop-local)
            "bb": (y0, y1, x0, x1),                # crop-local bounding box
            "fe_loc": fe_np[y0:y1, x0:x1, None].copy(),   # [bh,bw,1] composite weight
            "ring_loc": ring[y0:y1, x0:x1].copy(),        # [bh,bw] drift-check band
            "has_ring": bool(ring.any()),
            "lama_ok": lama_ok,
            "tile": (ty0, ty1, tx0, tx1),          # crop-local LaMa tile
            "tdims": (tih, tiw, thp, twp),         # downscaled + /8-padded tile dims
            "mp_tile": mp_tile,                    # torch [1,1,thp,twp] tile mask
            "bb_in_tile": (y0 - ty0, y1 - ty0, x0 - tx0, x1 - tx0),
            "plate": None,                         # numpy [bh,bw,3] background fill
            "seen": False,                         # is plate valid?
            "ref_ring": None,                      # numpy [n,3] ring at plate-capture time
            "reuse_run": 0,                        # consecutive reuses since last LaMa
        })

    framesize = cw * ch * 3
    ff1 = subprocess.Popen(
        [a.ffmpeg, "-v", "error", "-i", a.src, "-vf",
         f"crop={cw}:{ch}:{a.cx}:{a.cy}", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE)
    ff2 = subprocess.Popen(
        [a.ffmpeg, "-y", "-v", "error", "-i", a.src,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{cw}x{ch}", "-r", str(a.fps),
         "-i", "pipe:0", "-filter_complex", f"[0:v][1:v]overlay={a.cx}:{a.cy}",
         "-map", "0:a:0?", *venc, "-c:a", "copy", "-movflags", "+faststart", a.out],
        stdin=subprocess.PIPE)

    done = gpu_frames = lama_reg = reuse_reg = 0

    def run_region_lama(r, ks, frames_u8):
        """Batch one region's tile across the frames ``ks`` that need it, in a single
        model call, and return the upscaled tile predictions [n, th, tw, 3]."""
        ty0, ty1, tx0, tx1 = r["tile"]
        tih, tiw, thp, twp = r["tdims"]
        x = torch.from_numpy(np.stack([frames_u8[k][ty0:ty1, tx0:tx1] for k in ks])).to(dev)
        x = x.float().div_(255).permute(0, 3, 1, 2)          # [n,3,th,tw]
        xi = F.interpolate(x, size=(tih, tiw), mode="bilinear", align_corners=False)
        xi = F.pad(xi, (0, twp - tiw, 0, thp - tih))
        mask = r["mp_tile"].expand(len(ks), -1, -1, -1).contiguous()
        with torch.no_grad():
            pred = model(xi, mask)[..., :tih, :tiw]
        predf = F.interpolate(pred, size=(ty1 - ty0, tx1 - tx0),
                              mode="bilinear", align_corners=False)
        return predf.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

    def text_mask_tile(frame_u8, r):
        """Detect text inside region ``r``'s box (Apple Vision), then segment the actual
        glyph strokes within each detected box (keeps the background between/inside the
        letters). Detection runs on the tile (context helps) but is clipped to the user
        box, so only text the user marked is erased. Returns [th,tw] float or None."""
        mx, my, mw, mh = r["box"]
        ty0, ty1, tx0, tx1 = r["tile"]
        tile_img = frame_u8[ty0:ty1, tx0:tx1]
        th, tw = ty1 - ty0, tx1 - tx0
        boxes = vision_text.detect_text_boxes(tile_img)
        if not boxes:
            return None
        m = np.zeros((th, tw), np.float32)
        bx0, by0, bx1, by1 = mx - tx0, my - ty0, mx + mw - tx0, my + mh - ty0  # box in tile
        for (dx, dy, dw, dh) in boxes:
            ix0, iy0 = max(dx, bx0, 0), max(dy, by0, 0)
            ix1, iy1 = min(dx + dw, bx1, tw), min(dy + dh, by1, th)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            gm = vision_text.glyph_mask(tile_img[iy0:iy1, ix0:ix1])
            m[iy0:iy1, ix0:ix1] = np.maximum(m[iy0:iy1, ix0:ix1], gm.astype(np.float32))
        if m.sum() == 0:
            return None
        if a.text_dilate > 0:
            k = 2 * a.text_dilate + 1
            m = cv2.dilate(m, np.ones((k, k), np.uint8))
        return m

    def run_region_lama_masks(r, ks, tmasks, frames_u8):
        """Like run_region_lama but with a *different* mask per frame (the detected text),
        batched in one model call. Returns upscaled tile predictions [n, th, tw, 3]."""
        ty0, ty1, tx0, tx1 = r["tile"]
        tih, tiw, thp, twp = r["tdims"]
        x = torch.from_numpy(np.stack([frames_u8[k][ty0:ty1, tx0:tx1] for k in ks])).to(dev)
        x = x.float().div_(255).permute(0, 3, 1, 2)
        xi = F.pad(F.interpolate(x, (tih, tiw), mode="bilinear", align_corners=False),
                   (0, twp - tiw, 0, thp - tih))
        m = torch.from_numpy(np.stack(tmasks)).to(dev).float()[:, None]     # [n,1,th,tw]
        if a.scale > 1:                       # grow so the downscaled mask covers glyphs
            d = a.scale * 4
            m = (F.max_pool2d(m, 2 * d + 1, 1, d) > 0.5).float()
        mds = (F.interpolate(m, (tih, tiw), mode="bilinear", align_corners=False) > 0.5).float()
        mp = F.pad(mds, (0, twp - tiw, 0, thp - tih))
        with torch.no_grad():
            pred = model(xi, mp)[..., :tih, :tiw]
        predf = F.interpolate(pred, (ty1 - ty0, tx1 - tx0), mode="bilinear", align_corners=False)
        return predf.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

    def flush_reuse(frames_u8, ff, out, base):
        """Static-box reuse path: plate reuse + per-region tile LaMa (fills ``out``)."""
        nonlocal gpu_frames, lama_reg, reuse_reg
        b = len(frames_u8)
        pending = {}   # frame idx -> [regions needing LaMa this frame]

        for k in range(b):
            t = (base + k) / a.fps
            cur = ff[k]
            res = cur.copy()
            acts = [_active(r["intervals"], t) for r in regions]

            # 1) any region whose box is unmasked right now is truly visible ->
            #    grab a real, drift-free plate for it (the key win for gated subtitles).
            if a.reuse:
                for r, on in zip(regions, acts):
                    if on or not r["lama_ok"]:
                        continue
                    y0, y1, x0, x1 = r["bb"]
                    loc = cur[y0:y1, x0:x1]
                    r["plate"] = loc.copy()
                    r["ref_ring"] = loc[r["ring_loc"]].copy()
                    r["seen"] = True
                    r["reuse_run"] = 0

            # 2) decide per active region: reuse the plate or send to LaMa.
            lama_regions = []
            for r, on in zip(regions, acts):
                if not on or not r["lama_ok"]:
                    continue
                y0, y1, x0, x1 = r["bb"]
                can_reuse = False
                if a.reuse and r["seen"] and r["has_ring"]:
                    forced = a.reuse_keyint > 0 and r["reuse_run"] >= a.reuse_keyint
                    if not forced:
                        loc = cur[y0:y1, x0:x1]
                        resid = float(np.abs(loc[r["ring_loc"]] - r["ref_ring"]).mean())
                        can_reuse = resid <= a.reuse_thresh
                if can_reuse:
                    fe = r["fe_loc"]
                    res[y0:y1, x0:x1] = res[y0:y1, x0:x1] * (1 - fe) + r["plate"] * fe
                    r["reuse_run"] += 1
                    reuse_reg += 1
                else:
                    lama_regions.append(r)

            if lama_regions:
                pending[k] = lama_regions
                gpu_frames += 1
                lama_reg += len(lama_regions)
            out[k] = res

        # 3) run LaMa grouped by region — one batched model call per region over the
        #    frames that need it (small tiles) — then composite each box locally and
        #    reseed the plate. Reuse regions live in other boxes and are untouched.
        if pending:
            jobs = defaultdict(list)      # region id -> [frame idx, ...] (ascending)
            reg = {}
            for k in pending:
                for r in pending[k]:
                    jobs[id(r)].append(k)
                    reg[id(r)] = r
            for rid, ks in jobs.items():
                r = reg[rid]
                preds = run_region_lama(r, ks, frames_u8)     # [n,th,tw,3]
                y0, y1, x0, x1 = r["bb"]
                by0, by1, bx0, bx1 = r["bb_in_tile"]
                fe = r["fe_loc"]
                for j, k in enumerate(ks):
                    pbb = preds[j][by0:by1, bx0:bx1]           # prediction over the box
                    patch = out[k][y0:y1, x0:x1] * (1 - fe) + pbb * fe
                    out[k][y0:y1, x0:x1] = patch
                    r["plate"] = patch.copy()                  # reseed: future reuse
                    r["ref_ring"] = ff[k][y0:y1, x0:x1][r["ring_loc"]].copy()
                    r["seen"] = True
                    r["reuse_run"] = 0

    def flush_text(frames_u8, ff, out, base):
        """Text path: per active frame detect the text inside each box and erase only
        the glyphs (tight per-frame mask). Reuse is bypassed — the mask is so small that
        LaMa is cheap; batched per region with per-frame masks (fills ``out``)."""
        nonlocal gpu_frames, lama_reg
        b = len(frames_u8)
        jobs = defaultdict(list)      # region id -> [(frame idx, tile mask)]
        reg = {}
        for k in range(b):
            out[k] = ff[k].copy()
            t = (base + k) / a.fps
            for r in regions:
                if not r["lama_ok"] or not _active(r["intervals"], t):
                    continue
                tm = text_mask_tile(frames_u8[k], r)
                if tm is not None:
                    jobs[id(r)].append((k, tm))
                    reg[id(r)] = r

        touched = set()
        for rid, items in jobs.items():
            r = reg[rid]
            ks = [k for k, _ in items]
            preds = run_region_lama_masks(r, ks, [m for _, m in items], frames_u8)
            ty0, ty1, tx0, tx1 = r["tile"]
            for j, (k, tm) in enumerate(items):
                alpha = cv2.GaussianBlur(tm, (0, 0), 1.5)[..., None]   # soft edge
                out[k][ty0:ty1, tx0:tx1] = (out[k][ty0:ty1, tx0:tx1] * (1 - alpha)
                                            + preds[j] * alpha)
                lama_reg += 1
                touched.add(k)
        gpu_frames += len(touched)

    def flush_batch(frames_u8, base):
        nonlocal done
        ff = [f.astype(np.float32) / 255.0 for f in frames_u8]   # [ch,cw,3] each
        out = [None] * len(frames_u8)
        (flush_text if a.text else flush_reuse)(frames_u8, ff, out, base)
        stacked = np.clip(np.stack(out), 0, 1) * 255
        assert ff2.stdin is not None
        ff2.stdin.write(stacked.astype(np.uint8).tobytes())
        done += len(frames_u8)
        print(f"PROGRESS {done}/{a.total}", flush=True)

    batch, base, fidx = [], 0, 0
    assert ff1.stdout is not None
    while True:
        buf = _read_exact(ff1.stdout, framesize)
        if len(buf) < framesize:
            break
        batch.append(np.frombuffer(buf, np.uint8).reshape(ch, cw, 3))
        fidx += 1
        if len(batch) >= a.batch:
            flush_batch(batch, base)
            base = fidx
            batch = []
    if batch:
        flush_batch(batch, base)

    ff1.stdout.close()
    ff1.wait()
    ff2.stdin.close()
    if ff2.wait() != 0:
        print("ERROR: ffmpeg overlay/encode failed", flush=True)
        return 1
    print(f"STAT lama_reg={lama_reg} reuse_reg={reuse_reg} gpu_frames={gpu_frames} "
          f"frames={done}", flush=True)
    print("WORKER done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

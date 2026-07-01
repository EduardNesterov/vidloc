#!/usr/bin/env python
"""In-memory, batched LaMa inpaint worker with *reuse-first* acceleration.
Runs in ``.venv-inpaint`` (torch + MPS + cv2); no vidloc imports.

Streams a cropped region out of ffmpeg as raw frames, erases the marked regions,
and pipes the patched crop into a second ffmpeg that overlays it onto the original
video and encodes. No PNGs on disk.

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
from functools import reduce

import cv2
import numpy as np
import torch
import torch.nn.functional as F


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
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.jit.load(a.model, map_location=dev).eval()
    cw, ch = a.cw, a.ch
    regions_spec = json.loads(a.regions)   # [{"box":[mx,my,mw,mh],"intervals":[[t0,t1],..]}]
    venc = json.loads(a.venc)
    print(f"WORKER lama in-memory {cw}x{ch} scale={a.scale} batch={a.batch} "
          f"regions={len(regions_spec)} reuse={a.reuse} dev={dev}", flush=True)

    iw = max(8, round(cw / a.scale))
    ih = max(8, round(ch / a.scale))
    wp, hp = _ceil8(iw), _ceil8(ih)

    # Per-region precompute + mutable reuse state. ``box`` is relative to the crop;
    # ``intervals`` are in seconds. All reuse-path work is confined to each region's
    # bounding box ``bb`` (box + feather + ring), so cost scales with the box, not
    # the whole crop — important at 1440p+ where the crop is millions of pixels.
    regions = []
    for spec in regions_spec:
        box = spec["box"]
        mx, my, mw, mh = box
        hard = np.zeros((ch, cw), np.uint8)
        hard[max(0, my):my + mh, max(0, mx):mx + mw] = 1
        mp, fe_t = _region_mask(box, cw, ch, iw, ih, wp, hp, a.scale, dev)
        fe_np = fe_t[0, 0].cpu().numpy()          # [ch,cw] composite weight
        ring = _ring(hard, a.reuse_gap, a.reuse_pad)
        roi = (fe_np > 1e-6) | ring               # everything the reuse path touches
        ys, xs = np.where(roi)
        if ys.size:
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        else:
            y0 = y1 = x0 = x1 = 0                  # degenerate box -> always LaMa
        regions.append({
            "intervals": [tuple(iv) for iv in spec.get("intervals", [])],
            "mp": mp,                              # torch [1,1,hp,wp] LaMa mask
            "bb": (y0, y1, x0, x1),                # crop-local bounding box
            "fe_loc": fe_np[y0:y1, x0:x1, None].copy(),   # [bh,bw,1] composite weight
            "ring_loc": ring[y0:y1, x0:x1].copy(),        # [bh,bw] drift-check band
            "has_ring": bool(ring.any()),
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

    def run_lama(idxs, masks, frames_u8):
        """One GPU pass over the flagged sub-batch -> upscaled predictions [n,ch,cw,3]."""
        x = torch.from_numpy(np.stack([frames_u8[k] for k in idxs])).to(dev)
        x = x.float().div_(255).permute(0, 3, 1, 2)
        xi = F.interpolate(x, size=(ih, iw), mode="bilinear", align_corners=False)
        xi = F.pad(xi, (0, wp - iw, 0, hp - ih))
        mask = torch.cat(masks, dim=0).contiguous()
        with torch.no_grad():
            pred = model(xi, mask)[..., :ih, :iw]
        predf = F.interpolate(pred, size=(ch, cw), mode="bilinear", align_corners=False)
        return predf.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()

    def flush_batch(frames_u8, base):
        nonlocal done, gpu_frames, lama_reg, reuse_reg
        b = len(frames_u8)
        ff = [f.astype(np.float32) / 255.0 for f in frames_u8]   # [ch,cw,3] each
        out = [None] * b
        pending = {}   # frame idx -> (union_mask, [regions needing LaMa])

        for k in range(b):
            t = (base + k) / a.fps
            cur = ff[k]
            res = cur.copy()
            acts = [_active(r["intervals"], t) for r in regions]

            # 1) any region whose box is unmasked right now is truly visible ->
            #    grab a real, drift-free plate for it (the key win for gated subtitles).
            if a.reuse:
                for r, on in zip(regions, acts):
                    if on:
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
                if not on:
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
                pending[k] = (reduce(torch.maximum, (r["mp"] for r in lama_regions)),
                              lama_regions)
                gpu_frames += 1
                lama_reg += len(lama_regions)
            out[k] = res

        # 3) one GPU pass for every frame that needs LaMa, then composite + reseed
        #    each LaMa region's box locally (reuse regions live in other boxes).
        if pending:
            idxs = list(pending.keys())
            preds = run_lama(idxs, [pending[k][0] for k in idxs], frames_u8)
            for j, k in enumerate(idxs):
                pj = preds[j]
                for r in pending[k][1]:
                    y0, y1, x0, x1 = r["bb"]
                    fe = r["fe_loc"]
                    patch = out[k][y0:y1, x0:x1] * (1 - fe) + pj[y0:y1, x0:x1] * fe
                    out[k][y0:y1, x0:x1] = patch
                    r["plate"] = patch.copy()          # reseed: future frames can reuse
                    r["ref_ring"] = ff[k][y0:y1, x0:x1][r["ring_loc"]].copy()
                    r["seen"] = True
                    r["reuse_run"] = 0

        stacked = np.clip(np.stack(out), 0, 1) * 255
        assert ff2.stdin is not None
        ff2.stdin.write(stacked.astype(np.uint8).tobytes())
        done += b
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

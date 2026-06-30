#!/usr/bin/env python
"""In-memory, batched LaMa inpaint worker. Runs in ``.venv-inpaint`` (torch + MPS);
no vidloc imports.

Streams a cropped region out of ffmpeg as raw frames, inpaints them in batches on
the GPU (optionally at reduced resolution for speed), composites the result back at
full resolution (only the masked pixels change), and pipes the patched crop into a
second ffmpeg that overlays it onto the original video and encodes. No PNGs on disk.
"""

from __future__ import annotations

import argparse
import json
import subprocess

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
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.jit.load(a.model, map_location=dev).eval()
    cw, ch = a.cw, a.ch
    regions_spec = json.loads(a.regions)   # [{"box":[mx,my,mw,mh],"intervals":[[t0,t1],..]}]
    venc = json.loads(a.venc)
    print(f"WORKER lama in-memory {cw}x{ch} scale={a.scale} batch={a.batch} "
          f"regions={len(regions_spec)} dev={dev}", flush=True)

    iw = max(8, round(cw / a.scale))
    ih = max(8, round(ch / a.scale))
    wp, hp = _ceil8(iw), _ceil8(ih)

    # Precompute each region's masks once; per frame we OR the masks of the
    # regions whose time intervals cover that frame's timestamp.
    region_masks = []   # (intervals, mask_p[1,1,hp,wp], feather[1,1,ch,cw])
    for spec in regions_spec:
        mp, feather = _region_mask(spec["box"], cw, ch, iw, ih, wp, hp, a.scale, dev)
        region_masks.append(([tuple(iv) for iv in spec.get("intervals", [])], mp, feather))
    zero_mp = torch.zeros(1, 1, hp, wp, device=dev)
    zero_feather = torch.zeros(1, 1, ch, cw, device=dev)

    def frame_masks(idx):
        t = idx / a.fps
        mp, feather, any_active = zero_mp, zero_feather, False
        for (intervals, rmp, rfe) in region_masks:
            if _active(intervals, t):
                mp = rmp if not any_active else torch.maximum(mp, rmp)
                feather = rfe if not any_active else torch.maximum(feather, rfe)
                any_active = True
        return mp, feather, any_active

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

    done = 0

    def flush_batch(frames, base):
        nonlocal done
        x = torch.from_numpy(np.stack(frames)).to(dev).float().div_(255).permute(0, 3, 1, 2)
        b = x.shape[0]
        mps, fes, any_active = [], [], False
        for k in range(b):
            mp, fe, act = frame_masks(base + k)
            mps.append(mp)
            fes.append(fe)
            any_active = any_active or act
        if any_active:                        # at least one region active in this batch
            xi = F.interpolate(x, size=(ih, iw), mode="bilinear", align_corners=False)
            xi = F.pad(xi, (0, wp - iw, 0, hp - ih))
            with torch.no_grad():
                pred = model(xi, torch.cat(mps, dim=0).contiguous())
            pred = pred[..., :ih, :iw]
            predf = F.interpolate(pred, size=(ch, cw), mode="bilinear", align_corners=False)
            feather = torch.cat(fes, dim=0)
            x = x * (1 - feather) + predf.clamp(0, 1) * feather
        # else: nothing to erase on these frames -> pass the crop through untouched
        out = x.clamp(0, 1).mul_(255).byte().permute(0, 2, 3, 1).contiguous().cpu().numpy()
        assert ff2.stdin is not None
        ff2.stdin.write(out.tobytes())
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
    print("WORKER done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

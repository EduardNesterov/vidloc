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
import sys

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
    ap.add_argument("--boxes", required=True)   # json [[mx,my,mw,mh], ...]
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--total", type=int, default=0)
    ap.add_argument("--venc", required=True)    # json list of ffmpeg encoder args
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = torch.jit.load(a.model, map_location=dev).eval()
    cw, ch = a.cw, a.ch
    boxes = json.loads(a.boxes)
    venc = json.loads(a.venc)
    print(f"WORKER lama in-memory {cw}x{ch} scale={a.scale} batch={a.batch} dev={dev}",
          flush=True)

    # full-res mask (1 = inpaint)
    mfull = np.zeros((ch, cw), np.float32)
    for (mx, my, mw, mh) in boxes:
        mfull[max(0, my):my + mh, max(0, mx):mx + mw] = 1.0
    mask_full = torch.from_numpy(mfull)[None, None].to(dev)
    # When downscaling, the mask erodes — dilate to fully cover the object.
    if a.scale > 1:
        d = a.scale * 4
        mask_full = (F.max_pool2d(mask_full, 2 * d + 1, stride=1, padding=d) > 0.5).float()

    iw = max(8, round(cw / a.scale))
    ih = max(8, round(ch / a.scale))
    wp, hp = _ceil8(iw), _ceil8(ih)
    mds = F.interpolate(mask_full, size=(ih, iw), mode="bilinear", align_corners=False)
    mds = (mds > 0.5).float()
    mask_p = F.pad(mds, (0, wp - iw, 0, hp - ih))  # [1,1,hp,wp]

    # feather the composite mask only when downscaling; at full res use the exact
    # mask for a crisp, artifact-free result.
    if a.scale > 1:
        mask_feather = mask_full
        kf = a.scale * 4 + 1
        for _ in range(2):
            mask_feather = F.avg_pool2d(mask_feather, kf, stride=1, padding=kf // 2)
    else:
        mask_feather = mask_full

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

    def flush_batch(frames):
        nonlocal done
        x = torch.from_numpy(np.stack(frames)).to(dev).float().div_(255).permute(0, 3, 1, 2)
        xi = F.interpolate(x, size=(ih, iw), mode="bilinear", align_corners=False)
        xi = F.pad(xi, (0, wp - iw, 0, hp - ih))
        b = xi.shape[0]
        with torch.no_grad():
            pred = model(xi, mask_p.expand(b, -1, -1, -1).contiguous())
        pred = pred[..., :ih, :iw]
        predf = F.interpolate(pred, size=(ch, cw), mode="bilinear", align_corners=False)
        out = x * (1 - mask_feather) + predf.clamp(0, 1) * mask_feather
        out = out.clamp(0, 1).mul_(255).byte().permute(0, 2, 3, 1).contiguous().cpu().numpy()
        assert ff2.stdin is not None
        ff2.stdin.write(out.tobytes())
        done += b
        print(f"PROGRESS {done}/{a.total}", flush=True)

    batch = []
    assert ff1.stdout is not None
    while True:
        buf = _read_exact(ff1.stdout, framesize)
        if len(buf) < framesize:
            break
        batch.append(np.frombuffer(buf, np.uint8).reshape(ch, cw, 3))
        if len(batch) >= a.batch:
            flush_batch(batch)
            batch = []
    if batch:
        flush_batch(batch)

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

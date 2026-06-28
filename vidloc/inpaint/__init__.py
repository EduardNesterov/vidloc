"""Object removal (watermarks / hardcoded subtitles) via inpainting.

Two tiers:
  * FAST    -> ffmpeg ``delogo`` (one pass, instant, rough)
  * QUALITY -> neural LaMa via IOPaint (frame-by-frame, slow, clean)
"""

from __future__ import annotations

import shutil

from ..models import InpaintQuality


def inpaint_video(src, out, removal, info, output=None, reporter=None, on_progress=None) -> str:
    log = reporter.log if reporter else (lambda m: None)

    regions = [tuple(r) for r in (removal.watermark_regions or []) if r]
    if removal.remove_hardsubs and removal.hardsub_region:
        regions.append(tuple(removal.hardsub_region))
    if not regions:
        log("Немає виділених областей для затирання — пропускаю крок.")
        shutil.copy(src, out)
        return out

    if removal.quality == InpaintQuality.FAST:
        from .delogo import delogo_video
        log(f"Швидке затирання (delogo), областей: {len(regions)}")
        delogo_video(src, out, regions, info.width, info.height, output=output,
                     duration=info.duration, on_progress=on_progress, on_log=log)
    else:
        from .lama import lama_video
        scale = 1 if removal.quality == InpaintQuality.BEST else 2
        try:
            lama_video(src, out, regions, info, output=output, scale=scale,
                       reporter=reporter, on_progress=on_progress)
        except Exception as exc:  # noqa: BLE001 — degrade rather than fail the whole job
            log(f"⚠ Нейро-затирання не вдалося ({exc}); відкочуюсь на швидке (delogo).")
            from .delogo import delogo_video
            delogo_video(src, out, regions, info.width, info.height, output=output,
                         duration=info.duration, on_progress=on_progress, on_log=log)
    return out

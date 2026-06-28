"""vidloc — local, offline video localizer.

Features (built in phases):
  * EN/RU/IT -> UK translation as subtitles and/or dubbed voice
  * Watermark removal (neural video inpainting)
  * Hardcoded-subtitle removal (OCR-guided inpainting)
  * Output control: resolution, container, codec, quality

Everything runs locally on the machine; no data leaves the device.
"""

__version__ = "0.1.0"

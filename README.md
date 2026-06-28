# Vidloc — local video localizer (UA)

**English** · [Русский](README.ru.md)

A desktop app (macOS / Apple Silicon) that translates video **EN/RU/IT → Ukrainian**
(subtitles and/or dubbing) and — in later phases — removes watermarks and burned-in subtitles.
Everything runs **locally**, with no data sent to the cloud.

## Features

| Done | Feature |
|:---:|---|
| ✅ | Speech recognition (Whisper, mlx — Metal acceleration) |
| ✅ | Translation into Ukrainian (local LLM Aya / NLLB-200, Metal) |
| ✅ | Subtitles: soft (separate track) or burned-in (hardsub) |
| ✅ | Output: container (mp4/mkv/mov/webm), codec (H.264/H.265/copy), resolution, CRF, fps, audio bitrate |
| ✅ | GUI with every setting + headless CLI |
| 🚧 | Dubbing (Ukrainian voices: ukrainian-tts / Piper / Coqui) |
| 🚧 | Watermark removal (neural inpainting: LaMa / IOPaint) |
| 🚧 | Burned-in subtitle removal (OCR + inpainting) |

## Requirements

- macOS on Apple Silicon (tested on M2, 16 GB)
- Python 3.11 — installed via [uv](https://docs.astral.sh/uv/)
- `ffmpeg`/`ffprobe` — static Apple Silicon binaries in `bin/`

## Install

```sh
git clone <repo-url> vidloc && cd vidloc
uv sync          # creates .venv and installs dependencies
```

`ffmpeg`/`ffprobe` are not in the repo (large, git-ignored) — drop static Apple Silicon
binaries into `bin/` or install a system ffmpeg (`brew install ffmpeg`).
Models are not in the repo either and download automatically on first run (`models/`, ~26 GB).

## Run

GUI:
```sh
./run.sh
```

CLI (handy for testing):
```sh
PYTHONPATH=. .venv/bin/python -m vidloc.cli input.mp4 \
    --source en --mode subtitles --subs soft --asr balanced --mt fast
```
Flags: `--mode {subtitles,voice,both,none}`, `--subs {soft,burn}`,
`--asr {fast,balanced,best}`, `--mt {fast,best}`, `--beam N`,
`--height`, `--container`, `--codec`, `--crf`.

## Architecture

> A detailed walkthrough of the pipeline logic, models, and design decisions is in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

```
vidloc/
  settings.py        paths, env, device selection (MPS/CPU)
  models.py          job config: enums + JobConfig dataclass
  media/ffmpeg.py    ffmpeg wrappers (probe, extract_audio, render_video)
  asr/               speech recognition (mlx-whisper)
  translate/         translation (LLM Aya / NLLB-200)
  subtitles/         .srt / styled .ass generation
  tts/               dubbing (in progress)
  pipeline/          orchestration (runner) + progress/cancel (events)
  gui/               PySide6: window, worker thread
  cli.py             headless run
```

Folders: `models/` — model cache, `work/` — per-job temp files, `output/` — results
(all git-ignored).

## Status

- **Phase 1 (translation → subtitles)** — done and tested end-to-end.
- **Phase 1b (dubbing)** and **Phase 2 (object removal)** — in progress.

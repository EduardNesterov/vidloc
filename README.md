# Vidloc — local video localizer (UA)

**English** · [Русский](README.ru.md)

A desktop app (macOS / Apple Silicon) that translates video **EN/RU/IT → Ukrainian**
(subtitles and/or dubbing) and removes watermarks and burned-in subtitles.
Everything runs **locally**, with no data sent to the cloud.

## Features

| Done | Feature |
|:---:|---|
| ✅ | Speech recognition (Whisper, mlx — Metal acceleration) |
| ✅ | Translation into Ukrainian (local LLM Aya / Dragoman EN→UK / NLLB-200, Metal) |
| ✅ | Visual subtitle editor — captions are one source of truth for both burn-in and dubbing |
| ✅ | Dubbing (Ukrainian voices: StyleTTS2 default / ukrainian-tts / Piper / Coqui / F5-TTS clone) |
| ✅ | Adjustable original-track level under the dub + instant re-mix (no re-encode, no re-TTS) |
| ✅ | Watermark & burned-in subtitle removal — per-region, time-gated (neural LaMa + CRAFT text detection) |
| ✅ | Subtitle font library — 25 OFL families with full Ukrainian Cyrillic |
| ✅ | Output: container (mp4/mkv/mov/webm), codec (H.264/H.265/copy), resolution, CRF, fps, audio bitrate |
| ✅ | GUI wizard + workspace cache (repeat runs skip download & ASR) + headless CLI |
| ✅ | URL download (yt-dlp); Instagram via browser cookies |

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

### Optional isolated TTS environments

TTS engines intentionally live outside the main `.venv` because their dependency stacks
conflict. Install only the engines you plan to use:

```sh
uv venv .venv-styletts2 --python 3.11
uv pip install --python .venv-styletts2/bin/python \
  soundfile librosa scipy einops einops_exts munch pyyaml torch torchaudio \
  transformers "numpy<2" accelerate ctranslate2 sentencepiece huggingface-hub \
  git+https://github.com/patriotyk/ukrainian-word-stress.git \
  git+https://github.com/patriotyk/ipa-uk.git \
  git+https://github.com/patriotyk/styletts2-inference.git

uv venv .venv-uktts --python 3.11
uv pip install --python .venv-uktts/bin/python ukrainian-tts

uv venv .venv-piper --python 3.11
uv pip install --python .venv-piper/bin/python piper-tts onnxruntime ukrainian-word-stress

uv venv .venv-tts --python 3.11
uv pip install --python .venv-tts/bin/python coqui-tts torch torchaudio

uv venv .venv-f5 --python 3.11
uv pip install --python .venv-f5/bin/python f5-tts soundfile
```

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
`--asr {fast,balanced,best}`, `--mt {fast,best,natural,dragoman}`,
`--engine {styletts2_uk,uktts,piper,coqui_uk,f5_uk}`, `--beam N`,
`--height`, `--container`, `--codec`, `--crf`.

## Architecture

> A detailed walkthrough of the pipeline logic, models, and design decisions is in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

```
vidloc/
  settings.py        paths, env, device selection (MPS/CPU)
  config.py          user prefs (~/.vidloc: e.g. cookies browser)
  models.py          job config: enums + JobConfig dataclass
  workspace.py       per-video cache (source + ASR transcript + translation)
  media/             ffmpeg wrappers, yt-dlp fetch, H.264 proxy/master
  asr/               speech recognition (mlx-whisper)
  translate/         translation (Aya / Dragoman / NLLB-200) + shared service
  subtitles/         .srt / styled .ass generation + short-caption rechunk
  tts/               dubbing (StyleTTS2 / uktts / Piper / Coqui / F5) in isolated venvs
  inpaint/           watermark/subtitle removal (LaMa worker + CRAFT text detection)
  pipeline/          orchestration (runner) + progress/cancel (events)
  gui/               PySide6: window, subtitle & region editors, worker threads
  cli.py             headless run
```

Folders: `models/` — model cache, `work/` — per-job temp files, `workspace/` — per-video
cache, `output/` — results (all git-ignored).

## Status

- **Phase 1 (translation → subtitles)** — done and tested end-to-end.
- **Phase 1b (dubbing)** — done: StyleTTS2-Ukrainian by default, plus an adjustable
  original-track mix that can be re-tuned on a finished render without re-encoding.
- **Phase 2 (watermark / burned-in subtitle removal)** — working in the GUI wizard:
  per-region, time-gated erasing with LaMa plate-reuse and CRAFT text detection.

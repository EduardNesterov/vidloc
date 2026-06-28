# Vidloc architecture

**English** · [Русский](ARCHITECTURE.ru.md)

A technical walkthrough: how the pipeline works, which models are used, and the reasoning
behind the key engineering decisions.

## TL;DR

- **Fully local, offline.** All AI inference runs on the user's machine. The network is only used
  for (1) a one-time model download from the Hugging Face Hub and (2) optional video download by
  URL (`yt-dlp`). **There are no cloud inference APIs — no data leaves the device.**
- **Apple Metal acceleration.** ASR and the translation LLM run on **MLX**; NLLB runs on
  **PyTorch MPS**; neural inpainting (LaMa) runs on **MPS**; video encoding optionally uses
  **VideoToolbox**.
- **Engine decoupled from the UI.** The core (`PipelineRunner`) knows nothing about the GUI — it
  talks to a `Reporter` (progress/cancel), so the GUI and the CLI share the same pipeline.

## Pipeline

```mermaid
flowchart TD
    IN["Input: local file or URL (yt-dlp)"] --> PROBE["ffprobe: resolution, fps, audio?"]
    PROBE --> AUD["Extract 16 kHz audio (ffmpeg)"]
    AUD --> ASR["ASR — mlx-whisper (Metal / MLX)"]
    ASR --> MT{"Translate to Ukrainian"}
    MT -->|"NATURAL (default)"| LLM["Local LLM: Aya-expanse-8B-4bit (MLX) + video context"]
    MT -->|"FAST / BEST"| NLLB["NLLB-200 (PyTorch MPS)"]
    LLM --> SEG["Translated segments"]
    NLLB --> SEG

    SEG --> SUBS["Subtitles: SRT (soft) or styled ASS (burn)"]
    SEG --> DUB["Dubbing: UA TTS — uktts / piper / coqui"]
    IN --> INP["Watermark removal: LaMa / IOPaint (MPS) or ffmpeg delogo"]

    SUBS --> REND["Final render — ffmpeg: codec/container/CRF, VideoToolbox"]
    DUB --> REND
    INP --> REND
    REND --> OUT["Final video *_uk"]
```

Stage order lives in `pipeline/runner.py`. Stages are switched on by `JobConfig` (what actually
needs doing), and **inpainting (GPU) and dubbing (CPU) run in parallel** in a single `process`
stage — while the GPU cleans frames, the CPU synthesizes speech.

## Stages and models

| Stage | What it does | Model / engine | Acceleration |
|---|---|---|---|
| **ASR** | speech → timestamped text | `whisper-large-v3-mlx` (BEST, default), `whisper-large-v3-turbo` (BALANCED), `whisper-medium-mlx` (FAST) | Metal (MLX) |
| **Translate · NATURAL** | idiomatic, slang-aware, by meaning | `mlx-community/aya-expanse-8b-4bit` (Cohere Aya) | Metal (MLX) |
| **Translate · FAST/BEST** | literal, fast (fallback) | `facebook/nllb-200-distilled-600M` / `-1.3B` | Metal (PyTorch MPS) |
| **Dubbing** | text → Ukrainian speech | ukrainian-tts (5 voices), Piper ONNX (3 voices), Coqui `uk/mai/vits` (fallback) | CPU (subprocess) |
| **Watermark removal** | region inpainting | **LaMa** via IOPaint (QUALITY), `ffmpeg delogo` (FAST) | Metal (MPS) / ffmpeg |
| **Render** | assemble final video | ffmpeg: H.264 / H.265, mp4 / mkv / mov / webm | VideoToolbox (opt.) |

Device selection — `settings.torch_device()`: `mps` → `cuda` → `cpu`. On Apple Silicon that's MPS;
`PYTORCH_ENABLE_MPS_FALLBACK=1` routes the few ops not yet implemented on MPS to the CPU instead
of crashing.

## Why it's built this way — engineering decisions

- **No cloud APIs.** Privacy and offline operation are the priority. Everything runs locally; the
  only network traffic is the one-time weight download and (optionally) a video download by URL.

- **Isolated venvs + subprocess workers.** The dubbing and inpainting engines conflict on
  dependencies with the core (which uses `transformers 5`) and with each other (different torch
  versions, ESPnet / VITS / ONNX). So each lives in its own venv (`.venv-uktts`, `.venv-tts`,
  `.venv-piper`, `.venv-inpaint`) and is invoked as a separate process. The core stays clean and
  stable.

- **An LLM for translation instead of "literal".** NLLB translates word-for-word; for natural
  language the default is a local LLM (Aya) with a **context pass**: the model first summarizes the
  topic of the whole clip in one call, and that context plus the video's title/description (from
  `yt-dlp`) is mixed into every translation chunk — resolving word-sense ambiguity (e.g. "drift" in
  a drifting video is translated by meaning, not literally).

- **GPU ∥ CPU parallelism.** Inpainting loads the GPU, TTS loads the CPU, so they run at the same
  time (`ThreadPoolExecutor`). ASR reads the original audio track, which lets inpainting run after
  translation and overlap with dubbing.

- **Graceful degradation.** If LLM translation is unavailable for any reason, the pipeline
  automatically falls back to NLLB and finishes the job.

- **One source of truth for settings.** `JobConfig` (a dataclass in `models.py`) binds every GUI
  control to the engine, so the UI and the pipeline never drift apart.

- **ffmpeg filtergraph safety.** The project path contains a space, which trips the `subtitles`
  filter, so subtitle files are copied to a space-free temp path before burn-in.

## Code map

```
vidloc/
  settings.py          paths, env, device selection (MPS/CPU), HF_HOME → models/
  models.py            enums + JobConfig (single source of truth for a job)
  media/
    ffmpeg.py          probe / extract_audio / render_video, VideoToolbox
    fetch.py           URL download (yt-dlp), context extraction
  asr/                 speech recognition (mlx-whisper)
  translate/
    llm.py             natural translation (Aya, MLX) + context pass
    translator.py      literal translation (NLLB-200, MPS)
  subtitles/
    srt.py             soft subtitles (.srt)
    ass.py             styled burned-in subtitles (.ass, libass + Montserrat)
    rechunk.py         split into short captions (TikTok/Reels look)
  tts/
    dubbing.py         dubbing orchestration, time-fit to the original tempo
    uktts_worker.py    ukrainian-tts (subprocess in .venv-uktts)
    piper_worker.py    Piper ONNX (subprocess in .venv-piper)
    coqui_worker.py    Coqui VITS (subprocess in .venv-tts)
  inpaint/
    lama.py            neural LaMa inpainting via IOPaint (MPS)
    delogo.py          fast ffmpeg delogo
  pipeline/
    runner.py          orchestration of all stages (see the diagram above)
    events.py          progress and cancel (Reporter / StageTracker)
  gui/                 PySide6: window, worker thread, region selector
  cli.py               headless run
```

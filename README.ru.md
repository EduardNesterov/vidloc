# Vidloc — локальный локализатор видео (UA)

[English](README.md) · **Русский**

Десктопное приложение (macOS / Apple Silicon) для перевода видео **EN/RU/IT → украинский**
(субтитры и/или озвучка) и — в следующих этапах — удаления вотермарок и вшитых субтитров.
Всё работает **локально**, без отправки данных в сеть.

## Возможности

| Готово | Функция |
|:---:|---|
| ✅ | Распознавание речи (Whisper, mlx — ускорение Metal) |
| ✅ | Перевод на украинский (локальная LLM Aya / NLLB-200, Metal) |
| ✅ | Субтитры: мягкие (отдельная дорожка) или вшитые (hardsub) |
| ✅ | Вывод: контейнер (mp4/mkv/mov/webm), кодек (H.264/H.265/copy), разрешение, CRF, fps, битрейт аудио |
| ✅ | GUI со всеми настройками + headless CLI |
| 🚧 | Озвучка/дубляж (укр. голоса: ukrainian-tts / Piper / Coqui) |
| 🚧 | Удаление вотермарок (нейро-inpainting: LaMa / IOPaint) |
| 🚧 | Удаление вшитых субтитров (OCR + inpainting) |

## Требования

- macOS на Apple Silicon (проверено на M2, 16 ГБ)
- Python 3.11 — ставится через [uv](https://docs.astral.sh/uv/)
- `ffmpeg`/`ffprobe` — статические бинарники Apple Silicon в `bin/`

## Установка

```sh
git clone <repo-url> vidloc && cd vidloc
uv sync          # создаёт .venv и ставит зависимости
```

`ffmpeg`/`ffprobe` в репозиторий не входят (большие, в `.gitignore`) — положи статические
бинарники Apple Silicon в `bin/` или поставь системный ffmpeg (`brew install ffmpeg`).
Модели тоже не входят в git и подтягиваются автоматически при первом запуске (`models/`, ~26 ГБ).

## Запуск

GUI:
```sh
./run.sh
```

CLI (удобно для тестов):
```sh
PYTHONPATH=. .venv/bin/python -m vidloc.cli input.mp4 \
    --source en --mode subtitles --subs soft --asr balanced --mt fast
```
Флаги: `--mode {subtitles,voice,both,none}`, `--subs {soft,burn}`,
`--asr {fast,balanced,best}`, `--mt {fast,best}`, `--beam N`,
`--height`, `--container`, `--codec`, `--crf`.

## Архитектура

> Подробный разбор логики пайплайна, моделей и инженерных решений — в **[ARCHITECTURE.ru.md](ARCHITECTURE.ru.md)**.

```
vidloc/
  settings.py        пути, env, выбор устройства (MPS/CPU)
  models.py          конфиг задачи: enum'ы + dataclass JobConfig
  media/ffmpeg.py    обёртки ffmpeg (probe, extract_audio, render_video)
  asr/               распознавание речи (mlx-whisper)
  translate/         перевод (LLM Aya / NLLB-200)
  subtitles/         генерация .srt / стилизованных .ass
  tts/               озвучка (в разработке)
  pipeline/          оркестрация (runner) + прогресс/отмена (events)
  gui/               PySide6: окно, воркер-поток
  cli.py             headless-запуск
```

Папки: `models/` — кэш моделей, `work/` — временные файлы задачи, `output/` — результаты
(все в `.gitignore`).

## Статус

- **Phase 1 (перевод → субтитры)** — готово и протестировано end-to-end.
- **Phase 1b (озвучка)** и **Phase 2 (удаление объектов)** — в работе.

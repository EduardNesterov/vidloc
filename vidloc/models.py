"""Configuration model: enums + dataclasses describing one localization job.

The GUI binds its controls to a :class:`JobConfig`; the pipeline consumes it.
Every user-facing toggle the spec asked for lives here so the interface and the
engine share one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --------------------------------------------------------------------------- #
# Languages
# --------------------------------------------------------------------------- #
class SourceLang(str, Enum):
    AUTO = "auto"
    EN = "en"
    RU = "ru"
    IT = "it"


TARGET_LANG = "uk"  # the app always translates *into* Ukrainian

# Whisper language id -> NLLB-200 language code
NLLB_LANG = {
    "en": "eng_Latn",
    "ru": "rus_Cyrl",
    "it": "ita_Latn",
    "uk": "ukr_Cyrl",
    # a few neighbours Whisper might detect, mapped so AUTO rarely fails
    "pl": "pol_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
}


# --------------------------------------------------------------------------- #
# Translation
# --------------------------------------------------------------------------- #
class TranslateMode(str, Enum):
    NONE = "none"            # do not translate
    SUBTITLES = "subtitles"  # text only
    VOICE = "voice"          # dubbing only
    BOTH = "both"            # subtitles + dubbing


class SubtitlePlacement(str, Enum):
    SOFT = "soft"   # embedded, switchable track
    BURN = "burn"   # rendered into the picture (styled)


class SubtitlePosition(str, Enum):
    BOTTOM = "bottom"   # lower third
    CENTER = "center"   # middle of the screen


@dataclass
class SubtitleStyle:
    font: str = "Montserrat Black"  # family name; files live in assets/fonts/
    font_size_pct: float = 7.0      # caption height as % of video height
    position: SubtitlePosition = SubtitlePosition.CENTER
    max_chars: int = 28             # short-caption chunk size; 0 = whole segments
    uppercase: bool = False
    primary_color: str = "#FFFFFF"  # text colour (RGB hex)
    outline_color: str = "#000000"  # outline colour (RGB hex)


class ASRQuality(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    BEST = "best"


ASR_MODEL = {
    ASRQuality.FAST: "mlx-community/whisper-medium-mlx",
    ASRQuality.BALANCED: "mlx-community/whisper-large-v3-turbo",
    ASRQuality.BEST: "mlx-community/whisper-large-v3-mlx",
}


class TranslateQuality(str, Enum):
    FAST = "fast"          # NLLB-200 600M — fast, literal
    BEST = "best"          # NLLB-200 1.3B — slower, literal
    NATURAL = "natural"    # local LLM — idiomatic, slang-aware, not word-for-word


# NLLB sizes (literal but fast)
TRANSLATE_MODEL = {
    TranslateQuality.FAST: "facebook/nllb-200-distilled-600M",
    TranslateQuality.BEST: "facebook/nllb-200-distilled-1.3B",
}

# Local instruction-tuned LLM for natural/idiomatic translation (MLX, Apple Silicon).
# Aya-expanse is multilingual-optimized and clearly the strongest at Ukrainian of the
# locally-runnable options (Qwen-7B invented non-words; Aya is fluent and idiomatic).
LLM_TRANSLATE_MODEL = "mlx-community/aya-expanse-8b-4bit"


class TTSEngine(str, Enum):
    UKTTS = "uktts"        # ukrainian-tts (robinhad): 5 natural UA voices + stress
    PIPER = "piper"        # Piper (ONNX): 3 UA voices, different timbre
    COQUI_UK = "coqui_uk"  # Coqui tts_models/uk/mai/vits: single UA voice (fallback)


# Engine -> list of (display label, voice id) for the GUI voice picker.
TTS_VOICES = {
    TTSEngine.UKTTS: [
        ("Тетяна (жіночий)", "tetiana"),
        ("Лада (жіночий)", "lada"),
        ("Микита (чоловічий)", "mykyta"),
        ("Дмитро (чоловічий)", "dmytro"),
        ("Олекса (чоловічий)", "oleksa"),
    ],
    TTSEngine.PIPER: [          # voice id = Piper speaker_id
        ("Лада (жіночий)", "0"),
        ("Микита (чоловічий)", "1"),
        ("Тетяна (жіночий)", "2"),
    ],
    TTSEngine.COQUI_UK: [
        ("Стандартний (жіночий)", "default"),
    ],
}


# --------------------------------------------------------------------------- #
# Output / encoding
# --------------------------------------------------------------------------- #
class VideoCodec(str, Enum):
    H264 = "h264"
    H265 = "h265"
    KEEP = "keep"   # stream-copy when no filtering is required


class Container(str, Enum):
    MP4 = "mp4"
    MKV = "mkv"
    MOV = "mov"
    WEBM = "webm"


@dataclass
class OutputConfig:
    container: Container = Container.MP4
    codec: VideoCodec = VideoCodec.H264
    height: Optional[int] = None     # target height in px; None = keep source
    crf: int = 20                    # 0..51, lower = better quality
    fps: Optional[float] = None      # None = keep source
    audio_bitrate: str = "192k"
    hw_encode: bool = False          # VideoToolbox hardware encoder (fast, slightly lower quality)


class InpaintQuality(str, Enum):
    FAST = "fast"          # ffmpeg delogo — instant, rough
    BALANCED = "balanced"  # neural LaMa at half-res — fast & clean (default)
    BEST = "best"          # neural LaMa at full res — best, slowest


@dataclass
class TimedRegion:
    """A normalized rectangle (x, y, w, h in 0..1) to erase, active only during
    the given time intervals (seconds). Empty interval list = the whole video."""
    rect: tuple
    intervals: list = field(default_factory=list)  # [(start, end), ...] in seconds

    def active_at(self, t: float) -> bool:
        return (not self.intervals) or any(t0 <= t <= t1 for (t0, t1) in self.intervals)


@dataclass
class RemovalConfig:
    remove_watermark: bool = False
    # normalized rectangles (x, y, w, h) in 0..1, relative to frame size
    watermark_regions: list = field(default_factory=list)
    remove_hardsubs: bool = False
    # normalized rectangle for the burned-in subtitle band; None = auto-detect
    hardsub_region: Optional[tuple] = None
    quality: InpaintQuality = InpaintQuality.BALANCED
    # new-style per-region time-gated removal: list[TimedRegion]
    regions: list = field(default_factory=list)

    def timed_regions(self) -> list:
        """Unified TimedRegion list: new-style timed regions plus the legacy
        untimed watermark / hardsub rects (treated as whole-video)."""
        out = [r for r in self.regions if isinstance(r, TimedRegion)]
        if self.remove_watermark:
            out += [TimedRegion(rect=tuple(r)) for r in self.watermark_regions if r]
        if self.remove_hardsubs and self.hardsub_region:
            out.append(TimedRegion(rect=tuple(self.hardsub_region)))
        return out


@dataclass
class TranslateConfig:
    mode: TranslateMode = TranslateMode.SUBTITLES
    source: SourceLang = SourceLang.AUTO
    asr_quality: ASRQuality = ASRQuality.BEST
    translate_quality: TranslateQuality = TranslateQuality.NATURAL
    beam_size: int = 5
    # subtitles
    subtitle_placement: SubtitlePlacement = SubtitlePlacement.SOFT
    subtitle_style: SubtitleStyle = field(default_factory=SubtitleStyle)
    keep_source_subtitles: bool = False
    # dubbing
    tts_engine: TTSEngine = TTSEngine.UKTTS
    tts_voice: str = ""                       # voice id (see TTS_VOICES)
    tts_clone_reference: Optional[str] = None  # reserved for future cloning engines
    keep_background: bool = False              # default: replace original audio (clean voiceover)
    tts_speed: float = 1.0                     # tempo nudge on top of per-line matching


@dataclass
class JobConfig:
    input_path: str = ""
    output_path: str = ""
    source_context: str = ""   # extra context for translation (e.g. URL title/description)
    workspace: str = ""        # per-video cache dir (see vidloc/workspace.py); "" = no cache
    output: OutputConfig = field(default_factory=OutputConfig)
    removal: RemovalConfig = field(default_factory=RemovalConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)

    # convenience flags -------------------------------------------------- #
    @property
    def needs_subtitles(self) -> bool:
        return self.translate.mode in (TranslateMode.SUBTITLES, TranslateMode.BOTH)

    @property
    def needs_dubbing(self) -> bool:
        return self.translate.mode in (TranslateMode.VOICE, TranslateMode.BOTH)

    @property
    def needs_asr(self) -> bool:
        return self.needs_subtitles or self.needs_dubbing

    @property
    def needs_inpaint(self) -> bool:
        return (self.removal.remove_watermark or self.removal.remove_hardsubs
                or bool(self.removal.regions))

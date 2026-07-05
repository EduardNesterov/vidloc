"""Fetch videos from URLs (YouTube / Instagram / TikTok / 1000+ sites) with yt-dlp.

Besides downloading the file, we pull the title / description / tags and turn them
into a compact context string — the single best signal for disambiguating word
senses during translation (e.g. a "$10,000 DRIFT CHALLENGE" title makes "drift"
unambiguous).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .. import settings

ProgressCb = Optional[Callable[[float, str], None]]
LogCb = Optional[Callable[[str], None]]


@dataclass
class FetchResult:
    path: str
    title: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    uploader: str = ""
    webpage_url: str = ""

    def context(self) -> str:
        """Compact context string for the translator (kept short)."""
        parts: List[str] = []
        if self.title:
            parts.append(f"Title: {self.title}")
        if self.uploader:
            parts.append(f"Channel: {self.uploader}")
        if self.tags:
            parts.append("Tags: " + ", ".join(self.tags[:12]))
        if self.description:
            parts.append("Description: " + " ".join(self.description.split())[:400])
        return " | ".join(parts)


def is_url(s: str) -> bool:
    s = (s or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://")


# --------------------------------------------------------------------------- #
# Cookies for auth-gated sites (Instagram et al.)
# --------------------------------------------------------------------------- #
# Hosts that refuse anonymous requests → need a logged-in browser session.
_COOKIE_HOSTS = ("instagram.com", "facebook.com", "fb.watch", "threads.net")
# Auth failures we recognize as "you need cookies" (for a friendly message + retry).
_AUTH_HINTS = ("empty media response", "login required", "log in", "rate-limit",
               "requested content is not available", "cookies", "private")


def _needs_cookies(url: str) -> bool:
    u = (url or "").lower()
    return any(h in u for h in _COOKIE_HOSTS)


def _installed_browsers() -> List[str]:
    """macOS browsers that appear installed, ordered so the ones that don't trigger a
    Keychain prompt (Safari, Firefox) are tried before the Chromium family."""
    home = Path.home()
    support = home / "Library" / "Application Support"
    checks = [
        ("safari", home / "Library" / "Containers" / "com.apple.Safari"),
        ("firefox", support / "Firefox"),
        ("chrome", support / "Google" / "Chrome"),
        ("brave", support / "BraveSoftware" / "Brave-Browser"),
        ("edge", support / "Microsoft Edge"),
        ("chromium", support / "Chromium"),
        ("opera", support / "com.operasoftware.Opera"),
        ("vivaldi", support / "Vivaldi"),
    ]
    return [name for name, p in checks if p.exists()]


def _resolve_cookie_source() -> tuple:
    """(browser, cookies_file) from env (highest priority) then the GUI config. An
    empty / ``auto`` browser means auto-detect installed browsers."""
    from .. import config
    browser = (settings.COOKIES_BROWSER or config.get("cookies_browser")).strip().lower()
    cfile = (settings.COOKIES_FILE or config.get("cookies_file")).strip()
    if browser == "auto":
        browser = ""
    return browser, cfile


def _cookie_attempts(url: str) -> List[dict]:
    """Ordered list of extra yt-dlp options to try. An explicit cookies file / browser
    (from the GUI or env) is used as-is; otherwise we try anonymously first, then fall
    back to browser cookies for cookie-gated hosts."""
    browser, cfile = _resolve_cookie_source()
    if cfile and Path(cfile).exists():
        return [{"cookiefile": cfile}]
    if browser:
        return [{"cookiesfrombrowser": (browser, None, None, None)}]
    attempts: List[dict] = [{}]
    if _needs_cookies(url):
        attempts += [{"cookiesfrombrowser": (br, None, None, None)}
                     for br in _installed_browsers()]
    return attempts


class _QuietLogger:
    """Silences yt-dlp's cookie reader (it logs to stderr by default)."""
    def debug(self, *a, **k): pass
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass


def _has_ig_session(jar) -> bool:
    return any(getattr(c, "domain", "").endswith("instagram.com")
               and c.name == "sessionid" and c.value for c in jar)


def check_cookies(browser: str = "", cookies_file: str = "") -> tuple:
    """Local, no-network probe: can we read cookies from this source, and is there a
    logged-in Instagram session? Returns ``(ok, human_message)``. Reading a Chromium
    browser's cookies may raise a macOS Keychain prompt — call this off the UI thread."""
    if cookies_file:
        if not Path(cookies_file).exists():
            return False, f"Файл cookies не знайдено: {cookies_file}"
        try:
            import http.cookiejar
            jar = http.cookiejar.MozillaCookieJar(cookies_file)
            jar.load(ignore_discard=True, ignore_expires=True)
        except Exception as e:  # noqa: BLE001
            return False, f"Не вдалося прочитати cookies.txt: {e}"
        return ((True, "Файл cookies: сесію Instagram знайдено ✓") if _has_ig_session(jar)
                else (False, "Файл прочитано, але сесії Instagram немає — увійди в акаунт "
                             "і перевстанови файл."))

    from yt_dlp.cookies import extract_cookies_from_browser
    browsers = [browser] if browser and browser != "auto" else _installed_browsers()
    if not browsers:
        return False, "Не знайдено підтримуваних браузерів."
    problems: List[str] = []
    for br in browsers:
        try:
            jar = extract_cookies_from_browser(br, logger=_QuietLogger())
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "operation not permitted" in msg:
                problems.append(f"{br}: немає доступу (Safari потребує «Повний доступ до диска»)")
            elif "could not find" in msg or "does not exist" in msg:
                problems.append(f"{br}: не встановлено / немає профілю")
            else:
                problems.append(f"{br}: {str(e)[:70]}")
            continue
        if _has_ig_session(jar):
            return True, f"{br}: сесію Instagram знайдено ✓"
        problems.append(f"{br}: увійди в Instagram у цьому браузері")
    return False, "Instagram не підключено. " + "; ".join(problems[:4])


def _friendly_error(url: str, err: Optional[BaseException]) -> Exception:
    msg = str(err or "")
    low = msg.lower()
    if _needs_cookies(url) and (any(h in low for h in _AUTH_HINTS)
                                or "operation not permitted" in low
                                or "cookies database" in low):
        return RuntimeError(
            "Instagram вимагає авторизацію. Увійдіть в акаунт Instagram у "
            "Chrome або Firefox — застосунок автоматично візьме куки звідти "
            "(рекомендовано). Safari теж підтримується, але потребує дозволу "
            "«Повний доступ до диска» для терміналу/застосунку. Альтернатива — "
            "змінні середовища VIDLOC_COOKIES_BROWSER=chrome (чи інший браузер) "
            "або VIDLOC_COOKIES_FILE=/шлях/до/cookies.txt.\n\nДеталі: " + msg[:300])
    return err if isinstance(err, Exception) else RuntimeError(msg)


def _run_ydl(base_opts: dict, url: str, *, download: bool = True, on_log: LogCb = None):
    """Run yt-dlp, retrying with browser cookies for auth-gated hosts. Returns
    ``(info, ydl)``; the caller uses ``ydl`` (e.g. ``prepare_filename``) then closes it."""
    import yt_dlp

    last: Optional[BaseException] = None
    for extra in _cookie_attempts(url):
        ydl = yt_dlp.YoutubeDL({**base_opts, **extra})
        try:
            info = ydl.extract_info(url, download=download)
            if on_log:
                if extra.get("cookiesfrombrowser"):
                    on_log(f"Авторизація через куки браузера: {extra['cookiesfrombrowser'][0]}")
                elif extra.get("cookiefile"):
                    on_log("Використано cookies-файл для авторизації.")
            return info, ydl
        except Exception as e:  # noqa: BLE001,PERF203 — any per-source failure → next
            ydl.close()
            last = e
            continue
    raise _friendly_error(url, last)


def _result_from_info(info: Optional[dict], path: str) -> FetchResult:
    if not info:
        return FetchResult(path=path)
    tags = info.get("tags") or info.get("categories") or []
    if not isinstance(tags, list):
        tags = []
    return FetchResult(
        path=path,
        title=info.get("title") or "",
        description=info.get("description") or "",
        tags=[str(t) for t in tags][:20],
        uploader=info.get("uploader") or info.get("channel") or "",
        webpage_url=info.get("webpage_url") or "",
    )


def fetch_info(url: str) -> FetchResult:
    """Metadata only, no download — quick, for previewing the detected context."""
    opts = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    info, ydl = _run_ydl(opts, url, download=False)
    try:
        return _result_from_info(info, path="")
    finally:
        ydl.close()


QUALITY_FORMAT = {
    "best": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
    "1080": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/b",
    "720": "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b",
    "480": "bv*[height<=480][ext=mp4]+ba[ext=m4a]/b[height<=480][ext=mp4]/b",
}


def _make_hook(on_progress: ProgressCb):
    def hook(d):
        if not on_progress:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            got = d.get("downloaded_bytes") or 0
            on_progress(min(0.99, got / total) if total else 0.0, "Завантаження…")
        elif d.get("status") == "finished":
            on_progress(1.0, "Оброблення…")
    return hook


def _final_path(ydl, info: dict, audio_only: bool) -> str:
    rds = info.get("requested_downloads") or []
    if rds and rds[0].get("filepath"):
        p = rds[0]["filepath"]
    else:
        p = ydl.prepare_filename(info)
    if audio_only:
        mp3 = str(Path(p).with_suffix(".mp3"))
        return mp3 if Path(mp3).exists() else p
    if not Path(p).exists():
        mp4 = str(Path(p).with_suffix(".mp4"))
        if Path(mp4).exists():
            return mp4
    return p


def save_media(url: str, dest_dir: str, quality: str = "best", audio_only: bool = False,
               on_progress: ProgressCb = None, on_log: LogCb = None) -> str:
    """Download-only: save the video (or mp3 audio) into ``dest_dir`` at the chosen
    quality. Returns the saved file path."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    opts = {
        "outtmpl": str(dest / "%(title).180B [%(id)s].%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ffmpeg_location": str(settings.BIN_DIR),
        "progress_hooks": [_make_hook(on_progress)],
    }
    if audio_only:
        opts["format"] = "ba/b"
        opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192",
        }]
    else:
        opts["format"] = QUALITY_FORMAT.get(quality, QUALITY_FORMAT["best"])
        opts["merge_output_format"] = "mp4"

    info, ydl = _run_ydl(opts, url, download=True, on_log=on_log)
    try:
        path = _final_path(ydl, info, audio_only)
    finally:
        ydl.close()
    if on_log:
        on_log(f"Збережено: {path}")
    return path


def download(url: str, on_progress: ProgressCb = None, on_log: LogCb = None) -> FetchResult:
    """Download best mp4 into work/downloads and return its path + metadata."""
    outdir = Path(settings.WORK_DIR) / "downloads"
    outdir.mkdir(parents=True, exist_ok=True)

    opts = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(outdir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ffmpeg_location": str(settings.BIN_DIR),  # use our bundled ffmpeg to merge
        "progress_hooks": [_make_hook(on_progress)],
    }
    info, ydl = _run_ydl(opts, url, download=True, on_log=on_log)
    try:
        path = ydl.prepare_filename(info)
        if not Path(path).exists():
            alt = Path(path).with_suffix(".mp4")
            if alt.exists():
                path = str(alt)
    finally:
        ydl.close()
    if on_log:
        on_log(f"Завантажено: {path}")
    return _result_from_info(info, path=str(path))

"""Stage 1: download the video (<=1080p) and, if available, the chat replay."""
from __future__ import annotations

import json
import re
from pathlib import Path

import yt_dlp
from yt_dlp.utils import DownloadCancelled, DownloadError, ExtractorError, download_range_func

from .errors import BadLinkError, Cancelled, ClipperError, DiskSpaceError
from .events import Reporter
from .tools import check_disk, ffmpeg_dir, is_disk_full, js_runtimes

URL_RE = re.compile(r"^https?://\S+$", re.I)
MAX_HOURS_WARN = 4.1
FORMAT = ("bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba"
          "/b[height<=1080]/bv*+ba/b")


class _Logger:
    def __init__(self, rep: Reporter):
        self.rep = rep

    def debug(self, msg):  # yt-dlp routes info to debug too
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        if "nsig" in msg or "Falling back" in msg:
            return
        self.rep.log(f"yt-dlp warning: {msg}")

    def error(self, msg):
        pass  # surfaced via the raised DownloadError


def _base_opts(rep: Reporter) -> dict:
    return {
        "quiet": True,
        "no_warnings": False,
        "noprogress": True,
        "noplaylist": True,
        "logger": _Logger(rep),
        "ffmpeg_location": ffmpeg_dir(),
        "js_runtimes": js_runtimes(),
        "retries": 5,
        "fragment_retries": 10,
        "socket_timeout": 30,
    }


def _clean_error(exc: Exception) -> str:
    msg = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))
    return msg.replace("ERROR: ", "").strip()


def _bad_link(exc: Exception) -> BadLinkError:
    msg = _clean_error(exc)
    low = msg.lower()
    if "unsupported url" in low or "is not a valid url" in low:
        return BadLinkError("That link isn't a video page Clipper can read.", msg)
    if "private" in low:
        return BadLinkError("That video is private.", msg)
    if "sign in" in low or "age" in low and "confirm" in low:
        return BadLinkError("That video needs a sign-in (age-restricted or members-only).", msg)
    if "unavailable" in low or "does not exist" in low or "404" in low:
        return BadLinkError("That video is unavailable or was removed.", msg)
    if "getaddrinfo" in low or "timed out" in low or "network" in low:
        return BadLinkError("Couldn't reach the site. Check your internet connection.", msg)
    return BadLinkError("Couldn't open that link.", msg)


def validate_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise BadLinkError("Paste a YouTube or Twitch video link first.")
    if not URL_RE.match(url):
        raise BadLinkError(f"'{url[:80]}' doesn't look like a link.", "It should start with https://")
    return url


def probe(url: str, rep: Reporter) -> dict:
    """Read metadata without downloading. Raises BadLinkError."""
    url = validate_url(url)
    opts = _base_opts(rep) | {"format": FORMAT, "skip_download": True, "extract_flat": "in_playlist",
                              "playlist_items": "1"}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except (DownloadError, ExtractorError) as e:
        raise _bad_link(e) from e
    if not info:
        raise BadLinkError("Couldn't read that link.")
    if info.get("_type") == "playlist":
        raise BadLinkError("That's a playlist or channel. Paste a single video link.")
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        raise BadLinkError("That stream is live or hasn't started. Wait for the VOD and try again.")
    return info


def estimate_size(info: dict) -> int:
    total = 0
    for f in info.get("requested_formats") or [info]:
        total += f.get("filesize") or f.get("filesize_approx") or 0
    if not total:
        total = int((info.get("duration") or 3600) * 600_000)  # ~4.8 Mbit/s
    return int(total)


def download_video(info: dict, workdir: Path, rep: Reporter) -> Path:
    """Download the selected formats into workdir/source.mp4."""
    workdir.mkdir(parents=True, exist_ok=True)
    size = estimate_size(info)
    check_disk(workdir, int(size * 2.2) + 1_000_000_000, "the download")

    n_parts = max(1, len(info.get("requested_formats") or []))
    seen: list[str] = []

    def hook(d):
        if rep.cancelled:
            raise DownloadCancelled()
        if d.get("status") == "downloading":
            fn = d.get("filename", "")
            if fn not in seen:
                seen.append(fn)
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            frac = (d.get("downloaded_bytes", 0) / total) if total else 0.0
            idx = seen.index(fn)
            rep.progress("download", min(0.99, (idx + frac) / n_parts))

    opts = _base_opts(rep) | {
        "format": FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": str(workdir / "source.%(ext)s"),
        "progress_hooks": [hook],
        "concurrent_fragment_downloads": 4,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.process_ie_result(dict(info), download=True)
    except DownloadCancelled as e:
        raise Cancelled() from e
    except (DownloadError, ExtractorError) as e:
        cause = e.__cause__ or getattr(e, "exc_info", (None, None))[1]
        if (cause is not None and is_disk_full(cause)) or \
                "No space left" in str(e) or "not enough space" in str(e):
            raise DiskSpaceError("The disk filled up during the download.") from e
        raise _bad_link(e) from e
    except OSError as e:
        if is_disk_full(e):
            raise DiskSpaceError("The disk filled up during the download.") from e
        raise

    vids = sorted(p for p in workdir.glob("source.*") if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov"))
    if not vids:
        raise ClipperError("The download finished but no video file was produced.")
    rep.progress("download", 1.0)
    return vids[0]


# ---------------------------------------------------------------- chat replay

def chat_available(info: dict) -> str | None:
    subs = info.get("subtitles") or {}
    for key in ("live_chat", "rechat"):
        if key in subs:
            return key
    return None


def download_chat(info: dict, workdir: Path, rep: Reporter) -> list[float] | None:
    """Return message offsets in seconds, or None if no chat replay is available."""
    key = chat_available(info)
    if not key:
        rep.log("No chat replay available for this video; using audio and transcript only.")
        return None
    rep.log("Downloading chat replay...")

    def hook(d):
        if rep.cancelled:
            raise DownloadCancelled()

    opts = _base_opts(rep) | {
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": [key],
        "outtmpl": str(workdir / "chat.%(ext)s"),
        "progress_hooks": [hook],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.process_ie_result(dict(info), download=True)
    except DownloadCancelled as e:
        raise Cancelled() from e
    except Exception as e:  # chat is a bonus; never fail the run over it
        rep.log(f"Chat replay couldn't be downloaded ({_clean_error(e)[:160]}); continuing without it.")
        return None

    files = list(workdir.glob("chat*.json")) + list(workdir.glob("chat*.json3"))
    if not files:
        rep.log("Chat replay produced no file; continuing without it.")
        return None
    times = parse_chat(files[0])
    rep.log(f"Chat replay: {len(times):,} messages.")
    return times or None


def parse_chat(path: Path) -> list[float]:
    times: list[float] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    # YouTube live_chat: one JSON object per line.
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        act = obj.get("replayChatItemAction")
        if act:
            try:
                t = int(act.get("videoOffsetTimeMsec", 0)) / 1000.0
            except (TypeError, ValueError):
                continue
            n = sum(1 for a in act.get("actions", []) if "addChatItemAction" in a)
            times.extend([t] * n)
    if times:
        return sorted(times)
    # Twitch rechat style: {"comments": [{"content_offset_seconds": ...}]}
    try:
        obj = json.loads(text)
        for c in obj.get("comments", []) if isinstance(obj, dict) else obj:
            t = c.get("content_offset_seconds")
            if t is not None:
                times.append(float(t))
    except (ValueError, AttributeError, TypeError):
        pass
    return sorted(times)


# ---------------------------------------------------------- section download

def download_section(url: str, start: float, end: float, dest_dir: Path, rep: Reporter) -> Path:
    """Fetch only [start, end] of a video (used to re-render after the source was deleted)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = f"section_{int(start)}_{int(end)}"
    for p in dest_dir.glob(stem + ".*"):
        if p.suffix.lower() in (".mp4", ".mkv", ".webm"):
            return p
    opts = _base_opts(rep) | {
        "format": FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / (stem + ".%(ext)s")),
        "download_ranges": download_range_func(None, [(max(0.0, start), end)]),
        "force_keyframes_at_cuts": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(validate_url(url), download=True)
    except (DownloadError, ExtractorError) as e:
        raise _bad_link(e) from e
    for p in dest_dir.glob(stem + ".*"):
        if p.suffix.lower() in (".mp4", ".mkv", ".webm"):
            return p
    raise ClipperError("Couldn't fetch that part of the video for re-rendering.")

"""Locate external tools (ffmpeg, claude, node) and check disk space."""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from .errors import ClaudeNotFoundError, DiskSpaceError, ToolMissingError
from .paths import INSTALL_DIR

# Hide console windows for child processes when running as a GUI exe.
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

LOCAL = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
WINGET_LINKS = LOCAL / "Microsoft" / "WinGet" / "Links"
WINGET_PKGS = LOCAL / "Microsoft" / "WinGet" / "Packages"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    kw.setdefault("encoding", "utf-8")
    kw.setdefault("errors", "replace")
    return subprocess.run(cmd, creationflags=NO_WINDOW, **kw)


def _first_existing(paths) -> str | None:
    for p in paths:
        if p and Path(p).is_file():
            return str(p)
    return None


@lru_cache(maxsize=None)
def find_ffmpeg(name: str = "ffmpeg") -> str:
    exe = f"{name}.exe" if sys.platform == "win32" else name
    found = _first_existing([
        INSTALL_DIR / "ffmpeg" / exe,
        INSTALL_DIR / exe,
        shutil.which(name),
        WINGET_LINKS / exe,
        *sorted(glob.glob(str(WINGET_PKGS / "Gyan.FFmpeg*" / "*" / "bin" / exe)), reverse=True),
        *sorted(glob.glob(str(WINGET_PKGS / "*FFmpeg*" / "*" / "bin" / exe)), reverse=True),
    ])
    if not found:
        raise ToolMissingError(
            f"{name} was not found.",
            "Install it with:  winget install Gyan.FFmpeg   then restart Clipper.")
    # Some yt-dlp paths (partial downloads) only look on PATH, so put ours there.
    d = str(Path(found).parent)
    if d.lower() not in os.environ.get("PATH", "").lower().split(os.pathsep):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    return found


def ffmpeg_dir() -> str:
    return str(Path(find_ffmpeg()).parent)


@lru_cache(maxsize=None)
def find_claude() -> str:
    home = Path.home()
    candidates = [
        os.environ.get("CLIPPER_CLAUDE"),
        shutil.which("claude"),
        WINGET_LINKS / "claude.exe",
        *glob.glob(str(WINGET_PKGS / "Anthropic.ClaudeCode*" / "claude.exe")),
        home / ".local" / "bin" / "claude.exe",
        home / ".claude" / "local" / "claude.exe",
        Path(os.environ.get("APPDATA", "")) / "npm" / "claude.cmd",
    ]
    # Last resort: the copy bundled with the Claude desktop app (newest version).
    bundled = glob.glob(str(LOCAL / "Claude-Data" / "*" / "claude-code" / "*" / "*" / "claude.exe"))
    bundled.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    found = _first_existing([*candidates, *bundled])
    if not found:
        raise ClaudeNotFoundError(
            "The Claude Code CLI ('claude') was not found.",
            "Install it with:  winget install Anthropic.ClaudeCode   then run 'claude' once and log in.")
    return found


@lru_cache(maxsize=None)
def find_node() -> str | None:
    return shutil.which("node") or shutil.which("deno")


def js_runtimes() -> dict:
    """yt-dlp needs a JS runtime for YouTube's challenges; offer whatever is installed."""
    rt: dict = {}
    if shutil.which("deno"):
        rt["deno"] = {}
    if shutil.which("node"):
        rt["node"] = {}
    return rt or {"deno": {}}


def check_disk(path: Path, need_bytes: int, what: str) -> None:
    path = Path(path)
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < need_bytes:
        raise DiskSpaceError(
            f"Not enough disk space for {what} on {probe.anchor or probe}: "
            f"{free / 1e9:.1f} GB free, about {need_bytes / 1e9:.1f} GB needed.",
            "Free up space or pick an output folder on another drive.")


def is_disk_full(exc: BaseException) -> bool:
    import errno
    return isinstance(exc, OSError) and (exc.errno == errno.ENOSPC or getattr(exc, "winerror", 0) in (112, 39))


def media_duration(path: Path) -> float:
    r = run([find_ffmpeg("ffprobe"), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)])
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0

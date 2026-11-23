"""Where things live: app data, models, profiles, temp and output."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)

# Folder holding the exe (frozen) or the repo root (source checkout).
INSTALL_DIR = Path(sys.executable).parent if FROZEN else Path(__file__).resolve().parent.parent

# Old location (on C:). Anything found there is moved to APP_DIR on start-up.
LEGACY_APP_DIR = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Clipper"


def _writable(d: Path) -> bool:
    """Can we write in d? Each process uses its own probe file: two Clipper processes starting at
    the same moment used to collide on a shared one, and one of them then fell back to C:."""
    import uuid
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / f".write_test_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        probe.write_text("ok")
        try:
            probe.unlink()
        except OSError:
            pass  # wrote fine; a leftover probe file is harmless
        return True
    except OSError:
        return False


def _data_dir() -> Path:
    r"""Everything Clipper stores (models, temp downloads, settings) lives next to the app.

    Source checkout D:\Clipper -> D:\Clipper\data. Built exe in D:\Clipper\dist\Clipper -> the same
    D:\Clipper\data (dist is wiped on every build). Elsewhere: <exe folder>\data. Override with the
    CLIPPER_DATA environment variable. Falls back to %APPDATA%\Clipper only if that isn't writable.
    """
    env = os.environ.get("CLIPPER_DATA")
    if env:
        return Path(env)
    root = INSTALL_DIR
    if FROZEN and INSTALL_DIR.parent.name.lower() == "dist":
        root = INSTALL_DIR.parent.parent
    d = root / "data"
    if d.is_dir() or _writable(d):
        return d
    return LEGACY_APP_DIR


APP_DIR = _data_dir()
MODELS_DIR = APP_DIR / "models"
CACHE_DIR = APP_DIR / "cache"
TMP_DIR = APP_DIR / "tmp"
SETTINGS_FILE = APP_DIR / "settings.json"
USER_PROFILES_DIR = APP_DIR / "profiles"
BUNDLED_PROFILES_DIR = INSTALL_DIR / "profiles"
TEMP_ROOT = APP_DIR / "temp"

DEFAULT_SETTINGS = {
    "output_dir": r"D:\Clipper\output",
    "whisper_model": "",          # empty = use the profile's model
    "keep_source": False,         # keep the downloaded video for fast re-renders
    "claude_model": "",           # empty = CLI default
    "max_clips": 20,
    "extras": False,              # teaser, context banner, callouts, end card
    "extras_end_card": True,      # include the thumbnail end card in extras
    "overlay_name": "",           # name that must appear on screen in every clip (campaign rule)
}


_dirs_ready = False


def ensure_dirs() -> None:
    global _dirs_ready
    for d in (APP_DIR, MODELS_DIR, USER_PROFILES_DIR, TEMP_ROOT, CACHE_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if not _dirs_ready:
        _dirs_ready = True
        # Keep libraries' own caches and temp files off C: too.
        os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
        os.environ["TMP"] = os.environ["TEMP"] = str(TMP_DIR)
        import tempfile
        tempfile.tempdir = str(TMP_DIR)
        migrate_legacy()


def _legacy_dirs() -> list[Path]:
    dirs = [LEGACY_APP_DIR]
    local = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    # Copies made while running inside the Claude desktop app's sandbox.
    dirs += list((local / "Packages").glob("Claude_*/LocalCache/Roaming/Clipper"))
    return [d for d in dirs if d.is_dir() and d.resolve() != APP_DIR.resolve()]


def migrate_legacy(log=None) -> None:
    """Move models, settings, profiles and kept sources from the old C: location, then remove it."""
    import shutil
    for old in _legacy_dirs():
        if not any((old / n).exists() for n in ("models", "temp", "settings.json")):
            continue  # not ours
        try:
            for name in ("models", "profiles", "sources"):
                src = old / name
                if not src.is_dir():
                    continue
                for item in src.rglob("*"):
                    if item.is_file() and not item.name.endswith(".part"):
                        dest = APP_DIR / name / item.relative_to(src)
                        if not dest.exists():
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, dest)
            if (old / "settings.json").exists() and not SETTINGS_FILE.exists():
                shutil.copy2(old / "settings.json", SETTINGS_FILE)
        except OSError as e:
            if log:
                log(f"Couldn't move {old} to {APP_DIR} ({e}); left it in place.")
            continue  # keep the originals if anything failed to copy
        shutil.rmtree(old, ignore_errors=True)  # temp downloads, logs, and the moved files
        if log:
            log(f"Moved Clipper's data from {old} to {APP_DIR}.")


def prune_temp(max_age_days: float = 2.0) -> None:
    """Drop cached re-render sections and leftover render folders older than max_age_days.

    Unfinished job folders (which allow resuming) are kept for a week.
    """
    import shutil
    import time
    if not TEMP_ROOT.is_dir():
        return
    now = time.time()
    for d in TEMP_ROOT.iterdir():
        try:
            age = (now - d.stat().st_mtime) / 86400
        except OSError:
            continue
        limit = max_age_days if d.name.startswith(("sections_", "render_")) else 7.0
        if age > limit:
            shutil.rmtree(d, ignore_errors=True)


def load_settings() -> dict:
    ensure_dirs()
    data = dict(DEFAULT_SETTINGS)
    try:
        data.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return data


def save_settings(settings: dict) -> None:
    ensure_dirs()
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")


_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(text: str, max_len: int = 80) -> str:
    """Windows-safe file/folder name."""
    text = _BAD_CHARS.sub("", text or "").strip()
    text = re.sub(r"\s+", " ", text).strip(" .")
    if len(text) > max_len:
        text = text[:max_len].rstrip(" .")
    return text or "untitled"


def slug(text: str, max_len: int = 40) -> str:
    text = safe_name(text, 200).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return (text[:max_len].rstrip("_")) or "clip"

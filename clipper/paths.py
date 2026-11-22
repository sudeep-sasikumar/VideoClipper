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

APP_DIR = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Clipper"
MODELS_DIR = APP_DIR / "models"
SETTINGS_FILE = APP_DIR / "settings.json"
USER_PROFILES_DIR = APP_DIR / "profiles"
BUNDLED_PROFILES_DIR = INSTALL_DIR / "profiles"
TEMP_ROOT = APP_DIR / "temp"

DEFAULT_SETTINGS = {
    "output_dir": str(Path.home() / "Videos" / "Clipper"),
    "whisper_model": "",          # empty = use the profile's model
    "keep_source": False,         # keep the downloaded video for fast re-renders
    "claude_model": "",           # empty = CLI default
    "max_clips": 20,
}


def ensure_dirs() -> None:
    for d in (APP_DIR, MODELS_DIR, USER_PROFILES_DIR, TEMP_ROOT):
        d.mkdir(parents=True, exist_ok=True)


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

"""Niche profiles: JSON files in /profiles (next to the app) or %APPDATA%\\Clipper\\profiles.

Copy any file, rename it, edit it, and it shows up in the dropdown. Missing keys fall
back to gaming.json, so a profile can be as small as {"name": "...", "highlight_criteria": "..."}.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .paths import BUNDLED_PROFILES_DIR, USER_PROFILES_DIR

FALLBACK = {
    "name": "Gaming",
    "description": "",
    "highlight_criteria": "The most entertaining, surprising or emotional moments.",
    "candidates_per_hour": 6,
    "clip_length": {"min": 20, "max": 60, "target": 35},
    "whisper_model": "small",
    "language": None,
    "vad_filter": False,
    "layout": {"mode": "blur", "facecam": {"x": 0.0, "y": 0.0, "w": 0.3, "h": 0.3},
               "split_top_ratio": 0.38, "foreground_zoom": 1.0},
    "captions": {"enabled": True, "font": "Arial Black", "size": 84, "color": "#FFFFFF",
                 "highlight": "#FFE21A", "outline": "#000000", "outline_width": 7, "shadow": 2,
                 "uppercase": True, "words_per_line": 3, "position_y": 0.72},
    "hook": {"enabled": True, "font": "Arial Black", "size": 72, "color": "#FFFFFF",
             "box": "#E3122C", "duration": 3.0, "position_y": 0.14},
    "zoom": {"enabled": True, "amount": 0.08, "max_punches": 6, "length": 0.7},
    "voiceover": {"enabled": False, "voice": "am_michael", "speed": 1.05, "volume": 1.0, "duck": 0.35},
}


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def load_all(log=None) -> dict[str, dict]:
    """{profile_id: profile}. profile_id is the file stem. User files override bundled ones."""
    base = _read(BUNDLED_PROFILES_DIR / "gaming.json") or {}
    base = _deep_merge(FALLBACK, base)
    found: dict[str, dict] = {}
    for d in (BUNDLED_PROFILES_DIR, USER_PROFILES_DIR):
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            data = _read(p)
            if data is None:
                if log:
                    log(f"Skipping profile {p.name}: not valid JSON.")
                continue
            prof = _deep_merge(base, data)
            prof.setdefault("name", p.stem.title())
            if "name" not in data:
                prof["name"] = p.stem.replace("_", " ").title()
            prof["_id"] = p.stem
            prof["_path"] = str(p)
            found[p.stem] = prof
    if not found:
        found["gaming"] = dict(base, _id="gaming", _path="")
    return found


def default_id(profiles: dict[str, dict]) -> str:
    if "gaming" in profiles:
        return "gaming"
    for pid, p in profiles.items():
        if p.get("default"):
            return pid
    return next(iter(profiles))

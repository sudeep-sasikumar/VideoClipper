"""Timestamp helpers."""
from __future__ import annotations

import re


def fmt_ts(sec: float, force_hours: bool = False) -> str:
    sec = max(0.0, float(sec))
    whole = int(sec)
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    if h or force_hours:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def fmt_precise(sec: float) -> str:
    sec = max(0.0, float(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


_TS = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s*$")


def parse_ts(v) -> float | None:
    """Accept 75, 75.5, "75", "1:15", "0:01:15", "01:15.5". None if unparseable."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower().rstrip("s").strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = _TS.match(s)
    if not m:
        return None
    h, mi, se = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + float(se)

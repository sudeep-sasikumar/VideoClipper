"""Find clip-length limits in free-text campaign rules and combine them with the profile."""
from __future__ import annotations

import re

_NUM = r"(\d+(?:\.\d+)?)"
_UNIT = r"(s|sec|secs|second|seconds|m|min|mins|minute|minutes)\b"
_DUR = rf"(?:(\d+):(\d{{2}})|{_NUM}\s*-?\s*{_UNIT})"

_MAX_WORDS = (r"max(?:imum)?|at most|up to|no (?:longer|more) than|not (?:longer|more) than|under|"
              r"less than|shorter than|below|within|cap(?:ped)? at|limit(?:ed)? to|<=?|≤")
_MIN_WORDS = (r"min(?:imum)?|at least|no (?:shorter|less) than|not (?:shorter|less) than|"
              r"longer than|more than|over|above|>=?|≥")
_LEN_WORDS = r"(?:length|long|duration|clips?|videos?|runtime)"


def _seconds(mm, ss, num, unit) -> float:
    if mm is not None:
        return int(mm) * 60 + int(ss)
    v = float(num)
    return v * 60 if unit.lower().startswith("m") else v


def parse_length_limits(text: str) -> dict:
    """Return {"min": seconds|None, "max": seconds|None, "found": [matched phrases]}."""
    out = {"min": None, "max": None, "found": []}
    if not text:
        return out
    t = text.lower().replace("–", "-").replace("—", "-")

    # Ranges: "between 15 and 45 seconds", "15-45 sec", "15 to 45s", "30s - 1:30", "0:30-1:00"
    rng = re.compile(
        rf"(?:between\s+)?(?:(\d+):(\d{{2}})|{_NUM})\s*(?:{_UNIT})?\s*(?:-|to|and)\s*{_DUR}")
    for m in rng.finditer(t):
        a_mm, a_ss, a_num, a_unit, b_mm, b_ss, b_num, b_unit = m.groups()
        # Context must be about length, or both sides must have units/colons.
        ctx = t[max(0, m.start() - 40): m.end() + 20]
        if not (a_mm or a_unit or b_mm or b_unit or re.search(_LEN_WORDS, ctx)):
            continue
        b = _seconds(b_mm, b_ss, b_num, b_unit or "s")
        unit_a = a_unit or (b_unit if not b_mm else "s")
        a = _seconds(a_mm, a_ss, a_num, unit_a)
        if 0 < a < b <= 600:
            out["min"] = a if out["min"] is None else max(out["min"], a)
            out["max"] = b if out["max"] is None else min(out["max"], b)
            out["found"].append(m.group(0).strip())

    for words, key in ((_MAX_WORDS, "max"), (_MIN_WORDS, "min")):
        pat = re.compile(rf"(?:{words})\s*(?:(?:of|a|the|clip|video|length|duration|is|:)\s*)*{_DUR}")
        for m in pat.finditer(t):
            mm, ss, num, unit = m.groups()
            v = _seconds(mm, ss, num, unit)
            if not 0 < v <= 600:
                continue
            ctx = t[max(0, m.start() - 50): m.end() + 30]
            if re.search(r"\b(?:post|upload|views?|followers?|days?|hours?|watch(?:ed)?)\b", ctx) and \
                    not re.search(_LEN_WORDS, ctx):
                continue
            if key == "max":
                out["max"] = v if out["max"] is None else min(out["max"], v)
            else:
                out["min"] = v if out["min"] is None else max(out["min"], v)
            out["found"].append(m.group(0).strip())

    if out["min"] and out["max"] and out["min"] >= out["max"]:
        out["min"] = None  # contradictory; the cap is the safer one to keep
    return out


def effective_length(profile: dict, rules_text: str) -> dict:
    """Profile clip length, overridden by any limits found in the campaign rules."""
    cl = profile.get("clip_length", {})
    lo, hi = float(cl.get("min", 20)), float(cl.get("max", 60))
    target = float(cl.get("target", (lo + hi) / 2))
    r = parse_length_limits(rules_text)
    if r["min"] is not None:
        lo = r["min"]
        if r["max"] is None:
            hi = max(hi, lo + 10)
    if r["max"] is not None:
        hi = r["max"]
        if r["min"] is None:
            lo = min(lo, max(5.0, hi * 0.6))
    if lo >= hi:
        lo = max(3.0, hi - 5)
    target = min(max(target, lo), hi)
    return {"min": lo, "max": hi, "target": target, "from_rules": r["found"]}

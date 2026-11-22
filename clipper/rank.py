"""Stage 4: ask Claude for the best moments, hour by hour, then merge into a top list."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import claude_cli
from .errors import ClaudeOutputError, ClipperError
from .events import Reporter
from .timefmt import fmt_ts, parse_ts

CHUNK = 3600.0
MERGE_TAIL = 900.0  # a last chunk shorter than 15 min joins the previous one


def make_chunks(duration: float) -> list[tuple[float, float]]:
    bounds = []
    t = 0.0
    while t < duration:
        bounds.append((t, min(duration, t + CHUNK)))
        t += CHUNK
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < MERGE_TAIL:
        last = bounds.pop()
        bounds[-1] = (bounds[-1][0], last[1])
    return bounds or [(0.0, duration)]


def _transcript_text(transcript: dict, a: float, b: float, hours: bool) -> str:
    lines = []
    for s in transcript["segments"]:
        if s["end"] < a or s["start"] > b:
            continue
        lines.append(f"[{fmt_ts(s['start'], hours)}] {s['text']}")
    return "\n".join(lines)


def _signals_text(signals: dict, a: float, b: float, hours: bool) -> str:
    out = []
    loud = sorted((x for x in signals.get("loudness", []) if a <= x["t"] <= b), key=lambda x: -x["z"])[:25]
    if loud:
        out.append("Audio loudness spikes (time, strength; screaming/hype/explosions often show here):")
        out.append(", ".join(f"{fmt_ts(x['t'], hours)} ({x['z']:.1f})" for x in sorted(loud, key=lambda x: x["t"])))
    chat = sorted((x for x in signals.get("chat", []) if a <= x["t"] <= b), key=lambda x: -x["z"])[:15]
    if chat:
        out.append("Chat message-rate spikes (time, strength, msgs per 10 s). Chat reacts 5-20 s AFTER the moment:")
        out.append(", ".join(f"{fmt_ts(x['t'], hours)} ({x['z']:.1f}, {x['rate']})" for x in sorted(chat, key=lambda x: x["t"])))
    elif signals.get("has_chat"):
        out.append("Chat replay was available but showed no clear spikes in this part.")
    return "\n".join(out) or "No audio or chat signals stood out in this part."


def build_prompt(*, title: str, profile: dict, length: dict, rules: str, transcript: dict, signals: dict,
                 a: float, b: float, n: int, hours: bool, part: int, parts: int) -> str:
    rules_block = rules.strip() or "(none)"
    return f"""Find the {n} best short-form clips (TikTok / Reels / Shorts) in this part of a video.

VIDEO: "{title}"
PART {part} of {parts}: {fmt_ts(a, hours)} to {fmt_ts(b, hours)}

WHAT MAKES A GOOD CLIP FOR THIS NICHE ({profile.get('name')}):
{profile.get('highlight_criteria')}

CAMPAIGN RULES (follow them strictly; they override the niche guidance):
{rules_block}

CLIP LENGTH: between {length['min']:.0f} and {length['max']:.0f} seconds, ideally about {length['target']:.0f} s.
- Start a moment before the action or the first line of the setup, so it hooks in the first second.
- End right after the payoff (the reaction, punchline or result). Never cut mid-sentence.
- Clips must not overlap each other. Spread across the part only if quality allows; quality first.

SIGNALS:
{_signals_text(signals, a, b, hours)}

TRANSCRIPT (timestamps are the segment start times):
{_transcript_text(transcript, a, b, hours)}

Return ONLY a JSON array of exactly {n} objects (fewer only if the part truly has nothing usable), best first:
[
  {{
    "start": "{'H:MM:SS' if hours else 'MM:SS'}",
    "end": "{'H:MM:SS' if hours else 'MM:SS'}",
    "score": 1-10 integer (10 = would go viral, 5 = decent, 1 = filler),
    "hook_text": "on-screen hook for the first 3 s, max 8 words, punchy, no emojis, no hashtags",
    "title": "3-6 word title for the file name",
    "reason": "one sentence: why this works as a clip",
    "intro_line": "a natural spoken intro line for a voiceover, max 15 words, that teases the clip without spoiling it"
  }}
]"""


def _validate(items, a: float, b: float, duration: float, length: dict) -> list[dict]:
    if isinstance(items, dict):
        for k in ("clips", "candidates", "results", "items"):
            if isinstance(items.get(k), list):
                items = items[k]
                break
        else:
            items = [items]
    if not isinstance(items, list):
        raise ClaudeOutputError("Claude's JSON wasn't a list of clips.")
    good = []
    for it in items:
        if not isinstance(it, dict):
            continue
        s, e = parse_ts(it.get("start")), parse_ts(it.get("end"))
        if s is None or e is None or e <= s:
            continue
        if s < a - 60 or s > b + 30 or e > duration + 1:
            continue
        try:
            score = float(it.get("score", 0))
        except (TypeError, ValueError):
            continue
        score = max(1.0, min(10.0, score))
        hook = str(it.get("hook_text") or "").strip().strip('"')
        good.append({
            "start": max(0.0, s), "end": min(duration, e), "score": score,
            "hook_text": " ".join(hook.split()[:12]),
            "title": str(it.get("title") or hook or "clip").strip()[:80],
            "reason": str(it.get("reason") or "").strip(),
            "intro_line": str(it.get("intro_line") or "").strip(),
        })
    return good


def fit_to_words(c: dict, words: list[list], length: dict, duration: float) -> dict:
    """Clamp length to the limits and snap edges to word boundaries."""
    lo, hi = length["min"], length["max"]
    s, e = c["start"], c["end"]
    if e - s > hi:
        e = s + hi
    if e - s < lo:
        need = lo - (e - s)
        e = min(duration, e + need)
        if e - s < lo:
            s = max(0.0, e - lo)
    # Snap start back to the beginning of the word being spoken at s (or just before).
    for ws, we, _w in words:
        if we >= s:
            if ws < s and s - ws < 1.0:
                s = ws
            break
    s = max(0.0, s - 0.15)
    # Snap end forward to finish the word in progress, plus a short tail.
    for ws, we, _w in words:
        if ws <= e <= we + 0.05:
            e = we
            break
    e = min(duration, e + 0.35)
    if e - s > hi:  # snapping pushed it over the cap
        cap = s + hi
        e = cap
        for ws, we, _w in words:  # don't stop mid-word if cutting before that word still fits
            if ws < cap < we and ws - s >= lo:
                e = ws
                break
    # Hard guarantee: inside [lo, hi] whatever the snapping did.
    if e - s < lo:
        e = min(duration, s + lo)
        if e - s < lo:
            s = max(0.0, e - lo)
    if e - s > hi:
        e = s + hi
    c = dict(c, start=round(s, 2), end=round(e, 2))
    return c


def _overlap(a: dict, b: dict) -> float:
    inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
    if inter <= 0:
        return 0.0
    return inter / min(a["end"] - a["start"], b["end"] - b["start"])


def merge(cands: list[dict], max_clips: int) -> list[dict]:
    kept: list[dict] = []
    for c in sorted(cands, key=lambda x: (-x["score"], x["start"])):
        if any(_overlap(c, k) > 0.3 or abs(c["start"] - k["start"]) < 8 for k in kept):
            continue
        kept.append(c)
        if len(kept) >= max_clips:
            break
    return kept


def rank(*, title: str, transcript: dict, signals: dict, profile: dict, length: dict, rules: str,
         duration: float, rep: Reporter, cache: Path | None = None, max_clips: int = 20,
         model: str = "") -> list[dict]:
    if cache and cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("rules") == rules and data.get("profile") == profile.get("_id") and data.get("clips"):
                rep.log("Reusing ranking from an earlier run.")
                rep.progress("rank", 1.0)
                return data["clips"]
        except ValueError:
            pass

    words = [w for s in transcript["segments"] for w in s["words"]]
    if len(words) < 20:
        raise ClipperError("The video has almost no speech, so there's nothing to rank.",
                           "Clipper needs talking/commentary to find moments.")
    chunks = make_chunks(duration)
    hours = duration >= 3600
    cph = int(profile.get("candidates_per_hour", 3))
    rep.log(f"Asking Claude for highlights in {len(chunks)} part(s)...")

    def do_chunk(i: int, a: float, b: float) -> list[dict]:
        n = max(cph, round(cph * (b - a) / CHUNK))
        prompt = build_prompt(title=title, profile=profile, length=length, rules=rules,
                              transcript=transcript, signals=signals, a=a, b=b, n=n,
                              hours=hours, part=i + 1, parts=len(chunks))
        last_err: Exception | None = None
        for attempt in range(2):
            rep.check()
            p = prompt if attempt == 0 else (
                prompt + "\n\nIMPORTANT: your previous answer could not be parsed. Reply with the JSON "
                         "array only, starting with [ and ending with ]. Use the timestamp format shown.")
            try:
                text = claude_cli.ask(p, model=model)
                found = _validate(claude_cli.extract_json(text), a, b, duration, length)
                if found:
                    return found
                last_err = ClaudeOutputError("Claude's answer had no usable clips.", text[:300])
            except ClaudeOutputError as e:
                last_err = e
            rep.log(f"Part {i + 1}: answer was malformed{' - retrying once' if attempt == 0 else ''}.")
        raise last_err or ClaudeOutputError("No usable clips.")

    cands: list[dict] = []
    errors: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(do_chunk, i, a, b): i for i, (a, b) in enumerate(chunks)}
        for f in as_completed(futs):
            i = futs[f]
            try:
                got = f.result()
                cands.extend(got)
                rep.log(f"Part {i + 1}/{len(chunks)}: {len(got)} candidate(s).")
            except ClaudeOutputError as e:
                errors.append(f"part {i + 1}: {e.message}")
                rep.log(f"Part {i + 1}/{len(chunks)}: skipped ({e.message}).")
            done += 1
            rep.progress("rank", done / len(chunks))
    if not cands:
        raise ClaudeOutputError("Claude didn't return any usable clips.", "; ".join(errors))

    fitted = [fit_to_words(c, words, length, duration) for c in cands]
    clips = merge(fitted, max_clips)
    for n, c in enumerate(clips, 1):
        c["rank"] = n
    if cache:
        cache.write_text(json.dumps({"rules": rules, "profile": profile.get("_id"), "clips": clips},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    rep.log(f"Kept {len(clips)} clip(s) from {len(cands)} candidate(s).")
    return clips

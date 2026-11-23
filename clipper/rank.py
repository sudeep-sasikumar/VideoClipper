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


LOW_SPEECH_WORDS_PER_10MIN = 20


def _words_in(transcript: dict, a: float, b: float) -> int:
    return sum(len(s["words"]) for s in transcript["segments"] if a <= s["start"] < b)


def is_low_speech(transcript: dict, a: float, b: float) -> bool:
    return _words_in(transcript, a, b) < LOW_SPEECH_WORDS_PER_10MIN * max(1.0, (b - a) / 600)


def _signals_text(signals: dict, a: float, b: float, hours: bool, low_speech: bool = False) -> str:
    out = []
    n_loud, n_chat = (40, 25) if low_speech else (25, 15)
    loud = sorted((x for x in signals.get("loudness", []) if a <= x["t"] <= b), key=lambda x: -x["z"])[:n_loud]
    if loud:
        out.append("Audio loudness spikes (time, strength; screaming/hype/explosions often show here):")
        out.append(", ".join(f"{fmt_ts(x['t'], hours)} ({x['z']:.1f})" for x in sorted(loud, key=lambda x: x["t"])))
    chat = sorted((x for x in signals.get("chat", []) if a <= x["t"] <= b), key=lambda x: -x["z"])[:n_chat]
    if chat:
        out.append("Chat message-rate spikes (time, strength, msgs per 10 s). Chat reacts 5-20 s AFTER the moment:")
        out.append(", ".join(f"{fmt_ts(x['t'], hours)} ({x['z']:.1f}, {x['rate']})" for x in sorted(chat, key=lambda x: x["t"])))
    elif signals.get("has_chat"):
        out.append("Chat replay was available but showed no clear spikes in this part.")
    if low_speech:
        tl = [x for x in signals.get("timeline", []) if a <= x[0] < b]
        if tl:
            out.append("Average loudness every 30 s (time, dBFS; higher = louder, about -20 is loud, "
                       "-40 is quiet). Use it to see where action builds up and peaks:")
            out.append(", ".join(f"{fmt_ts(t, hours)} {d:.0f}" for t, d in tl))
    return "\n".join(out) or "No audio or chat signals stood out in this part."


def has_signals(signals: dict, a: float, b: float) -> bool:
    return any(a <= x["t"] <= b for x in signals.get("loudness", [])) or \
        any(a <= x["t"] <= b for x in signals.get("chat", []))


LOW_SPEECH_GUIDE = """NOTE: THIS PART HAS LITTLE OR NO SPEECH (music, gameplay sounds, or a silent streamer).
Choose clips from the signals instead of the transcript:
- Center each clip on the strongest loudness spikes and on where the loudness timeline rises to a peak.
- Start 5-10 s before a loudness spike so the build-up is visible; end a few seconds after it settles.
- Chat spikes come 5-20 s AFTER the moment, so the moment itself is earlier than the chat spike.
- Hook text and titles must not invent specific events you can't know. Use the video title and
  the niche to write a curiosity hook that fits ("Wait for the ending", "This got out of hand fast").
- Score honestly: signal-only picks are less certain, so cap them at 7 unless chat and loudness agree.
"""


def build_prompt(*, title: str, profile: dict, length: dict, rules: str, transcript: dict, signals: dict,
                 a: float, b: float, n: int, hours: bool, part: int, parts: int) -> str:
    rules_block = rules.strip() or "(none)"
    low = is_low_speech(transcript, a, b)
    guide = ("\n" + LOW_SPEECH_GUIDE) if low else ""
    tx = _transcript_text(transcript, a, b, hours) or "(no speech detected in this part)"
    return f"""Find the {n} best short-form clips (TikTok / Reels / Shorts) in this part of a video.

VIDEO: "{title}"
PART {part} of {parts}: {fmt_ts(a, hours)} to {fmt_ts(b, hours)}

WHAT MAKES A GOOD CLIP FOR THIS NICHE ({profile.get('name')}):
{profile.get('highlight_criteria')}

CAMPAIGN RULES (follow them strictly; they override the niche guidance):
{rules_block}

CLIP LENGTH: between {length['min']:.0f} and {length['max']:.0f} seconds, ideally about {length['target']:.0f} s.
This is a HARD limit: end minus start must never exceed {length['max']:.0f} s. If a moment needs more, start later.
- Start a moment before the action or the first line of the setup, so it hooks in the first second.
- End right after the payoff (the reaction, punchline or result). Never cut mid-sentence.
- Clips must not overlap each other. Spread across the part only if quality allows; quality first.
{guide}
SIGNALS:
{_signals_text(signals, a, b, hours, low)}

TRANSCRIPT (timestamps are the segment start times):
{tx}

Return ONLY a JSON array of exactly {n} objects (fewer only if the part truly has nothing usable), best first:
[
  {{
    "start": "{'H:MM:SS' if hours else 'MM:SS'}",
    "end": "{'H:MM:SS' if hours else 'MM:SS'}",
    "score": 1-10 integer (10 = would go viral, 5 = decent, 1 = filler),
    "hook_text": "on-screen hook for the first 3 s, max 8 words, punchy, no emojis, no hashtags",
    "title": "3-6 word title for the file name",
    "reason": "one sentence: why this works as a clip",
    "intro_line": "a natural spoken intro line for a voiceover, max 15 words, that teases the clip without spoiling it",
    "context": "2-5 word on-screen label saying what this is, e.g. the game and the situation; only facts from the title or transcript",
    "post_caption": "the caption to post with the clip: one short line that makes people watch, plus any names and hashtags the campaign rules require",
    "callouts": [{{"at": "{'H:MM:SS' if hours else 'MM:SS'}", "text": "2-5 word pop-up pointing out what happens at that moment"}}]
  }}
]
"callouts": 0 to 3 per clip, at moments inside the clip (not in its first 3 seconds), each saying only what the
transcript shows is happening then (a reaction, a reveal, a turn). Never invent facts, names or numbers."""


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
            "suggested": [round(s, 2), round(e, 2)],
            "hook_text": " ".join(hook.split()[:12]),
            "title": str(it.get("title") or hook or "clip").strip()[:80],
            "reason": str(it.get("reason") or "").strip(),
            "intro_line": str(it.get("intro_line") or "").strip(),
            "context": " ".join(str(it.get("context") or "").split()[:6])[:48],
            "post_caption": " ".join(str(it.get("post_caption") or "").split())[:300],
            "callouts": _callouts(it.get("callouts"), s, e),
        })
    return good


def required_hashtags(rules: str) -> list[str]:
    """Hashtags written in the campaign rules (e.g. "#KAICENAT"): every post caption gets them."""
    import re
    seen, out = set(), []
    for tag in re.findall(r"#[A-Za-z0-9_]{2,40}", rules or ""):
        if tag.lower() not in seen:
            seen.add(tag.lower())
            out.append(tag)
    return out


def finish_post_caption(caption: str, rules: str, hook: str = "") -> str:
    caption = (caption or hook or "").strip()
    have = caption.lower()
    missing = [t for t in required_hashtags(rules) if t.lower() not in have]
    return (caption + " " + " ".join(missing)).strip()


def _callouts(raw, s: float, e: float) -> list[list]:
    """[[absolute time, text], ...] inside the clip, at most 3, at least 2.5 s apart."""
    out: list[list] = []
    if not isinstance(raw, list):
        return out
    for c in raw:
        if not isinstance(c, dict):
            continue
        t = parse_ts(c.get("at"))
        text = " ".join(str(c.get("text") or "").split()[:7])[:42]
        if t is None or not text or not (s + 3.0 <= t <= e - 0.8):
            continue
        if all(abs(t - o[0]) >= 2.5 for o in out):
            out.append([round(t, 2), text])
        if len(out) >= 3:
            break
    return sorted(out)


def fit_to_words(c: dict, words: list[list], length: dict, duration: float) -> dict:
    """Clamp length to the limits and snap edges to word boundaries.

    Too long: trim the START (setup), never the end, because the prompt asks Claude to end right
    on the payoff. Too short: extend the end.
    """
    lo, hi = length["min"], length["max"]
    s, e = c["start"], c["end"]
    # Finish the word in progress at the end, plus a short tail.
    for ws, we, _w in words:
        if ws <= e <= we + 0.05:
            e = we
            break
    e = min(duration, e + 0.35)
    if e - s < lo:
        e = min(duration, s + lo)
        if e - s < lo:
            s = max(0.0, e - lo)
    if e - s > hi:
        s = e - hi
        # Start on a word boundary: the next word start, if the clip stays long enough.
        for ws, we, _w in words:
            if we > s:
                if ws > s and e - ws >= lo:
                    s = ws
                break
    else:
        # Start at the beginning of the word being spoken at s, if that keeps us under the cap.
        for ws, we, _w in words:
            if we >= s:
                if ws < s and s - ws < 1.0 and e - ws <= hi:
                    s = ws
                break
    s = max(0.0, s - 0.15) if e - (s - 0.15) <= hi else s
    # Hard guarantee: inside [lo, hi] whatever the snapping did.
    if e - s < lo:
        e = min(duration, s + lo)
        if e - s < lo:
            s = max(0.0, e - lo)
    if e - s > hi:
        s = e - hi
    return dict(c, start=round(s, 2), end=round(e, 2))


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
            if data.get("rules") == rules and data.get("profile") == profile.get("_id") and data.get("clips") \
                    and data.get("cph") == profile.get("candidates_per_hour") \
                    and data.get("max_len") == length["max"]:
                rep.log("Reusing ranking from an earlier run.")
                rep.progress("rank", 1.0)
                return data["clips"]
        except ValueError:
            pass

    words = [w for s in transcript["segments"] for w in s["words"]]
    chunks = make_chunks(duration)
    hours = duration >= 3600
    cph = int(profile.get("candidates_per_hour", 6))
    # A part is worth asking about if it has speech or at least one loudness/chat spike.
    usable = [(i, a, b) for i, (a, b) in enumerate(chunks)
              if not is_low_speech(transcript, a, b) or has_signals(signals, a, b)]
    if not usable:
        raise ClipperError("Nothing to rank: the video has almost no speech and no loudness or chat spikes.",
                           "Clipper needs talking, loud moments, or a chat replay to find highlights.")
    low = sum(1 for i, a, b in usable if is_low_speech(transcript, a, b))
    if low:
        rep.log(f"{low} of {len(chunks)} part(s) have little speech; ranking those from loudness/chat spikes.")
    if len(usable) < len(chunks):
        rep.log(f"Skipping {len(chunks) - len(usable)} part(s) with no speech and no spikes.")
    rep.log(f"Asking Claude for highlights in {len(usable)} part(s)...")

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
        futs = {pool.submit(do_chunk, i, a, b): i for i, a, b in usable}
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
            rep.progress("rank", done / len(usable))
    if not cands:
        raise ClaudeOutputError("Claude didn't return any usable clips.", "; ".join(errors))

    fitted = [fit_to_words(c, words, length, duration) for c in cands]
    for c in fitted:
        c["post_caption"] = finish_post_caption(c.get("post_caption", ""), rules, c.get("hook_text", ""))
    clips = merge(fitted, max_clips)
    for n, c in enumerate(clips, 1):
        c["rank"] = n
    if cache:
        cache.write_text(json.dumps({"rules": rules, "profile": profile.get("_id"), "clips": clips,
                                    "cph": profile.get("candidates_per_hour"), "max_len": length["max"]},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    rep.log(f"Kept {len(clips)} clip(s) from {len(cands)} candidate(s).")
    return clips

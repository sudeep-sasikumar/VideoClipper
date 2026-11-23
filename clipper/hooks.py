"""Second pass: rewrite each clip's on-screen hook so it stops the scroll.

The ranking pass picks moments and writes a first hook, which tends to read like a title
("Jeff opens the mystery bag"). This pass gives Claude each clip's full transcript and asks
for a hook built for the first 1-3 seconds of a TikTok, then keeps the best one.
"""
from __future__ import annotations

import re

from . import claude_cli
from .errors import ClaudeOutputError, ClipperError
from .events import Reporter
from .timefmt import fmt_ts

MAX_WORDS = 9
MAX_CHARS = 52

GUIDE = """A hook is the text on screen in the first 3 seconds. Its only job is to stop someone scrolling.
Use one of these patterns, whichever fits the moment best:
- curiosity gap: tease what happens without saying it ("Wait for Jeff's face at the end")
- stakes or conflict: what is on the line ("Jeff had ONE job in GTA 6")
- a specific detail or number from the clip ("Jeff lost $40,000 in 5 seconds")
- a bold claim or question the clip answers ("Is this the worst GTA 6 job ever?")
- POV / relatable framing ("POV: chat told you to shower for 4 hours")
Rules:
- max {max_words} words, readable in under 2 seconds; plain words, no hashtags, no emojis
- say only what the clip actually shows or says; never invent events, numbers or quotes
- don't give away the punchline or payoff; make them want to see it
- every hook in this batch must be clearly different from the others (no reused openings)
- no negative framing of the creator, no rage bait, no drama framing
{extra}"""


def _clip_text(transcript: dict, start: float, end: float) -> str:
    lines = []
    for s in transcript["segments"]:
        if s["end"] <= start or s["start"] >= end:
            continue
        lines.append(f"[{fmt_ts(max(0.0, s['start'] - start))}] {s['text']}")
    return "\n".join(lines) or "(no speech)"


def _name_tokens(overlay_name: str) -> list[str]:
    name = (overlay_name or "").strip()
    if not name:
        return []
    parts = re.findall(r"[A-Z][a-z]+|[a-z]+|[A-Z]+(?![a-z])|\d+", name) + name.split()
    return list(dict.fromkeys([name] + [p for p in parts if len(p) >= 3]))


def sharpen_hooks(clips: list[dict], transcript: dict, *, title: str, rules: str, rep: Reporter,
                  overlay_name: str = "", model: str = "") -> int:
    """Rewrite clips' hook_text in place. Returns how many hooks changed. Keeps old hooks on failure."""
    if not clips:
        return 0
    tokens = _name_tokens(overlay_name)
    extra = ""
    if tokens:
        extra = (f"- every hook must name the creator the way the current hooks do "
                 f"(e.g. \"{tokens[-1]}\"), since the campaign requires the name on screen")
    blocks = []
    for i, c in enumerate(clips, 1):
        blocks.append(f"CLIP {i}  ({c['end'] - c['start']:.0f} s)\ncurrent hook: {c.get('hook_text', '')}\n"
                      f"why it was picked: {c.get('reason', '')}\ntranscript (time from clip start):\n"
                      f"{_clip_text(transcript, c['start'], c['end'])}")
    prompt = (f"You write on-screen hooks for short-form clips (TikTok, Reels, Shorts) cut from \"{title}\".\n\n"
              + GUIDE.format(max_words=MAX_WORDS, extra=extra)
              + (f"\n\nCAMPAIGN RULES (they override everything else):\n{rules.strip()}\n" if rules.strip() else "\n")
              + "\n\n" + "\n\n".join(blocks)
              + "\n\nFor each clip, write 3 different hook options, then pick the strongest. Return ONLY a JSON array:\n"
              '[{"clip": 1, "options": ["...", "...", "..."], "hook": "the best one", "pattern": "curiosity|stakes|detail|question|pov"}]')

    result = None
    for attempt in range(2):
        rep.check()
        try:
            text = claude_cli.ask(prompt if attempt == 0 else prompt + "\n\nReply with the JSON array only.",
                                  model=model, timeout=600)
            data = claude_cli.extract_json(text)
            if isinstance(data, list) and data:
                result = data
                break
        except ClaudeOutputError:
            pass
        rep.log("Hook pass: answer was malformed" + (" - retrying once." if attempt == 0 else "."))
    if result is None:
        rep.log("Hook pass skipped; keeping the original hooks.")
        return 0

    changed = 0
    used: set[str] = set()
    for item in result:
        try:
            idx = int(item.get("clip")) - 1
        except (TypeError, ValueError, AttributeError):
            continue
        if not 0 <= idx < len(clips):
            continue
        c = clips[idx]
        cands = [item.get("hook")] + list(item.get("options") or [])
        for h in cands:
            h = " ".join(str(h or "").replace("#", "").split()).strip(' "')
            if not h or len(h.split()) > MAX_WORDS or len(h) > MAX_CHARS or h.lower() in used:
                continue
            if tokens and not any(t.lower() in h.lower() for t in tokens):
                continue  # the campaign needs the name on screen
            c.setdefault("hook_original", c.get("hook_text", ""))
            c["hook_options"] = [str(o) for o in (item.get("options") or [])][:3]
            c["hook_pattern"] = str(item.get("pattern") or "")
            if h != c.get("hook_text"):
                changed += 1
            c["hook_text"] = h
            used.add(h.lower())
            break
    rep.log(f"Hook pass: rewrote {changed} of {len(clips)} hooks.")
    return changed


def sharpen_project_hooks(proj: dict, rep: Reporter, model: str = "") -> int:
    """Run the hook pass on a saved project (uses its transcript.json)."""
    import json
    from pathlib import Path
    tr = json.loads((Path(proj["_folder"]) / "transcript.json").read_text(encoding="utf-8"))
    clips = sorted(proj["clips"], key=lambda c: c["rank"])
    if not clips:
        raise ClipperError("This project has no clips.")
    return sharpen_hooks(clips, tr, title=proj["video"]["title"], rules=proj.get("rules", ""), rep=rep,
                         overlay_name=proj.get("overlay_name", ""), model=model)

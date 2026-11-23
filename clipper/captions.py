"""Burned-in captions as ASS subtitles: word-by-word highlight plus the 3-second hook."""
from __future__ import annotations

import re

W, H = 1080, 1920


def ass_color(hex_rgb: str, alpha: int = 0) -> str:
    h = (hex_rgb or "#FFFFFF").lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        r, g, b = 255, 255, 255
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _ts(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _esc(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def build_lines(words: list[list], clip_start: float, clip_end: float, per_line: int = 3,
                gap: float = 0.6) -> list[dict]:
    """Group the clip's words into caption lines. Times are absolute (source) seconds."""
    ws = [w for w in words if w[1] > clip_start + 0.05 and w[0] < clip_end - 0.05]
    lines: list[dict] = []
    cur: list[list] = []
    for w in ws:
        if cur and (len(cur) >= per_line or w[0] - cur[-1][1] > gap or re.search(r"[.!?]$", cur[-1][2])):
            lines.append(cur)
            cur = []
        cur.append([max(w[0], clip_start), min(w[1], clip_end), w[2]])
    if cur:
        lines.append(cur)
    return [{"start": ln[0][0], "end": ln[-1][1], "words": ln} for ln in lines]


def retime_line(line: dict, new_text: str) -> dict:
    """Apply an edited caption line: keep word timings if the word count matches, else spread evenly."""
    new_words = new_text.split()
    if not new_words:
        return dict(line, words=[])
    old = line["words"]
    if len(new_words) == len(old):
        return dict(line, words=[[o[0], o[1], n] for o, n in zip(old, new_words)])
    s, e = line["start"], line["end"]
    step = (e - s) / len(new_words)
    return dict(line, words=[[s + i * step, s + (i + 1) * step, n] for i, n in enumerate(new_words)])


def _wrap_hook(text: str, max_chars: int = 18) -> str:
    words, rows, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            rows.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        rows.append(cur)
    return r"\N".join(rows)


def write_ass(path, *, lines: list[dict], clip_start: float, clip_end: float, cap_style: dict,
              hook_text: str, hook_style: dict, offset: float = 0.0, total: float | None = None,
              extra_styles: list[str] | None = None, extra_events: list[str] | None = None) -> None:
    """offset: seconds of teaser before the clip proper; total: full output length."""
    dur = clip_end - clip_start
    total = total if total is not None else dur + offset
    cs, hs = cap_style, hook_style
    cap_y = int(H * float(cs.get("position_y", 0.72)))
    hook_y = int(H * float(hs.get("position_y", 0.14)))
    upper = bool(cs.get("uppercase", True))
    hi = ass_color(cs.get("highlight", "#FFE21A"))
    base = ass_color(cs.get("color", "#FFFFFF"))
    box = ass_color(hs.get("box", "#E3122C"))

    out = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.709", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Cap,{cs.get('font', 'Arial Black')},{int(cs.get('size', 84))},{base},{hi},"
        f"{ass_color(cs.get('outline', '#000000'))},{ass_color('#000000', 0x60)},-1,0,0,0,100,100,1,0,1,"
        f"{float(cs.get('outline_width', 7))},{float(cs.get('shadow', 2))},5,60,60,0,1",
        f"Style: Hook,{hs.get('font', 'Arial Black')},{int(hs.get('size', 72))},"
        f"{ass_color(hs.get('color', '#FFFFFF'))},{box},{box},{box},-1,0,0,0,100,100,0,0,3,18,0,5,70,70,0,1",
        *(extra_styles or []),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    if cs.get("enabled", True):
        for li, ln in enumerate(lines):
            words = ln["words"]
            if not words:
                continue
            nxt_start = lines[li + 1]["start"] if li + 1 < len(lines) else clip_end
            line_end = min(nxt_start, ln["end"] + 0.6)
            for wi, w in enumerate(words):
                ws = w[0] - clip_start
                we = (words[wi + 1][0] if wi + 1 < len(words) else line_end) - clip_start
                if we <= ws:
                    we = ws + 0.05
                parts = []
                for wj, w2 in enumerate(words):
                    t = _esc(w2[2].upper() if upper else w2[2])
                    if wj == wi:
                        parts.append(r"{\c" + hi + r"\fscx112\fscy112}" + t + r"{\c" + base + r"\fscx100\fscy100}")
                    else:
                        parts.append(t)
                pop = r"\t(0,80,\fscx104\fscy104)\t(80,160,\fscx100\fscy100)" if wi == 0 else ""
                text = "{" + rf"\pos({W // 2},{cap_y}){pop}" + "}" + " ".join(parts)
                out.append(f"Dialogue: 1,{_ts(ws + offset)},{_ts(max(0, min(we, dur)) + offset)},Cap,,0,0,0,,{text}")

    hook_text = (hook_text or "").strip()
    if hs.get("enabled", True) and hook_text:
        hd = min(float(hs.get("duration", 3.0)) + offset, total)
        text = ("{" + rf"\pos({W // 2},{hook_y})\fad(120,220)\fscx80\fscy80\t(0,160,\fscx100\fscy100)" + "}"
                + _wrap_hook(_esc(hook_text.upper())))
        out.append(f"Dialogue: 2,{_ts(0)},{_ts(hd)},Hook,,0,0,0,,{text}")

    out.extend(extra_events or [])
    with open(path, "w", encoding="utf-8-sig", newline="\n") as f:
        f.write("\n".join(out) + "\n")


def extra_events(ex: dict, *, layout_mode: str, hook_style: dict, dur: float) -> tuple[list[str], list[str]]:
    """ASS styles + events for the optional extras: teaser label, context banner, callouts, end card.

    ex: {"pre", "post", "teaser", "teaser_text", "banner", "callouts": [(t, text)], "end_lines": [...]}
    Callout times are clip-relative (0 = clip start, before the teaser offset is added).
    """
    pre, post = float(ex.get("pre", 0)), float(ex.get("post", 0))
    hook_y = int(H * float(hook_style.get("position_y", 0.14)))
    hook_end = pre + float(hook_style.get("duration", 3.0))
    call_y = int(H * (0.52 if layout_mode == "split" else 0.58))
    yellow, black, white = ass_color("#FFE100"), ass_color("#101010"), ass_color("#FFFFFF")
    shade = ass_color("#000000", 0x50)
    styles = [
        f"Style: Teaser,Arial Black,96,{yellow},{yellow},{ass_color('#000000')},{ass_color('#000000', 0x64)},"
        "-1,0,0,0,100,100,2,0,1,8,3,5,40,40,0,1",
        f"Style: Banner,Arial Black,44,{white},{white},{shade},{shade},-1,0,0,0,100,100,1,0,3,14,0,5,60,60,0,1",
        f"Style: Callout,Arial Black,60,{black},{black},{yellow},{yellow},-1,0,0,0,100,100,0,0,3,16,0,5,60,60,0,1",
        f"Style: End,Arial Black,70,{white},{white},{ass_color('#000000')},{ass_color('#000000', 0x64)},"
        "-1,0,0,0,100,100,1,0,1,6,2,5,60,60,0,1",
    ]
    ev: list[str] = []
    if ex.get("teaser") and pre > 0:
        label = _esc((ex.get("teaser_text") or "WAIT FOR IT...").upper())
        tags = rf"\pos({W // 2},{int(H * 0.5)})\fad(80,120)\fscx70\fscy70\t(0,200,\fscx100\fscy100)"
        ev.append(f"Dialogue: 3,{_ts(0)},{_ts(pre)},Teaser,,0,0,0,,{{{tags}}}{label}")
    banner = (ex.get("banner") or "").strip()
    if banner:
        b0, b1 = hook_end, pre + dur
        if b1 - b0 > 1.0:
            tags = rf"\pos({W // 2},{hook_y})\fad(250,200)"
            ev.append(f"Dialogue: 2,{_ts(b0)},{_ts(b1)},Banner,,0,0,0,,{{{tags}}}{_esc(banner.upper())}")
    for i, (t, text) in enumerate(ex.get("callouts") or []):
        t0 = pre + float(t)
        t1 = min(t0 + 1.8, pre + dur)
        if t1 - t0 < 0.6:
            continue
        rot = -4 if i % 2 == 0 else 4
        tags = (rf"\pos({W // 2},{call_y})\frz{rot}\fad(60,150)\fscx60\fscy60"
                r"\t(0,120,\fscx105\fscy105)\t(120,200,\fscx100\fscy100)")
        ev.append(f"Dialogue: 4,{_ts(t0)},{_ts(t1)},Callout,,0,0,0,,{{{tags}}}{_esc(text.upper())}")
    if post > 0:
        e0, e1 = pre + dur, pre + dur + post
        lines = [_esc(x) for x in (ex.get("end_lines") or []) if x]
        if lines:
            tags = rf"\pos({W // 2},{int(H * 0.22)})\fad(150,0)"
            ev.append(f"Dialogue: 5,{_ts(e0)},{_ts(e1)},End,,0,0,0,,{{{tags}}}{lines[0].upper()}")
        if len(lines) > 1:
            tags = rf"\pos({W // 2},{int(H * 0.62)})\fad(150,0)"
            ev.append(f"Dialogue: 5,{_ts(e0)},{_ts(e1)},Banner,,0,0,0,,{{{tags}}}{_wrap_hook(lines[1], 28)}")
    return styles, ev

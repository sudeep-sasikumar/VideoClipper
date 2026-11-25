"""Beat-synced montage edits: short gameplay shots cut on a song's beat grid, biggest shot on the drop.

An edit is a JSON spec:
  {"name": "...", "text": "one short line", "text_until": 2.1, "length": 20.967,
   "grid": {"first": 0.395, "period": 0.428571}, "drop": 3.823, "punch_every": 2,
   "shots": [{"src": "path.mp4", "at": 1.2, "until_beat": 5, "speed": 1.0, "zoom": 1.1, "band": [y0, y1]}]}

Each shot runs from the previous shot's end to `until_beat` on the grid (or `until` seconds). `band`
(fractions of the source height) picks the clean gameplay strip of a repost; when left out it is
detected (the sharp, textured rows between blurred fill or black bars). The output is 1080x1920 at
30 fps with a silent audio track: the official sound is added in the app, at 0:00.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .tools import NO_WINDOW, find_ffmpeg

W, H, FPS = 1080, 1920, 30


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       creationflags=NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd[:6])}...\n{r.stderr[-2000:]}")


def probe(path: Path) -> dict:
    out = subprocess.run([find_ffmpeg("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, creationflags=NO_WINDOW).stdout
    j = json.loads(out)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"w": int(s["width"]), "h": int(s["height"]), "fps": float(num) / float(den),
            "duration": float(j["format"]["duration"])}


def _gray_frames(path: Path, times: list[float], w: int = 180) -> list[np.ndarray]:
    info = probe(path)
    h = int(round(info["h"] * w / info["w"] / 2) * 2)
    frames = []
    for t in times:
        raw = subprocess.run([find_ffmpeg(), "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                              "-vf", f"scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                             capture_output=True, creationflags=NO_WINDOW).stdout
        if len(raw) == w * h:
            frames.append(np.frombuffer(raw, np.uint8).reshape(h, w).astype(np.float32))
    return frames


def detect_band(path: Path, t0: float | None = None, t1: float | None = None) -> tuple[float, float]:
    """The sharp gameplay strip as (top, bottom) fractions: the longest run of textured rows.
    Blurred fill and black bars have almost no horizontal detail."""
    info = probe(path)
    a = t0 if t0 is not None else 0.3
    b = t1 if t1 is not None else info["duration"] - 0.3
    frames = _gray_frames(path, list(np.linspace(a, b, 6)))
    tex = np.mean([np.abs(np.diff(f, axis=1)).mean(axis=1) for f in frames], axis=0)
    h = len(tex)
    thr = 0.45 * np.percentile(tex, 90)
    on = tex > thr
    # close small gaps (dark sky, flat road) up to 4 % of the height
    gap = max(1, int(0.04 * h))
    runs, start = [], None
    for i, v in enumerate(list(on) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    top, bot = max(merged, key=lambda r: r[1] - r[0])
    # Snap edges to the strongest row-to-row change nearby (the band's hard border).
    rowdiff = np.mean([np.abs(np.diff(f, axis=0)).mean(axis=1) for f in frames], axis=0)
    k = max(2, int(0.025 * h))
    lo = max(0, top - k)
    top = lo + int(np.argmax(rowdiff[lo:top + k])) + 1
    hi = min(len(rowdiff), bot + k)
    bot = bot - k + int(np.argmax(rowdiff[bot - k:hi])) + 1
    inset = 2  # a couple of rows inside, so no border line survives the crop
    return (min(1.0, (top + inset) / h), max(0.0, (bot - inset) / h))


def trim_black(path: Path, band: tuple[float, float], t0: float, t1: float) -> tuple[float, float]:
    """Tighten a band past black letterbox rows the source shows during this shot (cinematic replays)."""
    info = probe(path)
    times = [min(max(0.0, t), info["duration"] - 0.1) for t in (t0 + 0.05, (t0 + t1) / 2, t1 - 0.05)]
    frames = _gray_frames(path, times, w=180)
    if not frames:
        return band
    h = frames[0].shape[0]
    rows = np.max([f.mean(axis=1) for f in frames], axis=0)
    a, b = int(round(band[0] * h)), int(round(band[1] * h))
    top, bot = a, b
    while top < b and rows[top] < 16:
        top += 1
    while bot > top and rows[bot - 1] < 16:
        bot -= 1
    if (top, bot) == (a, b) or (bot - top) < 0.5 * (b - a):
        return band
    pad = 2  # past the soft edge of the bar
    return ((min(top + pad, b)) / h, (max(bot - pad, top)) / h)


def beat_time(grid: dict, k: float) -> float:
    return float(grid["first"]) + float(k) * float(grid["period"])


def plan_cuts(spec: dict) -> list[tuple[float, float]]:
    """(start, end) output seconds for every shot, snapped to whole frames."""
    grid, t, out = spec["grid"], 0.0, []
    for s in spec["shots"]:
        end = float(s["until"]) if "until" in s else beat_time(grid, s["until_beat"])
        end = min(end, float(spec["length"]))
        out.append((round(t * FPS) / FPS, round(end * FPS) / FPS))
        t = end
    if out and out[-1][1] < float(spec["length"]) - 1 / FPS:
        raise ValueError(f"shots end at {out[-1][1]:.2f}s but the edit is {spec['length']}s long")
    return out


def _fill_filter(info: dict, band: tuple[float, float], speed: float, zoom: float, frames: int, grade: str,
                 cx: float, cy: float) -> str:
    """Full-screen 9:16: crop a vertical window out of the (wide) source, centred at (cx, cy) of the band."""
    bw = info["w"]
    by, bh = info["h"] * band[0], info["h"] * (band[1] - band[0])
    ch = bh / zoom
    cw = ch * 9 / 16
    if cw > bw:                     # very tall source: limit by width instead
        cw, ch = bw / zoom, bw / zoom * 16 / 9
    cw, ch = int(cw) // 2 * 2, int(ch) // 2 * 2
    x = int(min(max(cx * bw - cw / 2, 0), bw - cw))
    y = int(min(max(by + cy * bh - ch / 2, by), by + bh - ch))
    sharpen = ",unsharp=5:5:0.6:3:3:0.0" if cw < 900 else ""
    return (f"[0:v]crop={cw}:{ch}:{x}:{y},setpts=(PTS-STARTPTS)/{speed:.4f},fps={FPS},"
            f"scale={W}:{H}:flags=lanczos,setsar=1{sharpen}{grade},trim=end_frame={frames},format=yuv420p[v]")


SPLIT_GAP = 8                       # white divider between the two halves of a split-screen shot
LABEL_FONT = "C\\:/Windows/Fonts/arialbd.ttf"


def _half_filter(idx: int, info: dict, half: dict, speed: float, frames: int, grade: str, out_h: int,
                 tag: str) -> str:
    """One half of a split-screen shot: a W x out_h window cropped out of the source at (cx, cy)."""
    band = half.get("band") or [0.0, 1.0]
    zoom = float(half.get("zoom", 1.0))
    bw = info["w"]
    by, bh = info["h"] * band[0], info["h"] * (band[1] - band[0])
    ch = bh / zoom
    cw = ch * W / out_h
    if cw > bw:
        cw, ch = bw / zoom, bw / zoom * out_h / W
    cw, ch = int(cw) // 2 * 2, int(ch) // 2 * 2
    cx, cy = float(half.get("cx", 0.5)), float(half.get("cy", 0.5))
    x = int(min(max(cx * bw - cw / 2, 0), bw - cw))
    y = int(min(max(by + cy * bh - ch / 2, by), by + bh - ch))
    sharpen = ",unsharp=5:5:0.5:3:3:0.0" if cw < 900 else ""
    sp = float(half.get("speed", speed))
    label = ""
    if half.get("label"):
        txt = str(half["label"]).replace("'", "").replace(":", "\\:")
        label = (f",drawtext=fontfile='{LABEL_FONT}':text='{txt}':fontsize=54:fontcolor=white:"
                 f"x=40:y=h-th-40:box=1:boxcolor=black@0.55:boxborderw=14")
    return (f"[{idx}:v]crop={cw}:{ch}:{x}:{y},setpts=(PTS-STARTPTS)/{sp:.4f},fps={FPS},"
            f"scale={W}:{out_h}:flags=lanczos,setsar=1{sharpen}{half.get('grade', grade)}{label},"
            f"trim=end_frame={frames}[{tag}]")


def _split_filter(infos: list[dict], shot: dict, speed: float, frames: int, grade: str) -> str:
    """Two sources stacked: `top` above a thin white divider, `bottom` below."""
    hh = (H - SPLIT_GAP) // 2
    return ";".join([
        _half_filter(0, infos[0], shot["top"], speed, frames, grade, hh, "t"),
        _half_filter(1, infos[1], shot["bottom"], speed, frames, grade, H - SPLIT_GAP - hh, "b"),
        f"color=c=white:s={W}x{SPLIT_GAP}:r={FPS},trim=end_frame={frames}[d]",
        "[t][d][b]vstack=inputs=3:shortest=1,format=yuv420p[v]"])


def _shot_filter(info: dict, band: tuple[float, float], speed: float, zoom: float, frames: int,
                 grade: str) -> str:
    y0, y1 = band
    bh = int(round(info["h"] * (y1 - y0) / 2) * 2)
    by = int(round(info["h"] * y0))
    bw = info["w"] - (info["w"] % 2)
    # Foreground: the band scaled to the frame width times `zoom`, then cropped to fit.
    fw = int(round(W * zoom / 2) * 2)
    fh = int(round(fw * bh / bw / 2) * 2)
    crop_h = min(fh, H)
    sharpen = ",unsharp=5:5:0.7:3:3:0.0" if bw < 900 else ""
    g = [f"[0:v]crop={bw}:{bh}:0:{by},setpts=(PTS-STARTPTS)/{speed:.4f},fps={FPS},setsar=1{grade}[src]",
         "[src]split=2[a][b]",
         f"[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=14:3,"
         f"eq=brightness=-0.12:saturation=1.2,scale={W}:{H}[bg]",
         f"[b]scale={fw}:{fh}:flags=lanczos,crop={W}:{crop_h}{sharpen}[fg]",
         f"[bg][fg]overlay=(W-w)/2:(H-h)/2,trim=end_frame={frames},format=yuv420p[v]"]
    return ";".join(g)


def _zoom_expr(punches: list[tuple[float, float, float]]) -> str:
    """zoompan zoom: 1 + sum of decaying bumps (time, amount, length)."""
    terms = [f"{amt:.3f}*between(it,{t:.3f},{t + ln:.3f})*(1-(it-{t:.3f})/{ln:.3f})" for t, amt, ln in punches]
    return "1+" + "+".join(terms) if terms else "1"


def _ass(text: str, until: float, size: int, y: float) -> str:
    def ts(s: float) -> str:
        cs = int(round(s * 100))
        return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"
    return "\n".join([
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 2",
        "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: T,Arial Black,{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,1,0,1,6,3,5,60,60,0,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        f"Dialogue: 0,{ts(0)},{ts(until)},T,,0,0,0,,{{\\pos({W // 2},{int(H * y)})\\fad(0,180)}}{text}", ""])


def render(spec: dict, out_dir: Path, work: Path, preview_audio: Path | None = None, log=print) -> dict:
    """Render one edit. Returns the paths written (silent post file, preview with the song)."""
    F = find_ffmpeg()
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cuts = plan_cuts(spec)
    grade = spec.get("grade", "")
    seg_paths = []
    for i, (s, (a, b)) in enumerate(zip(spec["shots"], cuts)):
        frames = int(round((b - a) * FPS))
        speed = float(s.get("speed", 1.0))
        if s.get("mode", spec.get("mode", "band")) == "split":
            inputs, infos = [], []
            for side in ("top", "bottom"):
                h = s[side]
                p, at = Path(h["src"]), float(h.get("at", 0.0))
                info = probe(p)
                need = frames / FPS * float(h.get("speed", speed)) + 0.4
                if at + need - 0.4 > info["duration"] + 0.05:
                    raise ValueError(f"shot {i + 1} {side} ({p.name}) needs {need - 0.4:.2f}s from {at:.2f}s "
                                     f"but the clip is {info['duration']:.2f}s")
                inputs += ["-ss", f"{at:.3f}", "-t", f"{need:.3f}", "-i", str(p)]
                infos.append(info)
            seg = work / f"seg_{i:02d}.mp4"
            _run([F, "-v", "error", "-y", *inputs, "-filter_complex", _split_filter(infos, s, speed, frames, grade),
                  "-map", "[v]", "-frames:v", str(frames), "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast",
                  "-threads", "2", "-crf", "12", "-pix_fmt", "yuv420p", "-an", str(seg)])
            seg_paths.append(seg)
            log(f"  shot {i + 1:2d} {a:6.2f}-{b:6.2f}s  split {Path(s['top']['src']).name}@{s['top'].get('at', 0)} / "
                f"{Path(s['bottom']['src']).name}@{s['bottom'].get('at', 0)}")
            continue
        src = Path(s["src"])
        info = probe(src)
        need = frames / FPS * speed + 0.4
        at = float(s.get("at", 0.0))
        if at + need - 0.4 > info["duration"] + 0.05:
            raise ValueError(f"shot {i + 1} ({src.name}) needs {need - 0.4:.2f}s from {at:.2f}s "
                             f"but the clip is {info['duration']:.2f}s")
        band = tuple(s["band"]) if s.get("band") else detect_band(src)
        if s.get("trim_black", True):
            band = trim_black(src, band, at, at + need - 0.4)
        s["_band"] = [round(band[0], 4), round(band[1], 4)]
        seg = work / f"seg_{i:02d}.mp4"
        if s.get("mode", spec.get("mode", "band")) == "fill":
            fg = _fill_filter(info, band, speed, float(s.get("zoom", 1.0)), frames, s.get("grade", grade),
                              float(s.get("cx", 0.5)), float(s.get("cy", 0.5)))
        else:
            fg = _shot_filter(info, band, speed, float(s.get("zoom", 1.0)), frames, s.get("grade", grade))
        _run([F, "-v", "error", "-y", "-ss", f"{at:.3f}", "-t", f"{need:.3f}", "-i", str(src),
              "-filter_complex", fg,
              "-map", "[v]", "-frames:v", str(frames), "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-threads", "2",
              "-crf", "12", "-pix_fmt", "yuv420p", "-an", str(seg)])
        seg_paths.append(seg)
        log(f"  shot {i + 1:2d} {a:6.2f}-{b:6.2f}s  {src.name} @{at:.2f}s x{speed:g} band {s['_band']}")

    (work / "list.txt").write_text("".join(f"file '{p.name}'\n" for p in seg_paths), encoding="utf-8")
    _run([F, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "joined.mp4"], cwd=work)

    length = cuts[-1][1]
    drop = float(spec["drop"])
    g, cur = [], "0:v"
    if spec.get("drop_fx", False):
        # Zoom punches: big on the drop, small on every Nth cut after it; flash and colour split on the drop.
        punches = [(drop, 0.16, 0.45)]
        every = int(spec.get("punch_every", 2))
        for n, (a, _) in enumerate(cuts):
            if a > drop + 0.05 and every and n % every == 0:
                punches.append((a, 0.05, 0.3))
        g = [f"[0:v]zoompan=z='{_zoom_expr(punches)}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={W}x{H}:fps={FPS}[z]",
             f"[z]rgbashift=rh=14:bh=-14:enable='between(t,{drop:.3f},{drop + 0.2:.3f})'[r]",
             f"color=white:s={W}x{H}:r={FPS}:d=0.3,format=rgba,fade=t=out:st=0:d=0.3:alpha=1,"
             f"setpts=PTS+{drop:.3f}/TB[fl]",
             "[r][fl]overlay=eof_action=pass[f]"]
        cur = "f"
    if spec.get("text"):
        (work / "text.ass").write_text(_ass(spec["text"], float(spec.get("text_until", 2.1)),
                                            int(spec.get("text_size", 78)), float(spec.get("text_y", 0.16))),
                                       encoding="utf-8")
        g.append(f"[{cur}]ass=text.ass[t]")
        cur = "t"
    g.append(f"[{cur}]format=yuv420p[v]")
    name = spec["name"]
    post = out_dir / f"{name}.mp4"
    _run([F, "-v", "error", "-y", "-i", "joined.mp4", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
          "-filter_complex", ";".join(g), "-map", "[v]", "-map", "1:a", "-t", f"{length:.3f}",
          "-c:v", "libx264", "-preset", "medium", "-threads", "3", "-crf", "17", "-profile:v", "high", "-pix_fmt", "yuv420p",
          "-r", str(FPS), "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(post.resolve())], cwd=work)
    res = {"post": post, "length": length, "cuts": cuts}
    # Platform copies: the same cut with its head trimmed so the drop lands where that app's sound has it.
    for v in spec.get("variants", []):
        vp = out_dir / f"{name}{v['suffix']}.mp4"
        skip = int(round(float(v.get("trim", 0)) * FPS))
        if skip:
            _run([F, "-v", "error", "-y", "-i", str(post), "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                  "-filter_complex", f"[0:v]trim=start_frame={skip},setpts=PTS-STARTPTS[v]", "-map", "[v]",
                  "-map", "1:a", "-t", f"{length - skip / FPS:.3f}", "-c:v", "libx264", "-preset", "medium",
                  "-threads", "3", "-crf", "17", "-profile:v", "high", "-pix_fmt", "yuv420p", "-c:a", "aac",
                  "-b:a", "128k", "-movflags", "+faststart", str(vp)])
        else:
            shutil.copy2(post, vp)
        res.setdefault("variants", []).append(vp)
        if v.get("audio"):
            prev = out_dir / "preview_with_song" / f"{name}{v['suffix']}_PREVIEW.mp4"
            prev.parent.mkdir(exist_ok=True)
            _run([F, "-v", "error", "-y", "-i", str(vp), "-i", str(v["audio"]), "-map", "0:v", "-map", "1:a",
                  "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(prev)])
    if spec.get("variants"):
        post.unlink()                      # the platform copies replace the master
    if preview_audio:
        prev = out_dir / "preview_with_song" / f"{name}_PREVIEW.mp4"
        prev.parent.mkdir(exist_ok=True)
        _run([F, "-v", "error", "-y", "-i", str(post), "-i", str(preview_audio), "-map", "0:v", "-map", "1:a",
              "-t", f"{length:.3f}", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
              str(prev)])
        res["preview"] = prev
    shutil.rmtree(work, ignore_errors=True)
    return res

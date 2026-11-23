"""Stage 5: render one 1080x1920 clip with ffmpeg."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import captions
from .errors import ClipperError, DiskSpaceError
from .events import Reporter
from .tools import find_ffmpeg, run

W, H, FPS = 1080, 1920, 30


def has_audio(path: Path) -> bool:
    r = run([find_ffmpeg("ffprobe"), "-v", "error", "-select_streams", "a", "-show_entries",
             "stream=index", "-of", "csv=p=0", str(path)])
    return bool(r.stdout.strip())


def pick_punches(loudness: list[dict], start: float, end: float, zoom: dict) -> list[float]:
    """Clip-relative times of the strongest loudness spikes, at least 3 s apart."""
    if not zoom.get("enabled", True):
        return []
    inside = [x for x in loudness if start + 0.5 <= x["t"] <= end - 0.5]
    chosen: list[float] = []
    for x in sorted(inside, key=lambda x: -x["z"]):
        t = x["t"] - start
        if all(abs(t - c) >= 3.0 for c in chosen):
            chosen.append(t)
        if len(chosen) >= int(zoom.get("max_punches", 6)):
            break
    return sorted(chosen)


def _zoom_expr(punches: list[float], amount: float, length: float) -> str:
    """Zoom factor over time: snap in fast (0.08 s), ease back out over `length` seconds."""
    if not punches:
        return "1"
    T = f"(in/{FPS})"
    terms = []
    for c in punches:
        up = f"clip(({T}-{c - 0.08:.3f})/0.08,0,1)"
        down = f"clip(({c + length:.3f}-{T})/{length * 0.7:.3f},0,1)"
        terms.append(f"{up}*{down}")
    return f"1+{amount:.3f}*min(1,{'+'.join(terms)})"


def build_filtergraph(layout: dict, punches: list[float], zoom: dict, *, src_w: int, src_h: int,
                      vo: bool, vo_dur: float, vo_cfg: dict, audio: bool) -> str:
    mode = layout.get("mode", "blur")
    g = [f"[0:v]fps={FPS},setsar=1,split=2[a0][b0]"]
    if mode == "split":
        top_h = int(round(H * float(layout.get("split_top_ratio", 0.38)) / 2) * 2)
        bot_h = H - top_h
        fc = layout.get("facecam", {})
        fx, fy = float(fc.get("x", 0)), float(fc.get("y", 0))
        fw, fh = float(fc.get("w", 0.3)), float(fc.get("h", 0.3))
        g.append(f"[a0]crop=iw*{fw:.4f}:ih*{fh:.4f}:iw*{fx:.4f}:ih*{fy:.4f},"
                 f"scale={W}:{top_h}:force_original_aspect_ratio=increase,crop={W}:{top_h}[top]")
        g.append(f"[b0]scale={W}:{bot_h}:force_original_aspect_ratio=increase,crop={W}:{bot_h}[bot]")
        g.append("[top][bot]vstack=inputs=2[base]")
    else:
        fz = max(1.0, float(layout.get("foreground_zoom", 1.0)))
        fg_w = int(round(W * fz / 2) * 2)
        # Blurred cover background, computed small for speed then scaled up.
        g.append(f"[a0]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
                 f"boxblur=12:2,eq=brightness=-0.06:saturation=1.15,scale={W}:{H}[bg]")
        fg = f"[b0]scale={fg_w}:-2"
        if fz > 1.0:
            fg += f",crop={W}:min(ih\\,{H})"
        g.append(fg + "[fg]")
        g.append("[bg][fg]overlay=(W-w)/2:(H-h)/2[base]")
    if punches:
        expr = _zoom_expr(punches, float(zoom.get("amount", 0.08)), float(zoom.get("length", 0.7)))
        g.append(f"[base]zoompan=z='{expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:"
                 f"s={W}x{H}:fps={FPS}[zbase]")
        base = "zbase"
    else:
        base = "base"
    g.append(f"[{base}]ass=captions.ass,setsar=1,format=yuv420p[v]")

    # Audio
    a_in = "[0:a]" if audio else "[sil]"
    if not audio:
        g.append("anullsrc=r=48000:cl=stereo[sil]")
    if vo:
        delay = 0.25
        duck = float(vo_cfg.get("duck", 0.35))
        vol = float(vo_cfg.get("volume", 1.0))
        a, b = delay, delay + vo_dur
        g.append(f"{a_in}aresample=48000,volume='if(between(t,{a:.2f},{b:.2f}),{duck},1)':eval=frame[orig]")
        g.append(f"[1:a]aresample=48000,aformat=channel_layouts=stereo,volume={vol},"
                 f"adelay={int(delay * 1000)}:all=1[vo]")
        g.append("[orig][vo]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]")
    else:
        g.append(f"{a_in}aresample=48000,alimiter=limit=0.95[a]")
    return ";\n".join(g)


def text_positions(profile: dict, layout: dict) -> tuple[dict, dict]:
    """Caption and hook styles with positions that suit the layout.

    Split layout: the hook sits on the seam between facecam and gameplay (never over the face) and
    captions sit in the gameplay half. Profiles can override with "split_position_y".
    """
    cap, hook = dict(profile["captions"]), dict(profile["hook"])
    if layout.get("mode") == "split":
        seam = float(layout.get("split_top_ratio", 0.38))
        hook["position_y"] = float(hook.get("split_position_y", seam))
        cap["position_y"] = float(cap.get("split_position_y", max(seam + 0.25, 0.66)))
    return cap, hook


def probe_size(path: Path) -> tuple[int, int]:
    r = run([find_ffmpeg("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "json", str(path)])
    try:
        s = json.loads(r.stdout)["streams"][0]
        return int(s["width"]), int(s["height"])
    except (ValueError, KeyError, IndexError):
        return 1920, 1080


def render_clip(*, source: Path, source_offset: float, clip: dict, words: list[list], profile: dict,
                loudness: list[dict], out_path: Path, work: Path, rep: Reporter,
                voiceover_wav: Path | None = None, vo_dur: float = 0.0, layout: dict | None = None) -> Path:
    """Render clip["start"]..clip["end"] (absolute source seconds) to out_path.

    source_offset: the absolute time that `source` starts at (0 for the full video, or the start
    of a downloaded section when re-rendering after the full video was deleted).
    """
    # ffmpeg runs with cwd=work (for the ASS path), so every other path must be absolute.
    work, source, out_path = Path(work).resolve(), Path(source).resolve(), Path(out_path).resolve()
    voiceover_wav = Path(voiceover_wav).resolve() if voiceover_wav else None
    work.mkdir(parents=True, exist_ok=True)
    start, end = float(clip["start"]), float(clip["end"])
    dur = end - start
    if dur <= 1:
        raise ClipperError(f"Clip {clip.get('rank', '?')} is too short ({dur:.1f} s).")
    if shutil.disk_usage(out_path.parent if out_path.parent.exists() else work).free < 300_000_000:
        raise DiskSpaceError("Less than 300 MB free; not enough to render clips.")

    lines = clip.get("captions") or captions.build_lines(
        words, start, end, int(profile["captions"].get("words_per_line", 3)))
    clip["captions"] = lines
    layout = layout or profile["layout"]
    cap_style, hook_style = text_positions(profile, layout)
    captions.write_ass(work / "captions.ass", lines=lines, clip_start=start, clip_end=end,
                       cap_style=cap_style, hook_text=clip.get("hook_text", ""), hook_style=hook_style)

    punches = pick_punches(loudness, start, end, profile.get("zoom", {}))
    src_w, src_h = probe_size(source)
    audio = has_audio(source)
    vo = voiceover_wav is not None and voiceover_wav.exists() and clip.get("voiceover", False)
    graph = build_filtergraph(layout, punches, profile.get("zoom", {}), src_w=src_w, src_h=src_h,
                              vo=vo, vo_dur=vo_dur, vo_cfg=profile.get("voiceover", {}), audio=audio)
    (work / "graph.txt").write_text(graph, encoding="utf-8")

    tmp_out = out_path.with_name(out_path.stem + ".part.mp4")
    cmd = [find_ffmpeg(), "-y", "-hide_banner", "-v", "error", "-stats_period", "1", "-progress", "pipe:1",
           "-ss", f"{max(0.0, start - source_offset):.3f}", "-t", f"{dur:.3f}", "-i", str(source)]
    if vo:
        cmd += ["-i", str(voiceover_wav)]
    cmd += ["-/filter_complex", "graph.txt", "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-profile:v", "high",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart",
            "-t", f"{dur:.3f}", str(tmp_out)]

    import subprocess
    from .tools import NO_WINDOW
    proc = subprocess.Popen(cmd, cwd=str(work), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    try:
        for line in proc.stdout:
            if rep.cancelled:
                proc.kill()
                break
            if line.startswith("out_time_us="):
                try:
                    t = int(line.split("=", 1)[1]) / 1e6
                    clip["_render_frac"] = min(1.0, t / dur)
                except ValueError:
                    pass
        err = proc.stderr.read()
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    rep.check()
    if proc.returncode != 0:
        tmp_out.unlink(missing_ok=True)
        if "No space left" in err or "not enough space" in err.lower():
            raise DiskSpaceError("The disk filled up while rendering.")
        raise ClipperError(f"ffmpeg failed to render clip {clip.get('rank', '?')}.", err.strip()[-600:])
    tmp_out.replace(out_path)
    clip.pop("_render_frac", None)
    return out_path

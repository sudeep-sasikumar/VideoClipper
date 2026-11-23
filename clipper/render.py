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
REF_W = 1920  # facecam size is kept as it looks in a 1080p frame (native size for 1080p sources)


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


def overlay_geometry(layout: dict, src_w: int = 1920, src_h: int = 1080) -> dict:
    """Where things go in the 'overlay' layout (gameplay band + facecam in its original corner)."""
    gh = int(round(H * float(layout.get("overlay_game_ratio", 0.70)) / 2) * 2)
    top = (H - gh) // 2
    fc = layout.get("facecam", {})
    cw = int(round(min(float(fc.get("w", 0.22)) * REF_W * float(layout.get("cam_scale", 1.0)), W * 0.55) / 2) * 2)
    ch = int(round(cw * (float(fc.get("h", 0.22)) * src_h) / (float(fc.get("w", 0.22)) * src_w) / 2) * 2)
    m = 20
    right = float(fc.get("x", 0.78)) + float(fc.get("w", 0.22)) / 2 > 0.5
    bottom = float(fc.get("y", 0.0)) + float(fc.get("h", 0.22)) / 2 > 0.5
    x = W - cw - m - 8 if right else m
    y = top + gh - ch - m - 8 if bottom else top + m
    return {"game_h": gh, "top": top, "cam_w": cw, "cam_h": ch, "cam_x": x, "cam_y": y}


def build_filtergraph(layout: dict, punches: list[float], zoom: dict, *, src_w: int, src_h: int,
                      vo: bool, vo_dur: float, vo_cfg: dict, audio: bool, ex: dict | None = None) -> str:
    """ex (extras): {"teaser_input": idx | None, "pre": s, "post": s, "dur": s, "thumb_input": idx | None,
    "vo_input": idx}. The teaser comes from its own input (seeked straight to the peak moment), so
    ffmpeg never has to hold the frames in between in memory."""
    ex = ex or {}
    pre, post, dur = float(ex.get("pre", 0)), float(ex.get("post", 0)), float(ex.get("dur", 0))
    mode = layout.get("mode", "blur")
    g = []

    # ---- source video (+ teaser: a flash-forward of the peak moment before the clip starts)
    tin = ex.get("teaser_input")
    if tin is not None:
        g.append(f"[{tin}:v]fps={FPS},setsar=1,setpts=PTS-STARTPTS[vt]")
        g.append(f"[0:v]fps={FPS},setsar=1,setpts=PTS-STARTPTS[vm]")
        g.append("[vt][vm]concat=n=2:v=1:a=0[vin]")
    else:
        g.append(f"[0:v]fps={FPS},setsar=1[vin]")

    if mode == "split":
        g.append("[vin]split=2[a0][b0]")
        top_h = int(round(H * float(layout.get("split_top_ratio", 0.38)) / 2) * 2)
        bot_h = H - top_h
        fc = layout.get("facecam", {})
        fx, fy = float(fc.get("x", 0)), float(fc.get("y", 0))
        fw, fh = float(fc.get("w", 0.3)), float(fc.get("h", 0.3))
        g.append(f"[a0]crop=iw*{fw:.4f}:ih*{fh:.4f}:iw*{fx:.4f}:ih*{fy:.4f},"
                 f"scale={W}:{top_h}:force_original_aspect_ratio=increase,crop={W}:{top_h}[top]")
        g.append(f"[b0]scale={W}:{bot_h}:force_original_aspect_ratio=increase,crop={W}:{bot_h}[bot]")
        g.append("[top][bot]vstack=inputs=2[base]")
    elif mode == "overlay":
        geo = overlay_geometry(layout, src_w, src_h)
        fc = layout.get("facecam", {})
        g.append("[vin]split=3[a0][b0][c0]")
        g.append(f"[a0]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
                 f"boxblur=12:2,eq=brightness=-0.10:saturation=1.1,scale={W}:{H}[bg]")
        g.append(f"[b0]scale={W}:{geo['game_h']}:force_original_aspect_ratio=increase,crop={W}:{geo['game_h']}[game]")
        g.append(f"[c0]crop=iw*{float(fc['w']):.4f}:ih*{float(fc['h']):.4f}:iw*{float(fc['x']):.4f}:"
                 f"ih*{float(fc['y']):.4f},scale={geo['cam_w']}:{geo['cam_h']},"
                 f"pad=iw+8:ih+8:4:4:color=white[cam]")
        g.append(f"[bg][game]overlay=0:{geo['top']}[bg2]")
        g.append(f"[bg2][cam]overlay={geo['cam_x']}:{geo['cam_y']}[base]")
    else:
        g.append("[vin]split=2[a0][b0]")
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

    cur = "base"
    # End card part 1: hold the last frame. This must come before zoompan: after a teaser concat,
    # zoompan's output doesn't let tpad extend the stream.
    if post > 0:
        g.append(f"[{cur}]tpad=stop_mode=clone:stop_duration={post:.3f}[held0]")
        cur = "held0"
    if punches:
        expr = _zoom_expr([p + pre for p in punches], float(zoom.get("amount", 0.08)), float(zoom.get("length", 0.7)))
        g.append(f"[{cur}]zoompan=z='{expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:"
                 f"s={W}x{H}:fps={FPS}[zb]")
        cur = "zb"

    # ---- end card part 2: darken the held frame, show the video's thumbnail
    if post > 0:
        t0 = pre + dur
        g.append(f"[{cur}]drawbox=x=0:y=0:w=iw:h=ih:color=black@0.6:t=fill:enable='gte(t,{t0:.3f})'[held]")
        cur = "held"
        if ex.get("thumb_input") is not None:
            g.append(f"[{ex['thumb_input']}:v]scale=760:-2,pad=iw+12:ih+12:6:6:color=white,setsar=1[th]")
            g.append(f"[{cur}][th]overlay=(W-w)/2:{int(H * 0.30)}:enable='gte(t,{t0:.3f})'[endc]")
            cur = "endc"
    g.append(f"[{cur}]ass=captions.ass,setsar=1,format=yuv420p[v]")

    # ---- audio
    if audio:
        if tin is not None:
            g.append(f"[{tin}:a]aresample=48000,asetpts=PTS-STARTPTS[at]")
            g.append("[0:a]aresample=48000,asetpts=PTS-STARTPTS[am]")
            g.append("[at][am]concat=n=2:v=0:a=1[ain]")
        else:
            g.append("[0:a]anull[ain]")
    else:
        g.append("anullsrc=r=48000:cl=stereo[ain]")
    mutes = ex.get("mutes") or []
    if mutes:
        cond = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in mutes)
        g.append(f"[ain]volume=0:enable='{cond}'[ainm]")
        g.append("[ainm]anull[ain2]")
    else:
        g.append("[ain]anull[ain2]")
    if vo:
        delay = 0.25
        duck = float(vo_cfg.get("duck", 0.35))
        vol = float(vo_cfg.get("volume", 1.0))
        a, b = delay, delay + vo_dur
        g.append(f"[ain2]aresample=48000,volume='if(between(t,{a:.2f},{b:.2f}),{duck},1)':eval=frame[orig]")
        g.append(f"[{ex.get('vo_input', 1)}:a]aresample=48000,aformat=channel_layouts=stereo,volume={vol},"
                 f"adelay={int(delay * 1000)}:all=1[vo]")
        g.append("[orig][vo]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[amix]")
    else:
        g.append("[ain2]aresample=48000,alimiter=limit=0.95[amix]")
    post_f = ex.get("audio_post") or "anull"   # e.g. a loudnorm pass for the final delivery encode
    if post > 0:
        t0 = pre + dur
        g.append(f"[amix]apad=pad_dur={post:.3f},afade=t=out:st={max(0.0, t0 - 0.4):.3f}:d={post + 0.4:.3f},"
                 f"{post_f}[a]")
    else:
        g.append(f"[amix]{post_f}[a]")
    return ";\n".join(g)


def text_positions(profile: dict, layout: dict) -> tuple[dict, dict]:
    """Caption and hook styles with positions that suit the layout.

    Split: the hook sits on the seam between facecam and gameplay (never over the face) and captions
    sit in the gameplay half. Overlay: the hook sits in the band above the gameplay.
    Profiles can override with "split_position_y" / "overlay_position_y".
    """
    cap, hook = dict(profile["captions"]), dict(profile["hook"])
    mode = layout.get("mode")
    if mode == "split":
        seam = float(layout.get("split_top_ratio", 0.38))
        hook["position_y"] = float(hook.get("split_position_y", seam))
        cap["position_y"] = float(cap.get("split_position_y", max(seam + 0.25, 0.66)))
    elif mode == "overlay":
        band = (1 - float(layout.get("overlay_game_ratio", 0.70))) / 2
        hook["position_y"] = float(hook.get("overlay_position_y", band / 2))
        cap["position_y"] = float(cap.get("overlay_position_y", cap.get("position_y", 0.72)))
    return cap, hook


def mute_ranges(clip: dict, pre: float = 0.0, teaser=None) -> list[tuple[float, float]]:
    """clip["mute"] source-time ranges -> output-time ranges (main part, plus the teaser if it overlaps)."""
    start, end = float(clip["start"]), float(clip["end"])
    out = []
    for a, b in clip.get("mute") or []:
        a, b = max(float(a), start), min(float(b), end)
        if b <= a:
            continue
        out.append((a - start + pre, b - start + pre))
        if teaser:
            ts, te = teaser
            ta, tb = max(a, start + ts), min(b, start + te)
            if tb > ta:
                out.append((ta - start - ts, tb - start - ts))
    return out


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
                voiceover_wav: Path | None = None, vo_dur: float = 0.0, layout: dict | None = None,
                extras: dict | None = None, crf: int = 21, audio_bitrate: str = "160k",
                audio_post: str = "") -> Path:
    """Render clip["start"]..clip["end"] (absolute source seconds) to out_path.

    source_offset: the absolute time that `source` starts at (0 for the full video, or the start
    of a downloaded section when re-rendering after the full video was deleted).
    extras: optional plan from project.extras_plan (teaser, banner, callouts, end card).
    clip["mute"]: [[start, end], ...] source-time ranges to silence (e.g. a slur), kept on re-render.
    crf / audio_bitrate / audio_post: encode settings and a final audio filter (e.g. loudnorm).
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

    ex = dict(extras or {})
    pre = float(ex.get("pre", 0.0))
    post = float(ex.get("post", 0.0))
    thumb = Path(ex["thumb"]).resolve() if post > 0 and ex.get("thumb") and Path(ex["thumb"]).exists() else None
    total = pre + dur + post

    lines = clip.get("captions") or captions.build_lines(
        words, start, end, int(profile["captions"].get("words_per_line", 3)))
    clip["captions"] = lines
    layout = layout or profile["layout"]
    cap_style, hook_style = text_positions(profile, layout)
    extra_styles, extra_events = captions.extra_events(ex, layout_mode=layout.get("mode", "blur"),
                                                      hook_style=hook_style, dur=dur) if ex else ([], [])
    captions.write_ass(work / "captions.ass", lines=lines, clip_start=start, clip_end=end,
                       cap_style=cap_style, hook_text=clip.get("hook_text", ""), hook_style=hook_style,
                       offset=pre, total=total, extra_styles=extra_styles, extra_events=extra_events,
                       clean=bool(profile.get("censor_captions", True)))

    punches = pick_punches(loudness, start, end, profile.get("zoom", {}))
    src_w, src_h = probe_size(source)
    audio = has_audio(source)
    vo = voiceover_wav is not None and voiceover_wav.exists() and clip.get("voiceover", False)
    teaser = ex.get("teaser") if pre > 0 else None
    inputs = [["-ss", f"{max(0.0, start - source_offset):.3f}", "-t", f"{dur:.3f}", "-i", str(source)]]
    teaser_input = vo_input = thumb_input = None
    if teaser:
        ts, te = teaser
        teaser_input = len(inputs)
        inputs.append(["-ss", f"{max(0.0, start - source_offset + ts):.3f}", "-t", f"{te - ts:.3f}",
                       "-i", str(source)])
    if vo:
        vo_input = len(inputs)
        inputs.append(["-i", str(voiceover_wav)])
    if thumb:
        thumb_input = len(inputs)
        inputs.append(["-i", str(thumb)])
    graph = build_filtergraph(layout, punches, profile.get("zoom", {}), src_w=src_w, src_h=src_h,
                              vo=vo, vo_dur=vo_dur, vo_cfg=profile.get("voiceover", {}), audio=audio,
                              ex={"teaser_input": teaser_input, "pre": pre, "post": post, "dur": dur,
                                  "thumb_input": thumb_input, "vo_input": vo_input,
                                  "mutes": mute_ranges(clip, pre, teaser), "audio_post": audio_post})
    (work / "graph.txt").write_text(graph, encoding="utf-8")

    tmp_out = out_path.with_name(out_path.stem + ".part.mp4")
    cmd = [find_ffmpeg(), "-y", "-hide_banner", "-v", "error", "-stats_period", "1", "-progress", "pipe:1"]
    for part in inputs:
        cmd += part
    cmd += ["-/filter_complex", "graph.txt", "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf)), "-profile:v", "high",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000", "-movflags", "+faststart",
            "-t", f"{total:.3f}", str(tmp_out)]

    import subprocess
    from .tools import NO_WINDOW
    log_path = work / "ffmpeg.log"
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_f:  # a file never blocks ffmpeg
        proc = subprocess.Popen(cmd, cwd=str(work), stdout=subprocess.PIPE, stderr=log_f, text=True,
                                encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
        try:
            for line in proc.stdout:
                if rep.cancelled:
                    proc.kill()
                    break
                if line.startswith("out_time_us="):
                    try:
                        t = int(line.split("=", 1)[1]) / 1e6
                        clip["_render_frac"] = min(1.0, t / total)
                    except ValueError:
                        pass
            proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
    err = log_path.read_text(encoding="utf-8", errors="replace")
    rep.check()
    if proc.returncode != 0:
        tmp_out.unlink(missing_ok=True)
        if "No space left" in err or "not enough space" in err.lower():
            raise DiskSpaceError("The disk filled up while rendering.")
        raise ClipperError(f"ffmpeg failed to render clip {clip.get('rank', '?')}.", err.strip()[-600:])
    tmp_out.replace(out_path)
    clip.pop("_render_frac", None)
    return out_path

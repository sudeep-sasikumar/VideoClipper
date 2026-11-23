"""A finished job on disk: project.json + transcript.json + clips + reports.

The GUI edits clips through these functions so a single clip can be re-rendered later,
even after the downloaded source video was deleted (only that part is fetched again).
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import captions, facecam, render, voiceover
from .download import download_section
from .errors import ClipperError
from .events import Reporter
from .paths import TEMP_ROOT, safe_name, slug
from .timefmt import fmt_precise, fmt_ts

SECTION_PAD = 30.0


def clip_filename(clip: dict) -> str:
    return f"{clip['rank']:02d}_score{int(round(clip['score']))}_{slug(clip.get('title') or clip.get('hook_text') or 'clip')}.mp4"


def load(folder: Path) -> dict:
    p = Path(folder) / "project.json"
    if not p.exists():
        raise ClipperError(f"No project.json in {folder}.", "Open a folder that Clipper created.")
    proj = json.loads(p.read_text(encoding="utf-8"))
    proj["_folder"] = str(Path(folder).resolve())
    return proj


def save(proj: dict) -> None:
    folder = Path(proj["_folder"])
    data = {k: v for k, v in proj.items() if not k.startswith("_")}
    tmp = folder / "project.json.tmp"
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(folder / "project.json")
    write_reports(proj)


def words_of(proj: dict) -> list[list]:
    cache = proj.get("_words")
    if cache is None:
        tr = json.loads((Path(proj["_folder"]) / "transcript.json").read_text(encoding="utf-8"))
        cache = [w for s in tr["segments"] for w in s["words"]]
        proj["_words"] = cache
    return cache


def _covers(path: Path, offset: float, clip: dict) -> bool:
    """Both video and audio in `path` (which starts at `offset`) last through the clip's end."""
    from .tools import find_ffmpeg, run
    r = run([find_ffmpeg("ffprobe"), "-v", "error", "-show_entries", "stream=codec_type,duration",
             "-of", "json", str(path)])
    try:
        streams = json.loads(r.stdout).get("streams", [])
    except ValueError:
        return False
    need = clip["end"] - offset - 0.3
    durs = {}
    for st in streams:
        try:
            durs[st.get("codec_type")] = float(st.get("duration"))
        except (TypeError, ValueError):
            continue
    return "video" in durs and durs["video"] >= need and durs.get("audio", need) >= need


def _source_for(proj: dict, clip: dict, rep: Reporter) -> tuple[Path, float]:
    """(video path, absolute time it starts at) covering the clip."""
    src = proj.get("source_path")
    if src and Path(src).exists():
        return Path(src), 0.0
    dur = float(proj["video"]["duration"])
    a = int(max(0.0, clip["start"] - SECTION_PAD))
    b = int(min(dur - 0.5, clip["end"] + SECTION_PAD)) + 1   # never ask past the end of the video
    b = max(b, int(clip["end"]) + 1) if clip["end"] + 1 <= dur else b
    sec_dir = TEMP_ROOT / f"sections_{proj['video']['id']}"
    # Reuse a cached section that covers this clip (and really contains it: checked, not assumed).
    for p in sec_dir.glob("section_*_*.mp4"):
        try:
            s = float(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if s <= clip["start"]:
            if _covers(p, s, clip):
                return p, s
    rep.log(f"Fetching {fmt_ts(a)}-{fmt_ts(b)} of the source for re-rendering...")
    for precise in (True, False):
        p = download_section(proj["video"]["url"], a, b, sec_dir, rep, precise=precise)
        if _covers(p, float(a), clip):
            return p, float(a)
        p.unlink(missing_ok=True)
        if precise:
            rep.log("That part came back incomplete; fetching it again another way...")
    raise ClipperError("Couldn't fetch that part of the video cleanly for re-rendering.",
                       "Try again, or turn on Settings > Keep the downloaded video before the next run.")


def render_one(proj: dict, clip: dict, rep: Reporter, source: Path | None = None,
               source_offset: float = 0.0, work_root: Path | None = None) -> Path:
    folder = Path(proj["_folder"])
    profile = proj["profile"]
    work = (work_root or TEMP_ROOT) / f"render_{proj['video']['id']}_{clip['rank']:02d}"
    if source is None:
        source, source_offset = _source_for(proj, clip, rep)

    vo_wav, vo_dur = None, 0.0
    if clip.get("voiceover") and clip.get("intro_line"):
        vo_wav = work / "voiceover.wav"
        vo_dur = voiceover.synth(clip["intro_line"], vo_wav, profile["voiceover"].get("voice", "af_heart"),
                                 profile["voiceover"].get("speed", 1.0), rep)

    layout = effective_layout(proj, clip, source, source_offset, rep)
    plan = extras_plan(proj, clip)
    new_name = clip_filename(clip)
    old = clip.get("file")
    out = folder / new_name
    render.render_clip(source=source, source_offset=source_offset, clip=clip, words=words_of(proj),
                       profile=profile, loudness=proj.get("signals", {}).get("loudness", []),
                       out_path=out, work=work, rep=rep, voiceover_wav=vo_wav, vo_dur=vo_dur, layout=layout,
                       extras=plan)
    if old and old != new_name:
        (folder / old).unlink(missing_ok=True)
    clip["file"] = new_name
    clip["rendered_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    clip.pop("error", None)
    shutil.rmtree(work, ignore_errors=True)
    return out


LAYOUTS = {
    "overlay": "Facecam corner (original size)",
    "auto": "Facecam on top (big)",
    "split": "Facecam on top (forced)",
    "blur": "Blurred background, no facecam",
}


def detect_facecams(proj: dict, source: Path, source_offset: float, rep: Reporter) -> None:
    """Look for the facecam in every clip first, so clips where detection misses can use the
    position found in the rest of the video (a streamer's cam rarely moves during a VOD)."""
    w, h = render.probe_size(source)
    for c in proj["clips"]:
        if "facecam_box" not in c:
            c["facecam_box"] = facecam.detect(source, c["start"] - source_offset, c["end"] - source_offset,
                                              w, h, rep) or False
    found = sum(1 for c in proj["clips"] if c.get("facecam_box"))
    rep.log(f"Facecam found in {found} of {len(proj['clips'])} clip(s).")


def video_facecam(proj: dict, exclude: dict | None = None) -> dict | None:
    """The facecam box the video's clips agree on, or None.

    Needs at least 2 detections covering half the clips, most within a few percent of the median."""
    others = [c for c in proj.get("clips", []) if c is not exclude]
    boxes = [c["facecam_box"] for c in others if c.get("facecam_box")]
    if len(boxes) < 2 or len(boxes) < 0.5 * max(1, len(others)):
        return None
    med = {k: float(np.median([b[k] for b in boxes])) for k in ("x", "y", "w", "h")}
    mcx, mcy = med["x"] + med["w"] / 2, med["y"] + med["h"] / 2
    agree = [b for b in boxes if abs(b["x"] + b["w"] / 2 - mcx) < 0.06 and abs(b["y"] + b["h"] / 2 - mcy) < 0.06]
    return med if len(agree) >= 0.75 * len(boxes) else None


def webcam_shape(box: dict, src_w: int, src_h: int, max_aspect_h: float = 1 / 1.65) -> dict:
    """Trim a box that is taller than a webcam (16:9-ish): the extra rows are usually chat or labels
    under the cam (with platform icons in them), so the bottom is cut."""
    b = {k: float(box[k]) for k in ("x", "y", "w", "h")}
    max_h = b["w"] * src_w * max_aspect_h / src_h
    if b["h"] > max_h * 1.04:
        b["h"] = max_h
    # Overlays often put the newest chat line or a name tag over the cam's bottom edge.
    b["h"] = round(b["h"] * 0.93, 4)
    return b


def effective_layout(proj: dict, clip: dict, source: Path, source_offset: float, rep: Reporter) -> dict:
    """Layout for this clip. Per-clip choice ("layout") overrides the profile's mode.

    overlay -> facecam kept at its original size in its original corner, gameplay in the middle
    auto    -> facecam enlarged on top, gameplay below
    split   -> like auto, but forced even when no facecam was detected in this clip
    blur    -> whole frame on a blurred background
    overlay/auto fall back to the cam position agreed by the video's other clips, then to blur.
    """
    base = dict(proj["profile"]["layout"])
    forced = bool(clip.get("layout"))
    choice = clip.get("layout") or base.get("mode", "blur")
    if choice not in LAYOUTS:
        choice = "blur"
    clip["layout_used"] = "blur"
    if choice == "blur":
        return dict(base, mode="blur")
    if "facecam_box" not in clip:
        w, h = render.probe_size(source)
        clip["facecam_box"] = facecam.detect(source, clip["start"] - source_offset, clip["end"] - source_offset,
                                             w, h, rep) or False
    mode = "overlay" if choice == "overlay" else "split"
    w, h = render.probe_size(source)
    box = clip["facecam_box"]
    vbox = video_facecam(proj, exclude=clip)
    clip.pop("facecam_used", None)
    if box and vbox:
        area, varea = box["w"] * box["h"], vbox["w"] * vbox["h"]
        far = (abs(box["x"] + box["w"] / 2 - vbox["x"] - vbox["w"] / 2) > 0.08
               or abs(box["y"] + box["h"] / 2 - vbox["y"] - vbox["h"] / 2) > 0.08)
        if area > 2.5 * varea and not forced:
            # The streamer switched to a full-screen camera (e.g. "just chatting" at the end of a
            # stream): show the whole frame, zoomed in, instead of cropping the usual cam corner.
            clip["layout_used"] = "full frame (streamer on full-screen camera)"
            rep.log(f"Clip {clip['rank']}: streamer is on a full-screen camera here; showing the whole frame.")
            return dict(base, mode="blur", foreground_zoom=max(1.5, float(base.get("foreground_zoom", 1.0))))
        if far or area < 0.75 * varea:
            # A "cam" somewhere else (a still face in a cutscene) or a much tighter box than usual:
            # use the position and size the rest of the video agrees on, so every clip is framed alike.
            box = None
            clip["layout_used"] = f"{mode} (facecam position from the rest of the video)"
            rep.log(f"Clip {clip['rank']}: using the usual facecam spot.")
            used = webcam_shape(vbox, w, h)
            clip["facecam_used"] = used
            return dict(base, mode=mode, facecam=used)
    if box:
        clip["layout_used"] = mode
        used = webcam_shape(box, w, h)
        clip["facecam_used"] = used
        rep.log(f"Clip {clip['rank']}: {facecam.describe(used)}.")
        return dict(base, mode=mode, facecam=used)
    if vbox is None and forced:
        others = [c["facecam_box"] for c in proj.get("clips", []) if c is not clip and c.get("facecam_box")]
        if others:
            vbox = {k: float(np.median([b[k] for b in others])) for k in ("x", "y", "w", "h")}
    if vbox:
        clip["layout_used"] = f"{mode} (facecam position from the rest of the video)"
        rep.log(f"Clip {clip['rank']}: facecam not seen here; using its position from the rest of the video.")
        used = webcam_shape(vbox, w, h)
        clip["facecam_used"] = used
        return dict(base, mode=mode, facecam=used)
    rep.log(f"Clip {clip['rank']}: no facecam found.")
    if choice == "split":
        clip["layout_used"] = "split (profile box)"
        return dict(base, mode="split")
    return dict(base, mode="blur")


# ------------------------------------------------------------------ extras

TEASER_LEN = 1.4
END_CARD_LEN = 1.8


def extras_on(proj: dict, clip: dict) -> bool:
    return bool(clip["extras"]) if "extras" in clip else bool(proj.get("extras", False))


def name_label(proj: dict, clip: dict) -> str:
    """On-screen label that guarantees the required name (Settings > Name on every clip) appears.

    Empty when no name is set or the hook already says it; otherwise "NAME · context" (or the name)."""
    name = (proj.get("overlay_name") or "").strip()
    if not name:
        return ""
    first = name.split()[0].lower()
    context = (clip.get("context") or "").strip()
    if first in (clip.get("hook_text") or "").lower() and (not context or first in context.lower()):
        return context
    if context and first in context.lower():
        return context
    return f"{name} \u2022 {context}" if context else name


def extras_plan(proj: dict, clip: dict) -> dict | None:
    """What extra elements this clip gets, fitted inside the clip-length limit (end card goes first)."""
    if not extras_on(proj, clip):
        label = name_label(proj, clip)
        # Extras off but a name is required on screen: just the label.
        return {"pre": 0.0, "post": 0.0, "teaser": None, "banner": label, "callouts": []} if label else None
    start, end = float(clip["start"]), float(clip["end"])
    dur = end - start
    max_len = float(proj.get("length", {}).get("max", 60))
    plan: dict = {"pre": 0.0, "post": 0.0, "teaser": None}
    # Teaser: flash-forward to the loudest moment (not in the first 4 s: that's already the opening).
    spikes = [x for x in proj.get("signals", {}).get("loudness", []) if start + 4 <= x["t"] <= end - 0.5]
    if spikes:
        peak = max(spikes, key=lambda x: x["z"])["t"] - start
        ts = max(0.0, peak - 0.9)
        plan.update(teaser=(round(ts, 3), round(min(dur, ts + TEASER_LEN), 3)), pre=TEASER_LEN,
                    teaser_text="WAIT FOR IT...")
    # End card: the video's thumbnail and where to find the full video.
    thumb = proj.get("thumbnail")
    if thumb and proj.get("extras_end_card", True) and (Path(proj["_folder"]) / thumb).exists():
        uploader = proj["video"].get("uploader") or ""
        title = proj["video"]["title"]
        plan.update(post=END_CARD_LEN, thumb=str(Path(proj["_folder"]) / thumb),
                    end_lines=["Watch the full stream", f"{uploader}: {title}" if uploader else title])
    if dur + plan["pre"] + plan["post"] > max_len + 0.01:
        plan.update(post=0.0)
    if dur + plan["pre"] + plan["post"] > max_len + 0.01:
        plan.update(pre=0.0, teaser=None)
    plan["banner"] = name_label(proj, clip) or clip.get("context") or ""
    plan["callouts"] = [(float(t) - start, txt) for t, txt in clip.get("callouts", [])
                        if start + 3.2 <= float(t) <= end - 0.8]
    return plan


def edit_clip(proj: dict, clip: dict, *, start=None, end=None, hook_text=None, caption_lines=None,
              voiceover_on=None, intro_line=None, layout=None, extras=None) -> None:
    """Apply GUI edits. Changing start/end rebuilds captions from the transcript."""
    if start is not None or end is not None:
        s = float(start if start is not None else clip["start"])
        e = float(end if end is not None else clip["end"])
        s = max(0.0, s)
        e = min(float(proj["video"]["duration"]), e)
        if e - s < 1.0:
            raise ClipperError("End must be at least 1 second after start.")
        if (s, e) != (clip["start"], clip["end"]):
            clip["start"], clip["end"] = round(s, 2), round(e, 2)
            clip.pop("facecam_box", None)  # different moment: look for the facecam again
            clip["captions"] = captions.build_lines(
                words_of(proj), s, e, int(proj["profile"]["captions"].get("words_per_line", 3)))
    if hook_text is not None:
        clip["hook_text"] = hook_text.strip()
    if caption_lines is not None:
        lines = clip.get("captions") or captions.build_lines(
            words_of(proj), clip["start"], clip["end"], int(proj["profile"]["captions"].get("words_per_line", 3)))
        new = []
        for i, ln in enumerate(lines):
            text = caption_lines[i] if i < len(caption_lines) else ""
            if text.strip():
                new.append(captions.retime_line(ln, text))
        clip["captions"] = new
    if voiceover_on is not None:
        clip["voiceover"] = bool(voiceover_on)
    if intro_line is not None:
        clip["intro_line"] = intro_line.strip()
    if layout is not None and layout in LAYOUTS:
        clip["layout"] = layout
    if extras is not None:
        clip["extras"] = bool(extras)


def delete_clip(proj: dict, clip: dict) -> None:
    folder = Path(proj["_folder"])
    if clip.get("file"):
        (folder / clip["file"]).unlink(missing_ok=True)
    proj["clips"] = [c for c in proj["clips"] if c is not clip]
    save(proj)


def write_reports(proj: dict) -> None:
    folder = Path(proj["_folder"])
    v = proj["video"]
    rows = []
    for c in sorted(proj["clips"], key=lambda c: c["rank"]):
        rows.append({
            "rank": c["rank"], "file": c.get("file"), "score": c["score"],
            "start": fmt_precise(c["start"]), "end": fmt_precise(c["end"]),
            "start_seconds": c["start"], "end_seconds": c["end"],
            "duration": round(c["end"] - c["start"], 1),
            "hook_text": c.get("hook_text", ""), "title": c.get("title", ""), "reason": c.get("reason", ""),
            "voiceover": bool(c.get("voiceover")), "intro_line": c.get("intro_line", ""),
            "post_caption": c.get("post_caption", ""),
            "layout": c.get("layout_used", ""), "facecam": c.get("facecam_used") or c.get("facecam_box") or None,
            "extras": extras_on(proj, c), "context": c.get("context", ""), "callouts": c.get("callouts", []),
            "source_link": _link_at(v["url"], c["start"]),
            **({"claude_suggested": [fmt_precise(x) for x in c["suggested"]]} if c.get("suggested") else {}),
            **({"error": c["error"]} if c.get("error") else {}),
        })
    report = {"video": {"title": v["title"], "url": v["url"], "duration": v["duration"]},
              "profile": proj["profile"].get("name"), "campaign_rules": proj.get("rules", ""),
              "clip_length": proj.get("length"), "generated_at": proj.get("created_at"), "clips": rows}
    (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"Clipper report: {v['title']}", f"Source: {v['url']}", f"Profile: {proj['profile'].get('name')}",
             f"Clip length: {proj['length']['min']:.0f}-{proj['length']['max']:.0f} s", ""]
    if proj.get("rules"):
        lines += ["Campaign rules:", proj["rules"].strip(), ""]
    for r in rows:
        lines += [f"#{r['rank']:02d}  score {r['score']:.0f}/10  {fmt_ts(r['start_seconds'], True)} -> "
                  f"{fmt_ts(r['end_seconds'], True)}  ({r['duration']:.0f} s)",
                  f"    File:   {r['file'] or '(not rendered: ' + r.get('error', '') + ')'}",
                  f"    Hook:   {r['hook_text']}",
                  f"    Why:    {r['reason']}",
                  f"    Link:   {r['source_link']}"]
        if r["layout"]:
            lines.append(f"    Layout: {r['layout']}" + (f" ({facecam.describe(r['facecam'])})" if r["facecam"] else ""))
        if r["post_caption"]:
            lines.append(f"    Post:   {r['post_caption']}")
        if r["voiceover"]:
            lines.append(f"    Intro:  {r['intro_line']}")
        lines.append("")
    (folder / "report.txt").write_text("\n".join(lines), encoding="utf-8")


def _link_at(url: str, t: float) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={int(t)}s" if ("youtu" in url or "twitch" in url) else url


def output_folder(base: Path, title: str, video_id: str) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    name = safe_name(title, 70)
    folder = base / name
    n = 2
    while folder.exists() and any(folder.iterdir()):
        folder = base / f"{name} ({n})"
        n += 1
    folder.mkdir(parents=True, exist_ok=True)
    return folder

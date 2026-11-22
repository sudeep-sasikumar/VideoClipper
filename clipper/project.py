"""A finished job on disk: project.json + transcript.json + clips + reports.

The GUI edits clips through these functions so a single clip can be re-rendered later,
even after the downloaded source video was deleted (only that part is fetched again).
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from . import captions, render, voiceover
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


def _source_for(proj: dict, clip: dict, rep: Reporter) -> tuple[Path, float]:
    """(video path, absolute time it starts at) covering the clip."""
    src = proj.get("source_path")
    if src and Path(src).exists():
        return Path(src), 0.0
    a = max(0.0, clip["start"] - SECTION_PAD)
    b = min(proj["video"]["duration"], clip["end"] + SECTION_PAD)
    sec_dir = TEMP_ROOT / f"sections_{proj['video']['id']}"
    # Reuse a cached section that already covers this clip.
    for p in sec_dir.glob("section_*_*.mp4"):
        try:
            _, s, e = p.stem.split("_")
            if float(s) <= clip["start"] and float(e) >= clip["end"]:
                return p, float(s)
        except ValueError:
            continue
    rep.log(f"Fetching {fmt_ts(a)}-{fmt_ts(b)} of the source for re-rendering...")
    p = download_section(proj["video"]["url"], int(a), int(b) + 1, sec_dir, rep)
    return p, float(int(a))


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

    new_name = clip_filename(clip)
    old = clip.get("file")
    out = folder / new_name
    render.render_clip(source=source, source_offset=source_offset, clip=clip, words=words_of(proj),
                       profile=profile, loudness=proj.get("signals", {}).get("loudness", []),
                       out_path=out, work=work, rep=rep, voiceover_wav=vo_wav, vo_dur=vo_dur)
    if old and old != new_name:
        (folder / old).unlink(missing_ok=True)
    clip["file"] = new_name
    clip["rendered_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    clip.pop("error", None)
    shutil.rmtree(work, ignore_errors=True)
    return out


def edit_clip(proj: dict, clip: dict, *, start=None, end=None, hook_text=None, caption_lines=None,
              voiceover_on=None, intro_line=None) -> None:
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
            "source_link": _link_at(v["url"], c["start"]),
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

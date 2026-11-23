"""Run a whole job: link in, folder of ranked clips out."""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from . import claude_cli, download, hooks, project, rank, signals, transcribe
from .errors import BadLinkError, Cancelled, ClipperError, DiskSpaceError
from .events import Reporter
from .paths import APP_DIR, DEFAULT_SETTINGS, TEMP_ROOT, ensure_dirs, safe_name
from .rules import effective_length
from .tools import check_disk, find_claude, find_ffmpeg, media_duration


@dataclass
class JobConfig:
    url: str
    profile: dict
    rules: str = ""
    output_dir: str = ""
    whisper_model: str = ""      # "" = profile's choice
    keep_source: bool = False
    max_clips: int = 20
    claude_model: str = ""
    hook_pass: bool = True       # second Claude pass that rewrites hooks to stop the scroll
    audio_first: bool | None = None  # None = automatic (long VODs, or not enough disk for the video)
    extras: bool = False         # teaser, context banner, callouts, end card
    end_card: bool = True        # thumbnail end card as part of the extras
    overlay_name: str = ""       # name that must appear on screen in every clip


AUDIO_FIRST_HOURS = 5.0   # longer videos download audio first and fetch only the clips' video


def _save_thumbnail(info: dict, folder: Path, rep: Reporter) -> str | None:
    """Save the video's thumbnail next to the clips (used by the end card). Returns the file name."""
    import urllib.request
    url = info.get("thumbnail")
    if not url:
        return None
    ext = url.split("?")[0].rsplit(".", 1)[-1].lower()
    ext = ext if ext in ("jpg", "jpeg", "png", "webp") else "jpg"
    dest = folder / f"thumbnail.{ext}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            dest.write_bytes(r.read())
        return dest.name
    except OSError as e:
        rep.log(f"Couldn't fetch the video thumbnail ({e}); end cards will be skipped.")
        return None


def preflight(rep: Reporter, claude_model: str = "") -> None:
    rep.log("Checking tools...")
    find_ffmpeg()
    find_ffmpeg("ffprobe")
    rep.log(f"Claude CLI: {find_claude()}")
    claude_cli.check_login(claude_model)
    rep.log("Claude CLI is logged in.")
    rep.progress("check", 1.0)


def run_job(cfg: JobConfig, rep: Reporter) -> Path:
    """Returns the output folder. Raises ClipperError (or Cancelled) with a user-facing message."""
    ensure_dirs()
    t0 = time.monotonic()
    profile = cfg.profile
    find_ffmpeg()
    find_ffmpeg("ffprobe")

    # ---- 1. download (link is checked first: free and fast; then Claude, before hours of work)
    rep.log("Reading the link...")
    info = download.probe(cfg.url, rep)
    preflight(rep, cfg.claude_model)
    title = info.get("title") or info.get("id") or "video"
    vid = safe_name(f"{info.get('extractor_key', 'video')}_{info.get('id', 'x')}", 60).replace(" ", "_")
    duration = float(info.get("duration") or 0)
    rep.log(f"Video: \"{title}\" ({duration / 60:.0f} min).")
    if duration > 4.1 * 3600:
        rep.log(f"Note: this video is {duration / 3600:.1f} h, longer than the 4 h Clipper is tuned for.")

    out_base = Path(cfg.output_dir or DEFAULT_SETTINGS["output_dir"])
    check_disk(out_base, 800_000_000, "the finished clips")
    work = TEMP_ROOT / vid
    work.mkdir(parents=True, exist_ok=True)

    video = next((p for p in work.glob("source.*") if p.suffix in (".mp4", ".mkv", ".webm")
                  and ".temp" not in p.name and not p.name.endswith(".part")), None)
    if video and abs(media_duration(video) - duration) >= 5:
        video = None
    audio_first = cfg.audio_first
    if audio_first is None and video is None:
        need = int(download.estimate_size(info) * 2.2) + 1_000_000_000
        free = shutil.disk_usage(work).free
        audio_first = duration > AUDIO_FIRST_HOURS * 3600 or free < need
        if audio_first:
            why = (f"it is {duration / 3600:.1f} h long" if duration > AUDIO_FIRST_HOURS * 3600
                   else f"only {free / 1e9:.0f} GB is free for about {need / 1e9:.0f} GB of video")
            rep.log(f"Audio-first mode ({why}): downloading only the audio now; each clip's video is "
                    "fetched on its own when it is rendered.")
    if video:
        rep.log("Reusing the video downloaded in an earlier run.")
        rep.progress("download", 1.0)
        src = video
    elif audio_first:
        src = next((p for p in work.glob("source_audio.*") if not p.name.endswith((".part", ".ytdl"))), None)
        if src and abs(media_duration(src) - duration) < 30:
            rep.log("Reusing the audio downloaded in an earlier run.")
            rep.progress("download", 1.0)
        else:
            rep.log("Downloading the audio...")
            try:
                src = download.download_audio(info, work, rep)
            except BadLinkError as e:
                # Some sites refuse audio-only streams; the full video still works.
                rep.log(f"Audio-only download failed ({e.hint or e.message}); downloading the video instead.")
                for p in work.glob("source_audio.*"):
                    p.unlink(missing_ok=True)
                src = video = download.download_video(info, work, rep)
    else:
        rep.log("Downloading video (up to 1080p)...")
        src = video = download.download_video(info, work, rep)
    duration = media_duration(src) or duration

    chat_file = work / "chat_times.json"
    if chat_file.exists():
        chat_times = json.loads(chat_file.read_text())
    else:
        chat_times = download.download_chat(info, work, rep)
        chat_file.write_text(json.dumps(chat_times))
    rep.check()

    # ---- 2. transcribe
    wav = transcribe.extract_audio(src, work / "audio.wav", rep)
    model = cfg.whisper_model or profile.get("whisper_model") or "small"
    tr = transcribe.transcribe(wav, work / "transcript.json", duration, model, rep,
                               language=profile.get("language"), vad=bool(profile.get("vad_filter", False)))
    rep.check()

    # ---- 3. signals
    rep.log("Finding loudness and chat spikes...")
    sig = signals.compute(wav, chat_times, duration)
    rep.log(f"Signals: {len(sig['loudness'])} loudness spikes, {len(sig['chat'])} chat spikes.")
    rep.progress("signals", 1.0)

    # ---- 4. rank
    length = effective_length(profile, cfg.rules)
    if length["from_rules"]:
        rep.log(f"Campaign rules set clip length to {length['min']:.0f}-{length['max']:.0f} s "
                f"(found: {'; '.join(length['from_rules'])}).")
    rank_length = length
    if cfg.extras:
        # Leave room for the teaser and end card so the finished clip still fits the limit.
        room = project.TEASER_LEN + (project.END_CARD_LEN if cfg.end_card else 0.0)
        if length["max"] - room >= length["min"] + 5:
            rank_length = dict(length, max=length["max"] - room, target=min(length["target"], length["max"] - room))
    clips = rank.rank(title=title, transcript=tr, signals=sig, profile=profile, length=rank_length,
                      rules=cfg.rules, duration=duration, rep=rep, cache=work / "ranking.json",
                      max_clips=cfg.max_clips, model=cfg.claude_model)

    if cfg.hook_pass:
        rep.log("Sharpening the hooks...")
        try:
            hooks.sharpen_hooks(clips, tr, title=title, rules=cfg.rules, rep=rep,
                                overlay_name=cfg.overlay_name, model=cfg.claude_model)
        except ClipperError as e:
            if isinstance(e, Cancelled):
                raise
            rep.log(f"Hook pass skipped ({e.message}); keeping the original hooks.")

    # ---- 5/6. render (+ voiceover)
    folder = project.output_folder(out_base, title, vid)
    shutil.copyfile(work / "transcript.json", folder / "transcript.json")
    thumb_name = _save_thumbnail(info, folder, rep)
    vo_on = bool(profile.get("voiceover", {}).get("enabled"))
    proj = {
        "app": "Clipper", "version": 1, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "video": {"title": title, "url": info.get("webpage_url") or cfg.url, "id": vid, "duration": duration,
                  "uploader": info.get("uploader")},
        "profile": {k: v for k, v in profile.items() if not k.startswith("_")},
        "profile_id": profile.get("_id"), "rules": cfg.rules, "length": length,
        "signals": {"loudness": sig["loudness"], "chat": sig["chat"]},
        "source_path": None, "clips": [], "extras": bool(cfg.extras), "thumbnail": thumb_name,
        "extras_end_card": bool(cfg.end_card), "overlay_name": cfg.overlay_name.strip(),
        "_folder": str(folder),
    }
    for c in clips:
        c["voiceover"] = vo_on and bool(c.get("intro_line"))
        proj["clips"].append(c)
    project.save(proj)
    if video is not None and profile.get("layout", {}).get("mode", "blur") != "blur":
        rep.log("Looking for the facecam in each clip...")
        project.detect_facecams(proj, video, 0.0, rep)
        project.save(proj)

    rep.log(f"Rendering {len(clips)} clip(s) to {folder} ...")
    n = len(clips)
    failed = 0
    for i, c in enumerate(clips):
        rep.check()
        rep.log(f"Clip {c['rank']}/{n}: score {c['score']:.0f}, {c['end'] - c['start']:.0f} s - {c.get('hook_text', '')}")
        try:
            if video is not None:
                project.render_one(proj, c, rep, source=video, source_offset=0.0, work_root=work)
            else:  # audio-first: render_one fetches just this clip's section of the video
                project.render_one(proj, c, rep, work_root=work)
        except (Cancelled, DiskSpaceError):
            project.save(proj)
            raise
        except ClipperError as e:
            failed += 1
            c["error"] = e.message
            rep.log(f"Clip {c['rank']} failed: {e}")
        rep.progress("render", (i + 1) / n)
        project.save(proj)

    if failed == n:
        raise ClipperError("Every clip failed to render.", "See the log above for ffmpeg's error.")

    # ---- 7. clean up
    if cfg.keep_source and video is not None:
        keep_dir = APP_DIR / "sources"
        keep_dir.mkdir(parents=True, exist_ok=True)
        kept = keep_dir / f"{vid}{video.suffix}"
        shutil.move(str(video), kept)
        proj["source_path"] = str(kept)
        project.save(proj)
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(TEMP_ROOT / f"sections_{vid}", ignore_errors=True)
    rep.log(f"Done in {(time.monotonic() - t0) / 60:.1f} min: {n - failed} clip(s) in {folder}")
    return folder

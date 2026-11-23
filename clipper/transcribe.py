"""Stage 2: speech to text with word timestamps (faster-whisper)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .errors import ClipperError, DiskSpaceError
from .events import Reporter
from .paths import MODELS_DIR
from .tools import check_disk, find_ffmpeg, is_disk_full, run

WHISPER_SIZES_MB = {"tiny": 75, "base": 145, "small": 485, "medium": 1530,
                    "large-v3": 3100, "large-v3-turbo": 1620, "distil-large-v3": 1510}


def extract_audio(video: Path, out_wav: Path, rep: Reporter) -> Path:
    """Mono 16 kHz WAV: what Whisper wants, and cheap to scan for loudness."""
    if out_wav.exists() and out_wav.stat().st_size > 1000:
        return out_wav
    check_disk(out_wav.parent, 600_000_000, "the audio track")
    rep.log("Extracting audio...")
    tmp = out_wav.with_suffix(".part.wav")
    r = run([find_ffmpeg(), "-y", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(tmp)])
    if r.returncode != 0:
        if "No space left" in r.stderr:
            raise DiskSpaceError("The disk filled up while extracting audio.")
        raise ClipperError("Couldn't extract the audio track.", r.stderr.strip()[-400:])
    tmp.replace(out_wav)
    return out_wav


def load_wav_float(path: Path, start: float = 0.0, end: float | None = None):
    """16 kHz mono PCM [start, end) seconds -> float32 numpy (avoids PyAV, whose API drifts)."""
    import wave

    import numpy as np
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        first = int(start * sr)
        last = wf.getnframes() if end is None else min(wf.getnframes(), int(end * sr))
        wf.setpos(first)
        n = max(0, last - first)
        out = np.empty(n, dtype=np.float32)
        pos = 0
        while pos < n:
            raw = wf.readframes(min(16000 * 600, n - pos))
            if not raw:
                break
            a = np.frombuffer(raw, dtype=np.int16)
            out[pos:pos + len(a)] = a / 32768.0
            pos += len(a)
    return out[:pos]


def free_ram_gb() -> float:
    """Available physical memory in GB (Windows), or 0 if unknown."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullAvailPhys / 1e9
    except Exception:
        pass
    return 0.0


def pick_device() -> tuple[str, str]:
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def load_model(name: str, rep: Reporter):
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    import warnings
    warnings.filterwarnings("ignore", module="huggingface_hub")
    from faster_whisper import WhisperModel

    root = MODELS_DIR / "whisper"
    root.mkdir(parents=True, exist_ok=True)
    marker = root / f".have_{name}"
    if not marker.exists():
        size = WHISPER_SIZES_MB.get(name, 500)
        check_disk(root, size * 1_100_000, f"the Whisper '{name}' model")
        rep.log(f"First run: downloading Whisper model '{name}' (~{size} MB) to {root} ...")

    device, compute = pick_device()
    try:
        model = WhisperModel(name, device=device, compute_type=compute, download_root=str(root),
                             cpu_threads=os.cpu_count() or 4)
    except Exception as e:
        if device == "cuda":
            rep.log(f"GPU load failed ({str(e)[:120]}); falling back to CPU.")
            device, compute = "cpu", "int8"
            model = WhisperModel(name, device=device, compute_type=compute, download_root=str(root),
                                 cpu_threads=os.cpu_count() or 4)
        elif is_disk_full(e):
            raise DiskSpaceError("The disk filled up while downloading the Whisper model.") from e
        else:
            raise ClipperError(f"Couldn't load the Whisper model '{name}'.",
                               f"{e}\nCheck your internet connection on first run.") from e
    marker.touch()
    rep.log(f"Whisper '{name}' on {device.upper()} ({compute}).")
    return model, device


def transcribe(audio: Path, out_json: Path, duration: float, model_name: str, rep: Reporter,
               language: str | None = None, vad: bool = False) -> dict:
    """Write and return {"language", "duration", "segments": [{start,end,text,words:[[s,e,w],...]}]}."""
    if out_json.exists():
        try:
            data = json.loads(out_json.read_text(encoding="utf-8"))
            if data.get("complete") and data.get("model") == model_name and data.get("vad") == vad:
                rep.log("Reusing transcript from an earlier run.")
                rep.progress("transcribe", 1.0)
                return data
        except ValueError:
            pass

    need = {"tiny": 0.6, "base": 0.8, "small": 1.3, "medium": 2.6}.get(model_name, 3.5)
    free = free_ram_gb()
    if 0 < free < need:
        rep.log(f"Warning: only {free:.1f} GB of RAM is free; Whisper '{model_name}' needs about {need:.1f} GB. "
                "Close browsers or other programs, or pick a smaller model in Settings, if it fails.")
    model, device = load_model(model_name, rep)
    rep.log("Transcribing (this is the slow part)...")
    t0 = time.monotonic()
    total = wav_seconds(audio)
    duration = duration or total or 1.0
    cuts = piece_bounds(audio, total)

    # Resume: pieces already finished in an earlier (interrupted) run.
    partial = out_json.with_suffix(".partial.json")
    done = {}
    if partial.exists():
        try:
            prev = json.loads(partial.read_text(encoding="utf-8"))
            if prev.get("model") == model_name and prev.get("vad") == vad and prev.get("cuts") == cuts:
                done = {int(k): v for k, v in prev["pieces"].items()}
                language = language or prev.get("language")
                if done:
                    rep.log(f"Resuming transcription: {len(done)}/{len(cuts) - 1} pieces already done.")
        except (ValueError, KeyError):
            done = {}

    lang = language or None
    lang_prob = None
    last_log = 0.0
    resumed_s = sum(cuts[j + 1] - cuts[j] for j in done)  # audio finished before this run
    for i in range(len(cuts) - 1):
        a, b = cuts[i], cuts[i + 1]
        if i in done:
            continue
        rep.check()
        segs = info = None
        for attempt in range(OOM_RETRIES + 1):
            try:
                samples = load_wav_float(audio, a, b)
                segments_iter, info = model.transcribe(
                    samples,
                    language=lang,
                    language_detection_segments=6,  # intros are often music; don't decide from 30 s
                    language_detection_threshold=0.6,
                    word_timestamps=True,
                    # VAD is fast but drops shouted/noisy gameplay speech, so it's a per-profile choice.
                    vad_filter=vad,
                    vad_parameters={"min_silence_duration_ms": 500} if vad else None,
                    hallucination_silence_threshold=None if vad else 2.0,
                    beam_size=5 if device == "cuda" else 1,
                    temperature=0.0 if device == "cpu" else [0.0, 0.2, 0.4],  # fallbacks are slow on CPU
                    condition_on_previous_text=False,
                )
                segs = []
                for seg in segments_iter:
                    rep.check()
                    words = [[round(a + float(w.start), 2), round(a + float(w.end), 2), w.word.strip()]
                             for w in (seg.words or []) if w.word.strip()]
                    segs.append({"start": round(a + float(seg.start), 2), "end": round(a + float(seg.end), 2),
                                 "text": seg.text.strip(), "words": words})
                    frac = (a + float(seg.end)) / duration
                    rep.progress("transcribe", frac)
                    now = time.monotonic()
                    if now - last_log > 60:
                        last_log = now
                        el = now - t0
                        pos = a + float(seg.end)
                        rate = max(pos - resumed_s, 1.0) / el          # audio seconds per wall second
                        left = (duration - pos) / rate
                        rep.log(f"Transcribed {_hms(a + seg.end)} of {_hms(duration)} (about {_hms(left)} left).")
                break
            except (MemoryError, RuntimeError) as e:
                if isinstance(e, RuntimeError) and "alloc" not in str(e).lower() and "memory" not in str(e).lower():
                    raise
                samples = segments_iter = None
                import gc
                gc.collect()
                if attempt < OOM_RETRIES:
                    wait = 20 * (attempt + 1)
                    rep.log(f"Low on memory ({free_ram_gb():.1f} GB free) at {_hms(a)}; retrying this part "
                            f"in {wait} s ({attempt + 1}/{OOM_RETRIES})...")
                    for _ in range(wait):
                        rep.check()
                        time.sleep(1)
                    continue
                raise ClipperError(
                    f"Ran out of memory while transcribing ({free_ram_gb():.1f} GB of RAM free).",
                    "Close browsers and other programs, or pick a smaller Whisper model in Settings "
                    "(base uses about half the memory of small). Start again to resume where it stopped.") from e
        if lang is None and info.language_probability >= 0.5:
            lang, lang_prob = info.language, info.language_probability  # keep it for later pieces
            rep.log(f"Language: {lang} ({lang_prob:.0%}).")
        done[i] = segs
        partial.write_text(json.dumps({"model": model_name, "vad": vad, "cuts": cuts, "language": lang,
                                       "pieces": done}, ensure_ascii=False), encoding="utf-8")

    segs = [s for i in sorted(done) for s in done[i]]
    data = {"language": lang, "duration": duration, "segments": segs, "complete": True,
            "model": model_name, "vad": vad}
    out_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    partial.unlink(missing_ok=True)
    rep.progress("transcribe", 1.0)
    rep.log(f"Transcript: {len(segs)} segments, {sum(len(s['words']) for s in segs):,} words "
            f"in {_hms(time.monotonic() - t0)}.")
    return data


PIECE = 300.0     # seconds of audio handed to Whisper at a time (bounds memory on 4 h videos)
OOM_RETRIES = 3   # a piece that runs out of memory is retried after a pause
SEARCH = 20.0     # look this far either side of each boundary for the quietest moment


def wav_seconds(path: Path) -> float:
    import wave
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def piece_bounds(audio: Path, total: float) -> list[float]:
    """Cut points about every 10 min, moved to the quietest 0.5 s nearby so no word is split."""
    if total <= PIECE * 1.5:
        return [0.0, round(total, 2)]
    from .signals import loudness_series
    t, db = loudness_series(audio)
    cuts = [0.0]
    b = PIECE
    while b < total - PIECE * 0.5:
        lo, hi = b - SEARCH, b + SEARCH
        idx = [i for i in range(len(t)) if lo <= t[i] <= hi] if len(t) else []
        cut = float(t[min(idx, key=lambda i: db[i])]) if idx else b
        cuts.append(round(cut, 2))
        b = cut + PIECE
    cuts.append(round(total, 2))
    return cuts


def all_words(transcript: dict) -> list[list]:
    return [w for s in transcript["segments"] for w in s["words"]]


def _hms(sec: float) -> str:
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

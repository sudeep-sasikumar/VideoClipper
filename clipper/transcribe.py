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


def load_wav_float(path: Path):
    """16 kHz mono PCM -> float32 numpy (avoids PyAV, whose API drifts between versions)."""
    import wave

    import numpy as np
    with wave.open(str(path), "rb") as wf:
        n = wf.getnframes()
        out = np.empty(n, dtype=np.float32)
        pos = 0
        while pos < n:
            raw = wf.readframes(16000 * 600)
            if not raw:
                break
            a = np.frombuffer(raw, dtype=np.int16)
            out[pos:pos + len(a)] = a / 32768.0
            pos += len(a)
    return out[:pos]


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

    model, device = load_model(model_name, rep)
    rep.log("Transcribing (this is the slow part)...")
    t0 = time.monotonic()
    samples = load_wav_float(audio)
    segments_iter, info = model.transcribe(
        samples,
        language=language or None,
        word_timestamps=True,
        # VAD is fast but drops shouted/noisy gameplay speech, so it's a per-profile choice.
        vad_filter=vad,
        vad_parameters={"min_silence_duration_ms": 500} if vad else None,
        hallucination_silence_threshold=None if vad else 2.0,
        beam_size=5 if device == "cuda" else 1,
        temperature=0.0 if device == "cpu" else [0.0, 0.2, 0.4],  # fallback retries are slow on CPU
        condition_on_previous_text=False,
    )
    duration = duration or info.duration or 1.0
    rep.log(f"Language: {info.language} ({info.language_probability:.0%}).")
    segs = []
    last_log = 0.0
    for seg in segments_iter:
        rep.check()
        words = [[round(float(w.start), 2), round(float(w.end), 2), w.word.strip()] for w in (seg.words or []) if w.word.strip()]
        segs.append({"start": round(float(seg.start), 2), "end": round(float(seg.end), 2), "text": seg.text.strip(), "words": words})
        frac = seg.end / duration
        rep.progress("transcribe", frac)
        now = time.monotonic()
        if now - last_log > 60:
            last_log = now
            el = now - t0
            eta = el / max(frac, 1e-3) - el
            rep.log(f"Transcribed {_hms(seg.end)} of {_hms(duration)} (about {_hms(eta)} left).")
    data = {"language": info.language, "duration": duration, "segments": segs, "complete": True,
            "model": model_name, "vad": vad}
    out_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    rep.progress("transcribe", 1.0)
    rep.log(f"Transcript: {len(segs)} segments, {sum(len(s['words']) for s in segs):,} words "
            f"in {_hms(time.monotonic() - t0)}.")
    return data


def all_words(transcript: dict) -> list[list]:
    return [w for s in transcript["segments"] for w in s["words"]]


def _hms(sec: float) -> str:
    sec = int(max(0, sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

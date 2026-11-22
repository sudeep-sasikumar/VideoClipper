"""Stage 6: optional spoken intro, synthesized locally with Kokoro (kokoro-onnx, no PyTorch)."""
from __future__ import annotations

import threading
import urllib.request
from pathlib import Path

from .errors import ClipperError, DiskSpaceError
from .events import Reporter
from .paths import MODELS_DIR
from .tools import check_disk, is_disk_full

BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
FILES = {"kokoro-v1.0.int8.onnx": 92_000_000, "voices-v1.0.bin": 28_000_000}
KOKORO_DIR = MODELS_DIR / "kokoro"

VOICES = ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky", "am_adam", "am_michael",
          "am_fenrir", "am_puck", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"]

_lock = threading.Lock()
_kokoro = None


def _download(name: str, rep: Reporter) -> Path:
    dest = KOKORO_DIR / name
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    KOKORO_DIR.mkdir(parents=True, exist_ok=True)
    check_disk(KOKORO_DIR, FILES[name] + 50_000_000, "the Kokoro voice model")
    rep.log(f"First use: downloading Kokoro {name} (~{FILES[name] // 1_000_000} MB) to {KOKORO_DIR} ...")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "Clipper"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                rep.check()
                buf = r.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        if is_disk_full(e):
            raise DiskSpaceError("The disk filled up while downloading the voice model.") from e
        raise ClipperError("Couldn't download the Kokoro voice model.", f"{e}\nCheck your internet connection.") from e
    tmp.replace(dest)
    return dest


def get_kokoro(rep: Reporter):
    global _kokoro
    with _lock:
        if _kokoro is None:
            model = _download("kokoro-v1.0.int8.onnx", rep)
            voices = _download("voices-v1.0.bin", rep)
            from kokoro_onnx import Kokoro
            _kokoro = Kokoro(str(model), str(voices))
        return _kokoro


def synth(text: str, out_wav: Path, voice: str, speed: float, rep: Reporter) -> float:
    """Write a WAV of `text`; return its duration in seconds."""
    import soundfile as sf

    text = " ".join((text or "").split())
    if not text:
        raise ClipperError("The voiceover line is empty.")
    k = get_kokoro(rep)
    if voice not in k.get_voices():
        rep.log(f"Voice '{voice}' not found; using af_heart.")
        voice = "af_heart"
    lang = "en-gb" if voice.startswith("b") else "en-us"
    samples, sr = k.create(text, voice=voice, speed=float(speed or 1.0), lang=lang)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), samples, sr)
    return len(samples) / sr

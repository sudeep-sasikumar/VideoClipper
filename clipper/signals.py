"""Stage 3: loudness spikes from the audio and message-rate spikes from chat."""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

WIN = 0.5  # seconds per loudness window


def _rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) < 3:
        return np.full_like(x, np.median(x) if len(x) else 0.0)
    win = max(3, min(win, len(x)) | 1)
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(xp, win)
    out = np.empty(len(x))
    step = 20000  # bound memory on 4 h inputs
    for i in range(0, len(x), step):
        out[i:i + step] = np.median(view[i:i + step], axis=1)
    return out


def _peaks(t: np.ndarray, z: np.ndarray, thresh: float, min_gap: float) -> list[int]:
    """Indices of local maxima above thresh, at least min_gap seconds apart (strongest first)."""
    idx = np.where(z >= thresh)[0]
    chosen: list[int] = []
    for i in idx[np.argsort(-z[idx])]:
        if all(abs(t[i] - t[j]) >= min_gap for j in chosen):
            chosen.append(int(i))
    return sorted(chosen, key=lambda i: t[i])


def loudness_series(wav_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(times, dBFS) per 0.5 s window, streamed so 4 h of audio stays small in memory."""
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        n_win = int(sr * WIN)
        db = []
        while True:
            raw = wf.readframes(n_win * 600)
            if not raw:
                break
            a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            k = len(a) // n_win
            if k:
                blk = a[: k * n_win].reshape(k, n_win)
                rms = np.sqrt(np.mean(blk * blk, axis=1) + 1e-10)
                db.append(20 * np.log10(rms))
    db = np.concatenate(db) if db else np.zeros(0)
    t = np.arange(len(db)) * WIN + WIN / 2
    return t, db


def loudness_spikes(wav_path: Path, thresh: float = 2.0, min_gap: float = 6.0) -> list[dict]:
    t, db = loudness_series(wav_path)
    if len(db) < 10:
        return []
    db = np.maximum(db, -70.0)
    # Light smoothing (1.5 s) so single clicks don't count.
    k = np.ones(3) / 3
    sm = np.convolve(db, k, mode="same")
    base = _rolling_median(sm, int(90 / WIN))           # local loudness over 90 s
    dev = sm - base
    mad = np.median(np.abs(dev - np.median(dev))) * 1.4826 + 1e-6
    z = dev / max(mad, 1.0)
    out = []
    for i in _peaks(t, z, thresh, min_gap):
        if sm[i] < -45:   # ignore "spikes" that are still near-silence
            continue
        out.append({"t": round(float(t[i]), 2), "z": round(float(z[i]), 2), "db": round(float(sm[i]), 1)})
    return out


def chat_spikes(times: list[float] | None, duration: float, bin_s: float = 10.0,
                thresh: float = 2.5, min_gap: float = 30.0) -> list[dict]:
    if not times or duration <= 0:
        return []
    n = int(duration // bin_s) + 1
    counts = np.bincount(np.clip((np.asarray(times) // bin_s).astype(int), 0, n - 1), minlength=n).astype(float)
    if counts.sum() < 50:
        return []
    base = _rolling_median(counts, int(300 / bin_s))
    dev = counts - base
    scale = np.sqrt(np.maximum(base, 1.0))                # Poisson-ish noise
    z = dev / scale
    t = np.arange(n) * bin_s + bin_s / 2
    return [{"t": round(float(t[i]), 1), "z": round(float(z[i]), 2), "rate": int(counts[i])}
            for i in _peaks(t, z, thresh, min_gap) if counts[i] >= 5]


def loudness_timeline(wav_path: Path, step: float = 30.0) -> list[list[float]]:
    """[[t, mean dBFS], ...] every `step` seconds: a coarse map of where the action is."""
    t, db = loudness_series(wav_path)
    if not len(db):
        return []
    k = max(1, int(step / WIN))
    n = len(db) // k
    out = []
    for i in range(n + (1 if len(db) % k else 0)):
        blk = db[i * k:(i + 1) * k]
        # Mean of power, not of dB, so a loud burst isn't averaged away.
        p = np.mean(np.power(10.0, np.maximum(blk, -70.0) / 10.0))
        out.append([round(i * step, 1), round(float(10 * np.log10(p + 1e-12)), 1)])
    return out


def compute(wav_path: Path, chat_times: list[float] | None, duration: float) -> dict:
    return {
        "loudness": loudness_spikes(wav_path),
        "chat": chat_spikes(chat_times, duration),
        "has_chat": bool(chat_times),
        "timeline": loudness_timeline(wav_path),
    }

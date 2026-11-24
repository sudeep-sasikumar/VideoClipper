"""Beat analysis for music-synced edits: tempo, beat grid, and the bass drop.

Pure numpy (no librosa): onset strength from a spectral-flux envelope, tempo from its
autocorrelation, beat phase by maximising onset energy on the grid, and the drop as the
biggest jump in low-frequency (bass) energy.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .tools import NO_WINDOW, find_ffmpeg

SR = 22050
HOP = 512


def load_mono(path: Path, seconds: float | None = None) -> np.ndarray:
    cmd = [find_ffmpeg(), "-v", "error", "-i", str(path)]
    if seconds:
        cmd += ["-t", f"{seconds:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def _stft_mag(y: np.ndarray, n_fft: int = 2048) -> np.ndarray:
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    win = np.hanning(n_fft).astype(np.float32)
    frames = 1 + (len(y) - n_fft) // HOP
    idx = np.arange(n_fft)[None, :] + HOP * np.arange(frames)[:, None]
    return np.abs(np.fft.rfft(y[idx] * win, axis=1))  # (frames, bins)


def analyse(path: Path, seconds: float | None = 60.0) -> dict:
    y = load_mono(path, seconds)
    mag = _stft_mag(y)
    t = (np.arange(mag.shape[0]) * HOP + 1024) / SR
    freqs = np.fft.rfftfreq(2048, 1 / SR)
    logmag = np.log1p(mag * 10)
    flux = np.maximum(0, np.diff(logmag, axis=0, prepend=logmag[:1])).sum(axis=1)
    onset = (flux - flux.mean()) / (flux.std() + 1e-9)
    onset = np.maximum(onset, 0)

    # Tempo: autocorrelation of the onset envelope over 70-180 BPM.
    fps = SR / HOP
    ac = np.correlate(onset, onset, mode="full")[len(onset) - 1:]
    lags = np.arange(len(ac))
    bpm_of = lambda lag: 60 * fps / lag  # noqa: E731
    lo, hi = int(60 * fps / 180), int(60 * fps / 70)
    cand = lags[lo:hi]
    # Prefer tempos near 120-130 slightly (dance music), to settle half/double ambiguity.
    weight = np.exp(-0.5 * ((np.log2(bpm_of(cand) / 125)) / 0.6) ** 2)
    lag = int(cand[np.argmax(ac[lo:hi] * weight)])
    period = lag / fps

    # Beat phase: the offset whose grid collects the most onset energy.
    best, best_phase = -1.0, 0.0
    for ph in np.arange(0, period, 1 / fps):
        grid = np.arange(ph, t[-1], period)
        idx = np.clip(np.round((grid - t[0]) * fps).astype(int), 0, len(onset) - 1)
        score = onset[idx].sum()
        if score > best:
            best, best_phase = score, ph
    beats = np.arange(best_phase, t[-1], period)

    # Bass energy (40-150 Hz) and the drop: biggest rise from the second before to the second after.
    bass = mag[:, (freqs >= 40) & (freqs <= 150)].mean(axis=1)
    bass = np.convolve(bass, np.ones(5) / 5, mode="same")
    win = int(fps)
    rise = np.zeros_like(bass)
    for i in range(win, len(bass) - win):
        rise[i] = bass[i:i + win].mean() / (bass[i - win:i].mean() + 1e-6)
    drop_i = int(np.argmax(rise[: int(20 * fps)]))  # the first big drop, within 20 s
    drop_t = float(t[drop_i])
    # Snap the drop to the nearest beat.
    drop_beat = float(beats[np.argmin(np.abs(beats - drop_t))]) if len(beats) else drop_t
    return {"bpm": round(60 / period, 2), "beat_period": round(period, 4), "first_beat": round(float(best_phase), 3),
            "beats": [round(float(b), 3) for b in beats], "drop": round(drop_beat, 3), "drop_raw": round(drop_t, 3),
            "duration": round(len(y) / SR, 2),
            "bass_curve": [[round(float(a), 2), round(float(b), 4)] for a, b in zip(t[::int(fps / 4)], bass[::int(fps / 4)])]}

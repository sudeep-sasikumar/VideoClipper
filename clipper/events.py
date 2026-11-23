"""Progress reporting shared by the pipeline, the CLI and the GUI."""
from __future__ import annotations

import threading
import time
from typing import Callable

from .errors import Cancelled

# Rough share of total run time per stage, used for the overall progress bar.
STAGE_WEIGHTS = {
    "check": 0.01,
    "download": 0.12,
    "transcribe": 0.50,
    "signals": 0.04,
    "rank": 0.13,
    "render": 0.20,
}


def _console_utf8() -> None:
    """Windows consoles/pipes default to cp1252, which can't print emoji in video titles."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure") and \
                (getattr(stream, "encoding", "") or "").lower().replace("-", "") != "utf8":
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


class Reporter:
    """Collects log lines and progress; checks for cancel.

    on_log(str) and on_progress(stage, stage_fraction, overall_fraction) are optional callbacks.
    """

    def __init__(self, on_log: Callable[[str], None] | None = None,
                 on_progress: Callable[[str, float, float], None] | None = None):
        self.on_log = on_log
        self.on_progress = on_progress
        self._cancel = threading.Event()
        self._last_progress = 0.0

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        if self.on_log:
            self.on_log(line)
        else:
            _console_utf8()
            print(line, flush=True)

    def progress(self, stage: str, frac: float) -> None:
        frac = max(0.0, min(1.0, frac))
        done = 0.0
        for name, w in STAGE_WEIGHTS.items():
            if name == stage:
                done += w * frac
                break
            done += w
        overall = done / sum(STAGE_WEIGHTS.values())
        if self.on_progress:
            self.on_progress(stage, frac, overall)
        else:
            now = time.monotonic()
            if now - self._last_progress > 5 or frac >= 1.0:
                self._last_progress = now
                print(f"    {stage}: {frac * 100:5.1f}%  (overall {overall * 100:4.1f}%)", flush=True)

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def check(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

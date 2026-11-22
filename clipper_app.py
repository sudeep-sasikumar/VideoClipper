"""Entry point for the packaged exe (and `python clipper_app.py`).

No arguments: open the window.
With arguments: run headless like `python -m clipper URL --profile ...`; the exe has no console,
so output goes to %APPDATA%\\Clipper\\last_run.log.
"""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) > 1:
        from clipper.paths import APP_DIR, ensure_dirs
        ensure_dirs()
        if sys.stdout is None or getattr(sys, "frozen", False):
            log = open(APP_DIR / "last_run.log", "w", encoding="utf-8", buffering=1)
            sys.stdout = sys.stderr = log
        from clipper.cli import main as cli_main
        sys.exit(cli_main())
    from clipper.gui.app import main
    sys.exit(main())

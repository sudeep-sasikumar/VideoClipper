"""Entry point for the packaged exe (and `python clipper_app.py`)."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from clipper.gui.app import main
    sys.exit(main())

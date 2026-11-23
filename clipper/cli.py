"""Headless runner:  python -m clipper URL [--profile gaming] [--rules "..."] [--out DIR]"""
from __future__ import annotations

import argparse
import sys

from . import profiles
from .errors import ClipperError
from .events import Reporter
from .paths import load_settings
from .pipeline import JobConfig, run_job


def main(argv=None) -> int:
    from .events import _console_utf8
    _console_utf8()
    settings = load_settings()
    profs = profiles.load_all(print)
    ap = argparse.ArgumentParser(prog="clipper", description="Turn a long video into ranked TikTok clips.")
    ap.add_argument("url")
    ap.add_argument("--profile", default=profiles.default_id(profs), choices=sorted(profs))
    ap.add_argument("--rules", default="", help="campaign rules text (or @file.txt)")
    ap.add_argument("--out", default=settings["output_dir"])
    ap.add_argument("--model", default=settings.get("whisper_model", ""), help="whisper model override")
    ap.add_argument("--keep-source", action="store_true", default=settings.get("keep_source", False))
    ap.add_argument("--max-clips", type=int, default=settings.get("max_clips", 20))
    ap.add_argument("--extras", action=argparse.BooleanOptionalAction, default=settings.get("extras", False),
                    help="add a teaser, context banner, callouts and an end card")
    ap.add_argument("--end-card", action=argparse.BooleanOptionalAction,
                    default=settings.get("extras_end_card", True), help="include the thumbnail end card in extras")
    ap.add_argument("--audio-first", action=argparse.BooleanOptionalAction, default=None,
                    help="download only the audio, then just each clip's video (default: automatic for "
                         "videos over 5 h or when the disk is too full for the whole video)")
    ap.add_argument("--name", default=settings.get("overlay_name", ""),
                    help="name that must appear on screen in every clip (e.g. a campaign rule)")
    a = ap.parse_args(argv)
    rules = a.rules
    if rules.startswith("@"):
        with open(rules[1:], encoding="utf-8") as f:
            rules = f.read()
    try:
        folder = run_job(JobConfig(url=a.url, profile=profs[a.profile], rules=rules, output_dir=a.out,
                                   whisper_model=a.model, keep_source=a.keep_source, max_clips=a.max_clips,
                                   extras=a.extras, end_card=a.end_card, overlay_name=a.name,
                                   audio_first=a.audio_first),
                         Reporter())
    except ClipperError as e:
        print(f"\n{e.title}: {e}", file=sys.stderr)
        return 1
    print(folder)
    return 0


if __name__ == "__main__":
    sys.exit(main())

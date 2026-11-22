"""Talk to Claude through the local `claude` CLI (your logged-in subscription, no API key).

The prompt goes in on stdin: a 1-hour transcript is longer than Windows' 32k-character
command-line limit, so `claude -p "<prompt>"` would fail for real videos.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile

from .errors import ClaudeNotFoundError, ClaudeNotLoggedInError, ClaudeOutputError, ClipperError
from .tools import NO_WINDOW, find_claude

SYSTEM = ("You are a senior short-form video editor. You answer with JSON only: no prose, "
          "no markdown fences, no explanations outside the JSON.")

_LOGIN_HINT = "Open a terminal, run:  claude   then type  /login  and finish signing in. Then press Start again."


def ask(prompt: str, model: str = "", timeout: int = 900, system: str = SYSTEM) -> str:
    """Run one prompt and return Claude's text answer. Raises a ClipperError subclass."""
    exe = find_claude()
    cmd = [exe, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
           "--system-prompt", system]
    if model:
        cmd += ["--model", model]
    try:
        with tempfile.TemporaryDirectory(prefix="clipper_claude_") as cwd:  # no project CLAUDE.md
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=timeout, cwd=cwd, creationflags=NO_WINDOW)
    except FileNotFoundError as e:
        find_claude.cache_clear()
        raise ClaudeNotFoundError("The Claude Code CLI ('claude') could not be started.",
                                  "Install it with:  winget install Anthropic.ClaudeCode") from e
    except subprocess.TimeoutExpired as e:
        raise ClipperError(f"Claude didn't answer within {timeout // 60} minutes.",
                           "Check your connection or usage limits and try again.") from e

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    data = None
    if out:
        try:
            data = json.loads(out.splitlines()[-1] if not out.startswith("{") else out)
        except ValueError:
            data = None
    text = (data or {}).get("result", "") if isinstance(data, dict) else ""
    blob = f"{text}\n{out[:2000]}\n{err[:2000]}".lower()

    if "not logged in" in blob or "/login" in blob or "invalid api key" in blob or \
            "oauth token has expired" in blob or "please run /login" in blob:
        raise ClaudeNotLoggedInError("The Claude CLI is installed but not logged in.", _LOGIN_HINT)
    if isinstance(data, dict) and data.get("is_error"):
        if "usage limit" in blob or "rate limit" in blob or "limit reached" in blob:
            raise ClipperError("Claude usage limit reached.", text or "Wait for your limit to reset, then try again.")
        raise ClipperError("Claude returned an error.", (text or err or out)[:500])
    if proc.returncode != 0 and not text:
        raise ClipperError("The Claude CLI failed.", (err or out)[:500] or f"exit code {proc.returncode}")
    if not text:
        raise ClaudeOutputError("Claude returned an empty answer.")
    return text


def check_login(model: str = "") -> None:
    """Fail fast (before hours of transcription) if the CLI is missing or logged out."""
    ans = ask('Reply with exactly {"ok": true}', model=model, timeout=120)
    if "ok" not in ans.lower():
        raise ClaudeOutputError("Unexpected reply from the Claude CLI.", ans[:200])


def extract_json(text: str):
    """Parse the first JSON value in text, tolerating ```json fences and stray prose."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except ValueError:
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "[{":
            try:
                val, _ = dec.raw_decode(t[i:])
                return val
            except ValueError:
                continue
    raise ClaudeOutputError("Claude's answer wasn't valid JSON.", text[:300])

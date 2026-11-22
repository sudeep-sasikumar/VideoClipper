# Clipper — build plan

## This machine
- Python 3.12.10, Git 2.54, Node 24 (yt-dlp uses it for YouTube's JS challenges)
- ffmpeg 9.1 (installed with winget: Gyan.FFmpeg)
- Claude Code CLI 2.1.330 (installed with winget: Anthropic.ClaudeCode). **Not logged in yet**: run `claude` once and `/login`.
- No NVIDIA GPU (Intel Iris Xe, 4 cores, 7.7 GB RAM), so Whisper runs on CPU int8. Default model `small`, and you can pick `base` for speed or `medium` for accuracy. Expect about 0.5–1x real time on CPU, so a 4 h VOD takes a few hours to transcribe. A CUDA machine picks the GPU automatically.

## Layout
```
D:\Clipper
  clipper\            python package
    paths.py          %APPDATA%\Clipper (models, settings), temp, output
    tools.py          find ffmpeg / claude, check login, check disk space
    download.py       yt-dlp video (<=1080p) + chat replay
    transcribe.py     faster-whisper, word timestamps, progress
    signals.py        loudness spikes (ffmpeg -> numpy RMS), chat-rate spikes
    rules.py          parse length limits out of campaign rules
    claude_cli.py     run `claude -p` (prompt on stdin), parse/validate JSON, retry once
    rank.py           1 h chunks -> candidates -> merge / de-dupe -> top 20
    captions.py       ASS: word-by-word captions + hook text
    render.py         ffmpeg 9:16 render: blur-fill or facecam split, zoom punches
    voiceover.py      Kokoro (kokoro-onnx, local) TTS + mix
    project.py        project.json: every clip's data, so the GUI can edit/re-render
    pipeline.py       orchestrates stages, emits progress events, cancel
    cli.py            headless runner: python -m clipper URL --profile gaming
    gui\              PySide6 window
  profiles\           gaming.json (default), streamer.json, podcast.json
  clipper.spec        PyInstaller one-folder build
```

## Pipeline
1. **Download**: yt-dlp `bv*[height<=1080]+ba/b[height<=1080]`, merged to mp4, in a temp folder. Chat comes from the `live_chat` subtitle (YouTube replays) or Twitch rechat when yt-dlp can get it; if it can't, the app skips it with a log line. Free space is checked against the estimated size before downloading.
2. **Transcribe**: faster-whisper with `word_timestamps=True` and VAD filter. Device is CUDA float16 if available, else CPU int8. Progress is the segment end time / duration.
3. **Signals**: mono 16 kHz audio → 0.5 s RMS in dB → z-score peaks. Chat comes in as messages per 10 s → z-score peaks.
4. **Rank**: 1 h chunks of `[mm:ss] text` plus that chunk's spikes, the profile's criteria and the campaign rules. Each chunk goes through `claude -p --output-format json --tools ""` and asks for N candidates `{start,end,score,hook_text,reason,title,intro_line}`. The JSON is validated (fields, types, inside the chunk, length limits) and the call retried once on bad output. Candidates are merged, overlaps de-duplicated (higher score wins), and the top 20 kept. Clip edges snap to word boundaries.
5. **Render**: 1080x1920 30 fps H.264/AAC, 20–60 s (profile ∩ campaign rules). Layout is `blur` (blurred cover background + fitted foreground) or `split` (facecam crop from the profile on top, gameplay below). ASS captions show word by word with the active word highlighted, and the hook text shows for the first 3 s. A short zoom punch-in (about 8 %) lands on each loudness spike.
6. **Voiceover** (profile on/off): Claude writes the intro line in the same ranking call, Kokoro speaks it with the profile's voice, and the original audio ducks underneath.
7. **Output**: `[output]\[video title]\01_score9_<title>.mp4 …`, `report.json`, `report.txt`, `project.json`. Temp is deleted after success.

## Decisions I made where the brief left room (easy to change)
- **Prompt goes over stdin, not as a `-p "<prompt>"` argument.** A 1 h transcript (~50k chars) is over Windows' 32k command-line limit. It's the same `claude -p` call on your subscription with no API key.
- **3 candidates per hour vs top 20.** At 3/hour, even a 4 h video yields 12, so the top-20 cap never binds. `candidates_per_hour` is a profile field (default 3, as asked); set it to 6+ if you want up to 20 clips from long VODs. **Update: default raised to 6.**
- **Re-render after temp is deleted.** The source video is deleted on success as asked. Re-rendering a clip fetches only that clip's section (±30 s) with yt-dlp `download_ranges`, so editing doesn't re-download 4 hours. A setting "keep source video" (off by default) keeps it instead.
- **Facecam position** isn't auto-detected; the `split` layout uses a crop box in the profile (fractions of the frame). Copy a profile and adjust it per streamer.
- **Kokoro** runs via `kokoro-onnx` (ONNX Runtime, no PyTorch), so the exe stays small. Its int8 model (~90 MB) and voices download to `%APPDATA%\Clipper\models` on first use, as do Whisper models.

## Phases (commit + push after each, test each on a ~10 min public video)
1. Setup: venv, git, repo
2. Download (+ chat, disk check, bad-link errors)
3. Transcribe + signals
4. Rank (needs `claude` logged in; the not-installed / not-logged-in errors are tested first)
5. Render + voiceover + report
6. GUI
7. PyInstaller exe + README

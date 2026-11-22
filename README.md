# Clipper

Paste a YouTube or Twitch link (up to 4 hours), pick a niche profile, optionally paste campaign
rules, click **Start**. Clipper downloads the video, transcribes it, finds the best moments with
Claude, and renders ranked, TikTok-ready 9:16 clips with word-by-word captions, a hook, and
zoom punch-ins. You don't need to do anything after clicking Start.

```
D:\Clipper\output\[video title]\      (change the folder in Settings -> Output folder)
    01_score9_he_was_not_ready.mp4
    02_score8_...
    report.txt      timestamps, hook, why it was picked, link to the moment
    report.json     same, machine-readable
    project.json    everything the editor needs to re-render a clip
    transcript.json
```

## Setup (once)

1. **Install the tools** (PowerShell):

   ```powershell
   winget install Python.Python.3.12
   ```
   ```powershell
   winget install Gyan.FFmpeg
   ```
   ```powershell
   winget install Git.Git
   ```
   ```powershell
   winget install Anthropic.ClaudeCode
   ```

   Open a new terminal afterwards so PATH updates.

2. **Log in to Claude Code** (uses your Claude subscription; no API key):

   ```powershell
   claude
   ```
   Type `/login`, finish signing in, then `/exit`.

3. **Optional, for YouTube reliability:** Node.js or Deno on PATH. yt-dlp uses it for
   YouTube's JavaScript challenges. `winget install OpenJS.NodeJS.LTS`

### Run the packaged app

Open `dist\Clipper\Clipper.exe`. On first use it downloads the models into `%APPDATA%\Clipper\models`:
- Whisper `small`: about 485 MB (other sizes can be picked in Settings)
- Kokoro TTS: about 120 MB, and only when a profile with a voiceover is used

### Run from source

```powershell
git clone https://github.com/sudeep-sasikumar/VideoClipper D:\Clipper
```
```powershell
cd D:\Clipper; python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt
```
```powershell
.\.venv\Scripts\python clipper_app.py
```

Headless (no window), handy for batch runs:

```powershell
.\.venv\Scripts\python -m clipper "https://www.youtube.com/watch?v=..." --profile streamer --rules "Clips 15-45 seconds"
```

### Build the exe

```powershell
.\build.ps1
```
This produces `dist\Clipper\` (one folder). Copy that folder anywhere; ffmpeg and the `claude` CLI must
still be installed (step 1).

## How it works

| Stage | What happens |
|---|---|
| Download | yt-dlp downloads the best quality up to 1080p into `%APPDATA%\Clipper\temp`, plus the chat replay when one exists (YouTube live replays). |
| Transcribe | faster-whisper with word timestamps, fed 10 minutes at a time (cut at quiet moments) so memory stays under ~1 GB even for 4 h. It uses an NVIDIA GPU (CUDA) if present, otherwise CPU int8. On a 4-core laptop CPU, `small` runs about 3.5–4x real time (a 1 h 52 m stream took 30 min), so a 4 h VOD takes about 1–1.2 h. An interrupted transcription resumes from the last finished piece. |
| Signals | Loudness spikes from the audio; message-rate spikes from chat. |
| Rank | The transcript goes to Claude in 1-hour parts, with the signals, profile criteria and campaign rules, through `claude -p --output-format json`. Claude returns 6 candidates per hour, and at least 6 for videos shorter than an hour (JSON is validated, with one retry if malformed). Candidates are merged and de-duplicated, and the top 20 are kept. Parts with little or no speech are ranked from the loudness spikes, a 30-second loudness timeline and the chat spikes; parts with no speech and no spikes are skipped. Claude runs with no tools and no MCP connectors. |
| Render | ffmpeg renders 1080x1920 at 30 fps: blurred background fill or facecam-on-top split, ASS word-by-word captions, a hook for the first 3 s, and zoom punch-ins on loudness spikes. |
| Voiceover | Optional per profile. Claude writes an intro line, Kokoro speaks it locally, and the original audio ducks underneath. |
| Clean up | Temp files are deleted after success. |

An interrupted run resumes: start the same link again and finished stages (download, transcript,
ranking) are reused from the temp folder.

## Profiles

Profiles are JSON files in `profiles\` next to the app (or `%APPDATA%\Clipper\profiles`). To add a
niche, **copy a file, rename it, edit it**, then click *Profiles folder → OK* to reload. Any key
you leave out falls back to `gaming.json`.

| Key | Meaning |
|---|---|
| `highlight_criteria` | What counts as a good moment (goes into Claude's prompt) |
| `candidates_per_hour` | Candidates requested per hour of video (default 6, so a 4 h VOD can fill the top 20) |
| `clip_length` | `min` / `max` / `target` seconds |
| `whisper_model`, `vad_filter`, `language` | Transcription. VAD is faster but can miss shouted speech over game audio |
| `layout.mode` | `blur` (blurred fill) or `split` (facecam on top, gameplay below) |
| `layout.facecam` | Facecam box as fractions of the frame: `x, y, w, h` (e.g. top-left 30 %: `0,0,0.3,0.3`) |
| `layout.split_top_ratio` | Height share of the facecam panel in split mode |
| `layout.foreground_zoom` | >1 crops the sides of the foreground in blur mode (good for talking heads) |
| `captions` | Font, size, colors, highlight color, uppercase, words per line, vertical position |
| `hook` | On-screen hook style and duration |
| `zoom` | Punch-in amount, max punches per clip, length |
| `voiceover` | `enabled`, `voice` (e.g. `af_heart`, `am_michael`, `bf_emma`, `bm_george`), `speed`, `duck` |

## Campaign rules

Whatever you paste is added to Claude's instructions. Length limits in it are also enforced by
the app, for example "between 15 and 45 seconds", "max 30s", "at least 20 seconds and under 1 minute"
or "0:30-1:30". The line under the rules box shows the limits that will apply.

## Editing clips

Select a clip in the results list to:
- **Play** it in your default player (or double-click it)
- edit **start/end** (H:MM:SS.ss in the source), **hook text**, and **captions** (one row per caption
  line; keep the word count of a row to keep exact timing)
- toggle the **voiceover** and edit its intro line
- **Save & re-render** that one clip
- **Delete** it

Re-rendering after the temp files are gone downloads only that part of the video (±30 s). Turn
on *Settings → Keep the downloaded video* to skip that.
*Open project...* reopens any earlier output folder.

## Errors you might see

| Message | Fix |
|---|---|
| Bad link / unavailable / private / playlist | Paste a single public video link |
| Claude CLI not found | `winget install Anthropic.ClaudeCode` |
| Claude CLI not logged in | Run `claude`, type `/login` |
| Claude usage limit reached | Wait for your limit to reset; the run resumes from the ranking step |
| Nothing to rank | The video has no speech, no loud moments and no chat replay; there's nothing for Claude to go on |
| Not enough disk space | Free space, or point Settings → Output folder at another drive. A 4 h 1080p VOD needs about 8–15 GB of temp space |
| ffmpeg not found | `winget install Gyan.FFmpeg`, then restart Clipper |

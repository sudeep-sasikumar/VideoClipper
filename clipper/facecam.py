"""Find a streamer's facecam overlay in a clip, so the split layout crops the right area.

1. Sample ~12 frames from the clip (640 px wide).
2. Detect faces (YuNet, OpenCV). A facecam is a face that stays in about the same place in at
   least half the frames; faces in gameplay or cutscenes move around.
3. Find the overlay box around that face. For each side, look for a line that is
   - a static edge (present in most frames: the overlay's border), and/or
   - a boundary between calm pixels (the room behind the streamer) and busy pixels (gameplay).
   If neither shows up, fall back to the frame edge (corner cams) or a padded box around the face.
4. Reject boxes that are too big or that don't fit a webcam overlay.
Returns the box as fractions of the frame: {"x","y","w","h","confidence"} or None.
"""
from __future__ import annotations

import os
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # silence OpenCV's backend warnings in the log

from .errors import ClipperError
from .events import Reporter
from .paths import MODELS_DIR
from .tools import NO_WINDOW, find_ffmpeg

YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET_PATH = MODELS_DIR / "face" / "face_detection_yunet_2023mar.onnx"

SAMPLE_W = 960                # small webcam faces need the resolution
N_FRAMES = 12
MIN_CONSISTENCY = 0.5      # face in about the same spot in >= half the sampled frames
MAX_FACE_W = 0.22          # a facecam face is small relative to the whole frame
MAX_BOX_AREA = 0.40        # overlay covers at most 40 % of the frame
LINE_SCORE = 0.5           # minimum score to accept a border line


def _model(rep: Reporter) -> Path:
    if YUNET_PATH.exists() and YUNET_PATH.stat().st_size > 100_000:
        return YUNET_PATH
    YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
    rep.log("First use: downloading the face detector model (230 KB)...")
    tmp = YUNET_PATH.with_suffix(".part")
    try:
        req = urllib.request.Request(YUNET_URL, headers={"User-Agent": "Clipper"})
        with urllib.request.urlopen(req, timeout=60) as r:
            tmp.write_bytes(r.read())
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise ClipperError("Couldn't download the face detector model.", str(e)) from e
    tmp.replace(YUNET_PATH)
    return YUNET_PATH


def sample_frames(source: Path, start: float, dur: float, src_w: int, src_h: int,
                  n: int = N_FRAMES) -> list[np.ndarray]:
    """n BGR frames spread over [start, start+dur], scaled to SAMPLE_W wide."""
    w = SAMPLE_W
    h = int(round(SAMPLE_W * src_h / max(1, src_w) / 2) * 2)
    fps = n / max(dur, 1.0)
    cmd = [find_ffmpeg(), "-v", "error", "-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}",
           "-i", str(source), "-vf", f"fps={fps:.5f},scale={w}:{h}", "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    r = subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW)
    size = w * h * 3
    raw = r.stdout
    return [np.frombuffer(raw[i:i + size], np.uint8).reshape(h, w, 3)
            for i in range(0, len(raw) - size + 1, size)]


def _faces(frames: list[np.ndarray], model: Path) -> list[np.ndarray]:
    import cv2
    h, w = frames[0].shape[:2]
    det = cv2.FaceDetectorYN.create(str(model), "", (w, h), score_threshold=0.7, nms_threshold=0.3)
    out = []
    for f in frames:
        _, faces = det.detect(f)
        out.append(faces[:, :5] if faces is not None else np.zeros((0, 5), np.float32))
    return out


def _steady_face(per_frame: list[np.ndarray], w: int):
    """(median face box (x, y, w, h), fraction of frames, indices of those frames) or None."""
    cands = [f for faces in per_frame for f in faces if f[2] <= MAX_FACE_W * w]
    best_members: list[tuple[int, np.ndarray]] = []
    clusters: list[list[tuple[int, np.ndarray]]] = []
    for c in cands:
        cx, cy, cw = c[0] + c[2] / 2, c[1] + c[3] / 2, c[2]
        members = []
        for i, faces in enumerate(per_frame):
            for f in faces:
                fx, fy = f[0] + f[2] / 2, f[1] + f[3] / 2
                # Heads move: allow about a face-width of drift and some zoom.
                if abs(fx - cx) < 0.9 * cw and abs(fy - cy) < 0.9 * cw and 0.6 < f[2] / cw < 1.6:
                    members.append((i, f))
                    break
        clusters.append(members)
    # Among faces steady enough to be a cam, take the largest (in group streams the main
    # streamer's cam is usually the big one); otherwise fall back to the steadiest.
    n = max(1, len(per_frame))
    steady = [m for m in clusters if len(m) / n >= MIN_CONSISTENCY]
    if steady:
        best_members = max(steady, key=lambda m: (float(np.median([f[2] for _, f in m])), len(m)))
    elif clusters:
        best_members = max(clusters, key=len)
    if not best_members:
        return None
    frac = len(best_members) / max(1, len(per_frame))
    box = np.median(np.array([f for _, f in best_members])[:, :4], axis=0)
    return box, frac, [i for i, _ in best_members]


def _maps(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(static edge fraction, temporal std) per pixel."""
    import cv2
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    edges = np.mean([cv2.Canny(g, 60, 160) > 0 for g in grays], axis=0)
    stack = np.stack([g.astype(np.float32) for g in grays])
    std = cv2.blur(stack.std(axis=0), (5, 5))
    return edges, std


def _find_line(edges, std, side: str, face, w: int, h: int) -> float | None:
    """Best border position on one side of the face (pixels), or None."""
    fx, fy, fw, fh = face
    band = 8
    if side in ("left", "right"):
        a, b = int(max(0, fy - 0.5 * fh)), int(min(h, fy + 1.6 * fh))
        if side == "left":
            rng = range(int(fx - 0.3 * fw), int(max(1, fx - 4 * fw)), -1)
        else:
            rng = range(int(fx + 1.3 * fw), int(min(w - 1, fx + 5 * fw)))
    else:
        a, b = int(max(0, fx - 0.6 * fw)), int(min(w, fx + 1.6 * fw))
        if side == "top":
            rng = range(int(fy - 0.4 * fh), int(max(1, fy - 3 * fh)), -1)
        else:
            rng = range(int(fy + 1.4 * fh), int(min(h - 1, fy + 3.5 * fh)))
    if b - a < 4:
        return None
    scored: list[tuple[int, float]] = []
    for p in rng:
        if side in ("left", "right"):
            e = edges[a:b, max(0, p - 1):p + 2].max(axis=1).mean()
            inner = std[a:b, p:p + band] if side == "left" else std[a:b, max(0, p - band):p]
            outer = std[a:b, max(0, p - band):p] if side == "left" else std[a:b, p:p + band]
        else:
            e = edges[max(0, p - 1):p + 2, a:b].max(axis=0).mean()
            inner = std[p:p + band, a:b] if side == "top" else std[max(0, p - band):p, a:b]
            outer = std[max(0, p - band):p, a:b] if side == "top" else std[p:p + band, a:b]
        if inner.size == 0 or outer.size == 0:
            continue
        mi, mo = float(inner.mean()), float(outer.mean())
        contrast = max(0.0, (mo - mi) / (mo + mi + 1e-3))
        scored.append((p, max(0.5 * float(e) + 0.5 * contrast, 0.55 * float(e))))
    if not scored:
        return None
    best_s = max(sc for _, sc in scored)
    if best_s < LINE_SCORE:
        return None
    # The nearest line that is nearly as strong as the best one: with stacked cams the shared
    # border between them is what we want, not the far edge of the neighbouring cam.
    for p, sc in scored:  # rng runs outward from the face
        if sc >= max(LINE_SCORE, 0.85 * best_s):
            return float(p)
    return None


CAM_ASPECT = 1.5            # typical webcam overlay (between 4:3 and 16:9), used for missing sides


def find_box(frames: list[np.ndarray], model: Path) -> dict | None:
    if len(frames) < 4:
        return None
    h, w = frames[0].shape[:2]
    steady = _steady_face(_faces(frames, model), w)
    if steady is None:
        return None
    face, frac, idx = steady
    if frac < MIN_CONSISTENCY:
        return None
    # The cam must be there in both halves of the clip: compilations cut between streamers, and a
    # split layout that shows HUD in the top panel after a cut looks broken.
    half = len(frames) / 2
    first = sum(1 for i in idx if i < half)
    second = len(idx) - first
    if first < 0.35 * half or second < 0.35 * half:
        return None
    fx, fy, fw, fh = (float(v) for v in face)
    cx, cy = fx + fw / 2, fy + fh / 2
    # Borders are measured only on frames that show this face (compilations cut between streamers).
    edges, std = _maps([frames[i] for i in idx])

    found = {s: _find_line(edges, std, s, (fx, fy, fw, fh), w, h) for s in ("left", "right", "top", "bottom")}
    # A side with no line but the frame edge right next to the face: the cam touches the frame edge.
    snap = {"left": fx < 1.3 * fw, "right": w - (fx + fw) < 1.3 * fw,
            "top": fy < 1.0 * fh, "bottom": h - (fy + fh) < 1.8 * fh}
    edge_pos = {"left": 0.0, "right": float(w), "top": 0.0, "bottom": float(h)}
    side = {k: (found[k] if found[k] is not None else edge_pos[k] if snap[k] else None) for k in found}

    # Horizontal extent.
    if side["left"] is not None and side["right"] is not None:
        x0, x1 = side["left"], side["right"]
    elif side["left"] is not None:
        x0 = side["left"]; x1 = x0 + 2 * (cx - x0)
    elif side["right"] is not None:
        x1 = side["right"]; x0 = x1 - 2 * (x1 - cx)
    else:
        x0, x1 = None, None
    # Vertical extent.
    if side["top"] is not None and side["bottom"] is not None:
        y0, y1 = side["top"], side["bottom"]
        # Taller than any webcam overlay: one of the lines is HUD, not the cam border. Keep the nearer.
        if x0 is not None and (x1 - x0) / max(1.0, y1 - y0) < 0.9:
            if (cy - y0) <= (y1 - cy):
                side["bottom"] = None
            else:
                side["top"] = None
            y0 = y1 = None
    else:
        y0 = y1 = None
    # Fill in what's missing from the other axis with a webcam aspect ratio.
    if x0 is None and y0 is None:
        bw = max(3.2 * fw, 2.0 * fh * CAM_ASPECT)
        x0, x1 = cx - bw / 2, cx + bw / 2
    if x0 is not None and y0 is None:
        bh = max((x1 - x0) / CAM_ASPECT, 1.9 * fh)
        if side["top"] is not None:
            y0, y1 = side["top"], side["top"] + bh
        elif side["bottom"] is not None:
            y0, y1 = side["bottom"] - bh, side["bottom"]
        else:
            y0 = cy - 0.45 * bh; y1 = y0 + bh  # faces sit a bit above the middle of a cam
    if x0 is None:
        bw = (y1 - y0) * CAM_ASPECT
        x0, x1 = cx - bw / 2, cx + bw / 2
    # Shape check: webcam overlays are landscape-ish. Too tall -> trim height around the face
    # (face a bit above the middle); far too wide -> trim width around the face.
    if (x1 - x0) / max(1.0, y1 - y0) < 1.0:
        bh = (x1 - x0) / 1.3
        y0 = max(y0, cy - 0.45 * bh)
        y1 = min(y1, y0 + bh)
    if (x1 - x0) / max(1.0, y1 - y0) > 2.4:
        bw = (y1 - y0) * 2.0
        x0 = max(x0, cx - bw / 2)
        x1 = min(x1, x0 + bw)
    # The inferred box must still contain the face; then clamp to the frame.
    x0, x1 = min(x0, fx - 0.3 * fw), max(x1, fx + 1.3 * fw)
    y0, y1 = min(y0, fy - 0.3 * fh), max(y1, fy + 1.3 * fh)
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
    bw, bh = x1 - x0, y1 - y0
    if bw < 1.5 * fw or bh < 1.3 * fh or (bw * bh) / (w * h) > MAX_BOX_AREA:
        return None
    n_found = sum(v is not None for v in found.values())
    n_snap = sum(1 for k in found if found[k] is None and snap[k])
    if n_found == 0 and n_snap == 0 and frac < 0.75:
        return None  # a face with no visible overlay edge and not steady enough: likely not a facecam
    conf = round(min(1.0, 0.5 * frac + 0.5 * (n_found + 0.5 * n_snap) / 4), 2)
    return {"x": round(x0 / w, 4), "y": round(y0 / h, 4), "w": round(bw / w, 4), "h": round(bh / h, 4),
            "confidence": conf, "face_in_frames": round(frac, 2),
            "borders_found": [k for k, v in found.items() if v is not None]}


def detect(source: Path, start: float, end: float, src_w: int, src_h: int, rep: Reporter) -> dict | None:
    """Facecam box for this clip of `source` (times relative to the file), or None."""
    try:
        model = _model(rep)
        frames = sample_frames(source, start, end - start, src_w, src_h)
        return find_box(frames, model)
    except ClipperError as e:
        rep.log(f"Facecam detection skipped: {e.message}")
        return None
    except Exception as e:  # detection is a bonus; never fail a render over it
        rep.log(f"Facecam detection failed ({type(e).__name__}: {e}); using the profile layout.")
        return None


def describe(box: dict | None) -> str:
    if not box:
        return "no facecam found"
    cx, cy = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
    v = "top" if cy < 0.4 else "bottom" if cy > 0.6 else "middle"
    hz = "left" if cx < 0.4 else "right" if cx > 0.6 else "center"
    return f"facecam at {v}-{hz} ({box['w'] * 100:.0f}% x {box['h'] * 100:.0f}% of frame)"

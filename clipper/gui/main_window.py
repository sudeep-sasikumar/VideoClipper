"""Main window: link + profile + rules + Start, live log, and the editable results list."""
from __future__ import annotations

import os
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

from .. import __version__, profiles, project
from ..errors import Cancelled, ClipperError
from ..events import Reporter
from ..paths import USER_PROFILES_DIR, BUNDLED_PROFILES_DIR, load_settings, save_settings
from ..pipeline import JobConfig, run_job
from ..rules import effective_length
from ..timefmt import fmt_precise, fmt_ts, parse_ts

STAGE_LABELS = {"check": "Checking tools", "download": "Downloading", "transcribe": "Transcribing",
                "signals": "Finding spikes", "rank": "Ranking with Claude", "render": "Rendering"}


# --------------------------------------------------------------------------- workers

class Worker(QObject):
    log = Signal(str)
    progress = Signal(str, float, float)
    finished = Signal(object)
    failed = Signal(str, str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.rep = Reporter(on_log=self.log.emit, on_progress=self.progress.emit)

    def run(self):
        try:
            self.finished.emit(self.fn(self.rep))
        except Cancelled:
            self.failed.emit("Cancelled", "The job was cancelled.")
        except ClipperError as e:
            self.failed.emit(e.title, str(e))
        except MemoryError:
            self.failed.emit("Out of memory", "Close other programs or pick a smaller Whisper model in Settings.")
        except Exception as e:  # unexpected: show it rather than die silently
            self.log.emit(traceback.format_exc())
            self.failed.emit("Unexpected error", f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- settings

class SettingsDialog(QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.s = dict(settings)
        lay = QFormLayout(self)
        row = QHBoxLayout()
        self.out = QLineEdit(self.s["output_dir"])
        b = QPushButton("Browse...")
        b.clicked.connect(self._browse)
        row.addWidget(self.out)
        row.addWidget(b)
        lay.addRow("Output folder", row)
        self.model = QComboBox()
        self.model.addItem("Use the profile's model", "")
        for m, desc in (("tiny", "fastest, rough"), ("base", "fast"), ("small", "balanced"),
                        ("medium", "slow, accurate"), ("large-v3-turbo", "GPU recommended")):
            self.model.addItem(f"{m} ({desc})", m)
        self.model.setCurrentIndex(max(0, self.model.findData(self.s.get("whisper_model", ""))))
        lay.addRow("Whisper model", self.model)
        self.keep = QCheckBox("Keep the downloaded video (faster re-renders, uses disk space)")
        self.keep.setChecked(bool(self.s.get("keep_source")))
        lay.addRow("", self.keep)
        self.maxc = QLineEdit(str(self.s.get("max_clips", 20)))
        lay.addRow("Max clips", self.maxc)
        self.cmodel = QLineEdit(self.s.get("claude_model", ""))
        self.cmodel.setPlaceholderText("CLI default (e.g. sonnet, opus)")
        lay.addRow("Claude model", self.cmodel)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self.resize(560, 0)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.out.text())
        if d:
            self.out.setText(d)

    def values(self) -> dict:
        self.s["output_dir"] = self.out.text().strip()
        self.s["whisper_model"] = self.model.currentData()
        self.s["keep_source"] = self.keep.isChecked()
        try:
            self.s["max_clips"] = max(1, min(100, int(self.maxc.text())))
        except ValueError:
            pass
        self.s["claude_model"] = self.cmodel.text().strip()
        return self.s


# --------------------------------------------------------------------------- main window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Clipper {__version__}")
        self.resize(1280, 820)
        self.settings = load_settings()
        self.profiles: dict[str, dict] = {}
        self.proj: dict | None = None
        self.thread: QThread | None = None
        self.worker: Worker | None = None

        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)

        # --- input row
        r1 = QHBoxLayout()
        self.link = QLineEdit()
        self.link.setPlaceholderText("Paste a YouTube or Twitch video link (up to 4 hours)")
        self.profile_box = QComboBox()
        self.profile_box.setMinimumWidth(170)
        self.profile_box.currentIndexChanged.connect(self._update_rules_hint)
        b_prof = QPushButton("Profiles folder")
        b_prof.setToolTip("Copy a .json file here to add a new niche profile")
        b_prof.clicked.connect(self._open_profiles)
        b_set = QPushButton("Settings")
        b_set.clicked.connect(self._settings)
        r1.addWidget(QLabel("Link"))
        r1.addWidget(self.link, 1)
        r1.addWidget(QLabel("Profile"))
        r1.addWidget(self.profile_box)
        r1.addWidget(b_prof)
        r1.addWidget(b_set)
        v.addLayout(r1)

        self.rules = QPlainTextEdit()
        self.rules.setPlaceholderText("Campaign rules (optional). Added to Claude's instructions; length "
                                      "limits like \"15-45 seconds\" or \"max 30s\" are enforced.")
        self.rules.setFixedHeight(70)
        self.rules.textChanged.connect(self._update_rules_hint)
        v.addWidget(self.rules)

        r2 = QHBoxLayout()
        self.rules_hint = QLabel()
        self.rules_hint.setStyleSheet("color: #888")
        self.start_btn = QPushButton("Start")
        self.start_btn.setMinimumWidth(110)
        self.start_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self.start_btn.clicked.connect(self._start)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        b_open = QPushButton("Open project...")
        b_open.clicked.connect(self._open_project)
        r2.addWidget(self.rules_hint, 1)
        r2.addWidget(b_open)
        r2.addWidget(self.cancel_btn)
        r2.addWidget(self.start_btn)
        v.addLayout(r2)

        r3 = QHBoxLayout()
        self.stage = QLabel("Ready.")
        self.stage.setMinimumWidth(260)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        r3.addWidget(self.stage)
        r3.addWidget(self.bar, 1)
        v.addLayout(r3)

        # --- log | results | editor
        split = QSplitter(Qt.Horizontal)
        logbox = QGroupBox("Progress log")
        lv = QVBoxLayout(logbox)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setStyleSheet("font-family: Consolas, monospace; font-size: 9pt;")
        lv.addWidget(self.log)
        split.addWidget(logbox)

        resbox = QGroupBox("Clips (best first)")
        rv = QVBoxLayout(resbox)
        self.folder_label = QLabel("")
        self.folder_label.setWordWrap(True)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._select)
        self.list.itemDoubleClicked.connect(lambda _: self._play())
        b_folder = QPushButton("Open output folder")
        b_folder.clicked.connect(self._open_folder)
        rv.addWidget(self.folder_label)
        rv.addWidget(self.list, 1)
        rv.addWidget(b_folder)
        split.addWidget(resbox)

        edbox = QGroupBox("Edit clip")
        ev = QVBoxLayout(edbox)
        f = QFormLayout()
        self.e_info = QLabel("Select a clip.")
        self.e_info.setWordWrap(True)
        self.e_start = QLineEdit()
        self.e_end = QLineEdit()
        self.e_start.setToolTip("H:MM:SS.ss in the source video")
        self.e_end.setToolTip("H:MM:SS.ss in the source video")
        self.e_hook = QLineEdit()
        self.e_vo = QCheckBox("Voiceover intro")
        self.e_intro = QLineEdit()
        self.e_caps = QPlainTextEdit()
        self.e_caps.setToolTip("One caption line per row. Keep the same number of words in a row to keep "
                               "exact timing; otherwise the row's words are spread evenly.")
        f.addRow(self.e_info)
        f.addRow("Start", self.e_start)
        f.addRow("End", self.e_end)
        f.addRow("Hook text", self.e_hook)
        f.addRow(self.e_vo)
        f.addRow("Intro line", self.e_intro)
        ev.addLayout(f)
        ev.addWidget(QLabel("Captions (one line per row)"))
        ev.addWidget(self.e_caps, 1)
        eb = QHBoxLayout()
        self.b_play = QPushButton("Play")
        self.b_play.clicked.connect(self._play)
        self.b_render = QPushButton("Save && re-render")
        self.b_render.clicked.connect(self._rerender)
        self.b_delete = QPushButton("Delete clip")
        self.b_delete.clicked.connect(self._delete)
        eb.addWidget(self.b_play)
        eb.addWidget(self.b_render)
        eb.addWidget(self.b_delete)
        ev.addLayout(eb)
        split.addWidget(edbox)
        split.setSizes([430, 380, 470])
        v.addWidget(split, 1)

        self._load_profiles()
        self._set_editor_enabled(False)
        self._update_rules_hint()
        self._append(f"Clipper {__version__}. Output folder: {self.settings['output_dir']}")

    # ----------------------------------------------------------------- helpers
    def _append(self, line: str):
        self.log.appendPlainText(line)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _load_profiles(self):
        self.profiles = profiles.load_all(self._append)
        last = self.settings.get("last_profile")
        self.profile_box.blockSignals(True)
        self.profile_box.clear()
        for pid, p in sorted(self.profiles.items(), key=lambda kv: (kv[0] != "gaming", kv[1]["name"])):
            self.profile_box.addItem(p["name"], pid)
            self.profile_box.setItemData(self.profile_box.count() - 1, p.get("description", ""), Qt.ToolTipRole)
        idx = self.profile_box.findData(last if last in self.profiles else profiles.default_id(self.profiles))
        self.profile_box.setCurrentIndex(max(0, idx))
        self.profile_box.blockSignals(False)

    def _current_profile(self) -> dict:
        return self.profiles[self.profile_box.currentData()]

    def _update_rules_hint(self):
        if not self.profiles:
            return
        p = self._current_profile()
        ln = effective_length(p, self.rules.toPlainText())
        src = "from your rules" if ln["from_rules"] else "from the profile"
        vo = "voiceover on" if p.get("voiceover", {}).get("enabled") else "no voiceover"
        self.rules_hint.setText(f"Clip length {ln['min']:.0f}-{ln['max']:.0f} s ({src}) · layout: "
                                f"{p['layout'].get('mode')} · {vo}")

    def _busy(self, busy: bool):
        self.start_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.b_render.setEnabled(not busy and self.proj is not None)
        self.b_delete.setEnabled(not busy and self.proj is not None)

    def _run(self, fn, on_done, label: str):
        self._busy(True)
        self.stage.setText(label)
        self.thread = QThread(self)
        self.worker = Worker(fn)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self._on_done = on_done
        # Bound methods (not lambdas) so Qt queues these onto the GUI thread.
        self.worker.log.connect(self._append)
        self.worker.progress.connect(self._progress)
        self.worker.finished.connect(self._worker_finished)
        self.worker.failed.connect(self._worker_failed)
        self.thread.start()

    def _worker_finished(self, res):
        self._job_end()
        try:
            self._on_done(res)
        except ClipperError as e:
            QMessageBox.warning(self, e.title, str(e))

    def _worker_failed(self, title: str, msg: str):
        self._job_end()
        self._job_failed(title, msg)

    def _job_end(self):
        self._busy(False)
        if self.thread:
            self.thread.quit()
            self.thread.wait(5000)
        self.thread = None
        self.worker = None

    def _job_failed(self, title: str, msg: str):
        self.stage.setText(title)
        self._append(f"{title}: {msg}")
        if title != "Cancelled":
            QMessageBox.warning(self, title, msg)

    def _progress(self, stage: str, frac: float, overall: float):
        self.bar.setValue(int(overall * 1000))
        self.stage.setText(f"{STAGE_LABELS.get(stage, stage)}  {frac * 100:.0f}%")

    # ----------------------------------------------------------------- actions
    def _start(self):
        url = self.link.text().strip()
        if not url:
            QMessageBox.information(self, "No link", "Paste a YouTube or Twitch video link first.")
            return
        prof = self._current_profile()
        self.settings["last_profile"] = prof["_id"]
        save_settings(self.settings)
        cfg = JobConfig(url=url, profile=prof, rules=self.rules.toPlainText().strip(),
                        output_dir=self.settings["output_dir"], whisper_model=self.settings.get("whisper_model", ""),
                        keep_source=bool(self.settings.get("keep_source")),
                        max_clips=int(self.settings.get("max_clips", 20)),
                        claude_model=self.settings.get("claude_model", ""))
        self.list.clear()
        self.proj = None
        self._set_editor_enabled(False)
        self.bar.setValue(0)
        self._append(f"--- Starting: {url} (profile: {prof['name']})")
        self._run(lambda rep: run_job(cfg, rep), self._job_done, "Starting...")

    def _cancel(self):
        if self.worker:
            self.worker.rep.cancel()
            self.stage.setText("Cancelling...")

    def _job_done(self, folder):
        self.bar.setValue(1000)
        self.stage.setText("Done.")
        self._load_project(Path(folder))

    def _open_project(self):
        d = QFileDialog.getExistingDirectory(self, "Open a Clipper output folder", self.settings["output_dir"])
        if d:
            try:
                self._load_project(Path(d))
            except ClipperError as e:
                QMessageBox.warning(self, e.title, str(e))

    def _load_project(self, folder: Path):
        self.proj = project.load(folder)
        self.folder_label.setText(f"<b>{self.proj['video']['title']}</b><br>{folder}")
        self._refresh_list()
        self._busy(False)

    def _refresh_list(self, keep_row: int = 0):
        self.list.blockSignals(True)
        self.list.clear()
        for c in sorted(self.proj["clips"], key=lambda c: c["rank"]):
            dur = c["end"] - c["start"]
            flag = "  [failed]" if c.get("error") else ""
            vo = "  🎙" if c.get("voiceover") else ""
            it = QListWidgetItem(f"#{c['rank']:02d}  {c['score']:.0f}/10  {fmt_ts(c['start'])} ({dur:.0f}s){vo}{flag}\n"
                                 f"      {c.get('hook_text', '')}")
            it.setData(Qt.UserRole, c["rank"])
            it.setToolTip(c.get("reason", ""))
            self.list.addItem(it)
        self.list.blockSignals(False)
        if self.list.count():
            self.list.setCurrentRow(min(keep_row, self.list.count() - 1))
            self._select(self.list.currentRow())
        else:
            self._set_editor_enabled(False)

    def _clip(self) -> dict | None:
        it = self.list.currentItem()
        if not it or not self.proj:
            return None
        rank = it.data(Qt.UserRole)
        return next((c for c in self.proj["clips"] if c["rank"] == rank), None)

    def _set_editor_enabled(self, on: bool):
        for w in (self.e_start, self.e_end, self.e_hook, self.e_vo, self.e_intro, self.e_caps,
                  self.b_play, self.b_render, self.b_delete):
            w.setEnabled(on)
        if not on:
            self.e_info.setText("Select a clip.")

    def _select(self, _row: int):
        c = self._clip()
        if not c:
            self._set_editor_enabled(False)
            return
        self._set_editor_enabled(True)
        self._busy(self.thread is not None)
        info = f"<b>#{c['rank']:02d} · score {c['score']:.0f}/10</b> · {c.get('file') or 'not rendered'}<br>" \
               f"<i>{c.get('reason', '')}</i>"
        if c.get("error"):
            info += f"<br><span style='color:#c33'>Render failed: {c['error']}</span>"
        self.e_info.setText(info)
        self.e_start.setText(fmt_precise(c["start"]))
        self.e_end.setText(fmt_precise(c["end"]))
        self.e_hook.setText(c.get("hook_text", ""))
        self.e_vo.setChecked(bool(c.get("voiceover")))
        self.e_intro.setText(c.get("intro_line", ""))
        lines = c.get("captions") or []
        self.e_caps.setPlainText("\n".join(" ".join(w[2] for w in ln["words"]) for ln in lines))
        self._caps_original = self.e_caps.toPlainText()

    def _play(self):
        c = self._clip()
        if c and c.get("file"):
            p = Path(self.proj["_folder"]) / c["file"]
            if p.exists():
                os.startfile(str(p))  # default player
                return
        QMessageBox.information(self, "Not rendered", "This clip has no video file yet. Click Save & re-render.")

    def _rerender(self):
        c = self._clip()
        if not c:
            return
        s, e = parse_ts(self.e_start.text()), parse_ts(self.e_end.text())
        if s is None or e is None:
            QMessageBox.warning(self, "Bad time", "Start and end must look like 0:01:23.50 or 83.5")
            return
        caps_changed = self.e_caps.toPlainText() != self._caps_original
        try:
            project.edit_clip(self.proj, c, start=s, end=e, hook_text=self.e_hook.text(),
                              voiceover_on=self.e_vo.isChecked(), intro_line=self.e_intro.text())
            if caps_changed and (abs(s - c["start"]) < 0.01 and abs(e - c["end"]) < 0.01):
                project.edit_clip(self.proj, c, caption_lines=self.e_caps.toPlainText().splitlines())
        except ClipperError as err:
            QMessageBox.warning(self, err.title, str(err))
            return
        if c.get("voiceover") and not c.get("intro_line"):
            QMessageBox.warning(self, "No intro line", "Write an intro line for the voiceover first.")
            return
        project.save(self.proj)
        row = self.list.currentRow()
        proj = self.proj
        self._append(f"--- Re-rendering clip #{c['rank']:02d}")

        def job(rep):
            project.render_one(proj, c, rep)
            project.save(proj)
            return row

        def done(r):
            self.stage.setText(f"Clip #{c['rank']:02d} re-rendered.")
            self._append(f"Clip #{c['rank']:02d} saved: {c['file']}")
            self._refresh_list(r)

        self._run(job, done, f"Re-rendering clip #{c['rank']:02d}...")

    def _delete(self):
        c = self._clip()
        if not c:
            return
        if QMessageBox.question(self, "Delete clip", f"Delete clip #{c['rank']:02d} and its video file?\n"
                                f"{c.get('file') or ''}\n\nThis can't be undone.") != QMessageBox.Yes:
            return
        row = self.list.currentRow()
        project.delete_clip(self.proj, c)
        self._append(f"Deleted clip #{c['rank']:02d}.")
        self._refresh_list(row)

    def _open_folder(self):
        folder = self.proj["_folder"] if self.proj else self.settings["output_dir"]
        Path(folder).mkdir(parents=True, exist_ok=True)
        os.startfile(folder)

    def _open_profiles(self):
        target = BUNDLED_PROFILES_DIR if BUNDLED_PROFILES_DIR.is_dir() else USER_PROFILES_DIR
        target.mkdir(parents=True, exist_ok=True)
        os.startfile(str(target))
        QMessageBox.information(self, "Profiles", f"Copy any .json profile in\n{target}\n(or {USER_PROFILES_DIR}),\n"
                                "rename it and edit it. Click OK to reload the list.")
        self._load_profiles()
        self._update_rules_hint()

    def _settings(self):
        d = SettingsDialog(self.settings, self)
        if d.exec():
            self.settings = d.values()
            save_settings(self.settings)
            self._append(f"Settings saved. Output folder: {self.settings['output_dir']}")

    def closeEvent(self, ev):
        if self.thread is not None:
            if QMessageBox.question(self, "Job running", "A job is running. Cancel it and quit?") != QMessageBox.Yes:
                ev.ignore()
                return
            if self.worker:
                self.worker.rep.cancel()
            self.thread.quit()
            self.thread.wait(3000)
        ev.accept()

"""Draw the app icon (assets/clipper.ico) with Qt, so no image editor is needed."""
import sys
from pathlib import Path

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QGuiApplication, QImage, QPainter, QPainterPath

app = QGuiApplication(sys.argv)
out = Path(__file__).resolve().parent.parent / "assets"
out.mkdir(exist_ok=True)
img = QImage(256, 256, QImage.Format_ARGB32)
img.fill(Qt.transparent)
p = QPainter(img)
p.setRenderHint(QPainter.Antialiasing)
path = QPainterPath()
path.addRoundedRect(QRectF(8, 8, 240, 240), 56, 56)
p.fillPath(path, QBrush(QColor("#E3122C")))
# 9:16 frame
p.setBrush(QColor("#FFFFFF"))
p.setPen(Qt.NoPen)
p.drawRoundedRect(QRectF(78, 40, 100, 176), 14, 14)
p.setBrush(QColor("#E3122C"))
tri = QPainterPath()
tri.moveTo(112, 96)
tri.lineTo(112, 160)
tri.lineTo(160, 128)
tri.closeSubpath()
p.fillPath(tri, QBrush(QColor("#E3122C")))
p.end()
img.save(str(out / "clipper.ico"))
img.save(str(out / "clipper.png"))
print("wrote", out / "clipper.ico")

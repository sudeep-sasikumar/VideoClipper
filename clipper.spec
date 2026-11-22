# PyInstaller spec: one-folder build of Clipper.  Build with:  .\build.ps1
# Models are NOT bundled: Whisper and Kokoro download to %APPDATA%\Clipper\models on first use.
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []
for pkg in ("faster_whisper", "ctranslate2", "kokoro_onnx", "espeakng_loader", "phonemizer",
            "yt_dlp_ejs", "language_tags", "segments", "csvw", "tokenizers"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass
hiddenimports += collect_submodules("yt_dlp")
hiddenimports += ["soundfile", "huggingface_hub"]

a = Analysis(
    ["clipper_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[
        "tkinter", "matplotlib", "torch", "tensorflow", "IPython", "pytest",
        "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
        "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtQuick3D", "PySide6.Qt3DCore", "PySide6.QtCharts",
        "PySide6.QtDataVisualization", "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtPdfWidgets",
        "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtPositioning", "PySide6.QtLocation",
        "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtSql", "PySide6.QtTest",
        "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtOpenGL", "PySide6.QtSvg",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Clipper",
    console=False,
    icon="assets/clipper.ico",
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Clipper", upx=False)

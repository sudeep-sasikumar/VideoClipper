# Build dist\Clipper\Clipper.exe (one folder). Run from the repo root:  .\build.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No venv. Run:  python -m venv .venv ; .\.venv\Scripts\pip install -r requirements.txt" }
if (-not (Test-Path "assets\clipper.ico")) { & $py tools\make_icon.py }
& $py -m PyInstaller clipper.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
# Profiles live next to the exe so users can copy/edit them.
Copy-Item -Recurse -Force profiles "dist\Clipper\profiles"
Copy-Item -Force README.md "dist\Clipper\README.md"
Write-Host "Built: $PSScriptRoot\dist\Clipper\Clipper.exe"

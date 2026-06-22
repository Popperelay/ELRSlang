[CmdletBinding()]
param(
    [string]$VenvPath = ".venv"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment python not found: $VenvPython. Run scripts\bootstrap.ps1 first."
}

& $VenvPython -m unittest discover -s tests
& $VenvPython -m elrslang.viewer --frames 1 --graph slangpy_preview --backend automatic --width 32 --height 32
& $VenvPython -m elrslang.viewer --frames 1 --graph raster_forward --backend automatic --width 32 --height 32
& $VenvPython -m elrslang.viewer --frames 1 --graph dxr_pathtrace --backend automatic --width 32 --height 32

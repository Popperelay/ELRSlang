[CmdletBinding()]
param(
    [string]$PythonExe = "",
    [string]$VenvPath = ".venv",
    [switch]$InstallPython,
    [switch]$SkipSmoke,
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$Args = @()
    )
    Write-Host (">> " + $Exe + " " + ($Args -join " "))
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Exe $($Args -join ' ')"
    }
}

function New-PythonCandidate {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [string[]]$Args = @(),
        [string]$Label = ""
    )
    [PSCustomObject]@{
        Exe = $Exe
        Args = $Args
        Label = if ($Label) { $Label } else { $Exe }
    }
}

function Test-Python312 {
    param([Parameter(Mandatory = $true)]$Candidate)
    try {
        $code = "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
        $result = & $Candidate.Exe @($Candidate.Args + @("-c", $code)) 2>$null
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($result)) {
            return $null
        }
        $version = [string]$result
        if ($version -like "3.12.*") {
            return [PSCustomObject]@{
                Exe = $Candidate.Exe
                Args = $Candidate.Args
                Label = $Candidate.Label
                Version = $version
            }
        }
    } catch {
        return $null
    }
    return $null
}

function Find-Python312 {
    $candidates = @()

    if ($PythonExe) {
        $candidates += New-PythonCandidate -Exe $PythonExe -Label $PythonExe
    }

    $python312 = Get-Command python3.12 -ErrorAction SilentlyContinue
    if ($python312) {
        $candidates += New-PythonCandidate -Exe $python312.Source -Label "python3.12"
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        $candidates += New-PythonCandidate -Exe $python.Source -Label "python"
    }

    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        $candidates += New-PythonCandidate -Exe $py.Source -Args @("-3.12") -Label "py -3.12"
    }

    foreach ($candidate in $candidates) {
        $match = Test-Python312 $candidate
        if ($match) {
            return $match
        }
    }

    return $null
}

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

Write-Host "ELRSlang bootstrap"
Write-Host "Repo: $RepoRoot"

if ($InstallPython) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "winget was not found. Install CPython 3.12 x64 from https://www.python.org/downloads/release/python-31210/ and rerun bootstrap."
    }
    Invoke-External $winget.Source @(
        "install",
        "--id",
        "Python.Python.3.12",
        "-e",
        "--scope",
        "user",
        "--accept-package-agreements",
        "--accept-source-agreements"
    )
}

$Python = Find-Python312
if (-not $Python) {
    throw @"
Python 3.12 x64 was not found.

Install it with winget:
  winget install --id Python.Python.3.12 -e --scope user --accept-package-agreements --accept-source-agreements

Then close and reopen PowerShell, and rerun:
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bootstrap.ps1

If Python 3.12 is installed but not on PATH, pass its full path:
  powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bootstrap.ps1 -PythonExe "C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe"
"@
}

Write-Host "Using Python $($Python.Version): $($Python.Label)"

if ($Recreate -and (Test-Path $VenvPath)) {
    $target = Resolve-Path -LiteralPath $VenvPath
    if (-not ($target.Path.StartsWith($RepoRoot.Path))) {
        throw "Refusing to remove venv outside repo: $target"
    }
    Remove-Item -LiteralPath $target.Path -Recurse -Force
}

if (Test-Path $VenvPath) {
    $existingPython = Join-Path $VenvPath "Scripts\python.exe"
    if (Test-Path $existingPython) {
        $existingVersion = & $existingPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
        if (-not ($existingVersion -like "3.12.*")) {
            throw "Existing venv uses Python $existingVersion. Rerun bootstrap with -Recreate to rebuild it with Python 3.12."
        }
    }
}

if (-not (Test-Path $VenvPath)) {
    Invoke-External $Python.Exe @($Python.Args + @("-m", "venv", $VenvPath))
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment python not found: $VenvPython"
}

Invoke-External $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
Invoke-External $VenvPython @("-m", "pip", "install", "--only-binary=:all:", "-r", "requirements\lock-py312-win64.txt")
Invoke-External $VenvPython @("-m", "pip", "install", "--no-deps", "-e", ".")
Invoke-External $VenvPython @("-c", "import slangpy as spy; assert getattr(spy, '__version__', '') == '0.40.1'; print('slangpy', spy.__version__)")
Invoke-External $VenvPython @("-m", "unittest", "discover", "-s", "tests")

if (-not $SkipSmoke) {
    Invoke-External $VenvPython @("-m", "elrslang.viewer", "--frames", "1", "--graph", "slangpy_preview", "--backend", "automatic", "--width", "32", "--height", "32")
    Invoke-External $VenvPython @("-m", "elrslang.viewer", "--frames", "1", "--graph", "raster_forward", "--backend", "automatic", "--width", "32", "--height", "32")
    Invoke-External $VenvPython @("-m", "elrslang.viewer", "--frames", "1", "--graph", "dxr_pathtrace", "--backend", "automatic", "--width", "32", "--height", "32")
}

Write-Host "Bootstrap completed successfully."

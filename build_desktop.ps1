$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment python not found at $pythonExe"
}

Push-Location $projectRoot
try {
    & $pythonExe -m PyInstaller `
        --noconfirm `
        --clean `
        --onefile `
        --windowed `
        --name "UFC ML Desktop" `
        --collect-all streamlit `
        --collect-all webview `
        ufc_desktop_app.py

    Write-Host ""
    Write-Host "Build complete:"
    Write-Host "  dist\UFC ML Desktop.exe"
}
finally {
    Pop-Location
}

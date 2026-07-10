$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "Welcome to Texts to Transformer for Windows!" -ForegroundColor Cyan
Write-Host "This setup will install Python 3.11 and the private CUDA training environment."
Write-Host "The first run is a large download, so give it a little time."
Write-Host ""

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed yet." -ForegroundColor Yellow
    Write-Host "Install it from https://docs.astral.sh/uv/getting-started/installation/"
    Write-Host "Then reopen PowerShell in this folder and run this script again."
    exit 1
}

uv python install 3.11
uv sync
uv run imessage-cuda doctor

Write-Host ""
Write-Host "Setup finished." -ForegroundColor Green
Write-Host "The doctor report above should say cuda_available: true and show your NVIDIA GPU."
Write-Host "Next: open docs\WINDOWS_WALKTHROUGH.md and continue with Part 2."

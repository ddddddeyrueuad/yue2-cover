# 补下模型脚本：若 models/ 下模型缺失，运行本脚本重新下载
# 用法：powershell -ExecutionPolicy Bypass -File setup_download.ps1
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:HF_HOME = Join-Path $PSScriptRoot ".hfcache"
& ".venv\Scripts\hf.exe" download m-a-p/YuE2-3B --local-dir models/YuE2-3B
& ".venv\Scripts\hf.exe" download m-a-p/YuE2-Vae --local-dir models/YuE2-Vae
& ".venv\Scripts\hf.exe" download m-a-p/SheetSage2 --local-dir models/SheetSage2
& ".venv\Scripts\hf.exe" download m-a-p/MERT-v2-FullSong
Write-Host "模型下载完成。"

@echo off
title YuE2 翻唱工作台
cd /d "%~dp0"
set "HF_HOME=%~dp0.hfcache"
set "PYTHONIOENCODING=utf-8"
if not exist ".venv\Scripts\python.exe" (
  echo [错误] 未找到虚拟环境 .venv，请先完成部署。
  pause
  exit /b 1
)
echo 正在启动 YuE2 翻唱 WebUI...
echo 启动完成后请用浏览器打开:  http://127.0.0.1:7860
echo 关闭本窗口即可停止服务。
".venv\Scripts\python.exe" cover_webui.py
pause

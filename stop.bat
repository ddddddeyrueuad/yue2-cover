@echo off
title YuE2 翻唱工作台 - 停止服务
echo 正在停止 YuE2 翻唱 WebUI 服务并释放显存/内存...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ps = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -and ($_.CommandLine -like '*cover_webui.py*' -or $_.CommandLine -like '*SheetSage2*infer.py*') }; if ($ps) { foreach ($p in $ps) { Write-Host ('  已停止进程 PID ' + $p.ProcessId); Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue } } else { Write-Host '  未发现正在运行的服务进程。' }"
taskkill /FI "WINDOWTITLE eq YuE2 翻唱工作台" >nul 2>&1
ping -n 3 127.0.0.1 >nul
powershell -NoProfile -Command "$l = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue; if ($l) { Write-Host '  警告: 端口 7860 仍被占用，可能有其他程序在监听。' } else { Write-Host '  端口 7860 已释放。' }"
echo.
where nvidia-smi >nul 2>&1 && (echo 当前显卡显存占用: & nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader) || (echo 未找到 nvidia-smi，跳过显卡状态显示。)
echo.
echo 服务已停止，显存与内存已随进程结束自动释放。
pause

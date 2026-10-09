@echo off
setlocal
set "GIT_TERMINAL_PROMPT=0"
title YuE2 翻唱工作台 - 更新代码
cd /d "%~dp0"

if not exist "YuE\.git" (
  echo [错误] 未找到 YuE 仓库，请将本脚本放在部署目录内运行。
  goto :end
)
where git >nul 2>&1
if errorlevel 1 (
  echo [错误] 未找到 git 命令，无法更新。
  goto :end
)

cd /d "%~dp0YuE"
netstat -ano | findstr "LISTENING" | findstr ":7860" >nul 2>&1
if not errorlevel 1 echo [提示] WebUI 正在运行，本次更新将在下次启动服务后生效。

for /f "delims=" %%i in ('git rev-parse --abbrev-ref HEAD') do set "BRANCH=%%i"
echo 分支: %BRANCH%   正在连接 GitHub 检查更新...
git fetch origin >nul 2>&1
if errorlevel 1 (
  echo [错误] 无法连接 GitHub，请检查网络后重试。
  goto :end
)

set "LOCAL_CHANGES="
for /f "delims=" %%i in ('git status --porcelain -uno') do set "LOCAL_CHANGES=1"
set "STASHED="
if defined LOCAL_CHANGES (
  echo 检测到本地补丁，先暂存保护...
  git stash push -m "yue2-cover-local-patches" >nul 2>&1
  if not errorlevel 1 set "STASHED=1"
)

set "OLD_HEAD="
for /f %%i in ('git rev-parse HEAD') do set "OLD_HEAD=%%i"

git merge --ff-only "origin/%BRANCH%" >nul 2>&1
if errorlevel 1 (
  echo [错误] 快进合并失败，仓库状态如下，请手动处理:
  git status
  if defined STASHED git stash pop >nul 2>&1
  goto :end
)

set "NEW_HEAD="
for /f %%i in ('git rev-parse HEAD') do set "NEW_HEAD=%%i"

if "%OLD_HEAD%"=="%NEW_HEAD%" (
  echo.
  echo 已是最新代码，无需更新。
) else (
  echo.
  echo 代码已更新，最新提交:
  git --no-pager log --oneline -10 "%OLD_HEAD%..%NEW_HEAD%"
)

set "DEPS_CHANGED="
if not "%OLD_HEAD%"=="%NEW_HEAD%" for /f "delims=" %%i in ('git --no-pager diff --name-only "%OLD_HEAD%" "%NEW_HEAD%" -- pyproject.toml') do set "DEPS_CHANGED=1"

if defined STASHED (
  echo.
  echo 正在恢复本地 Windows 补丁...
  git stash pop >nul 2>&1
  if errorlevel 1 (
    echo [警告] 补丁恢复冲突: YuE\src\yue2\cuda_graph.py 的 Windows FlashAttention 修复。
    echo   补丁仍保存在 git stash 中，请按 README-翻唱部署说明.md 手动处理，
    echo   处理前生成功能可能报 USE_FLASH_ATTENTION 错误。
  ) else (
    echo 本地补丁已恢复。
  )
)

if defined DEPS_CHANGED (
  echo.
  echo [提示] pyproject.toml 有变化，Python 依赖可能需要同步。
  echo   注意: torch 必须保持 CUDA 版本，请勿直接从 PyPI 重装，详见 README。
)

echo.
echo 完成。模型文件不受影响; WebUI 代码下次启动服务时生效。
echo 如更新后生成报错，请查看 README-翻唱部署说明.md 的补丁与依赖说明。

:end
echo.
pause

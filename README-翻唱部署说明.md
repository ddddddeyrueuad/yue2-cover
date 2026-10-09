# YuE2 翻唱（Cover）工作台 — 部署说明

本项目从 [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)（YuE2）中
只提取了 **翻唱（cover）** 功能，并提供全中文 WebUI。

## 功能流程

1. **上传源歌曲**（wav / mp3 / flac 等任意常见格式）
2. **转录旋律**：SheetSage2 自动把源歌曲转成 ABC 旋律曲谱（可手动编辑）
3. **填写目标风格 + 新歌词**
4. **生成翻唱**：YuE2-3B（`cot=melody` 模式）按原曲旋律演唱新歌词，输出 48 kHz 立体声音频

## 启动方法

双击 `启动翻唱WebUI.bat`，或手动执行：

```powershell
$env:HF_HOME = "E:\yue2 cover\.hfcache"
.\.venv\Scripts\python.exe cover_webui.py
```

然后用浏览器打开 **http://127.0.0.1:7860**。停止服务：双击 `stop.bat`（自动结束相关进程并释放显存/内存），或直接关闭服务窗口。更新代码：双击 `update.bat`（拉取 YuE2 最新代码并自动保护本地补丁；模型文件不受影响）。

- 首次点击「生成翻唱」会自动加载模型（约 1–2 分钟），也可先点「预加载生成模型」。
- 生成一首完整歌曲（约 3–5 分钟音频）通常需要 5–15 分钟，界面日志区会实时显示进度。

## 目录结构

| 路径 | 说明 |
|---|---|
| `cover_webui.py` | 中文 WebUI 主程序（仅 cover 功能） |
| `启动翻唱WebUI.bat` | 一键启动脚本（ANSI/GBK 编码保存，请勿用 UTF-8 覆盖保存） |
| `stop.bat` | 停止服务：结束相关进程，自动释放显存与内存（ANSI/GBK 编码） |
| `update.bat` | 同步更新 YuE2 仓库代码（git pull，自动保护/恢复本地补丁；ANSI/GBK 编码） |
| `setup_download.ps1` | 模型缺失时重新下载模型 |
| `YuE\` | YuE2 官方仓库（推理代码 `YuE\src\yue2`） |
| `models\YuE2-3B` | YuE2 生成模型（7.3 GB） |
| `models\YuE2-Vae` | YuE2 音频 VAE |
| `models\SheetSage2` | 旋律转录模型 |
| `.hfcache\` | HuggingFace 缓存（含 MERT-v2-FullSong） |
| `.venv\` | YuE2 推理环境（Python 3.12，torch 2.10.0+cu126） |
| `.venv-sheetsage2\` | SheetSage2 转录环境（Python 3.11，torch 2.8.0+cu126） |
| `outputs\` | 转录与生成结果输出目录 |

## 环境要点（已配置完成）

- 两个独立虚拟环境（依赖互相冲突，不能合并）：
  - `.venv`：Python 3.12 + torch 2.10.0+cu126 + transformers 4.57.6 + gradio 6.17.3
  - `.venv-sheetsage2`：Python 3.11 + torch 2.8.0+cu126 + transformers 4.45.2
- WebUI 通过子进程调用 `.venv-sheetsage2` 完成转录，主进程用 `.venv` 完成生成。
- `yue2` 包未安装进 site-packages，`cover_webui.py` 通过 `sys.path` 引入 `YuE\src`。
- 必须设置 `HF_HOME=E:\yue2 cover\.hfcache`（启动脚本已处理）。

## 本机已打的补丁

- `YuE\src\yue2\cuda_graph.py`：Windows 版 PyTorch 未编译 FlashAttention 内核，
  原自动探测会误选 flash 后端导致 `USE_FLASH_ATTENTION was not enabled for build`；
  补丁在 Windows 上跳过 flash，改用 cuDNN attention。
- transformers 4.45 复制本地动态模块时只复制第一层 import；WebUI 每次转录前会自动把
  `models\SheetSage2\*.py` 补全进模块缓存（`ensure_sheetsage_module_cache`）。
- `YuE\src\yue2\nar.py`：Windows 版 PyTorch 对分组查询注意力（GQA）没有高效 SDPA 内核，
  生成整首歌时声学阶段会按整块查询实体化 `[heads, q, k]` 分数矩阵（GiB 级）直接 OOM。
  补丁在 Windows 上默认把查询按 512 行分块（只改变临时存储，不改变可见 key 集，
  数值不变）。实测同一负载峰值显存从 16+ GiB 降至 2 GiB。
- `models\SheetSage2\notation_sheetsage2.py`：模型节拍跟踪器在真实音乐上偶尔会漏发
  小节内的某个节拍行（如某小节 ID 为 [1,3,4]），原版 ABC 导出器会直接报
  `non-consecutive beat IDs` 并放弃整首歌。补丁在解析节拍网格后自动用相邻拍的
  时间插值补齐缺失行（含漏下拍的情形），并打印 `filled N missing beat row(s)` 日志。
  该文件会被 WebUI 无条件同步进 transformers 模块缓存后生效。
- 若系统配置了全局代理，Gradio 对 127.0.0.1 的自检会被代理拦截（502）；
  `cover_webui.py` 启动时自动清除代理变量并设置 `NO_PROXY=127.0.0.1,localhost`。

## 常见问题

- **显存不足（OOM）**：本机为 20 GB 显存（官方建议 24 GB），已做四层优化：
  `memory_budget_gib=18`（进程显存上限 16 GiB，为桌面应用留余量）、
  `offload_ar=True`（AR 权重按需卸载）、`vae_core_frames=512`（音频解码分块）、
  nar.py 的 Windows 查询分块补丁（详见上方补丁列表，这是整首歌 OOM 的关键修复）。
  整首歌（5 分钟级）生成实测可在 16 GiB 限额内完成，总耗时约 10 分钟属正常。
- **转录失败**：确认 `ffmpeg` 在 PATH 中（已安装 N-123858）。
- **生成被截断**：日志出现 ⚠️ 时，可缩短歌词或更换 seed。
- **界面英文残留**：日志区中的 `[YuE2] ...` 进度行来自官方库的英文输出，属正常现象。
- **双击 bat 报 `'et' 不是内部或外部命令` / `'姩' ...`**：bat 曾以 UTF-8 + `chcp 65001` 保存，
  中文版 cmd 切换代码页后会错位解析批处理字节（`set` 被吃掉首字母变成 `et`）。
  现已改为 ANSI(GBK) 编码保存；日后手动编辑该 bat 时请保持 ANSI(GBK) 编码。
- **update.bat 提示 pyproject.toml 有变化**：依赖需谨慎同步，torch 必须保持 cu126 CUDA 版本
  （直接从 PyPI 重装会变成 CPU 版）；建议按「环境要点」中两套环境的原始版本手动补齐变化的包。

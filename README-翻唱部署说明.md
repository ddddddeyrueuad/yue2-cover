# YuE2 翻唱（Cover）工作台 — 部署说明与构建复盘

本项目从 [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE)（YuE2）中
只提取了 **翻唱（cover）** 功能，并提供全中文 WebUI。
英文版见 [README.en.md](README.en.md)。

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
| `YuE\` | YuE2 官方仓库（推理代码 `YuE\src\yue2`），已含本项目的 Windows 补丁 |
| `patches\` | 位于 `models\`（被 gitignore）的模型侧补丁，可重新套用 |
| `tools\` | 显存/数值验证脚本，复盘中的实测数字由它们产出 |
| `models\YuE2-3B` | YuE2 生成模型（7.3 GB） |
| `models\YuE2-Vae` | YuE2 音频 VAE |
| `models\SheetSage2` | 旋律转录模型 |
| `.hfcache\` | HuggingFace 缓存（含 MERT-v2-FullSong） |
| `.venv\` | YuE2 推理环境（Python 3.12，torch 2.10.0+cu126） |
| `.venv-sheetsage2\` | SheetSage2 转录环境（Python 3.11，torch 2.8.0+cu126） |
| `outputs\` | 转录与生成结果输出目录 |

## 环境要点（已配置完成）

- 两个独立虚拟环境（依赖互相冲突，不能合并）：
  - `.venv`：Python 3.12.12 + torch 2.10.0+cu126 + transformers 4.57.6 + gradio 6.17.3 + accelerate 1.13.0 + numpy 2.2.6
  - `.venv-sheetsage2`：Python 3.11.14 + torch 2.8.0+cu126 + transformers 4.45.2
- WebUI 通过子进程调用 `.venv-sheetsage2` 完成转录，主进程用 `.venv` 完成生成。
- `yue2` 包未安装进 site-packages，`cover_webui.py` 通过 `sys.path` 引入 `YuE\src`。
- 必须设置 `HF_HOME=E:\yue2 cover\.hfcache`（启动脚本已处理）。
- 上游代码锁定在 YuE 提交 `1647252`（2026-10-06）。
- 硬件：2 × RTX 3080 20 GB（官方建议 24 GB），ffmpeg N-123858。

## 本机已打的补丁

| # | 文件 | 问题 | 补丁要点 |
|---|---|---|---|
| 1 | `YuE\src\yue2\cuda_graph.py` | Windows 版 PyTorch 未编译 FlashAttention 内核，原自动探测误选 flash 后端 → `USE_FLASH_ATTENTION was not enabled for build` | Windows 上跳过 flash，改用 cuDNN attention |
| 2 | `YuE\src\yue2\nar.py` | Windows 版 PyTorch 对 GQA 没有高效 SDPA 内核，整首歌的声学阶段实体化 `[heads, q, k]` 分数矩阵（GiB 级）直接 OOM | Windows 上默认把查询按 512 行分块（详见下方复盘） |
| 3 | `models\SheetSage2\notation_sheetsage2.py` | 节拍跟踪器偶尔漏发小节内某个节拍行（如 ID `[1,3,4]`），原版 ABC 导出器直接报 `non-consecutive beat IDs` 并放弃整首歌 | 解析节拍网格后用相邻拍时间线性插值补齐缺失行（含漏下拍），打印 `filled N missing beat row(s)` |
| 4 | `cover_webui.py`（新增，非上游） | transformers 4.45 复制本地动态模块时只复制第一层 import，模型目录里的源码补丁不生效 | 每次转录前 `ensure_sheetsage_module_cache()` 把 `models\SheetSage2\*.py` 无条件同步进 transformers 模块缓存 |

补丁 1、2 在 `YuE\` 里，已随本仓库一起提交。补丁 3 位于 `models\`（被 gitignore），
因此额外导出为 `patches\SheetSage2-beat-grid-fill.patch`；在新机器上下载完模型后，于仓库根目录执行：

```powershell
git apply patches\SheetSage2-beat-grid-fill.patch   # 校验：git apply --reverse --check 同一命令
```

补丁 4 由 `cover_webui.py` 自动完成，无需手动操作。

---

# 构建复盘：长音频（整首歌）为什么跑不通，怎么修好的

## 时间线（`.tmp\` 里的验证脚本按时间排列）

| 阶段 | 产出脚本 | 结论 |
|---|---|---|
| ① 环境/模型自检 | `check_index.py` | 模型分片与 HF 缓存完整 |
| ② 端到端短样本 | `e2e_test.py`、`quick_gen.py` | 短旋律片段能出音频，链路（转录→生成）打通 |
| ③ ABC 重建失败 | `rebuild_test.py`、`tinghai_rebuilt.abc` | 真实歌曲转录后 ABC 导出报 `non-consecutive beat IDs` → 补丁 3 |
| ④ 整首歌 OOM | `attn_probe.py` | 定位到 NAR 声学阶段的注意力分数矩阵 → 补丁 2 |
| ⑤ 补丁后整首歌 | `oomfix_validate.py` → `oomfix_song.wav` | 同一负载首次完整跑通，峰值显存 16+ GiB → 2 GiB |
| ⑥ WebUI 功能补齐 | `test_prepare_abc.py`、`cfg_validate.py` | 加入 Ins 声部/速度剥离与 `cfg_scale` 风格引导 |
| ⑦ 真实翻唱样本 | `outputs\cover_*` | 连续多首完整翻唱，无截断 |

## 症状

在 20 GB 显存上生成 5 分钟级整首歌，声学（NAR）阶段直接 CUDA OOM；
同样的代码在短旋律片段上完全正常，所以问题只在**长序列**上暴露。

## 根因

1. **注意力后端选择错误（补丁 1）**
   Windows 版 PyTorch wheel 里 `torch.ops.aten._flash_attention_forward` 的 **schema 存在但内核没编译**，
   上游的自动探测只看 schema，于是选中 flash 并在运行时抛
   `USE_FLASH_ATTENTION was not enabled for build`。
   → Windows 上跳过 flash，落到 cuDNN attention（AR 阶段实测 `attention=cudnn, execution=cuda_graph`）。

2. **GQA 分数矩阵实体化（补丁 2，关键）**
   YuE2-3B 的注意力是分组查询注意力（16 个 Q 头 / 8 个 KV 头，`head_dim=128`，`context=24576`）。
   Windows 版 PyTorch 既没有 flash 内核，也没有针对 GQA 的 memory-efficient SDPA 内核，
   SDPA 只能退到 math 后端：**整块查询一次性算出 `[heads, q, k]` 分数张量**。
   整首歌的 NAR 分块里 `q ≈ 12 800`、`k ≈ 14 000`，仅 bf16 分数就是
   `16 × 12800 × 14000 × 2 B ≈ 5.3 GiB`，softmax 的 fp32 中间量再翻倍到 ≈ 10.7 GiB，
   叠加权重与 KV 缓存后必然超过 16 GiB 的进程显存上限。

   **修复**：把查询按 512 行分块（`query_chunk_size=512`）。
   分块只改变临时存储，**不改变每个查询可见的 key 集合**，因此数值不变（`attn_probe.py` 用
   `torch.allclose` 与独立计算的前 1024 行对照验证）。
   单块分数降到 `16 × 512 × 14000 × 2 B ≈ 0.21 GiB`。
   实测同一负载（《听海》真实曲谱）峰值显存 **16+ GiB → 2 GiB**。

3. **显存预算与分块解码（配置层，非补丁）**
   - `memory_budget_gib=18`：给桌面应用留 2 GB 余量，进程内显存实际上限约 16 GiB；
   - `offload_ar=True`：AR 权重在 NAR 阶段按需卸载，NAR 结束后再恢复（`nar.py` 在取消/异常路径也会恢复，避免峰值失控）；
   - `vae_core_frames=512`（+ `halo_frames=16`）：音频 VAE 解码分块，避免整首歌一次性解码；
   - `PYTORCH_ALLOC_CONF=expandable_segments:True`：减少显存碎片（官方 OOM 提示）。

4. **转录侧的隐性失败（补丁 3）**
   不是显存问题，而是鲁棒性问题：真实音乐上节拍跟踪器偶尔漏发小节内的某个节拍行，
   原版导出器把它当作数据错误直接放弃整首歌。补丁在解析节拍网格后按相邻拍时间线性插值补齐，
   并打印 `filled N missing beat row(s)` 便于核对。

## 实测耗时分解（`outputs\cover_20261009_185602\result.json`）

| 阶段 | 耗时 | 备注 |
|---|---|---|
| AR / semantic token | 129.0 s | 8112 tokens，62.9 tok/s，cudnn + cuda graph |
| NAR / 声学 flow | 547.1 s | 占总时长约 80%，就是 OOM 的那一段 |
| VAE 解码 | 7.9 s | 512 帧分块 + 16 帧 halo |
| **端到端** | **686.0 s** | 输出音频 324.4 s（≈5.4 分钟），`truncated` 全 false |

配置：`cot=melody`、`cfg_scale=1.5`（2 个 CFG 分支）、`ode_steps=32`、`context=24576`、
`memory_budget_gib=18`、`offload_ar=true`、`vae_core_frames=512`。

结论：**瓶颈在 NAR，不在 AR**。优化显存时优先动 NAR 的查询分块与 AR 权重卸载，
而不是降低 `ode_steps`（会直接伤音质）。

## 复现验证

```powershell
# 1) GQA 分块的显存与数值一致性（脚本内 set_device 按需修改）
.\.venv\Scripts\python.exe tools\probe_gqa_attention.py
# 2) 整首歌端到端（传入真实整曲曲谱，打印峰值显存与耗时；默认用短样本）
.\.venv\Scripts\python.exe tools\validate_full_song.py outputs\cover_20261009_185602\score.abc
```

---

## 其他踩过的坑

- **bat 报 `'et' 不是内部或外部命令` / 乱码**：bat 曾以 UTF-8 + `chcp 65001` 保存，
  中文版 cmd 切代码页后字节错位（`set` 被吃掉首字母变成 `et`）。
  现已改为 ANSI(GBK) 编码保存；日后手动编辑这些 bat 请保持 ANSI(GBK)。
- **transformers 4.45 动态模块只复制第一层 import**：模型目录里的源码补丁不会进模块缓存。
  `cover_webui.py` 每次转录前无条件把 `models\SheetSage2\*.py` 同步进
  `.hfcache\modules\transformers_modules\SheetSage2\`，补丁 3 因此生效。
- **系统代理导致 Gradio 自检 502**：`cover_webui.py` 启动时清除代理变量并设置
  `NO_PROXY=127.0.0.1,localhost`。
- **`update.bat` 提示 `pyproject.toml` 有变化**：依赖需谨慎同步，torch 必须保持 cu126 CUDA 版本
  （直接从 PyPI 重装会变成 CPU 版）；建议按「环境要点」中两套环境的原始版本手动补齐变化的包。
  `update.bat` 会自动 `git stash` 保护补丁 1、2，pull 后再恢复。

## 翻唱质量相关开关

- **跟随原曲伴奏旋律（Ins 声部）**：勾选=编曲贴近原曲；不勾选=编曲自由、风格更突出（`prepare_abc` 剥离 Ins 声部）。
- **跟随原曲速度（`Q:` 标记）**：勾选=锁定原曲速度；不勾选=速度交给风格提示词。
  官方把速度/拍号当硬条件，想让风格主导就得放开这两个锚点。
- **风格引导强度 `cfg_scale`**（1.0–10.0，默认 1.2）：>1 时启用 CFG，风格更贴提示词，但过高会破音质。

## 常见问题

- **显存不足（OOM）**：按上面「显存预算与分块解码」四项检查，重点是 `nar.py` 的查询分块补丁（整首歌 OOM 的关键修复）。
- **转录失败**：确认 `ffmpeg` 在 PATH 中（已安装 N-123858）。
- **生成被截断**：日志出现 ⚠️ 时，可缩短歌词或更换 seed。
- **界面英文残留**：日志区中的 `[YuE2] ...` 进度行来自官方库的英文输出，属正常现象。

## 新机器复现步骤

```powershell
git clone https://github.com/ddddddeyrueuad/yue2-cover.git
cd yue2-cover
# 模型（约 10 GB+）
powershell -ExecutionPolicy Bypass -File setup_download.ps1
# 套用模型侧补丁（补丁 3）
git apply patches\SheetSage2-beat-grid-fill.patch
# 两套虚拟环境按「环境要点」的版本锁定创建；torch 必须装 cu126 版本
```

`YuE\` 已含补丁 1、2，无需额外操作。

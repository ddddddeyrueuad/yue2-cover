# YuE2 翻唱（Cover）工作台 / YuE2 Cover Workbench

**中文完整文档 → [README-翻唱部署说明.md](README-翻唱部署说明.md)**
**Full English docs → [README.en.md](README.en.md)**

上传源歌曲 → SheetSage2 转录出 ABC 旋律曲谱 → 填写目标风格与新歌词 → YuE2-3B（`cot=melody`）按原曲旋律演唱新歌词，输出 48 kHz 立体声翻唱。
Upload a source song → SheetSage2 transcribes it into an ABC melody score → enter a target style and new lyrics → YuE2-3B (`cot=melody`) sings the new lyrics over the original melody.

## 快速开始 / Quick start

```powershell
# 1) 下载模型（约 10 GB+）/ download the models
powershell -ExecutionPolicy Bypass -File setup_download.ps1
# 2) 套用模型侧补丁 / apply the model-side patch
git apply patches\SheetSage2-beat-grid-fill.patch
# 3) 启动 WebUI（需按文档配好两套 venv）/ start the UI (two venvs, see docs)
双击 启动翻唱WebUI.bat   →   http://127.0.0.1:7860
```

## 仓库要点 / Repository facts

| | |
|---|---|
| 上游 / Upstream | [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) @ `1647252`（2026-10-06），已含本项目的 Windows 补丁 / carries the local Windows patches |
| 本地补丁 / Local patches | 4 个（2 个在 `YuE\`，1 个在 `patches\`，1 个在 `cover_webui.py`）/ 2 in `YuE\`, 1 in `patches\`, 1 in `cover_webui.py` |
| 验证脚本 / Validation scripts | [`tools\probe_gqa_attention.py`](tools/probe_gqa_attention.py)、[`tools\validate_full_song.py`](tools/validate_full_song.py) |
| 硬件 / Hardware | 2 × RTX 3080 20 GB（官方建议 24 GB / official recommendation is 24 GB） |
| 实测 / Measured | 5.4 分钟整首歌 = 686 s（AR 129 s + NAR 547 s + VAE 8 s），峰值显存 2 GiB / a 5.4-min song in 686 s, 2 GiB peak VRAM |
| 不入库 / Not committed | `models\`、`outputs\`、`.venv*\`、`.hfcache\`（见 `.gitignore`） |

> 关键修复：**Windows 版 PyTorch 缺少 flash / GQA memory-efficient SDPA 内核**，整首歌的 NAR 阶段会实体化
> `[heads, q, k]` 分数矩阵而 OOM；把查询按 512 行分块后峰值显存从 16+ GiB 降到 2 GiB，数值不变。
> 详见文档「构建复盘」/ see the "Build retrospective" section in the docs.

# YuE2 Cover Workbench — Deployment Notes & Build Retrospective

This project extracts **only the cover feature** from
[multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) (YuE2) and wraps it in a
fully localized WebUI. Chinese original: [README-翻唱部署说明.md](README-翻唱部署说明.md).

## Pipeline

1. **Upload the source song** (wav / mp3 / flac, any common format)
2. **Transcribe the melody**: SheetSage2 converts the source song into an ABC melody score (editable by hand)
3. **Enter the target style + new lyrics**
4. **Generate the cover**: YuE2-3B (`cot=melody`) sings the new lyrics over the original melody, 48 kHz stereo

## Starting the app

Double-click `启动翻唱WebUI.bat` (the launcher name is intentionally Chinese), or run:

```powershell
$env:HF_HOME = "E:\yue2 cover\.hfcache"
.\.venv\Scripts\python.exe cover_webui.py
```

Then open **http://127.0.0.1:7860**. Stop the service with `stop.bat` (kills the related processes and frees
VRAM/RAM), or just close the window. Update the code with `update.bat` (pulls the latest YuE2 code and
automatically protects the local patches; model files are untouched).

- The first click on "Generate cover" loads the model (about 1–2 minutes); you can also press "Preload model".
- A full song (3–5 minutes of audio) takes roughly 5–15 minutes; the log pane streams progress live.

## Layout

| Path | Description |
|---|---|
| `cover_webui.py` | Localized WebUI main program (cover feature only) |
| `启动翻唱WebUI.bat` | One-click launcher (saved as ANSI/GBK — do not re-save as UTF-8) |
| `stop.bat` | Stops the service and frees VRAM/RAM (ANSI/GBK) |
| `update.bat` | Updates the YuE2 checkout (git pull, protects/restores local patches; ANSI/GBK) |
| `setup_download.ps1` | Re-downloads the models if `models\` is incomplete |
| `YuE\` | Official YuE2 checkout (inference code in `YuE\src\yue2`), already carries the Windows patches |
| `patches\` | Model-side patch that lives in `models\` (gitignored), re-appliable |
| `tools\` | VRAM / numerics validation scripts — the measured numbers below come from them |
| `models\YuE2-3B` | YuE2 generation model (7.3 GB) |
| `models\YuE2-Vae` | YuE2 audio VAE |
| `models\SheetSage2` | Melody transcription model |
| `.hfcache\` | HuggingFace cache (includes MERT-v2-FullSong) |
| `.venv\` | YuE2 inference env (Python 3.12, torch 2.10.0+cu126) |
| `.venv-sheetsage2\` | SheetSage2 transcription env (Python 3.11, torch 2.8.0+cu126) |
| `outputs\` | Transcription and generation outputs |

## Environment (already configured)

- Two separate virtualenvs (their dependencies conflict; they cannot be merged):
  - `.venv`: Python 3.12.12 + torch 2.10.0+cu126 + transformers 4.57.6 + gradio 6.17.3 + accelerate 1.13.0 + numpy 2.2.6
  - `.venv-sheetsage2`: Python 3.11.14 + torch 2.8.0+cu126 + transformers 4.45.2
- The WebUI transcribes through a subprocess using `.venv-sheetsage2` and generates in the main process with `.venv`.
- The `yue2` package is not installed into site-packages; `cover_webui.py` puts `YuE\src` on `sys.path`.
- `HF_HOME=E:\yue2 cover\.hfcache` is mandatory (the launcher sets it).
- Upstream code is pinned to YuE commit `1647252` (2026-10-06).
- Hardware: 2 × RTX 3080 20 GB (official recommendation is 24 GB), ffmpeg N-123858.

## Local patches

| # | File | Problem | Fix |
|---|---|---|---|
| 1 | `YuE\src\yue2\cuda_graph.py` | Windows PyTorch builds ship no compiled FlashAttention kernel; the original auto-detection picked flash → `USE_FLASH_ATTENTION was not enabled for build` | Skip flash on Windows, use cuDNN attention |
| 2 | `YuE\src\yue2\nar.py` | Windows PyTorch has no efficient SDPA kernel for grouped-query attention; the acoustic stage of a full song materializes the `[heads, q, k]` score tensor (GiB-scale) and OOMs | Tile the queries at 512 rows on Windows (see the retrospective below) |
| 3 | `models\SheetSage2\notation_sheetsage2.py` | The beat tracker occasionally drops a beat row inside a measure (e.g. IDs `[1,3,4]`); the stock ABC exporter aborts the whole song with `non-consecutive beat IDs` | Re-create missing rows by linear time interpolation between neighbouring beats (including a dropped downbeat), logging `filled N missing beat row(s)` |
| 4 | `cover_webui.py` (new, not upstream) | transformers 4.45 copies only first-level imports of local dynamic modules, so model-directory source patches never reach the module cache | `ensure_sheetsage_module_cache()` unconditionally syncs `models\SheetSage2\*.py` into the transformers module cache before every transcription |

Patches 1 and 2 live inside `YuE\` and are committed to this repository.
Patch 3 lives in `models\` (gitignored), so it is additionally exported as
`patches\SheetSage2-beat-grid-fill.patch`. After downloading the models on a fresh machine, run from the repo root:

```powershell
git apply patches\SheetSage2-beat-grid-fill.patch   # verify: git apply --reverse --check <same file>
```

Patch 4 is applied automatically by `cover_webui.py`; nothing to do manually.

---

# Build retrospective: why full-length audio failed, and how it was fixed

## Timeline (validation scripts in `.tmp\`, in chronological order)

| Stage | Script | Outcome |
|---|---|---|
| ① Env / model self-check | `check_index.py` | Model shards and HF cache complete |
| ② Short end-to-end sample | `e2e_test.py`, `quick_gen.py` | Short melody fragments produce audio; transcription → generation link works |
| ③ ABC rebuild failure | `rebuild_test.py`, `tinghai_rebuilt.abc` | Real songs abort with `non-consecutive beat IDs` → patch 3 |
| ④ Full-song OOM | `attn_probe.py` | Root cause isolated to the NAR attention score matrix → patch 2 |
| ⑤ Full song after the patch | `oomfix_validate.py` → `oomfix_song.wav` | First complete run under the same load; peak VRAM 16+ GiB → 2 GiB |
| ⑥ WebUI features | `test_prepare_abc.py`, `cfg_validate.py` | Added Ins-voice/tempo stripping and `cfg_scale` style guidance |
| ⑦ Real cover samples | `outputs\cover_*` | Several complete covers, no truncation |

## Symptom

Generating a 5-minute-scale full song on 20 GB of VRAM dies with CUDA OOM inside the acoustic (NAR) stage.
The same code works fine on short melody fragments, so the failure only surfaces on **long sequences**.

## Root causes

1. **Wrong attention backend (patch 1)**
   In Windows PyTorch wheels the *schema* of `torch.ops.aten._flash_attention_forward` exists but the *kernel is
   not compiled*. The upstream auto-detection only inspects the schema, so it selects flash and then fails at
   runtime with `USE_FLASH_ATTENTION was not enabled for build`.
   → Skip flash on Windows and fall back to cuDNN attention (the AR stage reports `attention=cudnn, execution=cuda_graph`).

2. **Materialized GQA score matrix (patch 2, the critical one)**
   YuE2-3B uses grouped-query attention (16 query heads / 8 KV heads, `head_dim=128`, `context=24576`).
   Windows PyTorch ships neither a flash kernel nor a memory-efficient SDPA kernel for GQA, so SDPA degrades to
   the math backend: **one query block computes the whole `[heads, q, k]` score tensor**.
   In a full-song NAR chunk `q ≈ 12,800` and `k ≈ 14,000`, so the bf16 scores alone are
   `16 × 12800 × 14000 × 2 B ≈ 5.3 GiB`, and the fp32 softmax intermediates double that to ≈ 10.7 GiB —
   on top of weights and the KV cache this always exceeds the 16 GiB per-process VRAM budget.

   **Fix**: tile the queries at 512 rows (`query_chunk_size=512`).
   Tiling only changes transient storage; it **never changes the key set visible to each query**, so the numbers are
   identical (`attn_probe.py` checks the first 1024 rows against an independent computation with `torch.allclose`).
   Per-tile scores drop to `16 × 512 × 14000 × 2 B ≈ 0.21 GiB`.
   Measured on the same load (a real score transcribed from a commercial song): peak VRAM **16+ GiB → 2 GiB**.

3. **VRAM budget and tiled decoding (configuration, not a patch)**
   - `memory_budget_gib=18`: leaves 2 GB of headroom for the desktop session; the effective in-process cap is ≈ 16 GiB;
   - `offload_ar=True`: AR weights are offloaded during the NAR stage and restored afterwards (`nar.py` also restores
     them on the cancel/failure paths, so the peak stays bounded);
   - `vae_core_frames=512` (with `halo_frames=16`): tiled audio VAE decoding instead of decoding the whole song at once;
   - `PYTORCH_ALLOC_CONF=expandable_segments:True`: reduces VRAM fragmentation (the official OOM hint).

4. **A silent failure on the transcription side (patch 3)**
   Not a VRAM issue but a robustness one: on real music the beat tracker sometimes omits a beat row inside a
   measure, and the stock exporter treats that as corrupt data and abandons the entire song. The patch repairs the
   grid by interpolating the missing rows and logs `filled N missing beat row(s)` for verification.

## Measured stage breakdown (`outputs\cover_20261009_185602\result.json`)

| Stage | Time | Notes |
|---|---|---|
| AR / semantic tokens | 129.0 s | 8112 tokens, 62.9 tok/s, cudnn + cuda graph |
| NAR / acoustic flow | 547.1 s | ≈ 80% of the total — exactly the stage that used to OOM |
| VAE decode | 7.9 s | 512-frame tiles + 16-frame halo |
| **End to end** | **686.0 s** | 324.4 s (≈5.4 min) of audio, `truncated` all false |

Configuration: `cot=melody`, `cfg_scale=1.5` (2 CFG branches), `ode_steps=32`, `context=24576`,
`memory_budget_gib=18`, `offload_ar=true`, `vae_core_frames=512`.

Takeaway: **the bottleneck is NAR, not AR.** When fighting for VRAM, tune NAR query tiling and AR weight offloading
first — do not lower `ode_steps`, which directly costs audio quality.

## Reproducing the measurements

```powershell
# 1) GQA tiling: VRAM peak + numerical equivalence (adjust set_device inside the script)
.\.venv\Scripts\python.exe tools\probe_gqa_attention.py
# 2) Full end-to-end song (pass a real full-length score; prints peak VRAM and wall time, defaults to the short sample)
.\.venv\Scripts\python.exe tools\validate_full_song.py outputs\cover_20261009_185602\score.abc
```

---

## Other pitfalls hit along the way

- **`.bat` fails with `'et' is not recognized` / mojibake**: the batch files were once saved as UTF-8 with
  `chcp 65001`; after the Chinese edition of `cmd` switches code pages the bytes are parsed out of alignment
  (`set` loses its first letter and becomes `et`). They are now saved as ANSI(GBK) — keep that encoding if you edit them.
- **transformers 4.45 copies only first-level imports of dynamic modules**: source patches sitting in the model
  directory never reach the module cache. Before every transcription `cover_webui.py` copies
  `models\SheetSage2\*.py` into `.hfcache\modules\transformers_modules\SheetSage2\`, which is what makes patch 3 effective.
- **System proxy causes a 502 in Gradio's self-check**: `cover_webui.py` clears the proxy variables at startup and
  sets `NO_PROXY=127.0.0.1,localhost`.
- **`update.bat` reports that `pyproject.toml` changed**: dependencies must be synced carefully — torch must stay on
  the cu126 CUDA build (reinstalling from PyPI silently gives you a CPU build). Reconcile the changed packages
  against the pinned versions listed under "Environment". `update.bat` automatically `git stash`es patches 1–2 before
  pulling and restores them afterwards.

## Cover-quality switches

- **Follow the original accompaniment (Ins voice)**: checked = arrangement close to the original; unchecked = freer arrangement, stronger style shift (`prepare_abc` strips the Ins voice).
- **Follow the original tempo (`Q:` marker)**: checked = tempo locked to the original; unchecked = tempo driven by the style prompt.
  Upstream treats tempo and time signature as hard conditions, so releasing these anchors is how you let the style dominate.
- **Style guidance `cfg_scale`** (1.0–10.0, default 1.2): above 1.0 CFG is enabled; higher follows the prompt more strictly but degrades quality past a point.

## FAQ

- **CUDA OOM**: check the four items under "VRAM budget and tiled decoding"; the query-tiling patch in `nar.py` is the key fix for full-song OOM.
- **Transcription fails**: make sure `ffmpeg` is on PATH (N-123858 installed here).
- **Generation gets truncated**: when the log shows ⚠️, shorten the lyrics or change the seed.
- **English left in the UI**: the `[YuE2] ...` progress lines come from the upstream library and are expected.

## Setting up a fresh machine

```powershell
git clone https://github.com/ddddddeyrueuad/yue2-cover.git
cd yue2-cover
# models (10 GB+)
powershell -ExecutionPolicy Bypass -File setup_download.ps1
# apply the model-side patch (patch 3)
git apply patches\SheetSage2-beat-grid-fill.patch
# create the two virtualenvs with the pinned versions above; torch must be the cu126 build
```

`YuE\` already contains patches 1 and 2 — nothing extra to do there.

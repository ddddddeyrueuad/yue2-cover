# -*- coding: utf-8 -*-
"""OOM 修复终极验证：用真实整曲曲谱完整生成（与当初 OOM 时同负载）

用法：python tools/validate_full_song.py [曲谱.abc]
默认使用 YuE/examples/melody.abc（短样本）；要复现整首歌的显存压力，
请传入 outputs/cover_*/score.abc 这类真实整曲曲谱。
"""
import os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hfcache"))
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(ROOT / "YuE" / "src"))
import torch
from yue2 import YuE2Pipeline

abc_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "YuE" / "examples" / "melody.abc"
device = os.environ.get("DEVICE", "cuda:1")  # 用空闲的那张卡；按需 set DEVICE=cuda:0
abc = abc_path.read_text(encoding="utf-8")
print(f"曲谱: {abc_path}   设备: {device}")
lyrics = ("[verse]\nMoonlight on the water tonight\nSinging softly, holding you tight\n"
          "[chorus]\nHold on, hold on to the light\nWe will make it through the night\n")
t0 = time.time()
with YuE2Pipeline.from_pretrained(str(ROOT / "models/YuE2-3B"), vae=str(ROOT / "models/YuE2-Vae"),
                                  device=device, memory_budget_gib=18,
                                  vae_core_frames=512, offload_ar=True) as pipe:
    song = pipe(style="Mandopop ballad, emotional female vocal, piano and strings",
                lyrics=lyrics, abc=abc, cot="melody", seed=831001)
    out = ROOT / "outputs" / "validate_full_song.wav"
    song.save(out)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"OK: {len(song.audio)/song.sample_rate:.1f}s audio in {time.time()-t0:.0f}s; peak VRAM {peak:.2f} GiB")
    print("saved", out)

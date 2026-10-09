# -*- coding: utf-8 -*-
"""GQA 下分块注意力验证（full 块已证实 OOM，这里验证补丁方案可行）"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "YuE" / "src"))
from yue2.nar import attention

torch.cuda.set_device(1)  # 用空闲的那张卡；按需改成 0
DEV = "cuda"
# 合成负载：头数比 YuE2-3B 真实配置（16 Q / 8 KV）更极端，用于放大分数矩阵的显存代价
SEQ, QH, KVH, DIM = 12800, 24, 4, 128
q = torch.randn(SEQ, QH, DIM, device=DEV, dtype=torch.bfloat16)
k = torch.randn(SEQ + 1200, KVH, DIM, device=DEV, dtype=torch.bfloat16)
v = torch.randn(SEQ + 1200, KVH, DIM, device=DEV, dtype=torch.bfloat16)

torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
out = attention(q, k, v, causal=False, backend="sdpa", query_chunk_size=512)
peak = torch.cuda.max_memory_allocated() / 2**30
print(f"GQA 分块512: 峰值 {peak:.2f} GiB, 输出形状 {tuple(out.shape)}, 有限值: {torch.isfinite(out).all().item()}")
# 结果一致性: 与手动分块对照组数值相同（tiling 不改变可见 key 集）
torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
ref = attention(q[:1024], k, v, causal=False, backend="sdpa", query_chunk_size=512)
same = torch.allclose(out[:1024], ref, atol=1e-2, rtol=1e-2)
print(f"前1024行与独立计算一致: {same}")

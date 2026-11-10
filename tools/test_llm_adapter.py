# -*- coding: utf-8 -*-
"""LLM 改词适配器的端到端测试：用本地 mock OpenAI 兼容服务验证完整链路。

覆盖：请求构造 → urllib 调用 → 代码块剥离 → 结构校验 → 成功/失败分支。
不需要真实 API key，也不产生费用。
"""
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cover_webui as cw  # noqa: E402

real = (ROOT / "outputs/cover_20261009_185602/score.abc").read_text(encoding="utf-8")
CN = "\n".join(["写信告诉我今天 海是什么颜色", "夜夜陪着你的海 心情又如何",
                "灰色是不想说 蓝色是忧郁", "而漂泊的你 狂浪的心 停在哪里"])

GRID = cw.score_grid(real)
CAPTURED = {}
CALLS = {}


def build_valid_lyrics():
    """按网格造一份合格歌词：只写歌唱段（乐器段省略——真实 LLM 常见行为）。"""
    out = []
    for g in GRID:
        if not g["targets"]:
            continue
        out.append(f"[{g['tag']}]")
        for t in g["targets"]:
            out.append("词" * t)
        out.append("")
    return "\n".join(out)


def build_with_empty_inst():
    """保留空乐器段标记的变体，校验器也应接受。"""
    out = []
    for g in GRID:
        out.append(f"[{g['tag']}]")
        for t in g["targets"]:
            out.append("词" * t)
        out.append("")
    return "\n".join(out)


def build_numbered():
    """段落带序号（[Verse 1]/[Verse 2]）的变体，校验器也应接受。"""
    txt, counts = build_valid_lyrics(), {}

    def repl(m):
        tag = m.group(1)
        counts[tag] = counts.get(tag, 0) + 1
        return f"[{tag} {counts[tag]}]" if tag == "Verse" else f"[{tag}]"

    return re.sub(r"\[(Verse|Chorus)\]", repl, txt)


def build_bad_syllables():
    """结构对、但某一行音节超标（复现用户遇到的 '[Verse] 第 2 行音节 15 偏离目标 10'）。"""
    lines = build_valid_lyrics().split("\n")
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("词"))
    lines[idx] = lines[idx] + "多" * 5
    return "\n".join(lines)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n).decode("utf-8"))
        CAPTURED["path"] = self.path
        CAPTURED["auth"] = self.headers.get("Authorization")
        CAPTURED["body"] = body
        mode = self.path.split("/")[1] if "/" in self.path else ""
        CALLS[mode] = CALLS.get(mode, 0) + 1
        if mode == "bad":                       # 结构不合格
            content = "[Verse]\n短\n"
        elif mode == "fence":                   # 带代码块围栏，应被剥离
            content = "```\n" + build_valid_lyrics() + "\n```"
        elif mode == "fix":                     # 第一次不合格，重试后合格
            content = build_valid_lyrics() if CALLS[mode] > 1 else build_bad_syllables()
        elif mode == "always_bad":              # 始终超标：应最终回落
            content = build_bad_syllables()
        elif mode == "boom":                    # 服务端错误
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"error":"boom"}')
            return
        else:
            content = build_valid_lyrics()
        payload = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


srv = HTTPServer(("127.0.0.1", 0), Handler)
port = srv.server_address[1]
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{port}"

print("=== 1. 正常返回（应通过校验） ===")
text, note = cw.llm_adapt_lyrics(real, CN, base + "/v1", "sk-test", "mock-model")
print("结果:", note)
assert text, note
assert cw.validate_structured(text, GRID)[0]
assert CAPTURED["path"] == "/v1/chat/completions", CAPTURED["path"]
assert CAPTURED["auth"] == "Bearer sk-test"
assert CAPTURED["body"]["model"] == "mock-model"
# 提示词里应带上网格与用户歌词
msgs = json.dumps(CAPTURED["body"]["messages"], ensure_ascii=False)
assert "旋律网格" in msgs and "写信告诉我今天" in msgs, "提示词应包含网格与用户歌词"

print("\n=== 2. 代码块围栏应被剥离 ===")
text2, note2 = cw.llm_adapt_lyrics(real, CN, base + "/fence/v1", "sk-test", "m")
print("结果:", note2)
assert text2 and not text2.startswith("```"), text2[:40]

print("\n=== 3. 结构不合格应拒绝 ===")
text3, note3 = cw.llm_adapt_lyrics(real, CN, base + "/bad/v1", "sk-test", "m")
print("结果:", (text3, note3))
assert text3 is None and "校验未通过" in note3

print("\n=== 4. 服务端 500 应安全失败 ===")
text4, note4 = cw.llm_adapt_lyrics(real, CN, base + "/boom/v1", "sk-test", "m")
print("结果:", (text4, note4))
assert text4 is None

print("\n=== 5. 空 key 不发请求 ===")
text5, note5 = cw.llm_adapt_lyrics(real, CN, base + "/v1", "  ", "m")
assert text5 is None and "key" in note5.lower()

print("\n=== 6. 校验器容忍无害变体（真实 LLM 常见输出） ===")
ok_a, why_a = cw.validate_structured(build_valid_lyrics(), GRID)      # 省略空乐器段
print("省略乐器段:", ok_a, why_a)
assert ok_a, why_a
ok_b, why_b = cw.validate_structured(build_with_empty_inst(), GRID)   # 保留空乐器段
print("保留空乐器段:", ok_b, why_b)
assert ok_b, why_b
ok_c, why_c = cw.validate_structured(build_numbered(), GRID)          # 段落带序号
print("段落带序号:", ok_c, why_c)
assert ok_c, why_c

print("\n=== 7. 真正的结构错误仍应拒绝 ===")
sing = [g for g in GRID if g["targets"]]
missing = build_valid_lyrics().replace("\n".join("词" * t for t in sing[0]["targets"]),
                                       "\n".join("词" * t for t in sing[0]["targets"][:-1]), 1)
ok_d, why_d = cw.validate_structured(missing, GRID)
print("缺行:", ok_d, why_d)
assert not ok_d
swapped = "\n".join(f"[{g['tag']}]\n" + "\n".join("词" * t for t in g["targets"]) + "\n"
                    for g in [sing[1]] + [sing[0]] + sing[2:])
ok_e, why_e = cw.validate_structured(swapped, GRID)
print("段落顺序错:", ok_e, why_e)
assert not ok_e

print("\n=== 8. 音节超标应带错误重试一次并成功 ===")
t8, n8 = cw.llm_adapt_lyrics(real, CN, base + "/fix/v1", "sk-test", "m")
print("结果:", n8, "| 调用次数:", CALLS.get("fix"))
assert t8, n8
assert CALLS.get("fix") == 2, f"应重试一次，实际调用 {CALLS.get('fix')} 次"
assert cw.validate_structured(t8, GRID)[0]

print("\n=== 9. 始终不合格应最终回落 ===")
t9, n9 = cw.llm_adapt_lyrics(real, CN, base + "/always_bad/v1", "sk-test", "m")
print("结果:", (t9, n9), "| 调用次数:", CALLS.get("always_bad"))
assert t9 is None and "校验未通过" in n9
assert CALLS.get("always_bad") == 2, "默认只重试一次"

srv.shutdown()
print("\nALL OK")

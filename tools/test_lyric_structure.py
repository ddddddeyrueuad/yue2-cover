# -*- coding: utf-8 -*-
"""结构化歌词对齐 + LLM 改词适配器 的回归测试。

不变量：
1. 段落标记与顺序完全来自 ABC 的 % 注释；
2. 每个歌唱段的歌词行数 == 该段曲谱音乐行数；
3. 每行音节数围绕目标（≈0.9×onset）浮动，不再强制逐行相等
   （官方参考词实测 337 音节 / 388 onset ≈ 0.87，允许一音节跨多音）；
4. 乐器段无歌词；整句优先、不拆词；
5. align_lyrics_to_abc 是纯函数，不改动 ABC；
6. LLM 适配器：无 key 时安全跳过；结构校验能识别合格/不合格输出。
"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cover_webui as cw  # noqa: E402

real = (ROOT / "outputs/cover_20261009_185602/score.abc").read_text(encoding="utf-8")
score = (ROOT / "YuE/examples/score.abc").read_text(encoding="utf-8")
CN = "\n".join(["写信告诉我今天 海是什么颜色", "夜夜陪着你的海 心情又如何",
                "灰色是不想说 蓝色是忧郁", "而漂泊的你 狂浪的心 停在哪里"])


def lyric_lines_of(structured, tag, occurrence=0):
    hits = re.findall(rf"\[{tag}\]\n(.*?)(?=\n\[|\Z)", structured, re.S)
    if len(hits) <= occurrence:
        return []
    return [ln for ln in hits[occurrence].splitlines() if ln.strip()]


def check(name, abc, lyrics, expect_tags):
    out, report = cw.align_lyrics_to_abc(abc, lyrics)
    print(f"\n[{name}] {report}")
    tags = re.findall(r"\[([^\]]+)\]", out)
    assert tags == expect_tags, f"段落标记不符：期望 {expect_tags}，得到 {tags}"
    occ = {}
    for label, lines in cw.abc_sections(abc):
        tag = cw._section_tag(label)
        if tag in cw._INST_TAGS:
            continue
        k = occ.get(tag, 0)
        occ[tag] = k + 1
        got = lyric_lines_of(out, tag, k)
        assert len(got) == len(lines), f"[{tag}]#{k} 歌词 {len(got)} 行 != 曲谱 {len(lines)} 行"
        targets = [max(1, round(o * 0.9)) for o, _ in lines]
        for j, (ln, t) in enumerate(zip(got, targets)):
            syl = cw.count_lyric_syllables(ln)
            if t >= 5:
                assert abs(syl - t) <= max(3, t * 0.5), \
                    f"[{tag}]#{k} 第 {j} 行音节 {syl} 偏离目标 {t} 过大"
        # 段内总量：允许 ±20%
        tot_syl = sum(cw.count_lyric_syllables(x) for x in got)
        tot_t = sum(targets)
        if tot_t >= 10:
            assert abs(tot_syl - tot_t) <= tot_t * 0.2, \
                f"[{tag}]#{k} 段内音节 {tot_syl} 偏离目标 {tot_t}"
    # 乐器段无歌词
    for tag in ("Intro", "Outro", "Inst"):
        for m in re.finditer(rf"\[{tag}\]", out):
            rest = out[m.end():].strip()
            assert rest.startswith("[") or rest == "", f"[{tag}] 后不应有歌词"
    return out


print("=== 英文音节估算 ===")
for w, want in [("Moonlight", 2), ("tonight", 2), ("make", 1), ("little", 2),
                ("water", 2), ("singing", 2), ("through", 1), ("beautiful", 3)]:
    got = cw._en_syllables(w)
    print(f"  {w:10s} → {got} (期望 {want})")
    assert got == want, f"{w} 音节估算 {got} != {want}"

print("\n=== 曲谱段落/行解析 ===")
for label, lines in cw.abc_sections(real):
    print(f"  {label:10s} 行数={len(lines):3d} onset={sum(o for o, _ in lines):4d}")

out = check("真实整曲谱 + 纯中文歌词", real, CN,
            ["Intro", "Verse", "Chorus", "Inst", "Verse", "Chorus", "Outro"])
print("\n--- 歌词全文（前 22 行）---")
print("\n".join(out.splitlines()[:22]))

# 整句优先：当曲谱行长与歌词行长接近时，必须整句保留
user_lines = cw._split_lyric_lines(CN)
verse0 = lyric_lines_of(out, "Verse", 0)
norm = lambda s: re.sub(r"\s+", "", s)
assert [norm(x) for x in verse0] == [norm(user_lines[i % 4]) for i in range(len(verse0))], \
    f"长度匹配的段落应整句保留，得到 {verse0}"

# 遗留标记剥离
legacy = "[Verse 1]\n" + CN
out2 = check("含遗留段落标记（应剥离重建）", real, legacy,
             ["Intro", "Verse", "Chorus", "Inst", "Verse", "Chorus", "Outro"])
assert "[Verse 1]" not in out2

# 英文歌词
en = "Moonlight on the water tonight\nSinging softly holding you tight"
check("真实整曲谱 + 英文歌词", real, en,
      ["Intro", "Verse", "Chorus", "Inst", "Verse", "Chorus", "Outro"])

# 官方两段谱
check("官方 score.abc（两段）", score, "C E G A G E D C", ["Verse", "Chorus"])

# align_lyrics_to_abc 是纯函数
before = real
cw.align_lyrics_to_abc(real, CN)
assert real == before, "align_lyrics_to_abc 不应修改输入 ABC"

# 边界
assert cw.align_lyrics_to_abc(real, "")[0] == ""
assert "曲谱没有" in cw.align_lyrics_to_abc("", CN)[1]

print("\n=== LLM 适配器 ===")
grid = cw.score_grid(real)
print("grid:", [(g["tag"], len(g["targets"])) for g in grid])
assert [g["tag"] for g in grid] == ["Intro", "Verse", "Chorus", "Inst", "Verse", "Chorus", "Outro"]
assert grid[0]["targets"] == [] and grid[3]["targets"] == []

# 无 key → 安全跳过
res, why = cw.llm_adapt_lyrics(real, CN, "", "", "")
assert res is None and "key" in why.lower(), (res, why)
print("无 key 时：", why)

# 结构校验：合格输出
good = ("[Intro]\n\n[Verse]\n" + "\n".join(["一" * t for t in grid[1]["targets"]]) +
        "\n\n[Chorus]\n" + "\n".join(["二" * t for t in grid[2]["targets"]]) +
        "\n\n[Inst]\n\n[Verse]\n" + "\n".join(["三" * t for t in grid[4]["targets"]]) +
        "\n\n[Chorus]\n" + "\n".join(["四" * t for t in grid[5]["targets"]]) + "\n\n[Outro]\n")
ok, why = cw.validate_structured(good, grid)
print("合格输出校验：", ok, why)
assert ok, why

# 结构校验：行数错误应被拒绝
bad = good.replace("一" * grid[1]["targets"][0] + "\n", "", 1)   # 少一行
ok2, why2 = cw.validate_structured(bad, grid)
print("缺行输出校验：", ok2, why2)
assert not ok2

# 结构校验：音节严重偏离应被拒绝
bad2 = good.replace("一" * grid[1]["targets"][0], "短", 1)
ok3, why3 = cw.validate_structured(bad2, grid)
print("音节偏离校验：", ok3, why3)
assert not ok3

# 解析：代码块围栏应被剥离
assert cw.parse_structured_lyrics("```\n[Verse]\n你好\n```") == [("Verse", ["你好"])]

print("\nALL OK")

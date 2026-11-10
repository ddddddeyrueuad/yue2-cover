# -*- coding: utf-8 -*-
"""结构化歌词对齐（align_lyrics_to_abc）回归测试。"""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cover_webui as cw  # noqa: E402

real = (ROOT / "outputs/cover_20261009_185602/score.abc").read_text(encoding="utf-8")
score = (ROOT / "YuE/examples/score.abc").read_text(encoding="utf-8")


def check(name, abc, lyrics, expect_tags, expect_inst_empty=True):
    out, report = cw.align_lyrics_to_abc(abc, lyrics)
    print(f"\n[{name}]\n{report}")
    print("--- 结构化歌词（前 16 行）---")
    print("\n".join(out.splitlines()[:16]))
    tags = re.findall(r"\[([^\]]+)\]", out)
    print("tags =", tags)
    assert tags == expect_tags, f"段落标记不符：期望 {expect_tags}，得到 {tags}"
    if expect_inst_empty:
        for tag in ("Intro", "Outro", "Inst"):
            # 乐器段落标记后必须直接跟下一个段落标记或 EOF（即无歌词）
            for m in re.finditer(rf"\[{tag}\]", out):
                rest = out[m.end():].strip()
                assert rest.startswith("[") or rest == "", f"[{tag}] 后不应有歌词：{rest[:40]!r}"
    return out


print("=== 曲谱段落解析 ===")
secs = cw.abc_sections(real)
for label, n, bars in secs:
    print(f"  {label:12s} onsets={n:4d} bars={len(bars)}")
assert [s[1] for s in secs] == [2, 110, 104, 7, 54, 102, 6], "段落 onset 数不符"

# 1) 纯中文歌词（不带任何段落标记）
cn = "\n".join(["写信告诉我今天 海是什么颜色", "夜夜陪着你的海 心情又如何",
                "灰色是不想说 蓝色是忧郁", "而漂泊的你 狂浪的心 停在哪里"])
out = check("真实整曲谱 + 纯中文歌词", real, cn,
            ["Intro", "Verse", "Chorus", "Inst", "Verse", "Chorus", "Outro"])
# 歌词被循环补齐：每个歌唱段落都必须有歌词
for tag in ("Verse", "Chorus"):
    for m in re.finditer(rf"\[{tag}\]", out):
        rest = out[m.end():].strip()
        assert rest and not rest.startswith("["), f"[{tag}] 段落歌词为空"

# 2) 带遗留 [Verse] 标记也应被剥离并重新按曲谱生成
legacy = "[Verse 1]\n" + cn + "\n\n[Chorus]\n听 海哭的声音\n这片海未免也太多情"
out2 = check("含遗留段落标记的歌词（应被剥离重建）", real, legacy,
             ["Intro", "Verse", "Chorus", "Inst", "Verse", "Chorus", "Outro"])
assert "[Verse 1]" not in out2 and "[Chorus]" in out2

# 3) 英文歌词（按词计音节）
en = "Moonlight on the water tonight\nSinging softly holding you tight\nHold on to the light\nWe will make it through the night"
out3 = check("真实整曲谱 + 英文歌词", real, en,
             ["Intro", "Verse", "Chorus", "Inst", "Verse", "Chorus", "Outro"])
assert "Moonlight" in out3

# 4) 官方 score.abc 含 % verse / % chorus 两段
out4, rep4 = cw.align_lyrics_to_abc(score, "C E G A G E D C")
print("\n[官方 score.abc 两段]", rep4)
assert re.findall(r"\[([^\]]+)\]", out4) == ["Verse", "Chorus"]
assert "C E G A" in out4.replace("  ", " "), "歌词内容应来自用户输入"
# 每段歌词音节数 == 该段音符数
for tag, want in (("Verse", 28), ("Chorus", 28)):
    m = re.search(rf"\[{tag}\]\n(.*?)(?=\n\[|\Z)", out4, re.S)
    got = sum(len(cw._syllable_units(ln)) for ln in m.group(1).splitlines() if ln.strip())
    assert got == want, f"[{tag}] 歌词音节 {got} != 音符 {want}"

# 5) 空歌词 / 空谱的边界
assert cw.align_lyrics_to_abc(real, "")[0] == ""
assert "曲谱没有" in cw.align_lyrics_to_abc("", cn)[1]

# 6) 组合流程：auto_align_abc 后再 align_lyrics_to_abc（生成路径）
user = "啊" * 120
abc2, r1 = cw.auto_align_abc(real, user)
lyr2, r2 = cw.align_lyrics_to_abc(abc2, user)
print("\n[生成路径] ", r1, "|", r2[:60])
assert cw.count_vocal_notes(abc2) == 120, "音符对齐未生效"
song_secs = sum(n for label, n, _ in cw.abc_sections(abc2)
                if cw._section_tag(label) not in cw._INST_TAGS)
print("对齐后全谱音符 =", cw.count_vocal_notes(abc2), "歌唱段音符 =", song_secs)
assert cw.count_vocal_notes(abc2) == 120, "音符对齐未生效"
assert song_secs <= 120, "歌唱段音符不应超过全谱"
# 结构化歌词的音节数（不含标记）应等于歌唱段音符数
total = sum(len(cw._syllable_units(ln)) for ln in lyr2.splitlines()
            if ln.strip() and not cw._STRUCT_TAG_RE.match(ln))
print("结构化歌词音节数 =", total, "歌唱段音符数 =", song_secs)
assert total == song_secs, "结构化歌词音节数应与歌唱段音符一一对应"
print("\nALL OK")

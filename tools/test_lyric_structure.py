# -*- coding: utf-8 -*-
"""结构化歌词对齐（align_lyrics_to_abc）回归测试。

核心不变量：
1. 段落标记与顺序完全来自 ABC 的 % 注释；
2. 歌词行数 == 曲谱歌唱段音乐行数（官方原始歌词与曲谱行是同粒度对应）；
3. 每行歌词音节数 == 该曲谱行的 onset 数；
4. 乐器段无歌词；
5. 不改动原曲音符（align_lyrics_to_abc 是纯函数，不碰 ABC）。
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
    """取某个 [tag] 段落下的歌词行（第 occurrence 次出现）。"""
    hits = re.findall(rf"\[{tag}\]\n(.*?)(?=\n\[|\Z)", structured, re.S)
    if len(hits) <= occurrence:
        return []
    return [ln for ln in hits[occurrence].splitlines() if ln.strip()]


def check(name, abc, lyrics, expect_tags, expect_inst_empty=True):
    out, report = cw.align_lyrics_to_abc(abc, lyrics)
    print(f"\n[{name}] {report}")
    tags = re.findall(r"\[([^\]]+)\]", out)
    assert tags == expect_tags, f"段落标记不符：期望 {expect_tags}，得到 {tags}"
    # 2) 行数与曲谱行数一一对应（按段落出现顺序匹配）
    occ = {}
    for label, lines in cw.abc_sections(abc):
        tag = cw._section_tag(label)
        if tag in cw._INST_TAGS:
            continue
        k = occ.get(tag, 0)
        occ[tag] = k + 1
        got = lyric_lines_of(out, tag, k)
        want = len(lines)
        assert len(got) == want, f"[{tag}]#{k} 歌词 {len(got)} 行 != 曲谱 {want} 行"
        # 3) 每行音节数 == 该行 onset 数
        for j, (onsets, _bars) in enumerate(lines):
            syl = cw.count_lyric_syllables(got[j])
            assert syl == onsets, f"[{tag}]#{k} 第 {j} 行音节 {syl} != onset {onsets}"
    if expect_inst_empty:
        for tag in ("Intro", "Outro", "Inst"):
            for m in re.finditer(rf"\[{tag}\]", out):
                rest = out[m.end():].strip()
                assert rest.startswith("[") or rest == "", f"[{tag}] 后不应有歌词：{rest[:40]!r}"
    # 5) 歌词文本来自用户输入（不是随机生成）
    units = cw._syllable_units(cw._split_lyric_lines(lyrics)[0])
    assert units, "应能从用户歌词解析出音节"
    assert units[0][0] in out, "歌词内容应来自用户输入"
    return out


print("=== 曲谱段落/行解析 ===")
for label, lines in cw.abc_sections(real):
    print(f"  {label:10s} 行数={len(lines):3d} onset={sum(o for o, _ in lines):4d} "
          f"每行小节={sorted({b for _, b in lines})}")

out = check("真实整曲谱 + 纯中文歌词", real, CN,
            ["Intro", "Verse", "Chorus", "Inst", "Verse", "Chorus", "Outro"])
print("\n--- 歌词全文 ---")
print(out)

# 遗留 [Verse] 标记应被剥离重建
legacy = "[Verse 1]\n" + CN + "\n\n[Chorus]\n听 海哭的声音"
out2 = check("含遗留段落标记（应剥离重建）", real, legacy,
             ["Intro", "Verse", "Chorus", "Inst", "Verse", "Chorus", "Outro"])
assert "[Verse 1]" not in out2

# 英文歌词
en = "Moonlight on the water tonight\nSinging softly holding you tight\nHold on to the light\nWe will make it through the night"
out3 = check("真实整曲谱 + 英文歌词", real, en,
             ["Intro", "Verse", "Chorus", "Inst", "Verse", "Chorus", "Outro"])

# 官方两段谱
out4 = check("官方 score.abc（两段）", score, "C E G A G E D C", ["Verse", "Chorus"])

# 边界
assert cw.align_lyrics_to_abc(real, "")[0] == ""
assert "曲谱没有" in cw.align_lyrics_to_abc("", CN)[1]

# 与官方原始歌词的行粒度对比：原词 28 行 ↔ 歌唱段 30 行
import json
req = json.loads((ROOT / "outputs/cover_20261009_185602/request.json").read_text(encoding="utf-8"))
orig_lines = [ln for ln in req["lyrics"].split("\n")
               if ln.strip() and not ln.strip().startswith("[")]
score_song_lines = sum(len(lines) for label, lines in cw.abc_sections(real)
                       if cw._section_tag(label) not in cw._INST_TAGS)
print(f"\n官方原始歌词 {len(orig_lines)} 行 ↔ 本谱歌唱段 {score_song_lines} 行"
      f"（官方同曲为 28 ↔ 30）")
assert abs(score_song_lines - len(orig_lines)) <= 4, "行粒度与官方形态差距过大"

print("\nALL OK")

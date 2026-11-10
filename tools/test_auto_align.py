# -*- coding: utf-8 -*-
"""自动对齐（auto_align_abc）回归测试：不改小节总时值、只动 Vocal 声部、可逆幂等。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cover_webui as cw  # noqa: E402


def measure_sums(abc):
    """每行 Vocal 声部按小节统计总时值（含休止；Z 整小节休止跳过该小节）。"""
    sums, voice = [], None
    for ln in abc.replace("\r\n", "\n").split("\n"):
        v = cw._voice_of(ln)
        if v is not None:
            voice = v
            continue
        s = ln.strip()
        if not (voice and voice.startswith("VOC")) or cw._is_field_header(s) or s.startswith("%"):
            continue
        measures, cur = [], 0.0
        for kind, text in cw._split_music_line(ln):
            if kind == "bar":
                measures.append(cur)
                cur = 0.0
                continue
            if kind in ("note", "rest"):
                try:
                    cur += cw._parse_note(text)[1] if kind == "note" else cw._parse_dur(
                        cw.re.match(r"^[zZx][,']*(.*)$", text).group(1))
                except ValueError:
                    return None          # 含复杂结构则跳过该谱面校验
        measures.append(cur)
        sums.append(measures)
    return sums


def strip_vocal(abc):
    """去掉 Vocal 声部内容后的剩余文本（用于确认其他声部/注释未被改动）。"""
    out, voice = [], None
    for ln in abc.replace("\r\n", "\n").split("\n"):
        v = cw._voice_of(ln)
        if v is not None:
            voice = v
            out.append(ln)
            continue
        if voice and voice.startswith("VOC") and not cw._is_field_header(ln.strip()) \
                and not ln.strip().startswith("%"):
            out.append("<vocal-music-line>")
        else:
            out.append(ln)
    return "\n".join(out)


def check(name, abc, lyrics, expect_exact=True):
    target = cw.count_lyric_syllables(lyrics)
    before = cw.count_vocal_notes(abc)
    new_abc, report = cw.auto_align_abc(abc, lyrics)
    after = cw.count_vocal_notes(new_abc)
    print(f"\n[{name}] 音节 {target} / 音符 {before} → {after} | {report}")
    assert after == target if expect_exact else after <= target, "对齐后 onset 数不符合预期"
    s0, s1 = measure_sums(abc), measure_sums(new_abc)
    if s0 is not None and s1 is not None:
        assert len(s0) == len(s1), "Vocal 行数变化"
        for i, (a, b) in enumerate(zip(s0, s1)):
            assert len(a) == len(b), f"第 {i} 行小节数变化"
            for j, (x, y) in enumerate(zip(a, b)):
                assert abs(x - y) < 1e-6, f"第 {i} 行第 {j} 小节总时值被改动：{x} → {y}"
    assert strip_vocal(abc) == strip_vocal(new_abc), "Vocal 以外的内容被改动"
    # 幂等
    again, report2 = cw.auto_align_abc(new_abc, lyrics)
    assert cw.count_vocal_notes(again) == after, "二次对齐不稳定"
    return new_abc


score = (ROOT / "YuE/examples/score.abc").read_text(encoding="utf-8")
melody = (ROOT / "YuE/examples/melody.abc").read_text(encoding="utf-8")
real = (ROOT / "outputs/cover_20261009_185602/score.abc").read_text(encoding="utf-8")

chords_before = score.count('"')
check("官方 melody.abc：56→29（合并）", melody, "啊" * 29)
new_score = check("官方 score.abc：56→80（拆分，保留和弦）", score, "啦" * 80)
assert new_score.count('"') == chords_before, f"和弦标记丢失：{chords_before} → {new_score.count(chr(34))}"
assert cw.abc_has_chords(new_score), "和弦标记应保留"
# 真实转录谱：复现「歌词 226 vs 音符多 17%」的场景
real_before = cw.count_vocal_notes(real)
check("真实整曲谱：合并到 226", real, "啊" * 226)
print(f"\n真实谱 onset（已排除注释行）：{real_before}")

# 超出可拆分上限时应有残差报告而不是崩溃
_, r = cw.auto_align_abc(melody, "啊" * 1000)
print("不可达目标报告：", r)
assert "未能完全对齐" in r or "本就匹配" in r

# Ins 声部逐字未改动
ins0 = [ln for ln in melody.split("\n") if ln.startswith("V: Ins")]
new_mel = cw.auto_align_abc(melody, "啊" * 29)[0]
ins1 = [ln for ln in new_mel.split("\n") if ln.startswith("V: Ins")]
assert ins0 == ins1, "Ins 声部头被改动"
print("\nALL OK")

# -*- coding: utf-8 -*-
"""快速验证 cover_webui 新增的纯函数（不起 UI、不加载模型）。"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cover_webui as cw  # noqa: E402

score = (ROOT / "YuE/examples/score.abc").read_text(encoding="utf-8")
melody = (ROOT / "YuE/examples/melody.abc").read_text(encoding="utf-8")
real = (ROOT / "outputs/cover_20261009_185602/score.abc").read_text(encoding="utf-8")

print("has_chords score.abc =", cw.abc_has_chords(score))      # True
print("has_chords melody.abc =", cw.abc_has_chords(melody))    # False
print("has_chords real      =", cw.abc_has_chords(real))       # False（melody-only 转录）
print("notes melody.abc =", cw.count_vocal_notes(melody), "(期望 56)")
print("notes score.abc  =", cw.count_vocal_notes(score), "(期望 56)")
print("notes real      =", cw.count_vocal_notes(real))

stripped = cw.strip_chord_symbols(score)
print("strip_chords → has_chords =", cw.abc_has_chords(stripped), "notes =", cw.count_vocal_notes(stripped))

lyrics = "[verse]\n月光洒在水面上\n轻轻地唱 紧紧拥抱\n[chorus]\nHold on, hold on to the light\nWe will make it through the night"
print("syllables =", cw.count_lyric_syllables(lyrics),
      "(期望 29：中文15字 + 英文14词)")
print("hint =", cw.alignment_hint(melody, lyrics, False))

# prepare_abc 四种组合
for keep_ins, keep_tempo, keep_chords in [(False, False, True), (True, True, True),
                                          (False, False, False), (True, True, False)]:
    out, note, cot = cw.prepare_abc(score, keep_ins, keep_tempo, keep_chords)
    print(f"prepare(keep_ins={keep_ins}, keep_tempo={keep_tempo}, keep_chords={keep_chords}) "
          f"→ cot={cot} note={note!r}")
    if not keep_ins:
        assert "V: Ins" not in out, "Ins 声部未被剥离"
    if not keep_tempo:
        assert "Q:" not in out, "速度标记未被剥离"
    if not keep_chords:
        assert not cw.abc_has_chords(out), "和弦未被剥离（音乐行中仍有引号和弦）"
    else:
        assert '"C"' in out, "和弦被误删"

# 真实转录谱（无和弦）+ keep_chords=True → 应回落 melody
out, note, cot = cw.prepare_abc(real, True, True, True)
print("real + keep_chords=True → cot =", cot, "| note =", note)
assert cot == "melody"
print("ALL OK")

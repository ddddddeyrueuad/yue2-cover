# -*- coding: utf-8 -*-
"""YuE2 翻唱（Cover）中文 WebUI

流程：上传源歌曲音频 → SheetSage2 转录出旋律+和弦 ABC 曲谱 → 用户编辑歌词/风格 →
YuE2 生成翻唱音频（cot=full 沿用原曲和弦走向，cot=melody 自由伴奏）。

仅包含 cover 功能；界面为中文。
"""
from __future__ import annotations

import os
import sys
import json
import queue
import re
import shutil
import subprocess
import threading
import time
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".hfcache"))
# 减少显存碎片（官方 OOM 提示推荐）
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, str(ROOT / "YuE" / "src"))

# 屏蔽 Gradio/Starlette 每次前端轮询都刷屏的弃用警告（只针对这一条）
warnings.filterwarnings("ignore", message="'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated")

import gradio as gr  # noqa: E402

MODEL_DIR = ROOT / "models" / "YuE2-3B"
VAE_DIR = ROOT / "models" / "YuE2-Vae"
SHEET_DIR = ROOT / "models" / "SheetSage2"
SHEET_PY = ROOT / ".venv-sheetsage2" / "Scripts" / "python.exe"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------- 日志
_log_q: "queue.Queue[str]" = queue.Queue()


_LOG_NOISE = ("StarletteDeprecationWarning", "queue_join_helper", "routes.py:1368", "HTTP_422")


class _TeeStderr:
    """把 stderr（含 yue2 的英文进度输出）同时转发到 UI 日志队列。"""

    def __init__(self, original):
        self.original = original

    def write(self, s):
        n = self.original.write(s)
        self.original.flush()
        for line in str(s).splitlines():
            line = line.strip()
            if line and not any(tag in line for tag in _LOG_NOISE):
                _log_q.put(line)
        return n

    def flush(self):
        self.original.flush()


sys.stderr = _TeeStderr(sys.stderr)


def log(msg: str):
    _log_q.put(str(msg))


# ---------------------------------------------------------------- 模型状态
_pipe = None
_gen_lock = threading.Lock()
_cancel = threading.Event()


def model_status_text():
    parts = []
    parts.append("✅ YuE2-3B 已下载" if (MODEL_DIR / "config.json").is_file() else "❌ YuE2-3B 缺失")
    parts.append("✅ YuE2-Vae 已下载" if (VAE_DIR / "config.json").is_file() else "❌ YuE2-Vae 缺失")
    parts.append("✅ SheetSage2 已下载" if (SHEET_DIR / "config.json").is_file() else "❌ SheetSage2 缺失")
    parts.append("✅ 生成模型已加载" if _pipe is not None else "⚪ 生成模型未加载（首次生成时自动加载）")
    return "　|　".join(parts)


def load_pipeline():
    global _pipe
    if _pipe is None:
        from yue2 import YuE2Pipeline
        log("正在加载 YuE2 生成模型（首次约需 1-2 分钟）……")
        _pipe = YuE2Pipeline.from_pretrained(
            str(MODEL_DIR), vae=str(VAE_DIR), device="cuda",
            memory_budget_gib=18, vae_core_frames=512, offload_ar=True, progress=True,
        )
        log("YuE2 生成模型加载完成。")
    return _pipe


# ---------------------------------------------------------------- 步骤 1：转录
def ensure_sheetsage_module_cache():
    """transformers 4.45 复制本地动态模块时只复制第一层 import，
    这里把模型目录下全部 .py 无条件同步进模块缓存（源码补丁随之生效）。"""
    modules_cache = Path(os.environ.get("HF_HOME", "")) / "modules" / "transformers_modules" / "SheetSage2"
    if (SHEET_DIR / "config.json").is_file():
        modules_cache.mkdir(parents=True, exist_ok=True)
        for f in SHEET_DIR.glob("*.py"):
            shutil.copyfile(f, modules_cache / f.name)


def transcribe(audio_path, max_seconds, mode):
    ensure_sheetsage_module_cache()
    if _pipe is not None:
        log("提示：生成模型已占用显存，转录与其共享 GPU，速度可能略慢。")
    if not audio_path:
        return None, "请先上传源歌曲音频。", model_status_text()
    if not (SHEET_DIR / "config.json").is_file():
        return None, "SheetSage2 模型缺失，请先运行 setup_download.ps1 下载模型。", model_status_text()
    tag = time.strftime("%Y%m%d_%H%M%S")
    work = OUT_DIR / f"transcribe_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    cmd = [str(SHEET_PY), str(SHEET_DIR / "infer.py"), str(audio_path), "--output", str(work)]
    if mode == "仅旋律":
        cmd += ["--melody-only"]
    if max_seconds:
        cmd += ["--max-seconds", str(float(max_seconds))]
    env = dict(os.environ, HF_HOME=str(ROOT / ".hfcache"))
    log(f"开始转录：{' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=env, timeout=3600)
    except subprocess.TimeoutExpired:
        return None, "转录超时（>60 分钟）。", model_status_text()
    for line in (proc.stderr or "").splitlines():
        if line.strip():
            log(line.strip())
    abc_file = work / "score.abc"
    if proc.returncode != 0 or not abc_file.is_file():
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        hint = "；含和弦模式失败时可切换「仅旋律」重试" if mode != "仅旋律" else ""
        return None, "转录失败：" + (err[-1] if err else f"退出码 {proc.returncode}") + hint, model_status_text()
    abc = abc_file.read_bytes().decode("utf-8")
    log(f"转录完成，曲谱已保存至 {abc_file}")
    chord_note = "曲谱含和弦标记，可勾选「跟随原曲和弦走向」COVER 和声进行" \
        if abc_has_chords(abc) else "曲谱为纯旋律（无和弦标记）"
    return abc, f"✅ 转录完成（{chord_note}；输出目录：{work}）。请在下方检查并编辑 ABC 曲谱。", model_status_text()


# ---------------------------------------------------------------- 步骤 2：生成翻唱
_INS_VOICES = {"INS", "INST", "INST.", "INSTRUMENTAL"}
# 音符发声（onset）：音名 + 八度记号 + 时值；休止符 z/Z/x 不计
_NOTE_RE = re.compile(r"[A-Ga-g][,']*(?:\d+|/\d+|/\d*)?")
_CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]+")


def _abc_lines(abc):
    return abc.replace("\r\n", "\n").split("\n")


def _voice_of(line):
    s = line.strip()
    if s[:2].upper() != "V:":
        return None
    parts = s[2:].strip().split()
    return parts[0].upper().rstrip(",") if parts else ""


def _is_field_header(s):
    return len(s) >= 2 and s[0].isalpha() and s[1] == ":"


def _music_line(line):
    """返回 (是音乐行, 去掉注释/装饰/引号内容后的文本)。"""
    s = line.strip()
    if _voice_of(line) is not None or _is_field_header(s):
        return False, line
    s = re.sub(r"%.*", "", s)            # ABC 注释
    s = re.sub(r"![^!]*!", "", s)        # !trill! 等装饰
    s = re.sub(r'"[^"]*"', "", s)        # 引号内容（和弦标记等）
    return True, s


def abc_has_chords(abc):
    """SheetSage2 完整模式把和弦写成 Vocal 声部里的引号标记（如 "C"、"Am"）。"""
    return any('"' in line for line in _abc_lines(abc) if _music_line(line)[0])


def strip_chord_symbols(abc):
    out = []
    for line in _abc_lines(abc):
        if _music_line(line)[0]:
            out.append(re.sub(r'"[^"]*"', "", line))
        else:
            out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------- 自动对齐（歌词 ↔ 旋律）
_MUSIC_TOKEN_RE = re.compile(
    r'(?P<chord>"[^"]*")'
    r"|(?P<note>(?:\^{1,2}|_{1,2}|=)?[A-Ga-g][,']*\d*(?:/\d*)?-?)"
    r"|(?P<rest>[zZx][,']*\d*(?:/\d*)?)"
    r"|(?P<bar>\[?\|+:?\]?)"
    r"|(?P<grace>\{[^}]*\})"
    r"|(?P<deco>![^!]*!)"
    r"|(?P<block>\[[^\]]*\])"
    r"|(?P<other>\S)"
)


def _parse_dur(text):
    """ABC 时值 → L: 单位的浮点数（支持 8、8/、/2、3/2 等写法）。"""
    if not text:
        return 1.0
    if "/" in text:
        num, _, den = text.partition("/")
        return (float(num) if num else 1.0) / (float(den) if den else 2.0)
    return float(text)


def _parse_note(text):
    """→ (音高文本, 时值, 是否连音到下一个音)。"""
    m = re.match(r"^(?P<acc>\^{1,2}|_{1,2}|=)?(?P<pitch>[A-Ga-g])(?P<oct>[,']*)"
                 r"(?P<dur>\d*(?:/\d*)?)(?P<tie>-?)$", text)
    if not m:
        raise ValueError(text)
    pitch = (m.group("acc") or "") + m.group("pitch") + m.group("oct")
    return pitch, _parse_dur(m.group("dur")), bool(m.group("tie"))


def _fmt_note(pitch, dur, tie):
    return f"{pitch}{int(dur)}{'-' if tie else ''}"


def _split_music_line(line):
    return [(m.lastgroup, m.group(0)) for m in _MUSIC_TOKEN_RE.finditer(line)]


def _scan_measure(tokens, prev_tie_out=False):
    """把一个小节的 token 列表解析成音符 onset 组。

    groups 每组 = 一个发声点（连音链算 1 个）；ok=False 表示该小节含无法安全
    改写的结构（装饰音群/和弦块/跨小节连音/分数时值），自动对齐会跳过它。
    """
    groups, ok = [], True
    if prev_tie_out:
        ok = False
    chord_pending, current, prev_tie = False, None, prev_tie_out
    for idx, (kind, text) in enumerate(tokens):
        if kind == "chord":
            chord_pending = True
            continue
        if kind in ("grace", "block", "other"):
            ok = False
            if current is not None:
                groups.append(current)
                current = None
            prev_tie = False
            continue
        if kind == "deco":
            continue
        if kind != "note":
            if current is not None:
                groups.append(current)
                current = None
            prev_tie = False
            continue
        try:
            pitch, dur, tie = _parse_note(text)
        except ValueError:
            ok = False
            continue
        if not float(dur).is_integer():
            ok = False
        if prev_tie and current is None:
            prev_tie = tie            # 跨小节连音的延续，不计新 onset
            continue
        if current is not None and prev_tie:
            current["tokens"].append(idx)
            current["dur"] += dur
            current["last_tie"] = tie
        else:
            if current is not None:
                groups.append(current)
            current = {"tokens": [idx], "dur": dur, "pitch": pitch,
                       "chord": chord_pending, "last_tie": tie}
            chord_pending = False
        prev_tie = tie
    if current is not None:
        groups.append(current)
    return groups, ok


def _vocal_struct(abc):
    """把 ABC 拆成结构化行：只有 Vocal 声部的音乐行可被改写，其余原样保留。"""
    structs, voice = [], None
    for ln in abc.replace("\r\n", "\n").split("\n"):
        v = _voice_of(ln)
        if v is not None:
            voice = v
            structs.append({"kind": "header", "raw": ln})
            continue
        s = ln.strip()
        if _is_field_header(s) or s.startswith("%"):
            structs.append({"kind": "other", "raw": ln})
            continue
        if voice and voice.startswith("VOC"):
            measures, bars, cur = [], [], []
            for kind, text in _split_music_line(ln):
                if kind == "bar":
                    measures.append(cur)
                    bars.append(text)
                    cur = []
                else:
                    cur.append((kind, text))
            measures.append(cur)
            structs.append({"kind": "vocal", "measures": measures, "bars": bars})
        else:
            structs.append({"kind": "other", "raw": ln})
    return structs


def _iter_measures(structs):
    """yield (struct_idx, measure_idx, groups, ok)，并跨小节传递连音状态。"""
    prev_tie = False
    for si, st in enumerate(structs):
        if st["kind"] != "vocal":
            continue
        for mi, tokens in enumerate(st["measures"]):
            groups, ok = _scan_measure(tokens, prev_tie)
            yield si, mi, groups, ok
            prev_tie = groups[-1]["last_tie"] if groups else False


def _count_onsets(structs):
    return sum(len(groups) for _, _, groups, _ in _iter_measures(structs))


def count_vocal_notes(abc):
    """Vocal 声部的音符 onset 数（连音链记 1 个，休止符/注释/装饰音不计）。"""
    try:
        return _count_onsets(_vocal_struct(abc))
    except Exception:
        return 0


def _best_merge(structs):
    """挑一组相邻 onset 合并：优先同音高、其次不吞和弦、再取最短的第二个音。"""
    best = None
    for si, mi, groups, ok in _iter_measures(structs):
        if not ok or len(groups) < 2:
            continue
        for gi in range(len(groups) - 1):
            g1, g2 = groups[gi], groups[gi + 1]
            if g2["last_tie"]:        # 连到下一小节，合并会破坏连音语义
                continue
            key = (0 if g1["pitch"] == g2["pitch"] else 1,
                   0 if not g2["chord"] else 1, g2["dur"])
            if best is None or key < best[0]:
                best = (key, si, mi, gi)
    return None if best is None else best[1:]


def _apply_merge(structs, si, mi, gi):
    tokens = structs[si]["measures"][mi]
    groups, _ = _scan_measure(tokens, False)
    g1, g2 = groups[gi], groups[gi + 1]
    last = g1["tokens"][-1]
    pitch = g1["pitch"]
    text = tokens[last][1]
    _, last_dur, _ = _parse_note(text)
    tokens[last] = ("note", _fmt_note(pitch, last_dur + g2["dur"], g1["last_tie"]))
    for idx in sorted(g2["tokens"], reverse=True):
        del tokens[idx]


def _best_rest_measure(structs):
    """整小节只有一个 onset 时可整体改为休止（歌词远少于音符时的兜底）。"""
    best = None
    for si, mi, groups, ok in _iter_measures(structs):
        if not ok or len(groups) != 1:
            continue
        g = groups[0]
        if g["last_tie"]:
            continue
        key = g["dur"]
        if best is None or key < best[0]:
            best = (key, si, mi)
    return None if best is None else best[1:]


def _apply_rest(structs, si, mi):
    tokens = structs[si]["measures"][mi]
    groups, _ = _scan_measure(tokens, False)
    dur = int(groups[0]["dur"])
    chords = [t for k, t in tokens if k == "chord"]      # 和弦标记保留在小节原位
    tokens[:] = [("chord", t) for t in chords] + [("rest", "z" if dur == 1 else f"z{dur}")]


def _best_split(structs):
    """挑时值最长的 onset 拆成两个（音节多于音符时用来容纳歌词）。"""
    best = None
    for si, mi, groups, ok in _iter_measures(structs):
        if not ok:
            continue
        for gi, g in enumerate(groups):
            if g["dur"] < 2:
                continue
            if best is None or g["dur"] > best[0]:
                best = (g["dur"], si, mi, gi)
    return None if best is None else best[1:]


def _apply_split(structs, si, mi, gi):
    tokens = structs[si]["measures"][mi]
    groups, _ = _scan_measure(tokens, False)
    g = groups[gi]
    dur1 = int(g["dur"]) // 2
    dur2 = int(g["dur"]) - dur1
    first = g["tokens"][0]
    pitch = g["pitch"]
    tokens[first] = ("note", _fmt_note(pitch, dur1, False))
    for idx in sorted(g["tokens"][1:], reverse=True):
        del tokens[idx]
    tokens.insert(first + 1, ("note", _fmt_note(pitch, dur2, False)))


def _emit_structs(structs):
    out = []
    for st in structs:
        if st["kind"] != "vocal":
            out.append(st["raw"])
            continue
        parts = []
        for i, meas in enumerate(st["measures"]):
            parts.extend(text for _, text in meas)
            if i < len(st["bars"]):
                parts.append(st["bars"][i])
        out.append("".join(parts))
    return "\n".join(out)


def auto_align_abc(abc, lyrics):
    """把歌词音节数自动落到 Vocal 声部：音节多则拆分长音，音节少则合并/改休止。

    只改 Vocal 声部的音符时值，保持每小节总时值、和弦标记、Ins 声部与注释不变。
    返回 (新 ABC, 说明文字)。"""
    target = count_lyric_syllables(lyrics)
    if target <= 0:
        return abc, "歌词为空，未做对齐。"
    structs = _vocal_struct(abc)
    before = _count_onsets(structs)
    if before <= 0:
        return abc, "Vocal 声部没有音符，未做对齐（请确认曲谱包含 Vocal 旋律）。"
    merges = splits = rests = 0
    guard = 0
    while guard < 100000:
        guard += 1
        cur = _count_onsets(structs)
        if cur == target:
            break
        cand = _best_merge(structs) if cur > target else None
        if cand is None and cur > target:
            cand = _best_rest_measure(structs)
            if cand is not None:
                _apply_rest(structs, *cand)
                rests += 1
                continue
        if cand is None and cur < target:
            cand = _best_split(structs)
            if cand is not None:
                _apply_split(structs, *cand)
                splits += 1
                continue
        if cand is not None:
            _apply_merge(structs, *cand)
            merges += 1
            continue
        break
    after = _count_onsets(structs)
    if not (merges or splits or rests):
        return abc, f"✅ 歌词音节 {target}，Vocal 音符 {before}，数量本就匹配，无需调整。"
    new_abc = _emit_structs(structs)
    acts = []
    if merges:
        acts.append(f"合并 {merges} 个音符")
    if splits:
        acts.append(f"拆分 {splits} 个音符")
    if rests:
        acts.append(f"{rests} 个小节改为休止")
    head = f"歌词音节 {target}，Vocal 音符 {before} → {after}"
    if after == target:
        return new_abc, f"✅ 已自动对齐：{head}（{'；'.join(acts)}）。"
    return new_abc, (f"⚠️ 未能完全对齐：{head}，还差 {abs(target - after)} 个 onset"
                     f"（受可拆分/合并的音符限制，可手动编辑曲谱）。")


def count_lyric_syllables(lyrics):
    """歌词音节数：CJK 按字、英文按估算音节（段落标记 [Verse] 等不计）。

    英文不再按「词」计——「Moonlight」是 2 个音节而非 1 个，按词计会系统性少算。"""
    text = re.sub(r"\[[^\]]*\]", " ", lyrics)
    return sum(w for _, w in _syllable_units(text))


def alignment_hint(abc, lyrics, keep_ins):
    """音节数 vs 音符数参考提示：错配过大时解释吞字/拖腔的成因与缓解办法。"""
    if not abc or not abc.strip() or not lyrics or not lyrics.strip():
        return ""
    notes, syl = count_vocal_notes(abc), count_lyric_syllables(lyrics)
    if notes <= 0 or syl <= 0:
        return ""
    pct = (syl - notes) / notes * 100.0
    if pct > 15:
        tip = (f"🎵 对齐检查：歌词音节 ≈ {syl}，Vocal 声部音符 ≈ {notes}（多 {pct:.0f}%）——"
               "音节多于音符时模型会吞字/抢拍。点「🎯 自动对齐」可拆分长音容纳歌词，或精简歌词。")
    elif pct < -15:
        tip = (f"🎵 对齐检查：歌词音节 ≈ {syl}，Vocal 声部音符 ≈ {notes}（少 {-pct:.0f}%）——"
               "音节少于音符时会出现拖长音/留白。点「🎯 自动对齐」可合并多余音符，或适当增加歌词。")
    else:
        tip = f"🎵 对齐检查：歌词音节 ≈ {syl}，Vocal 声部音符 ≈ {notes}，数量接近，咬字对齐应较自然。"
    if keep_ins and abs(pct) > 15:
        tip += "（当前正跟随伴奏旋律，咬字受原曲音符约束最紧）"
    return tip


# ---------------------------------------------------------------- 结构化歌词（曲谱段落驱动）
_STRUCT_TAG_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_SECTION_MAP = {
    "verse": "Verse", "chorus": "Chorus", "bridge": "Bridge",
    "pre-chorus": "Verse", "post-chorus": "Verse", "rap": "Verse",
    "inst": "Inst", "intro": "Intro", "outro": "Outro", "interlude": "Inst",
    "silence": "Inst", "instrumental": "Inst", "solo": "Inst",
}
_INST_TAGS = {"Inst", "Intro", "Outro"}


def _section_tag(label):
    """把 ABC 的 % 段落名映射成 YuE2 歌词段落标记。未知标签归入 Verse。"""
    key = label.strip().lower()
    if key in _SECTION_MAP:
        return _SECTION_MAP[key]
    for name, tag in (("verse", "Verse"), ("chorus", "Chorus"), ("bridge", "Bridge"),
                      ("intro", "Intro"), ("outro", "Outro"), ("interlude", "Inst"),
                      ("inst", "Inst"), ("solo", "Inst")):
        if name in key:
            return tag
    return "Verse"


def abc_sections(abc):
    """按 % 段落注释把 Vocal 声部切成 (标签, [(每行 onset 数, 该行小节数), ...]) 列表。

    分配单位是 **ABC 音乐行**（官方原生记谱每行 4 小节），不是单个小节——
    模型的歌词行与曲谱行是同粒度对应，这是官方原始歌词的形态
    （实测：原曲歌词 28 行 ↔ 歌唱段 30 个音乐行）。
    跨小节连音状态在整首谱内连续传递；纯休止行不承载歌词。"""
    structs = _vocal_struct(abc)
    sections, current, prev_tie = [], None, False
    for st in structs:
        if st["kind"] == "other" and st["raw"].strip().startswith("%"):
            current = {"label": st["raw"].strip()[1:].strip(), "lines": []}
            sections.append(current)
            prev_tie = False
            continue
        if st["kind"] != "vocal":
            continue
        if current is None:
            current = {"label": "verse", "lines": []}
            sections.append(current)
        line_onsets, line_bars = 0, 0
        for tokens in st["measures"]:
            groups, _ = _scan_measure(tokens, prev_tie)
            line_onsets += len(groups)
            line_bars += 1
            prev_tie = groups[-1]["last_tie"] if groups else False
        current["lines"].append((line_onsets, line_bars))
    if not sections:
        sections = [{"label": "verse", "lines": []}]
    out = []
    for s in sections:
        lines = [(o, b) for o, b in s["lines"] if o > 0]
        out.append((s["label"], lines))
    return out


def _split_lyric_lines(lyrics):
    """用户只输入纯歌词：按换行拆行，剥离遗留的 [Verse] 类标记与空行。"""
    lines = []
    for ln in lyrics.replace("\r\n", "\n").split("\n"):
        s = ln.strip()
        if not s or _STRUCT_TAG_RE.match(s):
            continue
        lines.append(s)
    return lines


_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+")


def _en_syllables(word):
    """英文单词音节数粗估（元音组启发式）。

    比「按词计 1 个」更接近实际演唱：「Moonlight」→2、「tonight」→2、
    「make」→1（词尾不发音 e）、「little」→2（le 结尾成音节）。
    精确音节需要发音词典；这里只用于目标估算与预览，最终由 LLM/人工把握。"""
    w = word.lower()
    if not re.search(r"[a-z]", w):
        return 1
    groups = len(_VOWEL_GROUP_RE.findall(w))
    if groups == 0:
        return 1
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and groups > 1:
        groups -= 1
    return max(1, groups)


def _syllable_units(text):
    """把一行歌词拆成音节单元 (文本, 音节数)。

    CJK 每字 1 音节；拉丁按词估算音节；标点并入前一个单元、不计音节。
    """
    units = []
    for m in re.finditer(r"[\u3400-\u9fff\uf900-\ufaff]|[A-Za-z]+(?:['’][A-Za-z]+)?|"
                         r"[^\sA-Za-z\u3400-\u9fff\uf900-\ufaff]+", text):
        tok = m.group(0)
        if _CJK_RE.match(tok):
            units.append((tok, 1))
        elif re.match(r"[A-Za-z]", tok):
            units.append((tok, _en_syllables(tok)))
        elif units:
            prev_text, prev_w = units[-1]
            units[-1] = (prev_text + tok, prev_w)   # 标点并入前一个单元
        else:
            units.append((tok, 0))
    return units


def _line_units(lines):
    return [_syllable_units(ln) for ln in lines]


def _join_units(units):
    """把音节单元拼回可读文本：CJK 相连不加空格，拉丁词用空格。"""
    out = ""
    for text, _w in units:
        if not out:
            out = text
        elif _CJK_RE.match(text[0]) or _CJK_RE.match(out[-1]):
            out += text
        else:
            out += " " + text
    return out


def _fit_phrases(targets, user_lines):
    """把用户歌词铺到曲谱各行：整句优先、允许浮动、不足时整句循环。

    targets 为每个曲谱行的 onset 数；每行目标音节 ≈ 0.9 × onset（官方参考词
    实测 337 音节 / 388 onset ≈ 0.87，允许一个音节跨多个音）。
    不拆词、不拆句：只有单句明显超长时才在词边界切开。
    """
    lines = [toks for toks in user_lines if sum(w for _, w in toks) > 0]
    if not lines or not targets:
        return []
    res, idx = [], 0
    toks = list(lines[0])
    for t in targets:
        want = max(1, int(round(t * 0.9)))
        if not toks:                       # 当前行已用尽 → 取下一句（循环）
            idx = (idx + 1) % len(lines)
            toks = list(lines[idx])
        cur = sum(w for _, w in toks)
        # 整句优先：当前句完整且长度在目标的 0.7–1.3 倍内 → 整句放进这一行
        if 0.7 * t <= cur <= 1.3 * t:
            res.append(list(toks))
            toks = []
            continue
        chunk, syl = [], 0
        while syl < want:
            if not toks:                   # 行内取完 → 续下一句（长句续写）
                idx = (idx + 1) % len(lines)
                toks = list(lines[idx])
            text, w = toks[0]
            if chunk and syl + w > want * 1.45:
                break
            chunk.append((text, w))
            syl += w
            toks.pop(0)
        res.append(chunk)
    return res


def align_lyrics_to_abc(abc, lyrics):
    """由 ABC 谱自动生成结构化歌词（段落标记 + 与曲谱行对应的歌词行）。

    用户歌词框只收纯歌词文本；段落标记、段落顺序、每行歌词量全部由曲谱的
    % section 注释与各行 Vocal 音符数决定。乐器段落输出空标记；歌词不足时
    按整句循环补齐。返回 (结构化歌词, 预览文本)。"""
    sections = abc_sections(abc)
    if not sections:
        return lyrics, "曲谱没有可识别的 Vocal 段落，未做结构化对齐。"
    sing = [(i, label, lines) for i, (label, lines) in enumerate(sections)
            if _section_tag(label) not in _INST_TAGS and sum(o for o, _ in lines) > 0]
    if not sing:
        return lyrics, "曲谱没有歌唱段落（全为乐器段），未做结构化对齐。"
    lines_txt = _split_lyric_lines(lyrics)
    if not lines_txt:
        return lyrics, "歌词为空，未做结构化对齐。"
    units = _line_units(lines_txt)
    total_syllables = sum(w for lu in units for _, w in lu)
    total_onsets = sum(o for _, _, ls in sing for o, _ in ls)
    seg_lines = {}
    for i, label, ls in sing:
        seg_lines[i] = _fit_phrases([o for o, _ in ls], units)
    out_lines, preview = [], []
    for i, (label, ls) in enumerate(sections):
        tag = _section_tag(label)
        out_lines.append(f"[{tag}]")
        if i in seg_lines:
            seg = seg_lines[i]
            for phrase in seg:
                out_lines.append(_join_units(phrase))
            got = sum(w for p in seg for _, w in p)
            preview.append(f"[{tag}] {len(ls)} 行 / {sum(o for o, _ in ls)} 音符 → {len(seg)} 句 / {got} 音节")
        else:
            preview.append(f"[{tag}] 乐器段，无歌词")
        out_lines.append("")
    result = "\n".join(out_lines).rstrip() + "\n"
    report = ("✅ 已按曲谱自动对齐歌词结构：" + "；".join(preview)
              + (f"。原始歌词 {total_syllables} 音节，曲谱歌唱段共 {total_onsets} 个 onset，"
                 "已按整句循环补齐。" if total_syllables < total_onsets else
                 f"。原始歌词 {total_syllables} 音节 / 曲谱歌唱段 {total_onsets} 个 onset。"))
    return result, report


# ---------------------------------------------------------------- LLM 改词（可选）
LLM_DEFAULT_BASE = "https://api.deepseek.com/v1"
LLM_DEFAULT_MODEL = "deepseek-chat"


def score_grid(abc):
    """给 LLM 的结构摘要：段落顺序 + 每段行数 + 每行目标音节数（≈0.9×onset）。"""
    grid = []
    for label, lines in abc_sections(abc):
        tag = _section_tag(label)
        onsets = [o for o, _ in lines]
        if tag in _INST_TAGS or not onsets or sum(onsets) == 0:
            grid.append({"tag": tag, "targets": []})
        else:
            grid.append({"tag": tag, "targets": [max(1, int(round(o * 0.9))) for o in onsets]})
    return grid


def _grid_text(grid):
    parts = []
    for g in grid:
        if not g["targets"]:
            parts.append(f"[{g['tag']}]（乐器段，不要写歌词）")
        else:
            parts.append(f"[{g['tag']}] 需要 {len(g['targets'])} 行，"
                         f"各行目标音节数: {g['targets']}")
    return "\n".join(parts)


def _llm_messages(grid, lyrics):
    system = (
        "你是中文/英文歌曲的作词与改编助手。任务：把用户歌词改编成能贴着给定旋律网格演唱的版本。\n"
        "硬性要求：\n"
        "1) 只输出歌唱段（网格里给了目标音节数的段），顺序必须与网格一致；"
        "标注为乐器段的段落可以整段省略，也可以保留一个空标记；段落标记可带序号如 [Verse 1]；\n"
        "2) 每个歌唱段的行数必须与网格给出的行数完全一致；\n"
        "3) 每行音节数尽量等于该行目标值（±1 可接受）；中文一字符一音节，英文按实际发音音节；\n"
        "4) 尽量保留用户原句与原意，只做必要的断句、增删字或整句重复，不要重新创作整首；\n"
        "5) 只输出歌词文本：先是 [Tag]，然后是该段歌词行，行与行之间换行；"
        "不要解释、不要代码块、不要编号。"
    )
    user = ("旋律网格：\n" + _grid_text(grid) +
            "\n\n用户歌词：\n" + lyrics.strip() +
            "\n\n请输出贴合上述网格的歌词。")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _strip_code_fence(text):
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[A-Za-z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _llm_chat(base_url, api_key, model, messages, temperature=0.7, timeout=180):
    """OpenAI 兼容 /chat/completions（标准库实现，不引入额外依赖）。"""
    import urllib.request
    url = (base_url or LLM_DEFAULT_BASE).rstrip("/") + "/chat/completions"
    payload = json.dumps({"model": model or LLM_DEFAULT_MODEL, "messages": messages,
                          "temperature": temperature}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key.strip()}",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def parse_structured_lyrics(text):
    """把 (可能是 LLM 输出的) 文本解析成 [(tag, [line, ...])]。"""
    sections, cur = [], None
    for ln in _strip_code_fence(text).replace("\r\n", "\n").split("\n"):
        s = ln.strip()
        if not s:
            continue
        m = _STRUCT_TAG_RE.match(s)
        if m:
            cur = (m.group(1).strip(), [])
            sections.append(cur)
        elif cur is not None:
            cur[1].append(s)
    return sections


def _base_tag(tag):
    """归一化段落标记：[Verse 1]/[Verse 2] → Verse，[Chorus] → Chorus。"""
    return re.sub(r"[\s\d#]+$", "", (tag or "").strip()).strip()


def validate_structured(text, grid):
    """校验 LLM 输出是否满足网格：歌唱段顺序、行数、每行音节接近目标。

    宽容两类无害差异：
    1) 乐器段（Intro/Inst/Outro）可以省略，也可以保留空标记——它们没有歌词；
    2) 段落标记可带序号（[Verse 1]/[Verse 2]），按基础名比较。"""
    got = [(t, [ln for ln in ls if ln.strip()]) for t, ls in parse_structured_lyrics(text)]
    got = [(t, ls) for t, ls in got if ls]            # 只看有歌词的段
    want = [g for g in grid if g["targets"]]           # 只比较歌唱段
    if len(got) != len(want):
        return False, (f"歌唱段数 {len(got)} != 期望 {len(want)}"
                       f"（期望顺序 {[g['tag'] for g in want]}，实际 {[t for t, _ in got]}）")
    for (tag, lines), g in zip(got, want):
        if _base_tag(tag) != g["tag"]:
            return False, f"段落标记 {tag} != 期望 {g['tag']}"
        want_n = len(g["targets"])
        if len(lines) != want_n:
            return False, f"[{tag}] 行数 {len(lines)} != 期望 {want_n}"
        for j, (ln, t) in enumerate(zip(lines, g["targets"]), 1):
            syl = count_lyric_syllables(ln)
            if abs(syl - t) > max(2, int(t * 0.35)):
                return False, f"[{tag}] 第 {j} 行音节 {syl} 偏离目标 {t}"
    return True, "ok"


def llm_adapt_lyrics(abc, lyrics, base_url, api_key, model, temperature=0.7):
    """让 LLM 把歌词改写成贴合网格的版本；失败时返回 (None, 原因) 由调用方回落。"""
    if not (api_key or "").strip():
        return None, "未配置 API key"
    grid = score_grid(abc)
    if not any(g["targets"] for g in grid):
        return None, "曲谱没有歌唱段落"
    try:
        content = _llm_chat(base_url, api_key, model, _llm_messages(grid, lyrics), temperature)
    except Exception as exc:
        return None, f"调用失败：{exc}"
    ok, why = validate_structured(content, grid)
    if not ok:
        return None, f"结构校验未通过（{why}）"
    return _strip_code_fence(content), "已通过结构校验"


def prepare_abc(abc, keep_ins, keep_tempo, keep_chords):
    """按选择预处理转录曲谱，并决定生成模式 cot：

    - 剥离原曲伴奏旋律（Ins 声部）/ 原曲速度标记（Q: 行）
    - keep_chords=True 且曲谱含和弦标记 → cot=full（COVER 原曲和弦走向）；
      否则剥离引号和弦标记，按 cot=melody 生成（伴奏与和声交由模型自由发挥）。

    官方文档指出：速度/拍号在 ABC 中是硬条件，想让风格提示词主导就放开锚点；
    翻唱想沿用原曲和声进行则用完整记谱（cot=full），想换风格更彻底则用旋律记谱（cot=melody）。"""
    lines = abc.replace("\r\n", "\n").split("\n")
    if not keep_chords:
        lines = [re.sub(r'"[^"]*"', "", ln) if _music_line(ln)[0] else ln for ln in lines]
    out, ins_count, tempo_count, chord_kept, skipping = [], 0, 0, False, False
    for line in lines:
        s = line.strip()
        if s[:2].upper() == "V:":
            parts = s[2:].strip().split()
            name = parts[0].upper() if parts else ""
            if name in _INS_VOICES and not keep_ins:
                skipping = True
                ins_count += 1
                continue
            skipping = False
            out.append(line)
            continue
        if skipping:
            # 段内仍保留结构性 ABC 头（K:/M:/L: 等可能出现在声部头之后），只丢音符与歌词行
            if len(s) >= 2 and s[0].isalpha() and s[1] == ":" and s[0].lower() not in {"w", "s"}:
                out.append(line)
            continue
        if not keep_tempo and s[:2].upper() == "Q:":
            tempo_count += 1
            continue
        if keep_chords and '"' in line:
            chord_kept = True
        out.append(line)
    notes = []
    if ins_count:
        notes.append("已移除原曲伴奏旋律声部（编曲更自由、风格更突出）")
    if tempo_count:
        notes.append("已移除原曲速度标记（速度交给风格提示词）")
    if not keep_chords:
        notes.append("已剥离和弦标记（cot=melody，和声交给模型按新风格编排）")
    elif chord_kept:
        notes.append("保留原曲和弦标记（cot=full，COVER 原曲和弦走向）")
    cot = "full" if (keep_chords and chord_kept) else "melody"
    return "\n".join(out), "；".join(notes), cot


def generate_cover(style, lyrics, abc, seed, keep_ins, keep_tempo, keep_chords, auto_align, fit_notes,
                   llm_on, llm_base, llm_key, llm_model, cfg_scale):
    if not abc or not abc.strip():
        return None, "缺少 ABC 曲谱。请先转录，或手动粘贴曲谱。", model_status_text()
    if not style or not style.strip():
        return None, "请填写目标风格描述。", model_status_text()
    if not lyrics or not lyrics.strip():
        return None, "请填写歌词。", model_status_text()
    if not _gen_lock.acquire(blocking=False):
        return None, "当前已有生成任务在进行中，请稍候。", model_status_text()
    _cancel.clear()
    try:
        pipe = load_pipeline()
        seed = int(seed or 831001)
        align_note = ""
        raw_lyrics = lyrics
        if auto_align:
            # ① 结构化歌词：曲谱段落 → [Verse]/[Chorus] 与逐行歌词（不改原曲音符）
            lyrics, struct_note = align_lyrics_to_abc(abc, raw_lyrics)
            log("歌词结构对齐：" + struct_note)
            align_note = struct_note
            # ② 可选：LLM 把**原始歌词**改写成贴合网格的版本（失败自动回落到 ①）
            if llm_on:
                llm_lyrics, llm_note = llm_adapt_lyrics(abc, raw_lyrics, llm_base, llm_key, llm_model)
                if llm_lyrics:
                    lyrics = llm_lyrics
                    log("LLM 改词：" + llm_note)
                    align_note += "；LLM 改词完成"
                else:
                    log("LLM 改词跳过：" + llm_note + "（使用确定性结构对齐结果）")
            # ③ 可选：按歌词音节数改写音符时值（会改变原曲节奏，默认关闭）
            if fit_notes:
                abc, fit_note = auto_align_abc(abc, lyrics)
                log("音符时值改写：" + fit_note)
                align_note = (align_note + "；" + fit_note).strip("；")
        abc, prep_note, cot = prepare_abc(abc, bool(keep_ins), bool(keep_tempo), bool(keep_chords))
        if prep_note:
            log("曲谱预处理：" + prep_note)
        mode_note = "保留原曲和弦走向（cot=full）" if cot == "full" else "无和弦/已剥离和弦，伴奏与和声自由发挥（cot=melody）"
        log(f"生成模式：{mode_note}")
        hint = alignment_hint(abc, lyrics, bool(keep_ins))
        if hint:
            log(hint)
        extra = {}
        try:
            cfg = float(cfg_scale)
        except (TypeError, ValueError):
            cfg = None
        if cfg is not None and cfg > 1.0:
            extra["cfg_scale"] = cfg
        log(f"开始生成翻唱（seed={seed}, cot={cot}, cfg_scale={cfg if cfg is not None else 1.0}）……")
        t0 = time.time()
        song = pipe(style=style.strip(), lyrics=lyrics, abc=abc,
                    cot=cot, seed=seed,
                    cancelled=_cancel.is_set, **extra)
        tag = time.strftime("%Y%m%d_%H%M%S")
        out = OUT_DIR / f"cover_{tag}"
        song.save_artifacts(out)
        wav = out / "audio.wav"
        song.save(wav)
        secs = len(song.audio) / song.sample_rate
        trunc = song.truncated
        warn = ""
        if any(trunc.values()):
            warn = f"　⚠️ 生成被截断（abc={trunc['abc']}, semantic={trunc['semantic']}），可尝试缩短歌词或换 seed。"
        log(f"翻唱生成完成，用时 {time.time() - t0:.1f} 秒，音频 {secs:.1f} 秒。")
        return str(wav), (f"✅ 生成完成：时长 {secs:.1f} 秒，用时 {time.time() - t0:.0f} 秒。"
                          f"产物目录：{out}{warn}"
                          + (f"\n自动对齐：{align_note}" if align_note else "")
                          + (f"\n曲谱预处理：{prep_note}" if prep_note else "")), model_status_text()
    except InterruptedError:
        return None, "已取消生成。", model_status_text()
    except Exception as exc:
        log(traceback.format_exc())
        return None, f"生成失败：{exc}", model_status_text()
    finally:
        _gen_lock.release()


def cancel_generation():
    _cancel.set()
    log("已请求取消生成……")
    return "已请求取消，将在当前 token/步骤后停止。"


def auto_align_ui(abc, lyrics):
    """手动触发：按曲谱生成结构化歌词并展示（不改动原曲音符）。"""
    if not abc or not abc.strip():
        return abc, "缺少 ABC 曲谱，无法对齐。"
    if not lyrics or not lyrics.strip():
        return abc, "请先填写歌词。"
    try:
        structured, report = align_lyrics_to_abc(abc, lyrics)
        log("歌词结构对齐：" + report)
        full = report + "\n\n—— 生成时实际发送给模型的歌词 ——\n" + structured
        return abc, full
    except Exception as exc:
        log(traceback.format_exc())
        return abc, f"自动对齐失败：{exc}"


def align_preview(abc, lyrics):
    """曲谱/歌词变化时实时预览结构化歌词（不写回任何输入框）。"""
    if not abc or not abc.strip() or not lyrics or not lyrics.strip():
        return ""
    try:
        _, report = align_lyrics_to_abc(abc, lyrics)
        return report
    except Exception as exc:
        return f"预览失败：{exc}"


def llm_preview(abc, lyrics, base, key, model):
    """试跑 LLM 改词：只返回预览文本，不改变生成参数、不写回输入框。"""
    if not abc or not abc.strip():
        return "缺少 ABC 曲谱。"
    if not lyrics or not lyrics.strip():
        return "请先填写歌词。"
    structured, report = align_lyrics_to_abc(abc, lyrics)
    text, note = llm_adapt_lyrics(abc, lyrics, base, key, model)   # 用原始歌词改写
    if not text:
        return (f"{report}\n\n❌ LLM 改词未采用：{note}\n"
                f"（生成时会回落到上面的本地确定性结果）")
    log("LLM 试跑成功：" + note)
    return (f"{report}\n\n✅ LLM 改词（{note}），生成时实际发送：\n\n{text}\n\n"
            f"—— 本地确定性结果（LLM 失败时的回退）——\n{structured}")


def poll_log(last):
    lines = []
    try:
        while True:
            lines.append(_log_q.get_nowait())
    except queue.Empty:
        pass
    if not lines:
        return last
    buf = (last or "").splitlines()[-120:]
    buf.extend(lines)
    return "\n".join(buf[-150:])


def preload_model():
    if _pipe is not None:
        return "生成模型已加载。", model_status_text()
    try:
        load_pipeline()
        return "生成模型加载完成。", model_status_text()
    except Exception as exc:
        log(traceback.format_exc())
        return f"模型加载失败：{exc}", model_status_text()


# ---------------------------------------------------------------- 界面（中文）
EXAMPLE_STYLE = ("English, jazz-funk, warm lead vocal, Rhodes, bass and drums")
EXAMPLE_ABC = (ROOT / "YuE" / "examples" / "melody.abc")

with gr.Blocks(title="YuE2 翻唱工作台") as demo:
    gr.Markdown(
        "# 🎵 YuE2 翻唱（Cover）工作台\n"
        "**流程：** ① 上传源歌曲音频 → ② SheetSage2 自动转录出旋律曲谱（ABC，含和弦）→ "
        "③ 填写目标风格与歌词 → ④ YuE2 生成翻唱（可 COVER 原曲和弦走向，歌词无需逐字对齐）。\n"
        "转录与生成共用 GPU，请顺序操作；生成一首通常需要数分钟，请耐心等待。"
    )
    status_md = gr.Markdown(model_status_text())

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 第 1 步 · 上传源歌曲并转录旋律")
            audio_in = gr.Audio(type="filepath", label="源歌曲音频（wav/mp3/flac 等）")
            max_sec = gr.Number(value=None, precision=0, label="转录最大时长（秒，可选）")
            mode_in = gr.Radio(choices=["含和弦（推荐）", "仅旋律"], value="含和弦（推荐）",
                               label="转录内容",
                               info="含和弦=曲谱带和弦标记，可 COVER 原曲和弦走向（cot=full）；"
                                    "仅旋律=旧模式，曲谱不带和弦，伴奏更自由（cot=melody）")
            btn_transcribe = gr.Button("🎼 转录旋律", variant="primary")
            abc_out = gr.Textbox(label="ABC 旋律曲谱（可编辑）", lines=14, interactive=True)
            step1_info = gr.Textbox(label="转录状态", lines=2, interactive=False)
        with gr.Column():
            gr.Markdown("### 第 2 步 · 设定风格与歌词，生成翻唱\n"
                        "💡 觉得风格不够明显？保持两个\"跟随原曲\"选项不勾选，并适当调高风格引导强度。")
            style_in = gr.Textbox(label="目标风格描述（英文效果最佳）", lines=3, value=EXAMPLE_STYLE)
            lyrics_in = gr.Textbox(label="歌词（只输入歌词正文，不要写 [Verse]/[Chorus] 等段落标记；系统按 ABC 谱自动对齐段落）", lines=12)
            align_out = gr.Textbox(label="对齐预览（曲谱段落 → 歌词结构）", lines=5, interactive=False)
            with gr.Row():
                btn_align = gr.Button("🎯 自动对齐（生成结构化歌词）")
                auto_align_in = gr.Checkbox(value=True, label="生成前自动对齐",
                                            info="按 ABC 的 % 段落自动生成 [Verse]/[Chorus] 结构，"
                                                 "并把歌词逐行分配到曲谱行（不改动原曲音符）")
                fit_notes_in = gr.Checkbox(value=False, label="同时按歌词量改写音符时值",
                                           info="勾选会把曲谱音符合并/拆分以匹配歌词音节数；"
                                                "默认关闭，因为改谱会破坏原曲旋律复刻")
            seed_in = gr.Number(value=831001, precision=0, label="随机种子 seed")
            with gr.Row():
                keep_ins_in = gr.Checkbox(value=False, label="跟随原曲伴奏旋律（Ins 声部）",
                                          info="勾选=旋律复刻更贴近原曲，但咬字受原曲音符约束、可能出现吞字；"
                                               "不勾选=咬字更自然、编曲更自由")
                keep_tempo_in = gr.Checkbox(value=False, label="跟随原曲速度（Q: 标记）",
                                            info="勾选=锁定原曲速度；不勾选=速度由风格提示词决定")
            keep_chords_in = gr.Checkbox(value=True, label="跟随原曲和弦走向（cot=full）",
                                         info="勾选且曲谱含和弦标记时按完整记谱生成，翻唱沿用原曲和声进行；"
                                              "不勾选=剥离和弦（cot=melody），和声交给模型按新风格编排")
            cfg_in = gr.Slider(minimum=1.0, maximum=10.0, value=1.2, step=0.1,
                               label="风格引导强度 cfg_scale",
                               info="越高越贴合风格提示词（官方建议从 1.2 起实验，过高可能影响音质；1.0=关闭引导且更快）")
            with gr.Accordion("🤖 LLM 改词（可选，OpenAI 兼容接口）", open=False):
                gr.Markdown(
                    "官方把「把歌词改写成能贴合乐句演唱的版本」交给 agent（LLM）；"
                    "本工具先把曲谱算成网格（段落/行数/每行目标音节数），再让 LLM 按网格改写，"
                    "输出会做结构校验，不合格自动回落到本地确定性结果。"
                )
                with gr.Row():
                    llm_on_in = gr.Checkbox(value=False, label="启用 LLM 改词",
                                            info="需要联网与 API 费用；失败不影响生成")
                    llm_model_in = gr.Textbox(value=LLM_DEFAULT_MODEL, label="模型", scale=1)
                llm_base_in = gr.Textbox(value=LLM_DEFAULT_BASE, label="接口地址（OpenAI 兼容 base_url）")
                llm_key_in = gr.Textbox(value=os.environ.get("LLM_API_KEY", ""), label="API Key",
                                        type="password", info="只保存在本进程内存；也可用环境变量 LLM_API_KEY")
                btn_llm = gr.Button("🧪 试跑 LLM 改词（只预览，不改生成参数）")
            with gr.Row():
                btn_generate = gr.Button("🎧 生成翻唱", variant="primary")
                btn_cancel = gr.Button("⏹ 取消生成")
            audio_out = gr.Audio(type="filepath", label="翻唱结果（48 kHz 立体声）")
            step2_info = gr.Textbox(label="生成状态", lines=3, interactive=False)

    with gr.Accordion("运行日志 / 模型管理", open=True):
        with gr.Row():
            btn_preload = gr.Button("⚡ 预加载生成模型")
            preload_info = gr.Textbox(label="模型状态", interactive=False)
        log_box = gr.Textbox(label="日志", lines=12, interactive=False)
        timer = gr.Timer(2.0)
        timer.tick(poll_log, inputs=[log_box], outputs=[log_box])

    btn_transcribe.click(transcribe, inputs=[audio_in, max_sec, mode_in],
                         outputs=[abc_out, step1_info, status_md])
    btn_generate.click(generate_cover,
                       inputs=[style_in, lyrics_in, abc_out, seed_in, keep_ins_in, keep_tempo_in,
                               keep_chords_in, auto_align_in, fit_notes_in,
                               llm_on_in, llm_base_in, llm_key_in, llm_model_in, cfg_in],
                       outputs=[audio_out, step2_info, status_md])
    btn_align.click(auto_align_ui, inputs=[abc_out, lyrics_in], outputs=[abc_out, align_out])
    btn_llm.click(llm_preview, inputs=[abc_out, lyrics_in, llm_base_in, llm_key_in, llm_model_in],
                  outputs=[align_out])
    btn_cancel.click(cancel_generation, outputs=[step2_info])
    btn_preload.click(preload_model, outputs=[preload_info, status_md])
    for comp in (abc_out, lyrics_in):
        comp.change(align_preview, inputs=[abc_out, lyrics_in], outputs=[align_out])

    if EXAMPLE_ABC.is_file():
        gr.Markdown("---\n💡 想先体验生成效果？可把仓库自带曲谱 `YuE/examples/score.abc`（含和弦走向）"
                    "或 `YuE/examples/melody.abc`（纯旋律）粘贴到曲谱框，再填写歌词与风格直接生成。")

if __name__ == "__main__":
    # 避免系统代理拦截 Gradio 对本机的启动自检请求
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft(), inbrowser=False)

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


def count_vocal_notes(abc):
    """统计 Vocal 声部音符发声数（休止符不计；[CEG] 和弦块记 1 个 onset）。

    仅用于「咬字对齐」参考提示，不是严格乐理解析。"""
    total, voice = 0, None
    for line in _abc_lines(abc):
        v = _voice_of(line)
        if v is not None:
            voice = v
            continue
        if voice != "VOCAL":
            continue
        s = line.strip()
        if _is_field_header(s):          # w:/K:/L: 等字段行
            continue
        s = re.sub(r"%.*", "", s)
        s = re.sub(r"![^!]*!", "", s)
        s = re.sub(r"\[[A-Za-z]:[^\]]*\]", "", s)   # 行内字段 [K:G]
        s = re.sub(r"\{[^}]*\}", "", s)  # 装饰音群
        s = re.sub(r"\[[^\]]*\]", "C1", s)          # 和弦块 [CEG] 记 1 个 onset
        s = re.sub(r'"[^"]*"', "", s)    # 和弦标记
        total += len(_NOTE_RE.findall(s))
    return total


def count_lyric_syllables(lyrics):
    """歌词音节数：中日韩字符按字计，拉丁文按词计（段落标记 [Verse] 等不计）。"""
    text = re.sub(r"\[[^\]]*\]", " ", lyrics)
    return len(_CJK_RE.findall(text)) + len(_LATIN_RE.findall(text))


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
               "音节明显多于音符时模型会吞字/抢拍。建议精简歌词，或取消勾选「跟随伴奏旋律」让咬字更自由。")
    elif pct < -15:
        tip = (f"🎵 对齐检查：歌词音节 ≈ {syl}，Vocal 声部音符 ≈ {notes}（少 {-pct:.0f}%）——"
               "音节明显少于音符时会出现拖长音或留白，可适当增加歌词。")
    else:
        tip = f"🎵 对齐检查：歌词音节 ≈ {syl}，Vocal 声部音符 ≈ {notes}，数量接近，咬字对齐应较自然。"
    if keep_ins and abs(pct) > 15:
        tip += "（当前正跟随伴奏旋律，咬字受原曲音符约束最紧）"
    return tip


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


def generate_cover(style, lyrics, abc, seed, keep_ins, keep_tempo, keep_chords, cfg_scale):
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
            lyrics_in = gr.Textbox(label="歌词（无需与原曲逐字对齐，模型按旋律自动咬字）", lines=12)
            align_out = gr.Textbox(label="咬字对齐检查（参考）", lines=2, interactive=False)
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
                               keep_chords_in, cfg_in],
                       outputs=[audio_out, step2_info, status_md])
    btn_cancel.click(cancel_generation, outputs=[step2_info])
    btn_preload.click(preload_model, outputs=[preload_info, status_md])
    for comp in (abc_out, lyrics_in, keep_ins_in):
        comp.change(alignment_hint, inputs=[abc_out, lyrics_in, keep_ins_in], outputs=[align_out])

    if EXAMPLE_ABC.is_file():
        gr.Markdown("---\n💡 想先体验生成效果？可把仓库自带曲谱 `YuE/examples/score.abc`（含和弦走向）"
                    "或 `YuE/examples/melody.abc`（纯旋律）粘贴到曲谱框，再填写歌词与风格直接生成。")

if __name__ == "__main__":
    # 避免系统代理拦截 Gradio 对本机的启动自检请求
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(var, None)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft(), inbrowser=False)

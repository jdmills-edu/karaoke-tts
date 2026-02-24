#!/usr/bin/env python3
"""
karaoke-tts MCP Server

Synthesizes speech from text, transcribes it with faster-whisper to get
word-level timestamps, then opens a karaoke-style browser player with
synchronized word highlighting.
"""

import json
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
from mcp.server.fastmcp import FastMCP

CONFIG_PATH = Path(__file__).parent / "config.json"

mcp = FastMCP("karaoke-tts")

_whisper_model = None
_whisper_lock = threading.Lock()

# ---------------------------------------------------------------------------
# HTML player template — placeholders replaced at generation time
# ---------------------------------------------------------------------------
PLAYER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>karaoke-tts</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0d0d1a;
      color: #c8c8e0;
      font-family: -apple-system, "Helvetica Neue", sans-serif;
      font-size: 26px;
      line-height: 1.9;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    #header {
      padding: 24px 60px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #1a1a2e;
      flex-shrink: 0;
    }
    #badge {
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.12em;
      color: #44447a;
    }
    #dur { font-size: 13px; color: #44447a; font-variant-numeric: tabular-nums; }
    #scroll { flex: 1; overflow-y: auto; padding: 60px; }
    #text { max-width: 860px; margin: 0 auto; }
    .w { color: #2a2a52; transition: color 0.07s ease, text-shadow 0.07s ease; }
    .w.active {
      color: #f5c518;
      text-shadow: 0 0 28px rgba(245, 197, 24, 0.45);
    }
    .w.past { color: #5858a0; }
    #footer {
      padding: 20px 60px;
      border-top: 1px solid #1a1a2e;
      display: flex;
      align-items: center;
      gap: 16px;
      flex-shrink: 0;
    }
    #track {
      flex: 1; height: 3px; background: #1a1a2e;
      border-radius: 2px; cursor: pointer;
    }
    #fill {
      height: 100%; background: #f5c518; border-radius: 2px;
      width: 0%; transition: width 0.1s linear; pointer-events: none;
    }
    #time {
      font-size: 13px; color: #44447a;
      font-variant-numeric: tabular-nums; min-width: 72px; text-align: right;
    }
    audio { display: none; }
  </style>
</head>
<body>
  <div id="header">
    <span id="badge">__ENGINE__ · __VOICE__</span>
    <span id="dur">—</span>
  </div>
  <div id="scroll"><div id="text"></div></div>
  <div id="footer">
    <div id="track"><div id="fill"></div></div>
    <span id="time">0:00</span>
  </div>
  <audio id="a" src="__AUDIO__"></audio>
  <script>
    const W = __WORDS__;
    const txt = document.getElementById("text");
    W.forEach((w, i) => {
      const s = document.createElement("span");
      s.className = "w"; s.id = "w" + i; s.textContent = w.word;
      txt.appendChild(s);
    });
    const audio = document.getElementById("a");
    const fill  = document.getElementById("fill");
    const timeEl = document.getElementById("time");
    const durEl  = document.getElementById("dur");
    const spans  = W.map((_, i) => document.getElementById("w" + i));
    function fmt(s) {
      return Math.floor(s / 60) + ":" + String(Math.floor(s % 60)).padStart(2, "0");
    }
    audio.addEventListener("loadedmetadata", () => { durEl.textContent = fmt(audio.duration); });
    let last = -1;
    audio.addEventListener("timeupdate", () => {
      const t = audio.currentTime;
      fill.style.width = (t / audio.duration * 100) + "%";
      timeEl.textContent = fmt(t);
      let lo = 0, hi = W.length - 1, idx = -1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if      (W[mid].end   <= t) lo = mid + 1;
        else if (W[mid].start >  t) hi = mid - 1;
        else { idx = mid; break; }
      }
      if (idx !== last) {
        if (last >= 0) spans[last].className = "w past";
        if (idx  >= 0) {
          spans[idx].className = "w active";
          spans[idx].scrollIntoView({ behavior: "smooth", block: "center" });
        }
        last = idx;
      }
    });
    document.getElementById("track").addEventListener("click", e => {
      const r = e.currentTarget.getBoundingClientRect();
      audio.currentTime = ((e.clientX - r.left) / r.width) * audio.duration;
    });
    audio.play().catch(() => {});
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def notify(title: str, message: str) -> None:
    subprocess.run(
        ["terminal-notifier", "-title", title, "-message", message],
        check=False,
    )


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def resolve(p: str) -> Path:
    return Path(p).expanduser().resolve()


def make_output_paths(config: dict) -> tuple[Path, Path]:
    output_dir = resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return output_dir / f"{stem}.ogg", output_dir / f"{stem}.html"


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip()
    if current:
        chunks.append(current)
    return chunks or [text]


# ---------------------------------------------------------------------------
# TTS engines
# ---------------------------------------------------------------------------

def generate_piper(text: str, voice: str, out_path: Path) -> None:
    from piper import PiperVoice

    piper_voice = PiperVoice.load(str(resolve(voice)))
    chunks = list(piper_voice.synthesize(text))
    if not chunks:
        raise RuntimeError("Piper produced no audio output")
    audio = np.concatenate([c.audio_float_array for c in chunks])
    sf.write(str(out_path), audio, chunks[0].sample_rate)


def generate_kokoro(text: str, voice: str, out_path: Path, config: dict) -> None:
    from kokoro_onnx import Kokoro

    k = config["kokoro"]
    kokoro = Kokoro(str(resolve(k["model_path"])), str(resolve(k["voices_path"])))
    lang, speed = k.get("lang", "en-us"), k.get("speed", 1.0)
    segments, sample_rate = [], None
    for chunk in chunk_text(text):
        samples, sr = kokoro.create(chunk, voice=voice, speed=speed, lang=lang)
        segments.append(samples)
        sample_rate = sr
    audio = np.concatenate(segments) if len(segments) > 1 else segments[0]
    sf.write(str(out_path), audio, sample_rate)


# ---------------------------------------------------------------------------
# Word timestamps via faster-whisper
# ---------------------------------------------------------------------------

def get_word_timestamps(audio_path: Path, model_size: str) -> list[dict]:
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        model = _whisper_model
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                words.append({
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                })
    return words


# ---------------------------------------------------------------------------
# HTML player generation
# ---------------------------------------------------------------------------

def generate_player(
    words: list[dict], engine: str, voice: str, wav_path: Path, html_path: Path
) -> None:
    html = (
        PLAYER_TEMPLATE
        .replace("__ENGINE__", engine.upper())
        .replace("__VOICE__", voice.upper())
        .replace("__AUDIO__", wav_path.name)
        .replace("__WORDS__", json.dumps(words))
    )
    html_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp.tool()
def generate_speech(
    text: str,
    engine: str,
    voice: str,
    output_path: str = None,
) -> str:
    """Synthesize speech then open a karaoke browser player with word-by-word
    highlighting synchronized to the audio.

    IMPORTANT: Always call list_kokoro_voices first to show the user the
    available voices and confirm which engine and voice to use before
    calling this tool. engine and voice are required — do not omit them.

    IMPORTANT: Before passing text to this tool, rewrite it to be
    TTS-friendly. Apply ALL of the following transformations:
      - Em dashes (—) and spaced hyphens ( - ) → comma or period for a natural pause
      - Hyphens in compound words → space (e.g. "well-known" → "well known")
      - % → "percent"
      - $ → "dollars" (e.g. "$1.5M" → "1.5 million dollars")
      - & → "and"
      - @ → "at"
      - # (number sign) → "number"
      - / (as a separator) → "or" or "and" depending on context
      - Large numbers → spoken form (e.g. 1,200,000 → "1.2 million",
        42,300 → "42 thousand 300", 9.5 → "nine point five")
      - Ordinals → spoken form (e.g. "1st" → "first", "3rd" → "third")
      - Acronyms that should be spelled out → spaced letters
        (e.g. "AI" → "A.I.", "GDP" → "G.D.P.")
      - URLs and email addresses → omit or replace with a short description
      - Markdown formatting (**, *, #, >, ---) → remove entirely
      - Parenthetical asides → replace parens with commas

    This tool returns immediately. Three stages run in the background:
      1. TTS synthesis (Piper or Kokoro)
      2. Word-timestamp extraction (faster-whisper)
      3. Karaoke player generation + browser launch
    A macOS notification fires at each stage. Do not wait or poll —
    tell the user generation is underway and they'll be notified.

    Args:
        text: TTS-friendly text to synthesize. Any length is supported.
        engine: TTS engine — must be "piper" or "kokoro".
        voice: Voice identifier (required).
               Piper: path to a .onnx model file.
               Kokoro: voice name (e.g. "af_heart", "am_michael").
        output_path: Optional custom path for the OGG file.
                     The HTML player is saved alongside it automatically.

    Returns:
        Confirmation with output paths.
    """
    config = load_config()
    if engine not in ("piper", "kokoro"):
        raise ValueError(f"Unknown engine {engine!r}. Must be 'piper' or 'kokoro'.")

    if output_path:
        wav_path = resolve(output_path)
        html_path = wav_path.with_suffix(".html")
    else:
        wav_path, html_path = make_output_paths(config)

    notify("Generating Speech…", f"{engine} · {voice} · {len(text):,} chars")

    def run():
        try:
            if engine == "piper":
                generate_piper(text, voice, wav_path)
            else:
                generate_kokoro(text, voice, wav_path, config)

            notify("Syncing words…", wav_path.name)
            words = get_word_timestamps(wav_path, config.get("whisper_model", "small"))

            generate_player(words, engine, voice, wav_path, html_path)
            notify("Karaoke Ready", wav_path.stem)
            subprocess.run(["open", str(html_path)], check=False)
        except Exception as e:
            notify("Speech Failed", str(e))

    threading.Thread(target=run, daemon=False).start()
    return (
        f"Speech generation is running in the background ({engine} · {voice}). "
        f"You'll receive macOS notifications at each stage:\n"
        f"  1. Generating Speech…\n"
        f"  2. Syncing words…\n"
        f"  3. Karaoke Ready (browser opens automatically)\n\n"
        f"Audio:  {wav_path}\n"
        f"Player: {html_path}\n\n"
        f"Do not wait or poll — let the user know it's underway."
    )


@mcp.tool()
def list_kokoro_voices() -> str:
    """List all available Kokoro voice names and present them to the user as a
    pick list so they can select one before generating speech. After the user
    makes a selection, proceed to call generate_speech with their chosen voice."""
    voices = {
        "af_heart":    "American Female — Heart (warm, natural)",
        "af_bella":    "American Female — Bella",
        "af_nicole":   "American Female — Nicole",
        "af_sarah":    "American Female — Sarah",
        "am_adam":     "American Male — Adam",
        "am_michael":  "American Male — Michael",
        "bf_emma":     "British Female — Emma",
        "bf_isabella": "British Female — Isabella",
        "bm_george":   "British Male — George",
        "bm_lewis":    "British Male — Lewis",
    }
    return "Kokoro voices:\n" + "\n".join(f"  {k}: {v}" for k, v in voices.items())


def main():
    mcp.run()


if __name__ == "__main__":
    main()

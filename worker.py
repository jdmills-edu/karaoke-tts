#!/usr/bin/env python3
"""
karaoke-tts background worker

Runs the full synthesis pipeline (TTS → word timestamps → karaoke player)
as a process completely detached from the MCP server. Invoked by server.py
via subprocess with start_new_session=True so it survives server exit.

Usage: python worker.py <params_json_file>
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# HTML player template
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


def resolve(p: str) -> Path:
    return Path(p).expanduser().resolve()


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
# Word timestamps
# ---------------------------------------------------------------------------

def get_word_timestamps(audio_path: Path, model_size: str) -> list[dict]:
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
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
# HTML player
# ---------------------------------------------------------------------------

def generate_player(
    words: list[dict], engine: str, voice: str, ogg_path: Path, html_path: Path
) -> None:
    html = (
        PLAYER_TEMPLATE
        .replace("__ENGINE__", engine.upper())
        .replace("__VOICE__", voice.upper())
        .replace("__AUDIO__", ogg_path.name)
        .replace("__WORDS__", json.dumps(words))
    )
    html_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    params_file = Path(sys.argv[1])
    params = json.loads(params_file.read_text())
    params_file.unlink(missing_ok=True)

    text         = params["text"]
    engine       = params["engine"]
    voice        = params["voice"]
    ogg_path     = Path(params["ogg_path"])
    html_path    = Path(params["html_path"])
    config       = params["config"]
    whisper_model = config.get("whisper_model", "small")

    try:
        if engine == "piper":
            generate_piper(text, voice, ogg_path)
        else:
            generate_kokoro(text, voice, ogg_path, config)

        notify("Syncing words…", ogg_path.name)
        words = get_word_timestamps(ogg_path, whisper_model)

        generate_player(words, engine, voice, ogg_path, html_path)
        notify("Karaoke Ready", ogg_path.stem)
        subprocess.run(["open", str(html_path)], check=False)
    except Exception as e:
        notify("Speech Failed", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

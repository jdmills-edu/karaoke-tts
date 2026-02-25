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
      height: 100vh;
      overflow: hidden;
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
    #playbtn {
      background: none; border: none; cursor: pointer;
      color: #f5c518; padding: 0; flex-shrink: 0;
      display: flex; align-items: center;
    }
    #playbtn:hover { color: #fff; }
    #playbtn svg { width: 28px; height: 28px; }
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
    <button id="playbtn" title="Play / Pause">
      <svg id="ico" viewBox="0 0 24 24" fill="currentColor">
        <path id="play-shape" d="M5,3 L19,12 L5,21 Z"/>
      </svg>
    </button>
    <div id="track"><div id="fill"></div></div>
    <span id="time">0:00</span>
  </div>
  <audio id="a" src="__AUDIO__"></audio>
  <script>
    const W = __WORDS__;
    const txt = document.getElementById("text");
    W.forEach((w, i) => {
      if (w.break) for (let b = 0; b < w.break; b++) txt.appendChild(document.createElement("br"));
      const s = document.createElement("span");
      s.className = "w"; s.id = "w" + i; s.textContent = w.word;
      txt.appendChild(s);
    });
    const audio   = document.getElementById("a");
    const fill    = document.getElementById("fill");
    const timeEl  = document.getElementById("time");
    const durEl   = document.getElementById("dur");
    const playbtn = document.getElementById("playbtn");
    const shape   = document.getElementById("play-shape");
    const spans   = W.map((_, i) => document.getElementById("w" + i));

    const PLAY_PATH  = "M5,3 L19,12 L5,21 Z";
    const PAUSE_PATH = "M6,3 L10,3 L10,21 L6,21 Z M14,3 L18,3 L18,21 L14,21 Z";

    function fmt(s) {
      return Math.floor(s / 60) + ":" + String(Math.floor(s % 60)).padStart(2, "0");
    }
    function updateBtn() {
      shape.setAttribute("d", audio.paused ? PLAY_PATH : PAUSE_PATH);
    }
    playbtn.addEventListener("click", () => {
      audio.paused ? audio.play() : audio.pause();
    });
    audio.addEventListener("play",  updateBtn);
    audio.addEventListener("pause", updateBtn);
    audio.addEventListener("ended", updateBtn);
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

def sanitize_text(text: str) -> str:
    """Normalize text for safe Kokoro/espeak-ng processing.

    Collapses whitespace/newlines and replaces Unicode punctuation and
    special characters that can cause native library crashes.
    """
    replacements = [
        # Curly quotes → straight quotes
        ("\u2018", "'"), ("\u2019", "'"),  # ' '
        ("\u201c", '"'), ("\u201d", '"'),  # " "
        # Dashes → comma or hyphen
        ("\u2014", ", "),  # em dash —
        ("\u2013", "-"),   # en dash –
        # Ellipsis
        ("\u2026", "..."),
        # Other common problem chars
        ("\u00b7", "."),   # middle dot ·
        ("\u2022", "."),   # bullet •
        ("\u00a0", " "),   # non-breaking space
        ("\u200b", ""),    # zero-width space
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    # Strip any remaining non-ASCII characters that aren't handled above
    text = text.encode("ascii", errors="ignore").decode("ascii")
    # Normalize line endings and tabs
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    # Collapse 3+ consecutive newlines to a double newline (paragraph break).
    # Double newlines (\n\n) are preserved so downstream code can render
    # paragraph spacing.  Single newlines are kept as-is.
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


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

def generate_kokoro(
    text: str, voice: str, out_path: Path, config: dict,
    _log=None,
) -> None:
    from kokoro_onnx import Kokoro
    k = config["kokoro"]
    kokoro = Kokoro(str(resolve(k["model_path"])), str(resolve(k["voices_path"])))
    lang, speed = k.get("lang", "en-us"), k.get("speed", 1.0)
    text = sanitize_text(text)
    chunks = chunk_text(text)
    if _log:
        _log(f"kokoro: {len(chunks)} chunks, sizes={[len(c) for c in chunks]}")
    # Stream-write each chunk directly to disk to avoid np.concatenate on a
    # large in-memory buffer, which can trigger heap corruption in the ONNX runtime.
    out_file = None
    try:
        for i, chunk in enumerate(chunks):
            if _log:
                _log(f"kokoro: chunk {i+1}/{len(chunks)} preview={repr(chunk[:80])}")
            samples, sr = kokoro.create(chunk, voice=voice, speed=speed, lang=lang)
            if _log:
                _log(f"kokoro: chunk {i+1} done, {len(samples)} samples")
            if out_file is None:
                out_file = sf.SoundFile(
                    str(out_path), mode="w", samplerate=sr, channels=1
                )
            out_file.write(samples)
    finally:
        if out_file is not None:
            out_file.close()


# ---------------------------------------------------------------------------
# Word timestamps
# ---------------------------------------------------------------------------

def get_word_timestamps(audio_path: Path, model_size: str, _log=None) -> list[dict]:
    import gc
    gc.collect()  # free Kokoro/ONNX memory before loading Whisper
    from faster_whisper import WhisperModel
    if _log:
        _log("whisper: loading model")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    if _log:
        _log("whisper: transcribing")
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
    if _log:
        _log(f"whisper: done, {len(words)} words")
    return words


# ---------------------------------------------------------------------------
# HTML player
# ---------------------------------------------------------------------------

def generate_player(
    words: list[dict], voice: str, ogg_path: Path, html_path: Path
) -> None:
    html = (
        PLAYER_TEMPLATE
        .replace("__ENGINE__", "KOKORO")
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
    voice        = params["voice"]
    ogg_path     = Path(params["ogg_path"])
    html_path    = Path(params["html_path"])
    config       = params["config"]
    whisper_model = config.get("whisper_model", "small")

    log_path = ogg_path.with_suffix(".worker.log")

    def log(msg: str) -> None:
        with open(log_path, "a") as f:
            f.write(msg + "\n")
            f.flush()

    log(f"worker started: voice={voice} text_len={len(text)}")

    try:
        log("step 1: TTS synthesis")
        log(f"step 1: text preview: {repr(text[:200])}")
        generate_kokoro(text, voice, ogg_path, config, _log=log)
        log(f"step 1 done: ogg_path={ogg_path} size={ogg_path.stat().st_size if ogg_path.exists() else 'MISSING'}")

        log("step 2: word timestamps")
        words = get_word_timestamps(ogg_path, whisper_model, _log=log)
        log(f"step 2 done: {len(words)} words")

        log("step 3: generate player")
        generate_player(words, voice, ogg_path, html_path)
        log("step 3 done: opening browser")
        subprocess.run(["open", str(html_path)], check=False)
        log("done")
    except Exception as e:
        import traceback
        log(f"EXCEPTION: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()

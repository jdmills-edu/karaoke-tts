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
  <title>__TITLE__</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%230d0d1a'/><g transform='translate(4,3) scale(0.75)'><path d='M16 2a5 5 0 0 0-5 5v8a5 5 0 0 0 10 0V7a5 5 0 0 0-5-5z' fill='%23f5c518'/><path d='M8 14v1a8 8 0 0 0 16 0v-1' stroke='%23f5c518' stroke-width='2' fill='none' stroke-linecap='round'/><line x1='16' y1='23' x2='16' y2='27' stroke='%23f5c518' stroke-width='2' stroke-linecap='round'/><line x1='12' y1='27' x2='20' y2='27' stroke='%23f5c518' stroke-width='2' stroke-linecap='round'/></g></svg>">
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
      justify-content: center;
      align-items: center;
      border-bottom: 1px solid #1a1a2e;
      flex-shrink: 0;
    }
    #header:empty { display: none; }
    #title {
      font-size: 16px;
      font-weight: 600;
      color: #2a2a52;
      letter-spacing: 0.01em;
    }
    #badge {
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.12em;
      color: #44447a;
      flex-shrink: 0;
    }
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
  <div id="header">__HEADER_CONTENT__</div>
  <div id="scroll"><div id="text"></div></div>
  <div id="footer">
    <button id="playbtn" title="Play / Pause">
      <svg id="ico" viewBox="0 0 24 24" fill="currentColor">
        <path id="play-shape" d="M5,3 L19,12 L5,21 Z"/>
      </svg>
    </button>
    <div id="track"><div id="fill"></div></div>
    <span id="time">0:00</span>
    <span id="badge">__ENGINE__ · __VOICE__</span>
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
    """Normalize text for Kokoro TTS processing.

    Handles whitespace/newline normalization and replaces a small set of
    Unicode characters that Kokoro cannot process natively.  Most
    punctuation and special characters (hyphens, %, $, &, etc.) are left
    intact — Kokoro handles them correctly.
    """
    replacements = [
        # Non-breaking / zero-width spaces
        ("\u00a0", " "),   # non-breaking space
        ("\u200b", ""),    # zero-width space
        # Bullets → period (not speakable)
        ("\u00b7", "."),   # middle dot ·
        ("\u2022", "."),   # bullet •
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    # Normalize line endings and tabs
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    # Normalize newlines: only preserve \n\n as a paragraph break when it
    # follows sentence-ending punctuation (.!?).  All other newlines
    # (including double newlines mid-sentence) are collapsed to spaces —
    # they're usually hard wraps from the source, not intentional breaks.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=[.!?])\n\n", "\x00PARA\x00", text)  # protect real breaks
    text = re.sub(r"\n", " ", text)                           # collapse all others
    text = text.replace("\x00PARA\x00", "\n\n")               # restore real breaks
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
# Whisper → source text alignment
# ---------------------------------------------------------------------------

def _clean(word: str) -> str:
    """Lowercase and strip non-alphanumeric chars for fuzzy comparison."""
    return re.sub(r"[^a-z0-9]", "", word.lower())


def _similarity(a: str, b: str) -> float:
    """Normalized character-level similarity (0.0–1.0) between two words."""
    a, b = _clean(a), _clean(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Optimistic short-circuit for exact match
    if a == b:
        return 1.0
    # Use ratio of longest common subsequence to max length.
    # LCS is more forgiving than Levenshtein for partial matches
    # (e.g. Whisper hearing "economy" as "economies").
    m, n = len(a), len(b)
    # Space-efficient LCS length via two rows
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs_len = prev[n]
    return lcs_len / max(m, n)


def align_words(
    source_text: str,
    whisper_words: list[dict],
    _log=None,
) -> list[dict]:
    """Align Whisper-detected words against known source text.

    Keeps the source words (ground truth) but uses Whisper's timing.
    Uses Needleman-Wunsch global sequence alignment with character-level
    similarity scoring.

    Args:
        source_text: The original text that was synthesized.
        whisper_words: Word dicts from Whisper with 'word', 'start', 'end'.

    Returns:
        Aligned word dicts using source words with Whisper timings.
    """
    # Tokenize source text into words (preserving order)
    src_tokens = source_text.split()
    if not src_tokens or not whisper_words:
        return whisper_words  # nothing to align

    whisper_tokens = [w["word"].strip() for w in whisper_words]

    n_src = len(src_tokens)
    n_wh = len(whisper_tokens)

    # --- Needleman-Wunsch alignment ---
    MATCH_BONUS = 2.0      # reward for high-similarity alignment
    MISMATCH_PENALTY = -1.0  # penalty for low-similarity alignment
    GAP_PENALTY = -0.5     # penalty for skipping a word
    SIM_THRESHOLD = 0.4    # minimum similarity to count as a match

    # Build score matrix
    dp = [[0.0] * (n_wh + 1) for _ in range(n_src + 1)]
    for i in range(1, n_src + 1):
        dp[i][0] = dp[i - 1][0] + GAP_PENALTY
    for j in range(1, n_wh + 1):
        dp[0][j] = dp[0][j - 1] + GAP_PENALTY

    for i in range(1, n_src + 1):
        for j in range(1, n_wh + 1):
            sim = _similarity(src_tokens[i - 1], whisper_tokens[j - 1])
            if sim >= SIM_THRESHOLD:
                score = MATCH_BONUS * sim
            else:
                score = MISMATCH_PENALTY
            diag = dp[i - 1][j - 1] + score
            up = dp[i - 1][j] + GAP_PENALTY    # skip source word
            left = dp[i][j - 1] + GAP_PENALTY  # skip whisper word
            dp[i][j] = max(diag, up, left)

    # Traceback
    pairs = []  # (src_idx | None, whisper_idx | None)
    i, j = n_src, n_wh
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sim = _similarity(src_tokens[i - 1], whisper_tokens[j - 1])
            if sim >= SIM_THRESHOLD:
                score = MATCH_BONUS * sim
            else:
                score = MISMATCH_PENALTY
            if dp[i][j] == dp[i - 1][j - 1] + score:
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + GAP_PENALTY:
            pairs.append((i - 1, None))  # source word, no Whisper match
            i -= 1
        else:
            pairs.append((None, j - 1))  # Whisper hallucination, no source
            j -= 1

    pairs.reverse()

    # --- Build aligned output ---
    # First pass: collect matched pairs, unmatched source words, and
    # unmatched Whisper words (with their timings for redistribution).
    aligned = []       # output list
    discarded = []     # (output_position, whisper_words[wh_idx]) for unmatched Whisper
    for src_idx, wh_idx in pairs:
        if src_idx is not None and wh_idx is not None:
            sim = _similarity(src_tokens[src_idx], whisper_tokens[wh_idx])
            if sim >= SIM_THRESHOLD:
                # Good match: source word + Whisper timing
                w = whisper_words[wh_idx].copy()
                w["word"] = " " + src_tokens[src_idx]
                aligned.append(w)
            else:
                # Low-quality diagonal: treat source as unmatched,
                # Whisper as discarded (timing available for redistribution)
                aligned.append({
                    "word": " " + src_tokens[src_idx],
                    "start": None,
                    "end": None,
                })
                discarded.append((len(aligned), whisper_words[wh_idx]))
        elif src_idx is not None and wh_idx is None:
            # Source word that Whisper missed — timing filled in below
            aligned.append({
                "word": " " + src_tokens[src_idx],
                "start": None,
                "end": None,
            })
        else:
            # Unmatched Whisper word — save its timing for redistribution
            discarded.append((len(aligned), whisper_words[wh_idx]))

    # Second pass: redistribute discarded Whisper timings to nearby
    # unmatched source words.  This handles cases like Whisper collapsing
    # "twenty twenty-two" into "2022" — the source words are unmatched
    # and the Whisper word is discarded, but they occupy the same time.
    for insert_pos, wh_word in discarded:
        # Find the nearest unmatched word to insert_pos, searching outward
        search_pos = min(insert_pos, len(aligned) - 1)
        found = None
        for offset in range(len(aligned)):
            for candidate in [search_pos - offset, search_pos + offset]:
                if 0 <= candidate < len(aligned) and aligned[candidate]["start"] is None:
                    found = candidate
                    break
            if found is not None:
                break
        if found is None:
            continue
        # Expand to the full contiguous span of unmatched words
        span_start = found
        while span_start > 0 and aligned[span_start - 1]["start"] is None:
            span_start -= 1
        span_end = found
        while span_end < len(aligned) - 1 and aligned[span_end + 1]["start"] is None:
            span_end += 1
        # Distribute this Whisper word's time range across the span
        count = span_end - span_start + 1
        wh_start = wh_word["start"]
        wh_end = wh_word["end"]
        duration = (wh_end - wh_start) / count
        for g in range(count):
            aligned[span_start + g]["start"] = round(wh_start + g * duration, 3)
            aligned[span_start + g]["end"] = round(wh_start + (g + 1) * duration, 3)

    # Third pass: interpolate any remaining unmatched source words from
    # their timed neighbors.
    for k in range(len(aligned)):
        if aligned[k]["start"] is not None:
            continue
        prev_end = None
        for p in range(k - 1, -1, -1):
            if aligned[p]["end"] is not None:
                prev_end = aligned[p]["end"]
                break
        next_start = None
        for p in range(k + 1, len(aligned)):
            if aligned[p]["start"] is not None:
                next_start = aligned[p]["start"]
                break
        if prev_end is not None and next_start is not None:
            gap_start = k
            while gap_start > 0 and aligned[gap_start - 1]["start"] is None:
                gap_start -= 1
            gap_end = k
            while gap_end < len(aligned) - 1 and aligned[gap_end + 1]["start"] is None:
                gap_end += 1
            gap_count = gap_end - gap_start + 1
            duration = (next_start - prev_end) / gap_count
            for g in range(gap_count):
                aligned[gap_start + g]["start"] = round(prev_end + g * duration, 3)
                aligned[gap_start + g]["end"] = round(prev_end + (g + 1) * duration, 3)
        elif prev_end is not None:
            aligned[k]["start"] = prev_end
            aligned[k]["end"] = round(prev_end + 0.2, 3)
        elif next_start is not None:
            aligned[k]["start"] = round(max(0, next_start - 0.2), 3)
            aligned[k]["end"] = next_start
        else:
            aligned[k]["start"] = 0.0
            aligned[k]["end"] = 0.2

    if _log:
        _log(f"align: {n_src} source, {n_wh} whisper → {len(aligned)} aligned")

    return aligned


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
    words: list[dict], voice: str, ogg_path: Path, html_path: Path,
    title: str = "",
) -> None:
    page_title = title if title else "karaoke-tts"
    header_content = f'<span id="title">{title}</span>' if title else ""
    html = (
        PLAYER_TEMPLATE
        .replace("__TITLE__", page_title)
        .replace("__HEADER_CONTENT__", header_content)
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
    title        = params.get("title", "")
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

        log("step 2b: align words to source text")
        words = align_words(sanitize_text(text), words, _log=log)
        log(f"step 2b done: {len(words)} aligned words")

        log("step 3: generate player")
        generate_player(words, voice, ogg_path, html_path, title=title)
        log("step 3 done: opening browser")
        subprocess.run(["open", str(html_path)], check=False)
        log("done")
    except Exception as e:
        import traceback
        log(f"EXCEPTION: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()

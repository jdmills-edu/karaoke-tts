#!/usr/bin/env python3
"""
karaoke-tts streaming worker

Runs a local HTTP + WebSocket server that streams Kokoro TTS audio chunks
to a browser-based karaoke player in real time.  Audio starts playing in
the browser within seconds while synthesis continues in the background.

After each chunk is synthesized, Whisper immediately refines that chunk's
word timings and sends the refinement to the browser — no waiting for full
synthesis to complete.  Both Kokoro and Whisper are loaded simultaneously
(~770MB combined).  An archival OGG + HTML pair is saved at the end.

Usage: python streaming_worker.py <params_json_file>
"""

import asyncio
import io
import json
import re
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# Reuse helpers from the existing worker
from worker import (
    PLAYER_TEMPLATE,
    align_words,
    generate_player,
    resolve,
    sanitize_text,
)


def chunk_text_streaming(text: str, max_chars: int = 200) -> list[str]:
    """Split text into chunks at sentence boundaries, preserving newlines.

    Unlike worker.chunk_text, the separators between sentences (including
    ``\\n``) are kept so that ``estimate_word_timings`` can detect
    paragraph breaks and pass them to the frontend.
    """
    # Split into alternating [sentence, separator, sentence, separator, ...]
    # The capture group causes re.split to include the matched separators.
    parts = re.split(r"((?<=[.!?])\s+)", text.strip())

    chunks: list[str] = []
    current = ""
    last_sep = ""
    for i, part in enumerate(parts):
        if i % 2 == 1:
            # separator — append to current chunk and remember it
            last_sep = part
            current += part
        else:
            # sentence text
            candidate = current + part
            if len(candidate) > max_chars and current.strip():
                chunks.append(current.rstrip())
                # If the separator that led here contained newlines,
                # prefix the new chunk so the breaks are detectable
                nl_count = last_sep.count("\n")
                prefix = "\n" * nl_count if nl_count > 0 else ""
                current = prefix + part
            else:
                current = candidate
    if current.strip():
        chunks.append(current.rstrip())
    return chunks or [text]

# ---------------------------------------------------------------------------
# Streaming player HTML template
# ---------------------------------------------------------------------------
STREAMING_PLAYER_HTML = """<!DOCTYPE html>
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
      line-height: 1.6;
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
      color: #f5c518;
      letter-spacing: 0.01em;
      transition: color 0.6s ease;
    }
    #title.dimmed { color: #2a2a52; }
    #badge {
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.12em;
      color: #44447a;
      flex-shrink: 0;
    }
    #status {
      font-size: 12px;
      color: #44447a;
      font-variant-numeric: tabular-nums;
      transition: opacity 0.3s;
      flex-shrink: 0;
    }
    #dur { font-size: 13px; color: #44447a; font-variant-numeric: tabular-nums; }
    #scroll { flex: 1; overflow-y: auto; padding: 60px; }
    #text br { display: block; content: ""; margin-top: 0.3em; }
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
      display: none; align-items: center;
    }
    #playbtn.ready { display: flex; }
    #playbtn:hover { color: #fff; }
    #playbtn svg { width: 28px; height: 28px; }
    #spinner {
      flex-shrink: 0;
      width: 28px; height: 28px;
      display: flex; align-items: center;
    }
    #spinner.hidden { display: none; }
    #spinner svg { width: 28px; height: 28px; }
    @keyframes spin { to { transform: rotate(360deg); } }
    #spinner svg { animation: spin 1s linear infinite; }
    #track {
      flex: 1; height: 3px; background: #1a1a2e;
      border-radius: 2px; cursor: pointer;
    }
    #fill {
      height: 100%; background: #f5c518; border-radius: 2px;
      width: 0%; transition: width 0.05s linear; pointer-events: none;
    }
    #time {
      font-size: 13px; color: #44447a;
      font-variant-numeric: tabular-nums; min-width: 72px; text-align: right;
    }
  </style>
</head>
<body>
  <div id="header">__HEADER_CONTENT__</div>
  <div id="scroll"><div id="text"></div></div>
  <div id="footer">
    <div id="spinner">
      <svg viewBox="0 0 24 24" fill="none" stroke="#44447a" stroke-width="2.5" stroke-linecap="round">
        <path d="M12 2 A10 10 0 1 1 2 12" />
      </svg>
    </div>
    <button id="playbtn" title="Play / Pause">
      <svg viewBox="0 0 24 24" fill="currentColor">
        <path id="play-shape" d="M5,3 L19,12 L5,21 Z"/>
      </svg>
    </button>
    <div id="track"><div id="fill"></div></div>
    <span id="time">0:00</span>
    <span id="status">connecting...</span>
    <span id="badge">__BADGE__</span>
  </div>
  <script>
    // --- State ---
    let audioCtx = null;
    const allBuffers = [];   // {buffer, offset, duration}
    let allWords = [];
    let spans = [];
    const chunkRanges = [];  // [{start, count}] — word index ranges per chunk
    let totalDuration = 0;
    let synthDone = false;
    let isPlaying = false;
    let startedAt = 0;       // audioCtx.currentTime when playback started
    let pausedAt = 0;        // playback offset when paused
    let lastIdx = -1;
    let activeSources = [];  // scheduled AudioBufferSourceNodes
    let pendingMeta = null;

    const STARTUP_DELAY = 0.15; // seconds — lets audio pipeline warm up

    const PLAY_PATH  = "M5,3 L19,12 L5,21 Z";
    const PAUSE_PATH = "M6,3 L10,3 L10,21 L6,21 Z M14,3 L18,3 L18,21 L14,21 Z";

    const txt      = document.getElementById("text");
    const fill     = document.getElementById("fill");
    const timeEl   = document.getElementById("time");
    const statusEl = document.getElementById("status");
    const playbtn  = document.getElementById("playbtn");
    const shape    = document.getElementById("play-shape");

    function fmt(s) {
      if (s < 0) s = 0;
      return Math.floor(s / 60) + ":" + String(Math.floor(s % 60)).padStart(2, "0");
    }

    function updateBtn() {
      shape.setAttribute("d", isPlaying ? PAUSE_PATH : PLAY_PATH);
    }

    // --- AudioContext bootstrap ---
    function ensureAudioCtx() {
      if (!audioCtx) audioCtx = new AudioContext();
    }

    async function tryResumeCtx() {
      ensureAudioCtx();
      if (audioCtx.state === "suspended") {
        try { await audioCtx.resume(); } catch(e) {}
      }
      return audioCtx.state === "running";
    }

    playbtn.addEventListener("click", async () => {
      await tryResumeCtx();
      if (isPlaying) {
        pausePlayback();
      } else {
        startPlayback();
      }
    });

    // --- Playback control ---
    function stopAllSources() {
      activeSources.forEach(s => { try { s.stop(); } catch(e) {} });
      activeSources = [];
    }

    function scheduleAll(fromTime) {
      stopAllSources();
      if (!audioCtx) return;
      const now = audioCtx.currentTime;
      allBuffers.forEach(({buffer, offset, duration}) => {
        const chunkEnd = offset + duration;
        if (chunkEnd <= fromTime) return;
        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(audioCtx.destination);
        const skipInto = Math.max(0, fromTime - offset);
        const when = now + (offset + skipInto - fromTime);
        source.start(Math.max(when, now), skipInto);
        activeSources.push(source);
      });
    }

    function startPlayback() {
      if (!audioCtx || audioCtx.state !== "running") return;
      if (allBuffers.length === 0) return; // nothing to play yet
      startedAt = audioCtx.currentTime - pausedAt + STARTUP_DELAY;
      isPlaying = true;
      scheduleAll(pausedAt);
      updateBtn();
      requestAnimationFrame(animLoop);
    }

    function pausePlayback() {
      if (!audioCtx) return;
      pausedAt = Math.max(0, audioCtx.currentTime - startedAt);
      isPlaying = false;
      stopAllSources();
      audioCtx.suspend();
      updateBtn();
    }

    // --- Progress bar seeking ---
    document.getElementById("track").addEventListener("click", async e => {
      if (totalDuration <= 0) return;
      await tryResumeCtx();
      const r = e.currentTarget.getBoundingClientRect();
      const seekTo = ((e.clientX - r.left) / r.width) * totalDuration;
      pausedAt = Math.max(0, Math.min(seekTo, totalDuration));
      if (isPlaying) {
        startedAt = audioCtx.currentTime - pausedAt;
        scheduleAll(pausedAt);
      }
      lastIdx = -1;
    });

    // --- Karaoke animation loop ---
    function animLoop() {
      if (!isPlaying) return;
      const t = audioCtx.currentTime - startedAt;

      if (totalDuration > 0) {
        fill.style.width = (Math.min(Math.max(t, 0) / totalDuration, 1) * 100) + "%";
      }
      timeEl.textContent = fmt(t);

      // Binary search for active word
      let lo = 0, hi = allWords.length - 1, idx = -1;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if      (allWords[mid].end   <= t) lo = mid + 1;
        else if (allWords[mid].start >  t) hi = mid - 1;
        else { idx = mid; break; }
      }
      if (idx !== lastIdx) {
        if (lastIdx >= 0 && lastIdx < spans.length) spans[lastIdx].className = "w past";
        if (idx >= 0 && idx < spans.length) {
          spans[idx].className = "w active";
          spans[idx].scrollIntoView({ behavior: "smooth", block: "center" });
        }
        lastIdx = idx;
      }

      // Auto-stop at end
      if (synthDone && t >= totalDuration + 0.5) {
        pausePlayback();
        pausedAt = 0;
        spans.forEach(s => s.className = "w");
        lastIdx = -1;
        return;
      }

      requestAnimationFrame(animLoop);
    }

    // --- Word display ---
    function insertBreaks(n) {
      for (let i = 0; i < (n || 0); i++) txt.appendChild(document.createElement("br"));
    }

    function addWords(words) {
      const startIdx = allWords.length;
      words.forEach(w => {
        allWords.push(w);
        if (w.break) insertBreaks(w.break);
        const s = document.createElement("span");
        s.className = "w";
        s.id = "w" + (spans.length);
        s.textContent = w.word;
        txt.appendChild(s);
        spans.push(s);
      });
      chunkRanges.push({start: startIdx, count: words.length});
    }

    function rebuildDOM() {
      const t = isPlaying ? audioCtx.currentTime - startedAt : pausedAt;
      txt.innerHTML = "";
      spans = [];
      allWords.forEach((w, i) => {
        if (w.break) insertBreaks(w.break);
        const s = document.createElement("span");
        s.className = w.end <= t ? "w past" : "w";
        s.id = "w" + i;
        s.textContent = w.word;
        txt.appendChild(s);
        spans.push(s);
      });
      lastIdx = -1;
    }

    function handleChunkRefined(msg) {
      const range = chunkRanges[msg.chunkIndex];
      if (!range) return;
      const oldCount = range.count;
      const newCount = msg.words.length;

      // Splice refined words into allWords
      allWords.splice(range.start, oldCount, ...msg.words);

      // Update chunk ranges
      const diff = newCount - oldCount;
      range.count = newCount;
      for (let i = msg.chunkIndex + 1; i < chunkRanges.length; i++) {
        chunkRanges[i].start += diff;
      }

      rebuildDOM();
    }

    // --- WebSocket ---
    const wsUrl = "ws://" + location.host + "/ws";
    let ws;

    function connect() {
      ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        statusEl.textContent = "connected";
        ws.send(JSON.stringify({type: "ready"}));
        ensureAudioCtx();
      };

      ws.onmessage = async (event) => {
        if (typeof event.data === "string") {
          const msg = JSON.parse(event.data);
          if (msg.type === "audio_chunk") {
            pendingMeta = msg;
            statusEl.textContent = "chunk " + (msg.chunkIndex + 1) + "/" + msg.totalChunks;
            totalDuration = msg.globalOffset + msg.duration;
          } else if (msg.type === "chunk_refined") {
            handleChunkRefined(msg);
            statusEl.textContent = "chunk " + (msg.chunkIndex + 1) + " refined";
          } else if (msg.type === "synthesis_complete") {
            totalDuration = msg.totalDuration;
            synthDone = true;
            statusEl.textContent = "done";
            setTimeout(() => { statusEl.style.opacity = "0"; }, 2000);
          }
        } else {
          // Binary WAV data
          if (!pendingMeta) return;
          ensureAudioCtx();
          try {
            const audioBuffer = await audioCtx.decodeAudioData(event.data.slice(0));
            onChunkDecoded(audioBuffer, pendingMeta);
          } catch (err) {
            console.error("Failed to decode audio chunk:", err);
          }
          pendingMeta = null;
        }
      };

      ws.onclose = () => {
        if (!synthDone) statusEl.textContent = "connection lost";
      };

      ws.onerror = () => {
        statusEl.textContent = "connection error";
      };
    }

    function onChunkDecoded(buffer, meta) {
      allBuffers.push({
        buffer: buffer,
        offset: meta.globalOffset,
        duration: meta.duration,
      });

      // Show play button once first chunk is ready
      if (allBuffers.length === 1) {
        document.getElementById("spinner").classList.add("hidden");
        playbtn.classList.add("ready");
        const titleEl = document.getElementById("title");
        if (titleEl) titleEl.classList.add("dimmed");
      }

      // Add word spans
      addWords(meta.words);

      // If already playing, schedule this new chunk immediately
      if (isPlaying && audioCtx) {
        const t = audioCtx.currentTime - startedAt;
        if (meta.globalOffset + meta.duration > t) {
          const source = audioCtx.createBufferSource();
          source.buffer = buffer;
          source.connect(audioCtx.destination);
          const skipInto = Math.max(0, t - meta.globalOffset);
          const when = audioCtx.currentTime + (meta.globalOffset + skipInto - t);
          source.start(Math.max(when, audioCtx.currentTime), skipInto);
          activeSources.push(source);
        }
      }
    }

    connect();
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Audio encoding
# ---------------------------------------------------------------------------

def samples_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 samples as a PCM-16 WAV file in memory."""
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Word timing estimation
# ---------------------------------------------------------------------------

def estimate_word_timings(
    chunk_text: str,
    chunk_duration: float,
    global_offset: float,
) -> list[dict]:
    """Estimate word-level timings proportionally from text and audio duration.

    Uses character counts with punctuation-pause bonuses as a proxy for
    phoneme counts.  Accuracy is typically within 200-400ms per word.

    Words that follow a newline in the source text get ``"break": True``
    so the frontend can insert a visible line break.
    """
    # Find each word and how many newlines precede it
    tokens = []  # (word_str, newline_count)
    pos = 0
    for m in re.finditer(r"\S+", chunk_text):
        gap = chunk_text[pos:m.start()]
        nl_count = gap.count("\n")
        tokens.append((m.group(), nl_count))
        pos = m.end()

    if not tokens:
        return []

    PAUSE_BONUS = 6
    COMMA_BONUS = 3
    weights = []
    for w, _ in tokens:
        weight = len(w)
        if w[-1] in ".!?":
            weight += PAUSE_BONUS
        elif w[-1] in ",;:":
            weight += COMMA_BONUS
        weights.append(weight)

    total_weight = sum(weights)
    if total_weight == 0:
        return []

    words = []
    t = global_offset
    for (w, nl_count), weight in zip(tokens, weights):
        word_duration = (weight / total_weight) * chunk_duration
        entry = {
            "word": " " + w,
            "start": round(t, 3),
            "end": round(t + word_duration, 3),
        }
        if nl_count > 0:
            entry["break"] = nl_count  # 1 = line break, 2 = paragraph break
        words.append(entry)
        t += word_duration

    return words


# ---------------------------------------------------------------------------
# Starlette / Uvicorn server
# ---------------------------------------------------------------------------

async def run_streaming_server(params: dict) -> None:
    """Start HTTP + WebSocket server, run synthesis, stream to browser."""
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocket, WebSocketDisconnect

    import uvicorn

    text = params["text"]
    voice = params["voice"]
    title = params.get("title", "")
    config = params["config"]
    ogg_path = Path(params["ogg_path"])
    html_path = Path(params["html_path"])

    log_path = ogg_path.with_suffix(".worker.log")

    def log(msg: str) -> None:
        with open(log_path, "a") as f:
            f.write(msg + "\n")
            f.flush()

    log(f"streaming worker started: voice={voice} text_len={len(text)}")

    # Title and badge for the player header
    page_title = title if title else "karaoke-tts streaming"
    header_content = f'<span id="title">{title}</span>' if title else ""
    badge = f"KOKORO \u00b7 {voice.upper()} \u00b7 STREAMING"
    player_html = (
        STREAMING_PLAYER_HTML
        .replace("__TITLE__", page_title)
        .replace("__HEADER_CONTENT__", header_content)
        .replace("__BADGE__", badge)
    )

    # Shared state between routes and synthesis
    client_ws: list[WebSocket | None] = [None]
    client_connected = asyncio.Event()
    synthesis_complete = asyncio.Event()
    all_done = asyncio.Event()
    loop = asyncio.get_event_loop()

    async def homepage(request):
        return HTMLResponse(player_html)

    async def websocket_handler(ws: WebSocket):
        await ws.accept()
        client_ws[0] = ws
        client_connected.set()
        log("client connected")

        try:
            while not all_done.is_set():
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=1.0)
                    data = json.loads(raw)
                    if data.get("type") == "close":
                        break
                except asyncio.TimeoutError:
                    continue
                except WebSocketDisconnect:
                    break
                except Exception:
                    break
        except Exception:
            pass
        finally:
            log("client disconnected")
            client_ws[0] = None
            all_done.set()

    app = Starlette(routes=[
        Route("/", homepage),
        WebSocketRoute("/ws", websocket_handler),
    ])

    # Find a free port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    log(f"server binding to 127.0.0.1:{port}")

    uvi_config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning",
    )
    server = uvicorn.Server(uvi_config)

    async def run_server():
        await server.serve()

    async def run_pipeline():
        """Wait for client, synthesize, refine, save archival."""
        # Open browser
        subprocess.run(["open", f"http://127.0.0.1:{port}"], check=False)

        # Wait for the browser to connect
        try:
            await asyncio.wait_for(client_connected.wait(), timeout=30)
        except asyncio.TimeoutError:
            log("client never connected, shutting down")
            server.should_exit = True
            return

        # --- Stage 1: Streaming synthesis with incremental Whisper ---
        log("stage 1: streaming synthesis + per-chunk whisper refinement")
        try:
            all_words = await asyncio.get_event_loop().run_in_executor(
                None, _blocking_synthesis, params, client_ws, loop, log,
            )
        except Exception as e:
            import traceback
            log(f"synthesis error: {e}\n{traceback.format_exc()}")
            all_done.set()
            server.should_exit = True
            return

        synthesis_complete.set()
        log("stage 1 done: synthesis + refinement complete")

        # --- Stage 2: Save archival files ---
        # No separate full-file Whisper pass needed — per-chunk refinement
        # already produced accurate word timings during synthesis.
        log("stage 2: archival output")
        generate_player(all_words, voice, ogg_path, html_path, title=title)
        log(f"archival HTML saved: {html_path}")

        ws = client_ws[0]
        if ws:
            await ws.send_json({
                "type": "archival_ready",
                "htmlPath": str(html_path),
                "oggPath": str(ogg_path),
            })

        log("done — waiting for client disconnect or timeout")

        # Wait for client to disconnect or timeout
        try:
            await asyncio.wait_for(all_done.wait(), timeout=300)
        except asyncio.TimeoutError:
            log("idle timeout, shutting down")

        server.should_exit = True

    # Run server and pipeline concurrently
    server_task = asyncio.create_task(run_server())
    pipeline_task = asyncio.create_task(run_pipeline())

    await asyncio.gather(server_task, pipeline_task, return_exceptions=True)
    log("server shut down")


def _blocking_synthesis(
    params: dict,
    client_ws: list,
    loop: asyncio.AbstractEventLoop,
    log,
) -> list[dict]:
    """Run Kokoro synthesis with incremental Whisper refinement.

    For each chunk: synthesize with Kokoro, send estimated timings + audio
    to the browser, then immediately run Whisper on that chunk's audio and
    send refined timings.  Both models are loaded simultaneously (~770MB).

    Returns the list of all word timings (refined where Whisper succeeded,
    estimated where it failed).
    """
    import tempfile

    from kokoro_onnx import Kokoro
    from faster_whisper import WhisperModel

    config = params["config"]
    voice = params["voice"]
    ogg_path = Path(params["ogg_path"])
    whisper_model_size = config.get("whisper_model", "small")

    k = config["kokoro"]
    kokoro = Kokoro(str(resolve(k["model_path"])), str(resolve(k["voices_path"])))
    lang, speed = k.get("lang", "en-us"), k.get("speed", 1.0)

    log(f"loading whisper model ({whisper_model_size}) for incremental refinement...")
    whisper_model = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
    log("both models loaded")

    text = sanitize_text(params["text"])
    chunks = chunk_text_streaming(text, max_chars=200)
    log(f"kokoro: {len(chunks)} chunks, sizes={[len(c) for c in chunks]}")

    global_offset = 0.0
    all_words = []
    chunk_word_ranges = []  # [(start_idx, count)] for each chunk
    out_file = None
    INTER_CHUNK_PAUSE = 0.25  # seconds

    try:
        for i, chunk in enumerate(chunks):
            log(f"kokoro: chunk {i+1}/{len(chunks)} preview={repr(chunk[:80])}")
            samples, sr = kokoro.create(chunk, voice=voice, speed=speed, lang=lang)
            chunk_duration = len(samples) / sr
            raw_samples = samples  # keep original without silence for Whisper
            log(f"kokoro: chunk {i+1} done, {len(samples)} samples, {chunk_duration:.2f}s")

            # Append a short silence after every chunk (except the last)
            if i < len(chunks) - 1:
                pause_samples = np.zeros(int(sr * INTER_CHUNK_PAUSE), dtype=samples.dtype)
                samples = np.concatenate([samples, pause_samples])
                chunk_duration += INTER_CHUNK_PAUSE

            # Write to archival OGG
            if out_file is None:
                out_file = sf.SoundFile(
                    str(ogg_path), mode="w", samplerate=sr, channels=1,
                )
            out_file.write(samples)

            # Encode to WAV for browser
            wav_bytes = samples_to_wav_bytes(samples, sr)

            # Estimate word timings
            estimated_words = estimate_word_timings(chunk, chunk_duration, global_offset)
            chunk_word_start = len(all_words)
            all_words.extend(estimated_words)
            chunk_word_ranges.append((chunk_word_start, len(estimated_words)))

            # Send estimated timings + audio to client
            meta = {
                "type": "audio_chunk",
                "chunkIndex": i,
                "totalChunks": len(chunks),
                "words": estimated_words,
                "duration": chunk_duration,
                "globalOffset": global_offset,
            }

            ws = client_ws[0]
            if ws:
                future = asyncio.run_coroutine_threadsafe(
                    _send_chunk(ws, meta, wav_bytes), loop,
                )
                try:
                    future.result(timeout=10)
                except Exception as e:
                    log(f"failed to send chunk {i}: {e}")

            # --- Incremental Whisper refinement ---
            try:
                tmp_path = Path(tempfile.mktemp(suffix=".wav"))
                sf.write(str(tmp_path), raw_samples, sr)

                segments, _ = whisper_model.transcribe(
                    str(tmp_path), word_timestamps=True,
                )
                refined_words = []
                for seg in segments:
                    if seg.words:
                        for w in seg.words:
                            refined_words.append({
                                "word": w.word,
                                "start": round(w.start + global_offset, 3),
                                "end": round(w.end + global_offset, 3),
                            })

                tmp_path.unlink(missing_ok=True)

                if refined_words:
                    # Align Whisper words to source text: keep source
                    # words (ground truth) with Whisper timings.
                    refined_words = align_words(chunk, refined_words, _log=log)

                    # Transfer break flags by proportional index.
                    # Time-based overlap is fragile because Whisper
                    # and estimated timings can differ significantly.
                    # Instead, map each break's position in the
                    # estimated list to the corresponding position
                    # in the refined list.
                    n_est = len(estimated_words)
                    n_ref = len(refined_words)
                    for j, ew in enumerate(estimated_words):
                        if ew.get("break"):
                            ri = round(j * n_ref / n_est) if n_est else 0
                            ri = max(0, min(ri, n_ref - 1))
                            refined_words[ri]["break"] = ew["break"]

                    # Update all_words with refined timings
                    start, count = chunk_word_ranges[i]
                    all_words[start:start + count] = refined_words
                    chunk_word_ranges[i] = (start, len(refined_words))
                    # Shift subsequent chunk ranges if word count changed
                    diff = len(refined_words) - count
                    for j in range(i + 1, len(chunk_word_ranges)):
                        s, c = chunk_word_ranges[j]
                        chunk_word_ranges[j] = (s + diff, c)

                    # Send refined timings for this chunk
                    ws = client_ws[0]
                    if ws:
                        future = asyncio.run_coroutine_threadsafe(
                            ws.send_json({
                                "type": "chunk_refined",
                                "chunkIndex": i,
                                "wordStartIdx": start,
                                "oldWordCount": count,
                                "words": refined_words,
                            }),
                            loop,
                        )
                        try:
                            future.result(timeout=10)
                        except Exception as e:
                            log(f"failed to send chunk_refined {i}: {e}")

                    log(f"whisper: chunk {i+1} refined, {count} est → {len(refined_words)} refined words")
                else:
                    log(f"whisper: chunk {i+1} produced no words, keeping estimates")

            except Exception as e:
                log(f"whisper: chunk {i+1} refinement failed: {e}")
                # Keep estimated timings for this chunk

            global_offset += chunk_duration
    finally:
        if out_file is not None:
            out_file.close()

    # Send synthesis_complete
    ws = client_ws[0]
    if ws:
        future = asyncio.run_coroutine_threadsafe(
            ws.send_json({
                "type": "synthesis_complete",
                "totalDuration": global_offset,
            }),
            loop,
        )
        try:
            future.result(timeout=5)
        except Exception as e:
            log(f"failed to send synthesis_complete: {e}")

    import gc
    del kokoro
    del whisper_model
    gc.collect()

    return all_words


async def _send_chunk(ws, meta: dict, wav_bytes: bytes) -> None:
    """Send JSON metadata followed by binary WAV data."""
    await ws.send_json(meta)
    await ws.send_bytes(wav_bytes)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    params_file = Path(sys.argv[1])
    params = json.loads(params_file.read_text())
    params_file.unlink(missing_ok=True)

    try:
        asyncio.run(run_streaming_server(params))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        import traceback
        ogg_path = Path(params.get("ogg_path", "/tmp/streaming_error"))
        log_path = ogg_path.with_suffix(".worker.log")
        with open(log_path, "a") as f:
            f.write(f"FATAL: {e}\n{traceback.format_exc()}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

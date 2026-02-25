# karaoke-tts: Development Notes & Lessons Learned

Hard-won notes from building this project. Read this before touching the synthesis
pipeline, the MCP server architecture, or the background worker.

---

## Table of Contents

1. [MCP Server Architecture](#1-mcp-server-architecture)
2. [Kokoro-ONNX: Crashes & Workarounds](#2-kokoro-onnx-crashes--workarounds)
3. [Text Sanitization for TTS](#3-text-sanitization-for-tts)
4. [Memory Management: Kokoro + Whisper Together](#4-memory-management-kokoro--whisper-together)
5. [Faster-Whisper Notes](#5-faster-whisper-notes)
6. [Audio Format: OGG via SoundFile](#6-audio-format-ogg-via-soundfile)
7. [Debugging Native Crashes](#7-debugging-native-crashes)
8. [Known Good Configuration](#8-known-good-configuration)
9. [Streaming TTS Architecture](#9-streaming-tts-architecture)

---

## 1. MCP Server Architecture

### The core problem: MCP kills background threads

The MCP framework (FastMCP / `mcp[cli]`) calls `os._exit()` when the transport
closes. This kills **all** threads — including daemon and non-daemon — with no
chance to clean up. Any threading-based background work will be silently
terminated.

**Do not use threads for long-running synthesis.** They will be killed.

### Solution: fully detached subprocess

Launch `worker.py` as a subprocess in a new session:

```python
subprocess.Popen(
    [str(venv_python), str(WORKER_PATH), str(params_file)],
    start_new_session=True,   # new session → immune to SIGHUP, parent death
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
```

`start_new_session=True` creates a new process group and session. The worker
survives the MCP server exiting or being killed.

### Use the venv Python explicitly

Do **not** use `sys.executable` when launching the worker. Under `uv run`,
`sys.executable` may resolve to something that can't load the project's packages
in the detached session.

Instead, hard-code the venv path:

```python
venv_python = Path(__file__).parent / ".venv" / "bin" / "python"
```

### Pass parameters via a temp JSON file

The worker receives a JSON file path as `sys.argv[1]` — not via stdin or env vars,
both of which are unreliable with fully detached processes.

```python
params_file = Path(tempfile.mktemp(suffix=".json"))
params_file.write_text(json.dumps({...}))
# worker deletes it after reading: params_file.unlink(missing_ok=True)
```

---

## 2. Kokoro-ONNX: Crashes & Workarounds

### Version situation

- **Installed**: `kokoro-onnx==0.5.0` (not on PyPI as of Feb 2026; PyPI latest is 0.4.7)
- **Dependency**: uses `phonemizer-fork` + `espeakng-loader` for phonemization
- **Stack**: Python → kokoro-onnx → phonemizer-fork → espeak-ng (native C library)

Crashes in espeak-ng produce **SIGSEGV** — a native signal that Python's
`try/except Exception` **cannot catch**. The process dies silently.

### ~~Crash #1: Double newlines~~ (Disproven)

Early development attributed espeak-ng segfaults to newlines in the input text.
This was a misdiagnosis — the crashes were actually caused by concurrent bugs
(heap corruption from `np.concatenate`, Unicode characters). Kokoro/espeak-ng
handles both single and double newlines without issues.

**Newlines are safe.** `sanitize_text()` preserves `\n` and `\n\n` (paragraph
breaks). Only runs of 3+ consecutive newlines are collapsed to `\n\n`.

### Crash #2: np.concatenate triggers heap corruption

After synthesizing many chunks (4–5+), calling `np.concatenate(segments)` on
the collected audio arrays reliably causes a SIGSEGV. The crash does not happen
inside `kokoro.create()` itself — all chunks succeed — but the subsequent
concatenation corrupts memory.

Root cause: repeated ONNX Runtime inference calls accumulate heap corruption in
the native runtime. Numpy's `concatenate` then touches the corrupted regions.

**Fix**: stream-write each chunk directly to the output file instead of
accumulating arrays in memory:

```python
out_file = None
try:
    for chunk in chunks:
        samples, sr = kokoro.create(chunk, ...)
        if out_file is None:
            out_file = sf.SoundFile(str(out_path), mode="w", samplerate=sr, channels=1)
        out_file.write(samples)
finally:
    if out_file is not None:
        out_file.close()
```

This avoids any large in-memory buffer and writes OGG frames incrementally.

### Crash #3: Unicode characters in input

Certain Unicode characters (em dashes, curly quotes, ellipsis, etc.) can trigger
crashes in espeak-ng's phonemizer. The MCP tool docstring instructs the LLM to
normalize text, but this cannot be relied upon — always sanitize in the worker.

See [§3 Text Sanitization](#3-text-sanitization-for-tts).

### Chunk size

Keep chunks to ≤ 800 characters. The kokoro-onnx library itself has a
`MAX_PHONEME_LENGTH` limit, and sentence-boundary chunking at 800 chars stays
well within it. Smaller chunks (300–400 chars) are safer but slower.

---

## 3. Text Sanitization for TTS

Always run `sanitize_text()` on input before passing to any TTS engine. The LLM
may skip normalization even when instructed. The worker must be defensive.

```python
def sanitize_text(text: str) -> str:
    replacements = [
        ("\u2018", "'"), ("\u2019", "'"),   # curly single quotes
        ("\u201c", '"'), ("\u201d", '"'),    # curly double quotes
        ("\u2014", ", "),                    # em dash → comma-space
        ("\u2013", "-"),                     # en dash → hyphen
        ("\u2026", "..."),                   # ellipsis
        ("\u00b7", "."),                     # middle dot
        ("\u2022", "."),                     # bullet
        ("\u00a0", " "),                     # non-breaking space
        ("\u200b", ""),                      # zero-width space
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    # Normalize line endings, preserve \n and \n\n, collapse 3+ to \n\n
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
```

The `encode("ascii", errors="ignore")` catches any Unicode that slipped through
the explicit replacements.

**Newlines are preserved** so that downstream code (chunking, word timing
estimation, and the HTML player) can render paragraph breaks. Single `\n`
produces a line break; `\n\n` produces a paragraph break (double `<br>`).

---

## 4. Memory Management: Kokoro + Whisper Together

### The problem

Kokoro loads a **310 MB** ONNX model. Faster-whisper (small) loads a **460 MB**
model. Running them back-to-back in the same Python process hits ~800 MB+ of
model memory, which can trigger macOS's memory pressure killer (jetsam) or
cause a native crash when the Whisper model tries to allocate.

### Fix: force garbage collection between stages

After `generate_kokoro()` returns, the local `kokoro` variable goes out of scope
but the ONNX Runtime may keep internal caches. Call `gc.collect()` before
loading Whisper:

```python
def get_word_timestamps(audio_path, model_size):
    import gc
    gc.collect()                    # release Kokoro/ONNX memory
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    ...
```

The stream-write approach in `generate_kokoro` (§2) also helps: since audio is
never accumulated in a large numpy array, peak memory during synthesis is lower.

### Streaming mode: both models loaded simultaneously

The streaming worker (`streaming_worker.py`) takes a different approach: it loads
both Kokoro and Whisper at startup (~770MB combined) and runs Whisper on each
chunk immediately after synthesis. This avoids waiting for full synthesis to
complete before refining timings. After all chunks are done, both models are
explicitly deleted and `gc.collect()` is called. This has worked reliably in
testing despite the higher peak memory.

---

## 5. Faster-Whisper Notes

- Model: `small` is a good default (460 MB, fast, accurate for clean TTS audio)
- `compute_type="int8"` halves memory vs `float16` with minimal accuracy loss
- `word_timestamps=True` is what drives the karaoke highlighting
- Faster-whisper works reliably when run after Kokoro as long as memory is freed
  first (see §4)
- The `segments` iterator is lazy — consume it fully before doing anything else,
  otherwise the model may be freed prematurely

```python
segments, _ = model.transcribe(str(audio_path), word_timestamps=True)
words = []
for segment in segments:           # fully consume the iterator
    if segment.words:
        for w in segment.words:
            words.append({"word": w.word, "start": ..., "end": ...})
```

---

## 6. Audio Format: OGG via SoundFile

OGG Vorbis is ~8× smaller than WAV for the same audio. `soundfile` writes OGG
natively — just use a `.ogg` extension in the output path.

**Streaming write** (preferred for multi-chunk synthesis):

```python
with sf.SoundFile(path, mode="w", samplerate=sr, channels=1) as f:
    for chunk_audio in chunks:
        f.write(chunk_audio)
```

**Single write**:
```python
sf.write(str(path), audio_array, sample_rate)
```

A 3446-byte OGG file means the file was created (headers written) but synthesis
was interrupted before any audio frames were flushed — diagnostic indicator of
a native crash mid-synthesis.

---

## 7. Debugging Native Crashes

Native library crashes (SIGSEGV) are invisible to Python's exception handling.
Symptoms:
- Worker log truncated mid-step with no exception message
- Tiny OGG file (headers only, ~3446 bytes), no HTML
- Worker log is 0 bytes (stderr redirect captures nothing from a SIGSEGV)

### Essential: write your own log file with explicit flush

Do **not** rely on stderr redirection to capture errors from native crashes.
Instead, write directly to a `.worker.log` file from Python with `flush()` after
every step:

```python
def log(msg):
    with open(log_path, "a") as f:
        f.write(msg + "\n")
        f.flush()          # CRITICAL: flush ensures partial logs survive kills
```

### Debugging workflow

1. Run `worker.py` directly from the command line to get stderr output
2. Reproduce with increasing text lengths to find the size threshold
3. Add per-chunk logging inside the synthesis loop to find which chunk crashes
4. Test the problematic text fragment in isolation to confirm the trigger

### The "0-byte log" problem

If the worker produces a 0-byte log AND a 3446-byte OGG, the crash happens
**after** `sf.SoundFile` is opened but **before** any audio frames are written.
This is typical of a crash during the first `kokoro.create()` call.

If the worker produces a 0-byte log AND **no** OGG, Python is crashing before
the synthesis even begins — check imports, venv path, and params file.

---

## 8. Known Good Configuration

Tested and working as of February 2026:

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.12.9 | via uv/cpython |
| kokoro-onnx | 0.5.0 | Not on PyPI; stream-write required |
| faster-whisper | ≥ 1.0.0 | `small` model, `int8` compute |
| soundfile | ≥ 0.12.0 | OGG Vorbis output |
| mcp[cli] | ≥ 1.0.0 | FastMCP |

**Kokoro models** (download separately):
- `kokoro-v1.0.onnx` — 310 MB
- `voices-v1.0.bin` — 28 MB

**Whisper model** — downloads automatically to `~/.cache/huggingface/` on first use.

---

## 9. Streaming TTS Architecture

### Overview

`streaming_worker.py` streams Kokoro audio to the browser in real time via
WebSocket, so the user hears audio within 2-4 seconds instead of waiting for
full synthesis + Whisper.

### How it works

1. **Server starts** — Starlette + Uvicorn on `127.0.0.1` with an OS-assigned
   port. Browser opens immediately.
2. **Browser connects** via WebSocket. An `AudioContext` is created for playback.
3. **Synthesis loop** — Both Kokoro (~310MB) and Whisper (~460MB) are loaded
   simultaneously (~770MB combined). Kokoro synthesizes small chunks (~200 chars,
   ~2-5s audio each). For each chunk:
   - Samples are encoded as in-memory PCM-16 WAV (`soundfile` → `BytesIO`)
   - Word timings are estimated proportionally from character counts
   - JSON metadata + binary WAV sent over WebSocket
   - Browser decodes WAV via `AudioContext.decodeAudioData()` and schedules
     playback with `AudioBufferSourceNode.start(when)`
   - 0.25s silence is appended between chunks for natural pacing
   - Whisper immediately refines that chunk's word timings and sends a
     `chunk_refined` message — the player rebuilds word spans in place
4. **Archival** — OGG + self-contained HTML saved to `~/Music/TTS/`.

### Key design decisions

**Web Audio API, not `<audio>` element**: The `<audio>` element can't append
buffers dynamically. Web Audio's `AudioBufferSourceNode` lets us schedule each
decoded chunk at a precise time offset, so chunks play seamlessly end-to-end.

**Estimated word timings**: Character-proportional with punctuation bonuses
(+6 weight for `.!?`, +3 for `,;:`). Accurate within ~200-400ms — good enough
for karaoke during streaming. Whisper refines each chunk immediately after it
is synthesized.

**Small chunks (200 chars)**: Kokoro synthesizes ~200 chars in 1-3 seconds,
giving low latency to first audio. The original non-streaming worker uses 800
chars. Too small (< 100 chars) produces choppy prosody.

**Inter-chunk silence**: 0.25 seconds of zero-samples appended to each chunk
(except the last). Without this, concatenated chunks sound rushed at boundaries.

**Paragraph breaks**: Newlines in the source text are tracked through the
chunking → timing estimation → WebSocket → JS rendering pipeline. The word
timing `"break"` field is an integer: `1` = line break (single `<br>`),
`2` = paragraph break (double `<br>`). When Whisper refines a chunk, break
flags are transferred from estimated to refined words via proportional index
mapping (not time-based overlap, which is fragile when timings diverge).

### WebSocket protocol

| Direction | Type | Format |
|-----------|------|--------|
| Server → Client | `audio_chunk` | JSON `{type, chunkIndex, totalChunks, words, duration, globalOffset}` then binary WAV |
| Server → Client | `chunk_refined` | JSON `{type, chunkIndex, wordStartIdx, oldWordCount, words}` |
| Server → Client | `synthesis_complete` | JSON `{type, totalDuration}` |
| Server → Client | `archival_ready` | JSON `{type, htmlPath, oggPath}` |
| Client → Server | `ready` | JSON `{type: "ready"}` |

### Playback

The streaming player does not autoplay. A buffering spinner is shown until the
first audio chunk is decoded, then replaced by a play/pause button. The user
clicks play to start; `AudioContext` is created and resumed on the first click.

### Karaoke animation

The streaming player uses `requestAnimationFrame` (~60Hz) for word highlighting,
compared to `timeupdate` (~4Hz) in the non-streaming player. The same binary
search algorithm finds the active word. Playback time is derived from
`audioCtx.currentTime - startedAt`, with a 0.15s startup delay to let the audio
pipeline warm up before the first word highlights.

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
6. [Piper TTS API Changes](#6-piper-tts-api-changes)
7. [Audio Format: OGG via SoundFile](#7-audio-format-ogg-via-soundfile)
8. [Debugging Native Crashes](#8-debugging-native-crashes)
9. [terminal-notifier on macOS 15](#9-terminal-notifier-on-macos-15)
10. [Known Good Configuration](#10-known-good-configuration)

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

### Crash #1: Double newlines create empty espeak-ng inputs

`phonemizer.phonemize()` splits its input on `\n`. A double newline (`\n\n`)
produces an empty string in the middle of the list. Passing `""` to espeak-ng
causes a segfault.

**Fix**: collapse all newlines to spaces before synthesizing.

```python
text = re.sub(r"[\r\n\t]+", " ", text)
```

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
    text = re.sub(r"[\r\n\t]+", " ", text)   # CRITICAL: collapse newlines
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
```

The `encode("ascii", errors="ignore")` catches any Unicode that slipped through
the explicit replacements.

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

## 6. Piper TTS API Changes

**Piper v1.4+ broke the old wave-file API.** `synthesize()` no longer accepts a
`wave.Wave_write` object. It now returns `Iterable[AudioChunk]`.

### Old (broken, ≤ v1.3):
```python
with wave.open(path, "w") as wf:
    voice.synthesize(text, wf)      # no longer works
```

### New (correct, v1.4+):
```python
chunks = list(voice.synthesize(text))
if not chunks:
    raise RuntimeError("Piper produced no audio")
audio = np.concatenate([c.audio_float_array for c in chunks])
sf.write(str(out_path), audio, chunks[0].sample_rate)
```

Key fields on `AudioChunk`:
- `audio_float_array` — float32 numpy array, range [-1.0, 1.0]
- `sample_rate` — typically 22050 Hz
- `sample_width` — bytes per sample (2 = 16-bit)
- `sample_channels` — number of channels

---

## 7. Audio Format: OGG via SoundFile

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

## 8. Debugging Native Crashes

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

## 9. terminal-notifier on macOS 15

Several `terminal-notifier` flags are silently broken on macOS 15 (Sequoia):

| Flag | Status |
|------|--------|
| `-sender com.apple.Terminal` | Silently **suppresses** all notifications |
| `-open <url>` | Does nothing |
| `-execute <command>` | Does nothing |

**Use only** `-title` and `-message`. No `-sender`, no `-open`, no `-execute`.

```python
subprocess.run(
    ["terminal-notifier", "-title", title, "-message", message],
    check=False,
)
```

---

## 10. Known Good Configuration

Tested and working as of February 2026:

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.12.9 | via uv/cpython |
| kokoro-onnx | 0.5.0 | Not on PyPI; stream-write required |
| faster-whisper | ≥ 1.0.0 | `small` model, `int8` compute |
| soundfile | ≥ 0.12.0 | OGG Vorbis output |
| piper-tts | ≥ 1.4.1 | New `AudioChunk` API |
| mcp[cli] | ≥ 1.0.0 | FastMCP |
| terminal-notifier | any | macOS 15: use `-title`/`-message` only |

**Kokoro models** (download separately):
- `kokoro-v1.0.onnx` — 310 MB
- `voices-v1.0.bin` — 28 MB

**Whisper model** — downloads automatically to `~/.cache/huggingface/` on first use.

**Piper model** (optional):
- `en_US-lessac-medium.onnx` + `.json` — ~60 MB — from HuggingFace rhasspy/piper-voices

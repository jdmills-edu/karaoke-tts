# karaoke-tts

A local MCP server that synthesizes speech and opens a karaoke-style browser player with word-by-word highlighting synchronized to the audio. Built on [Piper](https://github.com/rhasspy/piper) or [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) for TTS and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for word timestamps.

Forked from [local-hq-tts](https://github.com/jdmills-edu/local-hq-tts).

## How it works

### Streaming mode (default)

1. Claude calls `generate_speech` — a local HTTP + WebSocket server starts and the browser opens immediately
2. Kokoro synthesizes audio in small chunks (~200 chars each, ~2-5s of audio)
3. Each chunk streams to the browser over WebSocket and plays immediately with estimated word timings
4. After each chunk, Whisper immediately refines that chunk's word timings and sends them to the browser
5. When synthesis completes, an archival OGG + HTML pair is saved

Streaming mode typically plays the first audio within 2-4 seconds of starting synthesis, compared to 30-90+ seconds for full synthesis in standard mode.

### Standard mode (`streaming=False`)

1. Claude calls `generate_speech` with `streaming=False`
2. TTS synthesis runs in the background (Piper or Kokoro)
3. faster-whisper transcribes the audio to extract word-level timestamps
4. A self-contained HTML player is generated and opened in your browser
5. macOS notifications fire at each stage

## Tools

| Tool | Description |
|------|-------------|
| `generate_speech` | Synthesize text → karaoke player. Streams by default (Kokoro). |
| `list_kokoro_voices` | List available Kokoro voice names |

`engine` and `voice` are required. Claude will always call `list_kokoro_voices` first.

`streaming` defaults to `True`. When enabled with Kokoro, audio plays in the browser within seconds. Set `streaming=False` for standard mode (full synthesis before playback). Streaming is ignored for Piper.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- `terminal-notifier`: `brew install terminal-notifier`

## Setup

### 1. Install dependencies

```bash
cd ~/Documents/GitHub/karaoke-tts
uv sync
```

### 2. Download TTS models

**Piper** (~60MB):
```bash
mkdir -p ~/.local/share/piper
curl -L -o ~/.local/share/piper/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -o ~/.local/share/piper/en_US-lessac-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
```

**Kokoro** (~310MB):
```bash
mkdir -p ~/Documents/GitHub/karaoke-tts/models
curl -L -o ~/Documents/GitHub/karaoke-tts/models/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o ~/Documents/GitHub/karaoke-tts/models/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

The faster-whisper `small` model (~460MB) downloads automatically on first use to `~/.cache/huggingface/`.

### 3. Add to Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "karaoke-tts": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "/Users/YOUR_USERNAME/Documents/GitHub/karaoke-tts",
        "python",
        "/Users/YOUR_USERNAME/Documents/GitHub/karaoke-tts/server.py"
      ]
    }
  }
}
```

Restart Claude Desktop after saving.

## Configuration Reference

```json
{
  "output_dir": "~/Music/TTS",
  "whisper_model": "small",
  "piper": {
    "default_voice": "~/.local/share/piper/en_US-lessac-medium.onnx"
  },
  "kokoro": {
    "default_voice": "af_heart",
    "model_path": "~/Documents/GitHub/karaoke-tts/models/kokoro-v1.0.onnx",
    "voices_path": "~/Documents/GitHub/karaoke-tts/models/voices-v1.0.bin",
    "lang": "en-us",
    "speed": 1.0
  }
}
```

`whisper_model` can be `"tiny"`, `"base"`, `"small"`, `"medium"`, or `"large-v3"`. Larger models give more accurate word timestamps but are slower.

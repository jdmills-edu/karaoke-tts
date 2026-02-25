# karaoke-tts

A local MCP server that synthesizes speech and opens a karaoke-style browser player with word-by-word highlighting synchronized to the audio. Built on [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) for TTS and [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for word timestamps.

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
2. Kokoro synthesizes the full audio in the background
3. faster-whisper transcribes the audio to extract word-level timestamps
4. A self-contained HTML player is generated and opened in your browser

## Tools

| Tool | Description |
|------|-------------|
| `generate_speech` | Synthesize text → karaoke player. Streams by default (Kokoro). |
| `list_kokoro_voices` | List available Kokoro voice names |

`voice` is required. Claude will always call `list_kokoro_voices` first.

`streaming` defaults to `True`. Audio plays in the browser within seconds. Set `streaming=False` for standard mode (full synthesis before playback).

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)

## Setup

### 1. Install dependencies

From the project root:

```bash
uv sync
```

### 2. Download TTS models

Download to the `models/` directory inside the project.

**Kokoro** (~310MB):

macOS / Linux:
```bash
mkdir -p models
curl -L -o models/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Windows (PowerShell):
```powershell
New-Item -ItemType Directory -Force -Path models
Invoke-WebRequest -Uri "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx" -OutFile "models\kokoro-v1.0.onnx"
Invoke-WebRequest -Uri "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" -OutFile "models\voices-v1.0.bin"
```

The faster-whisper `small` model (~460MB) downloads automatically on first use to `~/.cache/huggingface/`.

### 3. Add to Claude Desktop

Add to your Claude Desktop configuration file:

| Platform | Config file location |
|----------|---------------------|
| macOS    | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux    | `~/.config/Claude/claude_desktop_config.json` |
| Windows  | `%APPDATA%\Claude\claude_desktop_config.json` |

```json
{
  "mcpServers": {
    "karaoke-tts": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "/path/to/karaoke-tts",
        "python",
        "/path/to/karaoke-tts/server.py"
      ]
    }
  }
}
```

Replace `/path/to/karaoke-tts` with the absolute path to your cloned repository. On Windows, use forward slashes (e.g. `C:/Users/you/karaoke-tts`).

Restart Claude Desktop after saving.

## Configuration Reference

```json
{
  "output_dir": "~/Music/TTS",
  "whisper_model": "small",
  "kokoro": {
    "default_voice": "af_heart",
    "model_path": "models/kokoro-v1.0.onnx",
    "voices_path": "models/voices-v1.0.bin",
    "lang": "en-us",
    "speed": 1.0
  }
}
```

Model paths are relative to the project directory. Absolute paths and `~` expansion are also supported.

`whisper_model` can be `"tiny"`, `"base"`, `"small"`, `"medium"`, or `"large-v3"`. Larger models give more accurate word timestamps but are slower.

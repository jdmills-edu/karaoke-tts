#!/usr/bin/env python3
"""
karaoke-tts MCP Server

Accepts generate_speech requests and immediately hands off the synthesis
pipeline to worker.py as a detached subprocess, so the server exiting
never interrupts generation.
"""

import json
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

CONFIG_PATH = Path(__file__).parent / "config.json"
WORKER_PATH = Path(__file__).parent / "worker.py"
STREAMING_WORKER_PATH = Path(__file__).parent / "streaming_worker.py"

mcp = FastMCP("karaoke-tts")


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


@mcp.tool()
def generate_speech(
    text: str,
    engine: str,
    voice: str,
    streaming: bool = True,
    output_path: str = None,
) -> str:
    """Synthesize speech and open a karaoke browser player with word-by-word
    highlighting synchronized to the audio.

    By default, streaming mode is enabled: audio starts playing in the browser
    almost immediately while synthesis continues in the background.  Streaming
    uses the Web Audio API and WebSocket to deliver audio chunk by chunk with
    estimated word timings, which are automatically refined by Whisper after
    synthesis completes.  Streaming is only supported with the kokoro engine;
    if engine is "piper", streaming is ignored and standard mode is used.

    In standard (non-streaming) mode, synthesis completes fully before the
    browser opens.

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

    Args:
        text: TTS-friendly text to synthesize. Any length is supported.
        engine: TTS engine — must be "piper" or "kokoro".
        voice: Voice identifier (required).
               Piper: path to a .onnx model file.
               Kokoro: voice name (e.g. "af_heart", "am_michael").
        streaming: Stream audio to the browser as it is synthesized
                   (default True). Only supported with kokoro engine;
                   ignored for piper. Set to False for standard mode
                   where the browser opens after synthesis completes.
        output_path: Optional custom path for the OGG file.
                     The HTML player is saved alongside it automatically.

    Returns:
        Confirmation with output paths.
    """
    config = load_config()
    if engine not in ("piper", "kokoro"):
        raise ValueError(f"Unknown engine {engine!r}. Must be 'piper' or 'kokoro'.")

    use_streaming = streaming and engine == "kokoro"

    if output_path:
        ogg_path = resolve(output_path)
        html_path = ogg_path.with_suffix(".html")
    else:
        ogg_path, html_path = make_output_paths(config)

    params_file = Path(tempfile.mktemp(suffix=".json"))
    params_file.write_text(json.dumps({
        "text": text,
        "engine": engine,
        "voice": voice,
        "ogg_path": str(ogg_path),
        "html_path": str(html_path),
        "config": config,
    }))

    venv_python = Path(__file__).parent / ".venv" / "bin" / "python"
    worker = STREAMING_WORKER_PATH if use_streaming else WORKER_PATH

    if use_streaming:
        notify("Streaming Speech…", f"{engine} · {voice} · {len(text):,} chars")
    else:
        notify("Generating Speech…", f"{engine} · {voice} · {len(text):,} chars")

    subprocess.Popen(
        [str(venv_python), str(worker), str(params_file)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if use_streaming:
        return (
            f"Streaming speech generation started ({engine} · {voice}).\n"
            f"A browser window will open with the streaming player.\n"
            f"Audio plays immediately as each chunk is synthesized.\n\n"
            f"Archival audio:  {ogg_path}\n"
            f"Archival player: {html_path}\n\n"
            f"Do not wait or poll — tell the user it's underway."
        )

    return (
        f"Speech generation is running in the background ({engine} · {voice}). "
        f"You'll receive macOS notifications at each stage:\n"
        f"  1. Generating Speech…\n"
        f"  2. Syncing words…\n"
        f"  3. Karaoke Ready (browser opens automatically)\n\n"
        f"Audio:  {ogg_path}\n"
        f"Player: {html_path}\n\n"
        f"Do not wait or poll — let the user know it's underway."
    )


@mcp.tool()
def list_kokoro_voices() -> str:
    """List all available Kokoro voice names and present them as a pick list
    so the user can select one before generating speech. After the user makes
    a selection, proceed to call generate_speech with their chosen voice."""
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

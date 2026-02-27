#!/usr/bin/env python3
"""End-to-end test: paragraph headers through the full TTS pipeline.

Runs sanitize → Kokoro TTS → Whisper timestamps → align_words → generate_player
on synthetic data with paragraph headers to verify line breaks survive.
"""

import json
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from worker import (
    sanitize_text,
    chunk_text,
    generate_kokoro,
    get_word_timestamps,
    align_words,
    generate_player,
)

# ---------------------------------------------------------------------------
# Synthetic test data: titles / headers separated by \n\n
# ---------------------------------------------------------------------------
TEST_TEXT = """\
The Solar System

The solar system consists of the Sun and everything that orbits around it. This includes eight major planets, their moons, and countless smaller objects.

The Inner Planets

Mercury, Venus, Earth, and Mars are the four inner planets. They are relatively small and composed mostly of rock and metal.

The Outer Planets

Jupiter, Saturn, Uranus, and Neptune are the outer planets. These gas giants are much larger and composed primarily of hydrogen and helium."""

OUTPUT_DIR = Path(__file__).parent / "test_output"
OUTPUT_DIR.mkdir(exist_ok=True)

OGG_PATH = OUTPUT_DIR / "paragraph_test.ogg"
HTML_PATH = OUTPUT_DIR / "paragraph_test.html"

CONFIG = json.loads((Path(__file__).parent / "config.json").read_text())


def log(msg: str) -> None:
    print(f"  {msg}")


def main() -> None:
    # Step 1: sanitize
    print("=== Step 1: sanitize_text ===")
    sanitized = sanitize_text(TEST_TEXT)
    print(f"  Input newlines:  {TEST_TEXT.count(chr(10))}")
    print(f"  Output newlines: {sanitized.count(chr(10))}")
    print(f"  Paragraph breaks (\\n\\n): {sanitized.count(chr(10)+chr(10))}")
    print(f"  Sanitized text:\n---\n{sanitized}\n---\n")

    # Verify paragraph breaks survived
    assert sanitized.count("\n\n") == 5, (
        f"Expected 5 paragraph breaks, got {sanitized.count(chr(10)+chr(10))}"
    )

    # Step 2: chunk
    print("=== Step 2: chunk_text ===")
    chunks = chunk_text(sanitized)
    for i, c in enumerate(chunks):
        print(f"  Chunk {i}: {repr(c[:100])}...")
    print()

    # Step 3: Kokoro TTS
    print("=== Step 3: Kokoro TTS ===")
    generate_kokoro(sanitized, "af_heart", OGG_PATH, CONFIG, _log=log)
    print(f"  Audio: {OGG_PATH} ({OGG_PATH.stat().st_size:,} bytes)\n")

    # Step 4: Whisper word timestamps
    print("=== Step 4: Whisper word timestamps ===")
    whisper_words = get_word_timestamps(OGG_PATH, CONFIG.get("whisper_model", "small"), _log=log)
    print(f"  Whisper detected {len(whisper_words)} words")
    for w in whisper_words[:10]:
        print(f"    {w['start']:6.2f}-{w['end']:6.2f}  {w['word']}")
    if len(whisper_words) > 10:
        print(f"    ... and {len(whisper_words) - 10} more\n")

    # Step 5: align
    print("=== Step 5: align_words ===")
    aligned = align_words(sanitized, whisper_words, _log=log)
    print(f"  Aligned {len(aligned)} words")
    for w in aligned[:10]:
        print(f"    {w['start']:6.2f}-{w['end']:6.2f}  {w['word']}")
    if len(aligned) > 10:
        print(f"    ... and {len(aligned) - 10} more\n")

    # Step 6: generate player HTML
    print("=== Step 6: generate_player ===")
    generate_player(aligned, "af_heart", OGG_PATH, HTML_PATH, title="Paragraph Header Test")
    print(f"  Player: {HTML_PATH} ({HTML_PATH.stat().st_size:,} bytes)\n")

    # Step 7: verify the HTML contains <br> tags for paragraph breaks
    print("=== Step 7: verify HTML paragraph breaks ===")
    html = HTML_PATH.read_text()
    words_json = html.split("const W = ")[1].split(";\n")[0]
    words_data = json.loads(words_json)
    break_words = [w for w in words_data if w.get("break")]
    print(f"  Words with 'break' property: {len(break_words)}")
    for w in break_words:
        print(f"    break={w['break']}  word={w['word']}")

    print("\n=== ALL DONE ===")
    print(f"Open {HTML_PATH} in a browser to see the result.")


if __name__ == "__main__":
    main()

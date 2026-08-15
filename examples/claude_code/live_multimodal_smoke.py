"""Live check that ``images=`` reaches Claude Code through a real LazyBridge Agent.

Draws a distinctive test picture (magenta square, black cross) so a correct
description proves the bytes really arrived — a generic "red square" could be
guessed. LazyBridge coerces the path to inline base64, which this engine sends
as an Anthropic ``image`` content block on the SDK's user-message stream.

    .venv\\Scripts\\python.exe examples\\claude_code\\live_multimodal_smoke.py
"""

from __future__ import annotations

import struct
import tempfile
import zlib
from pathlib import Path

from lazybridge import Agent, ClaudeCodeEngine

W = H = 96


def write_test_png(path: Path) -> Path:
    rows = bytearray()
    for y in range(H):
        rows.append(0)  # PNG filter type 0
        for x in range(W):
            on_cross = abs(x - W // 2) < 8 or abs(y - H // 2) < 8
            rows.extend((0, 0, 0) if on_cross else (220, 20, 160))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + chunk(b"IEND", b"")
    )
    return path


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        png = write_test_png(Path(tmp) / "cross.png")
        agent = Agent(name="claude-multimodal-smoke", engine=ClaudeCodeEngine(model="sonnet"))
        result = agent(
            "Describe this image in one short sentence: what colour is it, and what shape is drawn on it?",
            images=[str(png)],
        )

    text = result.text()
    print(text)
    described = "cross" in text.lower() or "plus" in text.lower()
    coloured = any(word in text.lower() for word in ("pink", "magenta"))
    print(f"[{'PASS' if result.ok and described and coloured else 'FAIL'}] image reached the model")
    if result.error:
        print(f"error={result.error.type}: {result.error.message}")


if __name__ == "__main__":
    main()

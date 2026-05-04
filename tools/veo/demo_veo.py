"""
demo_veo.py  —  manual demo for lazybridge.tools.veo
=====================================================
Run with F5 in Spyder or:
    python tools/veo/demo_veo.py

NOT a pytest test — intentionally named demo_* to avoid automatic test discovery.

All sections require GOOGLE_API_KEY in .env (Veo always calls the live API).

Sections
--------
    1. Text-to-video (8s, 720p, with audio)
    2. Image-to-video (first frame)
    3. First + last frame interpolation
    4. Via Agent
"""

import os
import sys
from pathlib import Path

# ── Load .env ──────────────────────────────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

api_key = os.getenv("GOOGLE_API_KEY")
print(f"GOOGLE_API_KEY present: {bool(api_key)}")
if not api_key:
    print("Set GOOGLE_API_KEY in .env or as an environment variable.")
    sys.exit(1)

# ── Shared setup ───────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # repo root

from lazybridge.tools.veo import veo_tool

OUTPUT_DIR = "generated_videos/demo"

# ══════════════════════════════════════════════════════════════════════════════
# 1. Text-to-video — 8s, 720p, with audio
# ══════════════════════════════════════════════════════════════════════════════

print("\n── Section 1: text-to-video ──")

tool = veo_tool(api_key=api_key, output_dir=OUTPUT_DIR)

result = tool.invoke(
    prompt=(
        "A cinematic close-up of a luxury perfume bottle on black glass. "
        "Golden particles float in the air. Dramatic side lighting, shallow depth of field."
    ),
    duration_seconds=8,
    aspect_ratio="16:9",
    resolution="720p",
    generate_audio=True,
    negative_prompt="blurry, shaky, low quality, watermark",
    enhance_prompt=True,
)

print(f"Saved to: {result['output_path']}")
print(result)

# ══════════════════════════════════════════════════════════════════════════════
# 2. Image-to-video
# ══════════════════════════════════════════════════════════════════════════════

# Uncomment and set FIRST_FRAME_PATH to a real image file to run this section.

# print("\n── Section 2: image-to-video ──")
#
# FIRST_FRAME_PATH = "path/to/your/image.jpg"
#
# result2 = tool.invoke(
#     prompt="The flowers sway gently in the breeze.",
#     duration_seconds=6,
#     first_frame=FIRST_FRAME_PATH,
#     aspect_ratio="16:9",
#     resolution="720p",
#     generate_audio=True,
# )
# print(f"Saved to: {result2['output_path']}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. First + last frame interpolation
# ══════════════════════════════════════════════════════════════════════════════

# Uncomment and set paths to run this section.

# print("\n── Section 3: frame interpolation ──")
#
# FIRST_FRAME = "path/to/start.jpg"
# LAST_FRAME  = "path/to/end.jpg"
#
# result3 = tool.invoke(
#     prompt="A hand gently places a glass of milk next to a plate of cookies.",
#     duration_seconds=8,   # must be 8 for interpolation
#     first_frame=FIRST_FRAME,
#     last_frame=LAST_FRAME,
#     generate_audio=True,
# )
# print(f"Saved to: {result3['output_path']}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Via Agent
# ══════════════════════════════════════════════════════════════════════════════

# Uncomment to run. Requires GOOGLE_API_KEY.

# print("\n── Section 4: via Agent ──")
#
# from lazybridge import Agent
#
# agent = Agent("google/gemini-2.5-pro", tools=[tool])
# resp  = agent(
#     "Generate an 8-second cinematic video of a sunset over the ocean, "
#     "wide angle, warm colors, with natural ambient audio.",
# )
# print(resp.text())

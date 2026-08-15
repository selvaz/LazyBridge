"""``CodexEngine`` — a standard LazyBridge ``Engine`` backed by the locally
authenticated Codex CLI (``codex app-server``).

Requires only the ``codex`` CLI itself (``npm install -g @openai/codex``,
then ``codex --login``) — no extra Python dependency, and no API key. See
:doc:`/guides/full/codex-engine` for setup, usage, and the verified protocol
surface.
"""

from .app_server import CodexAppServerClient, CodexRunResult, codex_executable
from .engine import CodexEngine

__all__ = [
    "CodexAppServerClient",
    "CodexEngine",
    "CodexRunResult",
    "codex_executable",
]

"""
Spyder bootstrap for the local LazyBridge checkout.

Run this file once in a Spyder console before importing the package or running
tests. It makes sure the local repository wins over any pip-installed copy and
provides a small helper to launch pytest from the same session.

Usage:
    runfile(r"D:/LazyBridge/setup_spyder_paths.py")

Then:
    import lazybridge
    run_tests()
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PATHS = [
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "tests"),
]
PACKAGES = ["lazybridge", "tests"]
ENV_FILE = PROJECT_ROOT / ".env"


def _evict_modules() -> list[str]:
    evicted = [
        name
        for name in list(sys.modules)
        if any(name == pkg or name.startswith(pkg + ".") for pkg in PACKAGES)
    ]
    for name in evicted:
        del sys.modules[name]
    return evicted


def _prepend_paths() -> None:
    for path in reversed(PATHS):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _enable_asyncio() -> None:
    try:
        import nest_asyncio

        nest_asyncio.apply()
        print("[setup_spyder_paths] nest_asyncio applied")
    except ImportError:
        print(
            "[setup_spyder_paths] nest_asyncio not installed; "
            "asyncio.run() may fail inside Spyder"
        )


def _load_dotenv() -> None:
    if not ENV_FILE.exists():
        print(f"[setup_spyder_paths] no .env file found at {ENV_FILE}")
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE, override=True)
        print(f"[setup_spyder_paths] loaded .env via python-dotenv: {ENV_FILE}")
        return
    except ImportError:
        pass

    with ENV_FILE.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip("'\"")
            if value:
                os.environ[key.strip()] = value
    print(f"[setup_spyder_paths] loaded .env manually: {ENV_FILE}")


def run_tests(*pytest_args: str) -> int:
    """Run the project's pytest suite from the current Spyder session."""
    import pytest

    args = list(pytest_args) if pytest_args else ["-q", "tests"]
    print(f"[setup_spyder_paths] running: pytest {' '.join(args)}")
    return pytest.main(args)


def check_env_keys() -> None:
    """Print whether the main provider API keys are available in this session."""
    keys = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    ]
    print("[setup_spyder_paths] API key status:")
    for key in keys:
        value = os.environ.get(key)
        if value:
            print(f"  {key}: FOUND (len={len(value)}) preview={value[:8]}...")
        else:
            print(f"  {key}: MISSING")


def configure() -> None:
    evicted = _evict_modules()
    if evicted:
        print(f"[setup_spyder_paths] removed from sys.modules: {evicted}")

    _prepend_paths()
    print("[setup_spyder_paths] local paths added to the front of sys.path:")
    for path in PATHS:
        print(f"  + {path}")

    _enable_asyncio()
    _load_dotenv()
    print("[setup_spyder_paths] ready: import lazybridge, call run_tests(), or call check_env_keys()")


configure()

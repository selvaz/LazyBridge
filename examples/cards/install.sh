#!/usr/bin/env bash
# LazyBridge cards installer (macOS / Linux).
#   curl -fsSL https://raw.githubusercontent.com/selvaz/LazyBridge/main/examples/cards/install.sh | bash
# Installs into ~/lazybridge-cards: a Python venv with LazyBridge + LazyPulse,
# the Claude Code and Codex CLIs if missing, and the two card scripts.
# It never asks for or stores credentials: you sign in yourself at the end.
set -euo pipefail

DIR="${LAZYBRIDGE_CARDS_DIR:-$HOME/lazybridge-cards}"
RAW="https://raw.githubusercontent.com/selvaz/LazyBridge/main/examples/cards"
LAZYBRIDGE="lazybridge[claude-code]==1.8.7"
LAZYPULSE="lazypulse[telegram,cron] @ git+https://github.com/selvaz/LazyPulse.git@v0.5.7"

say() { printf '\n==> %s\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1; }

say "Checking Python 3.11+ and git"
PY=""
for c in python3.13 python3.12 python3.11 python3; do
  if need "$c" && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "Python 3.11+ not found: install it from python.org, then rerun."; exit 1; }
need git || { echo "git not found: install it, then rerun."; exit 1; }

say "Creating $DIR and its Python environment"
mkdir -p "$DIR"
[ -x "$DIR/.venv/bin/python" ] || "$PY" -m venv "$DIR/.venv"
"$DIR/.venv/bin/python" -m pip install --quiet --upgrade pip
"$DIR/.venv/bin/python" -m pip install --quiet "$LAZYBRIDGE" "$LAZYPULSE"

say "Claude Code CLI"
if need claude; then echo "already installed"; else curl -fsSL https://claude.ai/install.sh | bash; fi

say "Codex CLI"
if need codex; then echo "already installed"
elif need npm; then npm install -g @openai/codex
else echo "npm not found: install Node.js, then run: npm install -g @openai/codex"; fi

say "Downloading the cards"
for f in research_desk.py research_desk_pulse.py; do curl -fsSL "$RAW/$f" -o "$DIR/$f"; done

cat <<MSG

Done. Next:
  1. Sign in once:   claude        (then /exit)
                     codex login
  2. Card 1:         cd "$DIR" && .venv/bin/python research_desk.py "your question"
  3. Card 2:         export TELEGRAM_TOKEN=...  TELEGRAM_OWNER_ID=...
                     cd "$DIR" && .venv/bin/python research_desk_pulse.py
  Quick check of every engine:  .venv/bin/python -m lazybridge.demo
MSG

#!/usr/bin/env bash
# Convenience wrapper for running the full pipeline from a Linux shell,
# with a venv bootstrap so it works on a bare machine.
set -euo pipefail

ROWS="${1:-200000}"
K="${2:-4}"

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  echo "Creating virtualenv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install -q --upgrade pip
pip install -q -r requirements.txt

python -m src.pipeline run --rows "$ROWS" --k "$K"

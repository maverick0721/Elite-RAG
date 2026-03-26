#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "== Elite-RAG One-Command Demo =="
echo "Project root: $ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  echo ""
  echo "[1/5] Creating virtual environment (.venv)"
  python -m venv .venv
else
  echo ""
  echo "[1/5] Virtual environment already exists"
fi

echo ""
echo "[2/5] Activating virtual environment"
# shellcheck disable=SC1091
source ".venv/bin/activate"

echo ""
echo "[3/5] Installing dependencies"
python -m pip install --upgrade pip >/dev/null
pip install -r requirements.txt >/dev/null

echo ""
echo "[4/5] Running quickstart Q&A walkthrough"
python main.py --quickstart --question "What is retrieval augmented generation?"
python main.py --quickstart --question "Why is hybrid retrieval useful?"
python main.py --quickstart --question "What does reflection do in this system?"

echo ""
echo "[5/5] Running quickstart evaluation"
python evaluate.py --quickstart

echo ""
echo "Demo complete."
echo "Interactive mode: python main.py --quickstart"
echo "Full research mode (GPU/CUDA expected): python main.py"

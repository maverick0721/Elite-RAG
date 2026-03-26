#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-demo}"

case "$MODE" in
  demo)
    exec bash run_project_demo.sh
    ;;
  quickstart)
    exec python main.py --quickstart
    ;;
  smoke)
    exec python scripts/smoke_test.py
    ;;
  eval)
    exec python evaluate.py --quickstart
    ;;
  ui)
    exec streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Supported modes: demo | quickstart | smoke | eval | ui"
    exit 1
    ;;
esac

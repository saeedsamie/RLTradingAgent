#!/bin/bash
set -e

# Start training in the background
python - m scripts.run_rl_pipeline &

# Start FastAPI web server (serves dashboard)
exec uvicorn web.main_webview:app --host 0.0.0.0 --port 8000 
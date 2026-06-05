#!/bin/bash
# Double-click to open the web control panel on your Mac.

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  osascript -e 'display alert "Trading Agent" message "Run setup first:\ncd trading-agent\npython3 -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt\npip install fastapi uvicorn"'
  exit 1
fi

source .venv/bin/activate

pip install -q fastapi uvicorn 2>/dev/null || true

echo "Control panel at http://localhost:8080"
open "http://localhost:8080" 2>/dev/null || true
exec python -m uvicorn web.control_app:app --host 127.0.0.1 --port 8080

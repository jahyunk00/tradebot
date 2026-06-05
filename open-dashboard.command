#!/bin/bash
# Double-click this file on Mac to open the trading dashboard in your browser.

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  osascript -e 'display alert "Trading Agent" message "Run setup first:\ncd trading-agent\npython3 -m venv .venv\nsource .venv/bin/activate\npip install -r requirements.txt"'
  exit 1
fi

source .venv/bin/activate

echo "Starting dashboard at http://localhost:8501"
echo "Press Ctrl+C in Terminal to stop."

open "http://localhost:8501" 2>/dev/null || true
exec streamlit run dashboard.py --server.headless true

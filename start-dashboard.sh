#!/bin/bash
# Start the local dashboard (same as open-dashboard.command, for terminal use).

cd "$(dirname "$0")"
source .venv/bin/activate
streamlit run dashboard.py

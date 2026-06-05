#!/usr/bin/env python3
"""Railway entrypoint — cron trade bot or always-on control panel by service name."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    service = os.getenv("RAILWAY_SERVICE_NAME", "").strip().lower()
    if service == "tradebot-control":
        import uvicorn

        port = int(os.getenv("PORT", "8080"))
        uvicorn.run("web.control_app:app", host="0.0.0.0", port=port, log_level="info")
        return

    from scripts.railway_trade import main as trade_main

    trade_main()


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Copy base64 OAuth token to clipboard for Railway variable ROBINHOOD_OAUTH_B64
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOKEN="$ROOT/.tokens/robinhood_oauth.json"
if [[ ! -f "$TOKEN" ]]; then
  echo "Missing $TOKEN — run Robinhood login on your Mac first (dashboard or trade dry-run)."
  exit 1
fi
base64 < "$TOKEN" | tr -d '\n' | pbcopy
echo "Copied base64 token to clipboard. Paste into Railway → Variables → ROBINHOOD_OAUTH_B64"

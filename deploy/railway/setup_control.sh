#!/usr/bin/env bash
# Deploy the tradebot-control web panel on Railway.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RAILWAY="npx --yes @railway/cli"

echo "=== Tradebot control panel (Railway) ==="
echo ""

if ! $RAILWAY whoami &>/dev/null; then
  echo "Log in to Railway first..."
  $RAILWAY login
fi
echo "Logged in as: $($RAILWAY whoami)"

echo ""
echo "Step 1 — In Railway dashboard:"
echo "  • + New → GitHub Repo → jahyunk00/tradebot"
echo "  • Name the service: tradebot-control"
echo "  • Settings → Config file: railway.control.toml"
echo "  • Settings → Networking → Generate domain"
echo ""
read -r -p "Press Enter after tradebot-control service exists..."

echo ""
echo "Step 2 — Link CLI to tradebot-control (select it when prompted):"
$RAILWAY link || true

echo ""
echo "Step 3 — Set control panel variables..."
read -r -p "Control PIN (4-6 digits, blank = no PIN): " PIN
if [[ -n "${PIN}" ]]; then
  $RAILWAY variables set "CONTROL_PIN=${PIN}"
fi

# tradebot cron service id (from existing tradebot service)
CRON_ID="8cceaaa7-dedb-482c-8da5-c5fd257270ef"
$RAILWAY variables set "RAILWAY_CRON_SERVICE_ID=${CRON_ID}"

echo ""
echo "Step 4 — Create a Railway API token:"
echo "  https://railway.app/account/tokens → New token"
read -r -p "Paste RAILWAY_API_TOKEN: " API_TOKEN
if [[ -n "${API_TOKEN}" ]]; then
  $RAILWAY variables set "RAILWAY_API_TOKEN=${API_TOKEN}"
  echo "Toggle will sync ACTIVE_INVESTING to the tradebot cron service."
else
  echo "Skipped API token — toggle only updates this service's local state."
fi

echo ""
echo "Step 5 — Deploy control panel..."
$RAILWAY up --detach 2>/dev/null || $RAILWAY redeploy --yes 2>/dev/null || true

echo ""
echo "=== Done ==="
echo "Open your Railway domain for tradebot-control (Settings → Networking)."
echo "Local panel: ./open-control.command"

#!/usr/bin/env bash
# One-shot Railway setup: login → link project → set env vars → redeploy
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "=== Railway setup for tradebot ==="
echo ""

# --- OAuth token (from your Mac login) ---
TOKEN_FILE="$ROOT/.tokens/robinhood_oauth.json"
if [[ ! -f "$TOKEN_FILE" ]]; then
  echo "ERROR: Missing $TOKEN_FILE"
  echo "Connect Robinhood on your Mac first (dashboard or: python main.py trade)"
  exit 1
fi
base64 < "$TOKEN_FILE" | tr -d '\n' > "$ROOT/deploy/railway/.oauth.b64"
OAUTH_B64="$(cat "$ROOT/deploy/railway/.oauth.b64")"
echo "✓ Robinhood OAuth token encoded"

# --- Railway CLI ---
RAILWAY="npx --yes @railway/cli"

if ! $RAILWAY whoami &>/dev/null; then
  echo ""
  echo "Opening Railway login in your browser..."
  $RAILWAY login
fi
echo "✓ Logged in as: $($RAILWAY whoami)"

# --- Link project (skip if already linked) ---
if [[ ! -f "$ROOT/.railway/config.json" ]] && [[ ! -f "$HOME/.railway/config.json" ]]; then
  echo ""
  echo "Select your tradebot project and service when prompted:"
  $RAILWAY link
fi
echo "✓ Project linked"

# --- Environment variables ---
echo ""
echo "Setting Railway variables..."
$RAILWAY variables set "ACTIVE_INVESTING=true"
$RAILWAY variables set "ROBINHOOD_MCP_URL=https://agent.robinhood.com/mcp/trading"
$RAILWAY variables set "ROBINHOOD_OAUTH_B64=${OAUTH_B64}"

# Optional email from .env
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env" 2>/dev/null || true
  set +a
  if [[ "${SMTP_USER:-}" != "you@gmail.com" && -n "${SMTP_USER:-}" ]]; then
    $RAILWAY variables set "SMTP_HOST=${SMTP_HOST:-smtp.gmail.com}"
    $RAILWAY variables set "SMTP_PORT=${SMTP_PORT:-587}"
    $RAILWAY variables set "SMTP_USER=${SMTP_USER}"
    $RAILWAY variables set "SMTP_PASSWORD=${SMTP_PASSWORD}"
    $RAILWAY variables set "NOTIFY_EMAIL=${NOTIFY_EMAIL}"
    $RAILWAY variables set "SMTP_FROM=${SMTP_FROM:-$SMTP_USER}"
    echo "✓ Email notification vars set"
  fi
fi
echo "✓ Core variables set"

# --- Cron note ---
echo ""
echo "Cron schedule (from railway.toml): 35 13 * * 1-5  (= 9:35 AM Eastern, Mon–Fri)"
echo "Start command: python scripts/railway_trade.py"

# --- Redeploy ---
echo ""
read -r -p "Redeploy now? [Y/n] " REDEPLOY
REDEPLOY="${REDEPLOY:-Y}"
if [[ "$REDEPLOY" =~ ^[Yy] ]]; then
  $RAILWAY up --detach 2>/dev/null || $RAILWAY redeploy --yes 2>/dev/null || {
    echo "Push to GitHub to trigger deploy (already done if you pulled latest main)."
  }
  echo "✓ Deploy triggered"
fi

echo ""
read -r -p "Run a live test now? (only works during market hours 9:30–4 ET) [y/N] " TEST
if [[ "$TEST" =~ ^[Yy] ]]; then
  $RAILWAY run python scripts/railway_trade.py
fi

echo ""
echo "=== Done ==="
echo "Check logs: npx @railway/cli logs"
echo "Turn OFF Active investing on your Mac dashboard so only Railway trades live."

# Deploy the trading bot on Railway (cron ~$5/mo)

Run the boss agent once each weekday at market open. Your Mac can be off.

## What you get

- **Cron job** at **9:35 AM Eastern** (Mon–Fri)
- Runs `python scripts/railway_trade.py` → `main.py trade --execute` logic
- **$15/order cap** and guardrails from `guardrails.yaml`
- Optional **email** when each run finishes

---

## Before you start (on your Mac)

1. Robinhood OAuth already works locally (you’ve logged in once).
2. **Active investing** can stay ON locally — Railway uses its own flag via env var.
3. Push this repo to GitHub (or connect Railway to your repo).

---

## Step 1 — Create a Railway account

1. Go to [railway.com](https://railway.com) and sign up.
2. Subscribe to **Hobby ($5/mo)** when prompted (includes $5 usage credit).

---

## Step 2 — New project from GitHub

1. **New Project** → **Deploy from GitHub repo**.
2. Select your repo (`tradebot` or this project).
3. If the repo root is not `trading-agent`, set **Root Directory** → `trading-agent`  
   (Project → Service → Settings → Root Directory).

---

## Step 3 — Configure as a cron service

1. Open your service → **Settings**.
2. **Cron Schedule** should pick up from `railway.toml`:
   ```
   35 13 * * 1-5
   ```
   That is **9:35 AM EDT** (13:35 UTC).  
   **Nov–Mar (EST):** change to `35 14 * * 1-5` in `railway.toml` or in the dashboard.
3. **Start Command** (should match `railway.toml`):
   ```
   python scripts/railway_trade.py
   ```
4. Do **not** run the Streamlit dashboard on this service — cron only.

---

## Step 4 — Environment variables

In Railway → your service → **Variables**:

| Variable | Value |
|----------|--------|
| `ACTIVE_INVESTING` | `true` |
| `ROBINHOOD_MCP_URL` | `https://agent.robinhood.com/mcp/trading` |
| `ROBINHOOD_OAUTH_B64` | *(see below)* |
| `SMTP_HOST` | `smtp.gmail.com` *(optional, for email alerts)* |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | your email |
| `SMTP_PASSWORD` | Gmail app password |
| `NOTIFY_EMAIL` | where to send alerts |
| `SMTP_FROM` | same as SMTP_USER |

### OAuth token (required)

On your Mac, from `trading-agent/`:

```bash
chmod +x deploy/railway/encode_oauth.sh
./deploy/railway/encode_oauth.sh
```

Paste the clipboard into Railway as **`ROBINHOOD_OAUTH_B64`**.

If the token expires later, re-run login on your Mac and update this variable.

---

## Step 5 — Resources

In **Settings → Resources**, use at least:

- **512 MB – 1 GB RAM** (HMM + scipy; bump to 1 GB if builds fail)
- **Shared CPU** is fine

---

## Step 6 — Deploy and test

1. Click **Deploy** (or push to GitHub to trigger deploy).
2. **Manual test:** Settings → **Run now** (or Redeploy) while the US market is open.
3. Check **Logs** for:
   - `Boss pick: ...` or `CASH`
   - `EXECUTED` or `Blocked: Outside US regular market hours`
4. Check email if SMTP is configured.

**First dry test during market hours** is strongly recommended before relying on cron.

---

## Step 7 — Turn off local Mac trading (optional)

Once Railway works:

- Turn **Active investing OFF** on your Mac dashboard so only Railway trades live.
- Or leave Mac on paper-only.

---

## Timezone cheat sheet (9:35 AM Eastern)

| Season | UTC cron |
|--------|----------|
| EDT (Mar–Nov) | `35 13 * * 1-5` |
| EST (Nov–Mar) | `35 14 * * 1-5` |

Railway cron always uses **UTC**.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| OAuth / auth errors | Re-run `encode_oauth.sh`, update `ROBINHOOD_OAUTH_B64` |
| Out of memory | Raise RAM to 1 GB |
| Blocked: market hours | Normal outside 9:30–4 ET; test during open |
| Blocked: analyze only | Set `ACTIVE_INVESTING=true` on Railway |
| Cron skipped | Previous run still running; ensure script exits (it should) |
| No email | Set all `SMTP_*` vars; check spam |

---

## Files added for Railway

```
railway.toml              # cron schedule + start command
nixpacks.toml             # Python 3.12 + deploy deps
requirements-deploy.txt   # lighter than full requirements (no Streamlit/Kronos)
scripts/railway_trade.py  # cron entrypoint
deploy/railway/boss_weights.json   # seed weights if logs/ missing
deploy/railway/encode_oauth.sh   # helper for OAuth secret
```

---

## Cost

- **Hobby plan:** $5/mo base
- Daily ~3–5 min runtime usually stays within included credit → **~$5 total**

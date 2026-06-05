# Web control panel

Toggle **Active investing** from your phone — no Streamlit required.

## Local (Mac)

Double-click **`open-control.command`** or:

```bash
cd trading-agent
source .venv/bin/activate
pip install fastapi uvicorn
python -m uvicorn web.control_app:app --host 127.0.0.0 --port 8080
```

Open http://localhost:8080

Optional PIN: `CONTROL_PIN=1234 python -m uvicorn web.control_app:app --port 8080`

---

## Railway (always-on URL)

### Quick setup

```bash
chmod +x deploy/railway/setup_control.sh
./deploy/railway/setup_control.sh
```

Or manually:

1. Railway → **+ New** → GitHub repo `tradebot`
2. Name: **`tradebot-control`**
3. **Settings → Build** → Config file: **`railway.control.toml`**
4. **Settings → Deploy** → leave cron **empty** (web server, not cron)
5. **Networking** → Generate domain
6. **Variables**:

| Variable | Value |
|----------|--------|
| `CONTROL_PIN` | Your PIN (e.g. `4829`) |
| `RAILWAY_API_TOKEN` | From [railway.app/account/tokens](https://railway.app/account/tokens) |
| `RAILWAY_CRON_SERVICE_ID` | `8cceaaa7-dedb-482c-8da5-c5fd257270ef` (tradebot cron) |

Railway auto-injects `RAILWAY_PROJECT_ID` and `RAILWAY_ENVIRONMENT_ID`.

### How sync works

When you tap **Turn ON/OFF**, the panel:

1. Saves `runtime_state.json` locally
2. Calls Railway API → sets `ACTIVE_INVESTING` on the **tradebot** cron service

The next cron run (9:35 AM ET) reads that env var — **no shared volume needed**.

---

## Two services

| Service | Type | URL |
|---------|------|-----|
| `tradebot` | Cron | No public URL |
| `tradebot-control` | Web | Your Railway domain |

---

## API

`GET /api/status` — JSON with toggle state, equity, guardrails, last run.

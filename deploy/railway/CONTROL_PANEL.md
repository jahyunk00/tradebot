# Control panel on Railway (toggle active investing from your phone)

The **tradebot** service is a cron job (no always-on UI). Add a second **web** service for the toggle page.

## Fix `pip: command not found` build errors

That error happens when Railway uses **Nixpacks/Railpack** with `RAILPACK_INSTALL_CMD` instead of the **Dockerfile**.

1. In Railway → **tradebot** → **Settings** → **Build**
   - Builder: **Dockerfile**
   - Dockerfile path: `Dockerfile`
2. **Delete** these variables if present (they break the build):
   - `RAILPACK_INSTALL_CMD`
   - `RAILPACK_START_CMD`
3. Redeploy

The repo’s `railway.toml` already points at `Dockerfile`.

---

## Step 1 — Create a shared volume

1. Railway project → **+ New** → **Volume**
2. Name: `tradebot-state`
3. Attach to **tradebot** service → mount path: **`/data`**
4. (After step 2) attach the same volume to **tradebot-control** at **`/data`**

## Step 2 — Add control web service

1. **+ New** → **GitHub Repo** → same `tradebot` repo
2. Name the service: **`tradebot-control`**
3. **Settings** → **Build**:
   - Config file path: **`railway.control.toml`**
   - (Uses `Dockerfile.control` — always-on web app, not cron)
4. **Settings** → **Deploy**:
   - **No cron schedule** (leave empty — this is a web server)
   - Generate domain: **Settings → Networking → Generate domain**
5. **Variables** (tradebot-control):

| Variable | Value |
|----------|--------|
| `STATE_DIR` | `/data` |
| `CONTROL_PIN` | your 4–6 digit PIN (e.g. `1234`) |

6. **Variables** on **tradebot** (cron) — add:

| Variable | Value |
|----------|--------|
| `STATE_DIR` | `/data` |

Both services read/write **`/data/runtime_state.json`** on the shared volume.

## Step 3 — Use the control page

Open your Railway URL, e.g. `https://tradebot-control-production.up.railway.app`

- **Green / Turn ON** → live trades allowed at next cron (9:35 AM ET)
- **Red / Turn OFF** → cron runs but stays paper / blocks live orders

---

## Local control (Mac)

You already have this — no Railway needed:

```bash
cd trading-agent
./open-dashboard.command
```

Toggle **Active investing** in Streamlit (uses `logs/runtime_state.json` locally, not Railway).

---

## Two services summary

| Service | Type | Purpose |
|---------|------|---------|
| `tradebot` | Cron `35 13 * * 1-5` | Runs boss trade at market open |
| `tradebot-control` | Web (always on) | Toggle active investing |

Both share volume **`/data`** via `STATE_DIR=/data`.

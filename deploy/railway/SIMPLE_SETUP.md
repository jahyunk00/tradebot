# Railway — simplified setup (read this first)

Your **tradebot** service should be ONE cron job. Ignore old FAILED rows in the dashboard — those were from earlier broken configs.

## Current correct setup

| Setting | Value |
|---------|--------|
| Builder | **Dockerfile** (not Nixpacks) |
| Start command | `python scripts/railway_trade.py` |
## Recommended: always-on worker (trades all day)

Cron often **skips runs** when each cycle takes ~60–90s (overlap) or when Robinhood OAuth fails silently. Use a **worker** instead:

| Setting | Value |
|---------|--------|
| Start command | `python scripts/railway_entry.py` |
| **Cron schedule** | **None / empty** (not a cron service) |
| `TRADE_LOOP_SECONDS` | `300` (trade every 5 min during market hours) |
| `ACTIVE_INVESTING` | `true` |
| `ROBINHOOD_OAUTH_B64` | your token (from encode_oauth.sh) |

Check Railway logs for `RAILWAY_RUN_COMPLETE` after each cycle.

## Alternative: cron (less reliable)

| Setting | Value |
|---------|--------|
| Cron | `*/5 13-20 * * 1-5` (every 5 min, US market hours UTC) |
| `ACTIVE_INVESTING` | `true` or `false` (controls live vs paper) |
| `ROBINHOOD_OAUTH_B64` | your token (from encode_oauth.sh) |

**Do NOT set** `STATE_DIR`, `RAILPACK_INSTALL_CMD`, or `RAILPACK_START_CMD` unless you add a volume (advanced).

## Fix failures in Railway UI

1. **tradebot → Settings → Build**
   - Builder: **Dockerfile**
   - Dockerfile path: `Dockerfile`
   - Clear any custom **Build command** field

2. **Variables** — delete if present:
   - `RAILPACK_INSTALL_CMD`
   - `RAILPACK_START_CMD`
   - `STATE_DIR` (unless you mounted a volume)

3. **Redeploy** (Deploy → Redeploy latest)

4. **Ignore old FAILED deployments** — Railway keeps history; only the **top SUCCESS** row matters.

## Turn active trading ON/OFF (simple)

**Option 1 — Railway dashboard (works now, no extra service)**

1. tradebot → **Variables**
2. Set `ACTIVE_INVESTING` = `true` (live) or `false` (paper/no orders)
3. Next cron run (9:35 AM ET weekday) uses the new value

**Option 2 — Mac dashboard**

```bash
./open-dashboard.command
```

Toggle **Active investing** (local only, not Railway).

**Option 3 — Web control page** (optional, extra service)

See `CONTROL_PANEL.md` only if you want a phone-friendly toggle with a shared volume.

## Verify it works

During market hours (9:30 AM–4 PM ET):

```bash
cd trading-agent
npx @railway/cli link -p reasonable-balance -s tradebot
npx @railway/cli run python scripts/railway_trade.py
```

Check logs:

```bash
npx @railway/cli logs
```

Success looks like: boss pick or CASH, not `pip not found` or `Read-only file system`.

## What went wrong before

| Error | Cause |
|-------|--------|
| `pip: command not found` | Nixpacks + RAILPACK_INSTALL_CMD (fixed: use Dockerfile) |
| `main.py: error: command required` | Wrong start command (fixed: railway_trade.py) |
| `Read-only file system: /data` | STATE_DIR=/data without a volume (fixed: remove STATE_DIR) |

## Cost

~$5/month Hobby plan. Cron runs ~1×/day for a few minutes.

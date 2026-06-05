"""Email notifications for briefings and trade activity."""

from __future__ import annotations

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


def _configured() -> bool:
    return bool(
        os.getenv("SMTP_HOST")
        and os.getenv("SMTP_USER")
        and os.getenv("SMTP_PASSWORD")
        and os.getenv("NOTIFY_EMAIL")
    )


def send_email(subject: str, body: str, *, html: bool = False) -> bool:
    """Send notification email. Returns True if sent, False if skipped or failed."""
    if not _configured():
        logger.info("Email not configured — skip notify (set SMTP_* and NOTIFY_EMAIL in .env).")
        return False

    host = os.environ["SMTP_HOST"]
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASSWORD"]
    to_addr = os.environ["NOTIFY_EMAIL"]
    from_addr = os.getenv("SMTP_FROM", user)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.attach(MIMEText(body, "plain"))
    if html:
        msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(from_addr, [to_addr], msg.as_string())
        logger.info("Email sent to %s: %s", to_addr, subject)
        return True
    except Exception as exc:
        logger.warning("Email failed: %s", exc)
        return False


def notify_briefing(run_id: str, briefing_text: str, *, bankroll: dict | None = None) -> bool:
    subject = f"Tradebot daily brief — {run_id}"
    extra = ""
    if bankroll:
        extra = (
            f"\n\nAccount: ${bankroll.get('equity_usd', '?')} equity "
            f"({bankroll.get('gain_pct', 0):+.1f}% vs ${bankroll.get('initial_usd', '?')} start)"
        )
    body = f"Your daily market brief:\n\n{briefing_text}{extra}\n\n— Tradebot (analyze-only)"
    return send_email(subject, body)


def notify_run_summary(
    run_id: str,
    analysis_excerpt: str,
    trade_intents: list[dict],
    *,
    bankroll: dict | None = None,
    mode: str = "analyze_only",
) -> bool:
    subject = f"Tradebot weekly run — {run_id}"
    lines = [f"Mode: {mode}\n"]
    if bankroll:
        lines.append(
            f"Bankroll: ${bankroll.get('equity_usd')} equity, "
            f"${bankroll.get('cash_usd')} cash ({bankroll.get('gain_pct', 0):+.1f}%)\n"
        )
    if trade_intents:
        lines.append("Trade intents:")
        for t in trade_intents:
            status = "ALLOWED" if t.get("allowed") else "BLOCKED"
            lines.append(f"  [{status}] {t.get('intent')}")
    else:
        lines.append("No trade intents this run.")
    lines.append(f"\n--- Analysis excerpt ---\n{analysis_excerpt[:3000]}")
    return send_email(subject, "\n".join(lines))

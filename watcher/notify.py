"""Push alerts to the phone. ntfy is the default (no account needed); Telegram and
Discord light up if their env vars are set. Every send is also logged to the day's
alerts.jsonl by the caller, so nothing is lost if a transport is down."""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger("notify")

# ntfy priority: 1 min, 2 low, 3 default, 4 high, 5 urgent
CATEGORY_TAGS = {
    "TRADE": "moneybag",
    "NEWS": "newspaper",
    "MARKET_MOVE": "chart_with_upwards_trend",
    "PLAY": "dart",
    "CHAT_SPIKE": "speech_balloon",
    "SELF_DEV": "brain",
    "BRIEF": "memo",
    "SYSTEM": "robot",
    "OTHER": "eyes",
}


def send(title: str, body: str, priority: int = 3, category: str = "OTHER", click: str | None = None) -> bool:
    priority = max(1, min(5, int(priority)))
    ok = False
    topic = os.environ.get("NTFY_TOPIC")
    if topic:
        ok |= _ntfy(topic, title, body, priority, category, click)
    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        ok |= _telegram(title, body, click)
    if os.environ.get("DISCORD_WEBHOOK_URL"):
        ok |= _discord(title, body, click)
    if not (topic or os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("DISCORD_WEBHOOK_URL")):
        log.warning("no notification transport configured; would send: %s | %s", title, body)
    return ok


def _ntfy(topic, title, body, priority, category, click) -> bool:
    server = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    headers = {
        # Headers must be latin-1; the JSON publish API avoids that limit for emoji/titles.
    }
    payload = {
        "topic": topic,
        "title": title[:250],
        "message": body[:3900],
        "priority": priority,
        "tags": [CATEGORY_TAGS.get(category, "eyes")],
    }
    if click:
        payload["click"] = click
    if os.environ.get("NTFY_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['NTFY_TOKEN']}"
    for attempt in range(3):
        try:
            r = requests.post(server, json=payload, headers=headers, timeout=15)
            if r.status_code < 300:
                return True
            log.warning("ntfy %s: %s", r.status_code, r.text[:200])
            if r.status_code == 429:
                return False
        except requests.RequestException as e:
            log.warning("ntfy error (attempt %d): %s", attempt + 1, e)
    return False


def _telegram(title, body, click) -> bool:
    tok, chat = os.environ["TELEGRAM_BOT_TOKEN"], os.environ["TELEGRAM_CHAT_ID"]
    text = f"*{title}*\n{body}" + (f"\n{click}" if click else "")
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{tok}/sendMessage",
            json={"chat_id": chat, "text": text[:4000], "parse_mode": "Markdown", "disable_web_page_preview": True},
            timeout=15,
        )
        if r.status_code >= 300:  # markdown parse failures: retry as plain text
            r = requests.post(
                f"https://api.telegram.org/bot{tok}/sendMessage",
                json={"chat_id": chat, "text": text[:4000]},
                timeout=15,
            )
        return r.status_code < 300
    except requests.RequestException as e:
        log.warning("telegram error: %s", e)
        return False


def _discord(title, body, click) -> bool:
    content = f"**{title}**\n{body}" + (f"\n<{click}>" if click else "")
    try:
        r = requests.post(os.environ["DISCORD_WEBHOOK_URL"], json={"content": content[:1990]}, timeout=15)
        return r.status_code < 300
    except requests.RequestException as e:
        log.warning("discord error: %s", e)
        return False

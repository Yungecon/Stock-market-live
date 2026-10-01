"""Chat analytics: message rate vs baseline, tickers people are naming, owner/mod posts."""
from __future__ import annotations

import collections
import json
import logging
import re
import statistics
import time
from pathlib import Path

import requests

from .config import DATA
from .youtube import ChatMsg

log = logging.getLogger("chat")

# Uppercase words that are tickers on paper but almost always just words in chat.
STOP = set("""
A I AM AN AND ARE AS AT BE BIG BY CAN CEO CFO DD DO FOR GO GOOD HAS HE HI HIS IF IN IPO IS IT ITS ME MY NEW NO NOT NOW OF OK ON ONE OR OUT
PM SO THE TO UP US USA WE YES YOU LOL LMAO OMG WTF ATH ETF GDP CPI FED FOMC PPI EPS AI TV ALL ANY ARE BUY SELL HOLD LONG SHORT CALL CALLS PUTS PUT
REAL FUN LOVE LIFE NICE HOPE BEST SEE WOW JUST GOD MAN BRO EV RUN FAST NEXT OPEN TRUE CASH EAT KEY PLAY SAVE MOVE WELL WAY DAY ELSE EDIT TIME
""".split())
CASHTAG = re.compile(r"\$([A-Za-z]{1,5})\b")
CAPS = re.compile(r"\b([A-Z]{2,5})\b")
PLAY_WORDS = re.compile(
    r"\b(calls?|puts?|strike|exp(iry|iration)?|0dte|long|short|bought|sold|entry|target|pt|stop|loading|yolo|position|shares|contracts?)\b",
    re.I,
)


def load_tickers() -> set[str]:
    cache = DATA / "tickers.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < 30 * 86400:
        return set(json.loads(cache.read_text()))
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "stock-market-live-watcher contact@example.com"},
            timeout=20,
        )
        r.raise_for_status()
        tick = sorted({v["ticker"].upper().replace("-", ".") for v in r.json().values()})
        tick += ["SPY", "QQQ", "IWM", "DIA", "VIX", "SPX", "NDX", "TLT", "GLD", "SLV", "USO", "UVXY", "SQQQ", "TQQQ", "SOXL", "ARKK", "BTC", "ETH"]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(sorted(set(tick))))
        return set(tick)
    except Exception as e:  # noqa: BLE001
        log.warning("ticker list unavailable (%s); cashtags only", e)
        return set(json.loads(cache.read_text())) if cache.exists() else set()


def extract_tickers(text: str, known: set[str]) -> set[str]:
    out = {m.upper() for m in CASHTAG.findall(text)}
    for w in CAPS.findall(text):
        if w in known and w not in STOP:
            out.add(w)
    return out


class ChatStats:
    def __init__(self, known: set[str], spike_ratio: float, spike_min: int, ticker_min_authors: int):
        self.known = known
        self.spike_ratio, self.spike_min, self.ticker_min_authors = spike_ratio, spike_min, ticker_min_authors
        self.per_min: collections.Counter[int] = collections.Counter()
        self.window: list[ChatMsg] = []  # messages since the last analysis window
        self.ticker_authors: dict[str, set[str]] = collections.defaultdict(set)
        self.ticker_hist: collections.Counter[str] = collections.Counter()  # whole-day mentions
        self.peak_rate = 0

    def add(self, msgs: list[ChatMsg]):
        for m in msgs:
            self.per_min[int(m.ts // 60)] += 1
            self.window.append(m)
            for t in extract_tickers(m.text, self.known):
                self.ticker_authors[t].add(m.author_id or m.author)
                self.ticker_hist[t] += 1

    def rate_now(self) -> tuple[float, float]:
        """(messages in the last full minute, trailing-30-minute median per minute)."""
        cur = int(time.time() // 60) - 1
        last = self.per_min.get(cur, 0)
        trail = [self.per_min.get(cur - i, 0) for i in range(1, 31)]
        trail = [x for x in trail if x] or [0]
        return float(last), float(statistics.median(trail))

    def spike(self) -> tuple[bool, float, float]:
        last, base = self.rate_now()
        self.peak_rate = max(self.peak_rate, int(last))
        return (last >= self.spike_min and last >= base * self.spike_ratio and base > 0), last, base

    def take_window(self) -> dict:
        """Summarize and reset the per-window buffer."""
        msgs, self.window = self.window, []
        authors_by_ticker, self.ticker_authors = self.ticker_authors, collections.defaultdict(set)
        hot = sorted(((t, len(a)) for t, a in authors_by_ticker.items()), key=lambda x: -x[1])[:12]
        flagged = [m for m in msgs if m.role in ("owner", "moderator") or m.paid]
        plays = [m for m in msgs if PLAY_WORDS.search(m.text) and extract_tickers(m.text, self.known)]
        last, base = self.rate_now()
        return {
            "count": len(msgs),
            "rate_last_min": last,
            "rate_baseline": base,
            "hot_tickers": hot,
            "crowd_tickers": [t for t, n in hot if n >= self.ticker_min_authors],
            "owner_mod": [f"[{m.role}] {m.author}: {m.text}" for m in flagged][-20:],
            "plays": [f"{m.author}: {m.text}" for m in plays][-30:],
            "sample": [f"{m.author}: {m.text}" for m in msgs[-40:]],
        }


class ChatLog:
    """Append-only raw chat for the day (gzip keeps a busy chat small enough to commit)."""

    def __init__(self, path: Path):
        import gzip

        self.f = gzip.open(path, "at", encoding="utf-8")

    def write(self, msgs: list[ChatMsg]):
        for m in msgs:
            self.f.write(json.dumps(m.to_json(), ensure_ascii=False) + "\n")

    def close(self):
        self.f.close()

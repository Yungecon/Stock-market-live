"""The judgment layer: reads each ~90s window of transcript + chat and decides what's
worth a ping. Claude when ANTHROPIC_API_KEY is set; keyword heuristics otherwise."""
from __future__ import annotations

import base64
import datetime as dt
import logging
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

log = logging.getLogger("brain")

Category = Literal["TRADE", "NEWS", "MARKET_MOVE", "PLAY", "CHAT_SPIKE", "SELF_DEV", "OTHER"]
Segment = Literal[
    "market_news", "trading", "market_analysis", "self_development", "chat_interaction",
    "entertainment", "ad_or_sponsor", "break_or_silence", "other",
]


class Alert(BaseModel):
    category: Category
    priority: int  # 1..5
    title: str
    body: str
    tickers: list[str]
    evidence: str  # short verbatim quote from the transcript or chat


class WindowAnalysis(BaseModel):
    segment: Segment
    signal_level: int  # 0 (pure fluff) .. 5 (must-know)
    alerts: list[Alert]
    notes: list[str]  # facts worth keeping for the daily brief, even if not ping-worthy
    look_at_screen: bool
    screen_reason: str


class ScreenRead(BaseModel):
    summary: str
    tickers: list[str]
    position_details: str


# Prices per million tokens (input, output) for the budget guard.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
}

SYSTEM = """You are the ears of a busy viewer who keeps a stock-market YouTube livestream on in the background. \
The streamer reads market news all day, comments on it, and makes his own trades live. He is also an entertainer: \
jokes, games, banter, tangents, sponsor reads, and off-topic chat are common and are NOT signal.

The viewer wants a phone ping for:
- TRADE — the streamer buys, sells, adds, trims, opens or closes a position (stock or options), or says he is about to. \
Include ticker, direction, size/price/strike/expiry when stated. Highest priority.
- NEWS — market-moving headlines he reads or reacts to (macro data, Fed, earnings, guidance, M&A, regulation, \
analyst moves, big company news) with why it matters.
- MARKET_MOVE — he calls out a sharp move in an index or a ticker, halts, breakouts, a sell-off.
- PLAY — a concrete trade idea from him or a crowd of chatters converging on the same ticker/setup.
- CHAT_SPIKE — chat suddenly erupts; say what it is about (a ticker, a headline, a call) in one line.
- SELF_DEV — the daily philosophy / self-development segment starts (usually around {self_dev_hint} {tz}). \
Ping once when it begins; afterwards capture its key lessons as notes, not more pings, unless something is exceptional.

Do NOT ping for: jokes, games (Minecraft etc.), personal stories, chat banter, merch/sponsor reads, repeated \
mentions of something already alerted, or vague musing with no market content.

Priority: 5 = he traded / is trading right now; 4 = big news or the self-dev segment starting; 3 = useful news, \
notable move, concrete play, real chat spike; 2 = minor but relevant; 1 = barely relevant. Early on, the viewer \
prefers MORE signal over less — when in doubt between 2 and nothing, emit a 2.

Titles: under 70 chars, start with the ticker(s) when there are any (e.g. "NVDA — bought 200 shares"). \
Body: 1-3 short sentences, plain words, no hype. evidence: a short verbatim quote. Never invent prices or numbers \
that are not in the transcript or chat; transcripts are machine speech-to-text, so fix obvious mis-hearings of \
tickers only when context makes it clear.

Never repeat an alert listed under ALREADY ALERTED unless there is a genuinely new development (then say what is new).
Set look_at_screen=true only when the speech clearly refers to something visible that matters (his positions, \
an order ticket, a chart he is pointing at) and the words alone don't say what it is.
notes: 0-4 terse factual bullets worth keeping for the end-of-day brief (news items, his views, positions, \
self-development lessons). Empty when the window was fluff.

=== HANDOFF (what previous days learned about this stream; trust it, but the transcript wins) ===
{handoff}
"""


class Brain:
    def __init__(self, cfg, handoff: str):
        self.cfg = cfg
        self.handoff = handoff
        self.spent_usd = 0.0
        self.calls = 0
        self.failures = 0
        self.client = None
        if cfg.has_llm:
            import anthropic

            self.client = anthropic.Anthropic(max_retries=3, timeout=120)
        tz = cfg["stream"]["timezone"]
        self.system = SYSTEM.format(self_dev_hint=cfg["stream"]["self_dev_hint"], tz=tz, handoff=handoff or "(none yet — first day)")

    @property
    def llm_on(self) -> bool:
        return self.client is not None and self.spent_usd < self.cfg["analysis"]["daily_budget_usd"]

    def _cost(self, model, usage) -> float:
        pin, pout = PRICES.get(model, (5.0, 25.0))
        cache_r = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_w = getattr(usage, "cache_creation_input_tokens", 0) or 0
        return (usage.input_tokens * pin + cache_w * pin * 1.25 + cache_r * pin * 0.1 + usage.output_tokens * pout) / 1e6

    def analyze(self, window_text: str) -> WindowAnalysis | None:
        if not self.llm_on:
            return None
        model = self.cfg["analysis"]["live_model"]
        try:
            resp = self.client.messages.parse(
                model=model,
                max_tokens=4000,
                system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": window_text}],
                output_config={"effort": self.cfg["analysis"]["live_effort"]},
                output_format=WindowAnalysis,
            )
            self.calls += 1
            self.spent_usd += self._cost(model, resp.usage)
            if resp.stop_reason == "refusal" or resp.parsed_output is None:
                log.warning("no parsed output (stop_reason=%s)", resp.stop_reason)
                return None
            return resp.parsed_output
        except Exception as e:  # noqa: BLE001 — fall back to heuristics on any API trouble
            self.failures += 1
            log.warning("LLM analyze failed: %s", str(e)[:300])
            return None

    def read_screen(self, image: Path, context: str) -> ScreenRead | None:
        if not self.llm_on:
            return None
        model = self.cfg["analysis"]["live_model"]
        try:
            b64 = base64.standard_b64encode(image.read_bytes()).decode()
            resp = self.client.messages.parse(
                model=model,
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": "This is a frame from a stock-market livestream. The streamer just said:\n"
                         f"{context}\n\nRead what's on screen that relates to it: tickers, positions, order details, "
                         "prices, chart levels. Only report what is legible."},
                    ],
                }],
                output_config={"effort": "low"},
                output_format=ScreenRead,
            )
            self.spent_usd += self._cost(model, resp.usage)
            return resp.parsed_output
        except Exception as e:  # noqa: BLE001
            log.warning("screen read failed: %s", str(e)[:300])
            return None


# ---------- heuristic fallback (no API key, budget exhausted, or API down) ----------

TRADE_RX = re.compile(
    r"\b(i|we)('m| am| just| have| had|'ve)?\s*(just\s+)?(bought|sold|buying|selling|added|adding|trimmed|trimming|"
    r"took profits?|taking profits?|closed|closing|opened|opening|entered|exited|loaded|loading|scaled (in|out))\b"
    r"|\b(buying|selling) (some|more|the)\b|\bstopped out\b|\b(my|our) (position|entry|calls|puts|shares)\b",
    re.I,
)
NEWS_RX = re.compile(
    r"\b(breaking|just in|headline|announced|announces|reports?|earnings|guidance|beat|miss(ed)?|downgrade[sd]?|"
    r"upgrade[sd]?|fed|powell|fomc|rate (cut|hike)|cpi|ppi|jobs report|payrolls|unemployment|tariffs?|sec|lawsuit|"
    r"acquir(e|es|ed|ing)|merger|buyback|halted|bankrupt)\b",
    re.I,
)
SELF_DEV_RX = re.compile(r"\b(self[- ]?development|philosophy|mindset|discipline|stoic|personal growth|life lesson)\b", re.I)


def heuristic(new_lines: list[str], chat: dict, now: dt.datetime, known_tickers: set[str], self_dev_hint: str) -> WindowAnalysis:
    from .chat import extract_tickers

    alerts: list[Alert] = []
    text = " ".join(new_lines)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    trade = [s for s in sentences if TRADE_RX.search(s)]
    news = [s for s in sentences if NEWS_RX.search(s)]
    def tix(ss):  # speech-to-text has no $ or caps, so match words against the ticker list loosely
        return sorted(extract_tickers(" ".join(ss).upper(), known_tickers))[:5]

    if trade:
        tick = tix(trade)
        alerts.append(Alert(category="TRADE", priority=4, title=f"{' '.join(tick) + ' — ' if tick else ''}possible trade talk",
                            body=" ".join(trade)[:400], tickers=tick, evidence=trade[0][:200]))
    if len(news) >= 2:
        tick = tix(news)
        alerts.append(Alert(category="NEWS", priority=2, title=f"{' '.join(tick) + ' — ' if tick else ''}news on stream",
                            body=" ".join(news[:3])[:400], tickers=tick, evidence=news[0][:200]))
    h, m = (int(x) for x in self_dev_hint.split(":"))
    near = abs((now.hour * 60 + now.minute) - (h * 60 + m)) <= 30
    seg: Segment = "self_development" if (near and SELF_DEV_RX.search(text)) else ("trading" if trade else ("market_news" if news else "other"))
    for line in chat.get("owner_mod", []):
        if line.startswith("[owner]"):
            alerts.append(Alert(category="PLAY", priority=3, title="Streamer posted in chat", body=line[:300], tickers=[], evidence=line[:200]))
    return WindowAnalysis(segment=seg, signal_level=3 if alerts else 0, alerts=alerts, notes=[],
                          look_at_screen=False, screen_reason="")

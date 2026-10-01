"""End-of-day wrap: market brief for the viewer, capture-quality brief for the agent,
and a rewritten HANDOFF.md that tomorrow's run reads before it does anything."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

from . import notify
from .config import BRIEFS, HANDOFF, day_dir

log = logging.getLogger("daily")


class Wrap(BaseModel):
    market_brief_md: str
    capture_brief_md: str
    handoff_md: str
    push_summary: str  # <= 600 chars, for the phone


WRAP_PROMPT = """You are the end-of-day editor for an agent that watches a stock-market YouTube livestream on behalf of a viewer \
and pings their phone when something important happens. Below is everything captured today.

Write three things:

1. market_brief_md — for the viewer. Markdown, skimmable in 2 minutes. Sections:
   "## The day in one paragraph", "## His trades" (every trade or position change, with time, ticker, direction, size/price when known; \
"No trades caught" if none), "## News that mattered" (ranked, with why it matters), "## Plays & chat" (crowd tickers, notable \
calls), "## Self-development hour" (the lessons, in his framing — the viewer loves this segment; say plainly if it was missed or \
didn't happen), "## Watch tomorrow". Times in {tz}. No invented numbers.

2. capture_brief_md — for the agent itself, honest QA. Coverage (minutes live vs minutes captured, audio restarts, transcription lag, \
chat errors), alert quality (which pings were probably noise, what was probably missed, judged from the transcript), how the \
streamer structured the day, and 3-6 concrete adjustments for tomorrow (filters, timing, keywords, thresholds).

3. handoff_md — the FULL new contents of HANDOFF.md, the living document the agent reads before every window tomorrow. \
Start from the current handoff below and update it: keep what's still true, fix what today disproved, add what today taught. \
Keep it under ~900 words. Sections: "# Handoff", "## How the stream runs" (schedule, recurring segments, the self-dev hour), \
"## The streamer" (style, tells for when he's actually trading vs. joking, his current positions/watchlist), \
"## Signal vs noise" (what earned pings, what was fluff), "## Open threads" (stories/positions to follow up), \
"## Capture notes" (technical lessons), and a dated "## Changelog" line for today.

push_summary — 2-4 plain sentences for a phone notification: the headline of the day, his trades, one self-dev takeaway.

=== CURRENT HANDOFF ===
{handoff}

=== CAPTURE METRICS ===
{metrics}

=== ALERTS SENT TODAY ===
{alerts}

=== WINDOW NOTES ===
{notes}

=== FULL TRANSCRIPT ({tz}, machine speech-to-text) ===
{transcript}
"""


def _read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def wrap_day(cfg, day: str, repo_url: str | None = None) -> None:
    d = day_dir(day)
    metrics = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
    alerts = _read_jsonl(d / "alerts.jsonl")
    notes = _read_jsonl(d / "notes.jsonl")
    transcript = (d / "transcript.md").read_text() if (d / "transcript.md").exists() else ""
    handoff = HANDOFF.read_text() if HANDOFF.exists() else ""
    BRIEFS.mkdir(exist_ok=True)
    market_p, capture_p = BRIEFS / f"{day}-market.md", BRIEFS / f"{day}-capture.md"

    wrap: Wrap | None = None
    if cfg.has_llm and (transcript or alerts):
        import anthropic

        client = anthropic.Anthropic(max_retries=3)
        prompt = WRAP_PROMPT.format(
            tz=cfg["stream"]["timezone"], handoff=handoff or "(empty)", metrics=json.dumps(metrics, indent=1),
            alerts="\n".join(f"- {a['time']} [{a['category']} p{a['priority']}] {a['title']} — {a['body']}" for a in alerts) or "(none)",
            notes="\n".join(f"- {n['time']} ({n.get('segment','')}) {n['note']}" for n in notes) or "(none)",
            transcript=transcript or "(no transcript captured)",
        )
        try:
            with client.messages.stream(
                model=cfg["analysis"]["wrap_model"],
                max_tokens=32000,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": cfg["analysis"]["wrap_effort"], "format": {
                    "type": "json_schema", "schema": _schema()}},
            ) as stream:
                msg = stream.get_final_message()
            txt = "".join(b.text for b in msg.content if b.type == "text")
            wrap = Wrap.model_validate_json(txt)
        except Exception as e:  # noqa: BLE001
            log.warning("LLM wrap failed, writing plain wrap: %s", str(e)[:300])

    if wrap is None:
        wrap = _plain_wrap(day, metrics, alerts, notes, handoff)

    market_p.write_text(f"# Stock Market Live — {day}\n\n" + wrap.market_brief_md.strip() + "\n")
    capture_p.write_text(f"# Capture quality — {day}\n\n" + wrap.capture_brief_md.strip() + "\n")
    HANDOFF.write_text(wrap.handoff_md.strip() + "\n")
    (d / "WRAPPED").write_text("1")
    link = f"{repo_url}/blob/main/briefs/{day}-market.md" if repo_url else None
    notify.send(f"Daily brief — {day}", wrap.push_summary, priority=3, category="BRIEF", click=link)


def _schema() -> dict:
    from anthropic import transform_schema

    return transform_schema(Wrap.model_json_schema())


def _plain_wrap(day, metrics, alerts, notes, handoff) -> Wrap:
    trades = [a for a in alerts if a["category"] == "TRADE"]
    lines = lambda xs: "\n".join(f"- {a['time']} — {a['title']}: {a['body']}" for a in xs) or "- none caught"
    market = (
        "## The day in one paragraph\nAuto-generated without the LLM (no API key or wrap failed) — a list of what was pinged.\n\n"
        f"## His trades\n{lines(trades)}\n\n## Everything pinged\n{lines(alerts)}\n\n"
        "## Notes\n" + ("\n".join(f"- {n['time']} {n['note']}" for n in notes) or "- none")
    )
    capture = "## Metrics\n```json\n" + json.dumps(metrics, indent=1) + "\n```\n"
    ho = handoff or "# Handoff\n\n(no LLM yet — add ANTHROPIC_API_KEY to let the agent learn the stream)\n"
    ho = ho.rstrip() + f"\n- {day}: plain wrap (no LLM). {len(alerts)} alerts, {len(trades)} trade pings.\n"
    summary = f"{len(alerts)} pings today, {len(trades)} possible trades." + (f" Top: {alerts[0]['title']}" if alerts else "")
    return Wrap(market_brief_md=market, capture_brief_md=capture, handoff_md=ho, push_summary=summary)

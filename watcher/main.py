"""Entry point.

  python -m watcher run          watch today's stream until it ends (or the runtime budget is hit)
  python -m watcher probe        60-second end-to-end smoke test: live check, audio, transcript, chat, one ping
  python -m watcher wrap [DAY]   write the daily briefs + handoff for DAY (default today)
  python -m watcher test-notify  send one test ping
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import logging
import os
import subprocess
import sys
import threading
import time

from . import brain as brain_mod
from . import notify
from .chat import ChatLog, ChatStats, load_tickers
from .config import HANDOFF, ROOT, WORK, day_dir, load
from .daily import wrap_day
from .transcribe import Line, Transcriber
from .youtube import AudioPipe, ChatPoller, grab_frame, resolve_live

log = logging.getLogger("watcher")
REPO_URL = (
    f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ['GITHUB_REPOSITORY']}"
    if os.environ.get("GITHUB_REPOSITORY") else None
)


def _fmt(cfg, ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, cfg.tz).strftime("%H:%M:%S")


def git_checkpoint(msg: str):
    """Commit + push day data mid-run so a crash or runner loss loses at most a few minutes."""
    if os.environ.get("WATCHER_AUTOCOMMIT") != "1":
        return
    try:
        subprocess.run(["git", "add", "data", "briefs", "HANDOFF.md"], cwd=ROOT, check=False)
        r = subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, capture_output=True)
        if r.returncode != 0:
            return
        for _ in range(3):
            subprocess.run(["git", "pull", "-q", "--rebase", "-X", "theirs", "origin", "main"], cwd=ROOT, check=False)
            if subprocess.run(["git", "push", "-q", "origin", "HEAD:main"], cwd=ROOT).returncode == 0:
                return
            time.sleep(5)
    except Exception as e:  # noqa: BLE001
        log.warning("checkpoint failed: %s", e)


class Session:
    def __init__(self, cfg, live, deadline: float):
        self.cfg, self.live, self.deadline = cfg, live, deadline
        self.day = cfg.today()
        self.dir = day_dir(self.day)
        self.handoff = HANDOFF.read_text() if HANDOFF.exists() else ""
        self.known = load_tickers()
        c = cfg["chat"]
        self.stats = ChatStats(self.known, c["spike_ratio"], c["spike_min_per_min"], c["ticker_min_authors"])
        self.chatlog = ChatLog(self.dir / "chat.jsonl.gz")
        self.brain = brain_mod.Brain(cfg, self.handoff)
        self.lines: list[Line] = []
        self.lock = threading.Lock()
        self.cursor = 0  # index into self.lines already analyzed
        self.state = self._load_state()
        self.cooldowns: dict[str, float] = {}
        self.last_screen = 0.0
        a = cfg["audio"]
        self.audio = AudioPipe(live.url, WORK / "chunks", a["chunk_seconds"])
        self.tx = Transcriber(WORK / "chunks", a["chunk_seconds"], a["whisper_model"], a["max_backlog_chunks"], self._on_line)
        self.chat = ChatPoller(live.video_id, self._on_chat)
        self.started = time.time()

    # ---- persistence ----
    def _load_state(self) -> dict:
        p = self.dir / "state.json"
        base = {"alerts_sent": 0, "self_dev_pinged": False, "segments": {}, "recent_titles": [], "runs": 0,
                "llm_usd": 0.0, "live_seconds": 0, "video_id": None, "title": None}
        if p.exists():
            base.update(json.loads(p.read_text()))
        base["runs"] += 1
        base["video_id"], base["title"] = self.live.video_id, self.live.title
        return base

    def save(self):
        self.state["llm_usd"] = round(self.state.get("llm_usd_prev", 0) + self.brain.spent_usd, 4)
        (self.dir / "state.json").write_text(json.dumps(self.state, indent=1))
        metrics = {
            "video_id": self.live.video_id, "title": self.live.title, "runs": self.state["runs"],
            "watched_minutes": round((self.state["live_seconds"] + time.time() - self.started) / 60, 1),
            "audio_restarts": self.audio.restarts, "audio_downtime_s": round(self.audio.downtime_s),
            "chunks_transcribed": self.tx.done_chunks, "chunks_dropped": self.tx.dropped_chunks,
            "transcribe_lag_s": round(self.tx.lag_s, 1), "whisper_model": self.tx.model_name,
            "chat_messages": self.chat.total, "chat_errors": self.chat.errors, "chat_peak_per_min": self.stats.peak_rate,
            "top_chat_tickers": self.stats.ticker_hist.most_common(15),
            "llm_calls": self.brain.calls, "llm_failures": self.brain.failures, "llm_usd": self.state["llm_usd"],
            "alerts_sent": self.state["alerts_sent"], "segments_min": self.state["segments"],
            "mode": "llm" if self.brain.client else "heuristic",
        }
        (self.dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
        self.chatlog.f.flush()

    def _append(self, name: str, obj: dict):
        with open(self.dir / name, "a") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # ---- callbacks from worker threads ----
    def _on_line(self, line: Line):
        with self.lock:
            self.lines.append(line)
        with open(self.dir / "transcript.md", "a") as f:
            f.write(f"[{_fmt(self.cfg, line.ts)}] {line.text}\n")

    def _on_chat(self, msgs):
        self.stats.add(msgs)
        self.chatlog.write(msgs)

    # ---- alerting ----
    def ping(self, a: brain_mod.Alert, source: str):
        n = self.cfg["notify"]
        pr = max(1, min(5, a.priority))
        if pr < n["min_priority"] or self.state["alerts_sent"] >= n["max_per_day"]:
            return
        key = f"{a.category}:{','.join(sorted(a.tickers)) or a.title.lower()[:30]}"
        if pr < 5 and time.time() - self.cooldowns.get(key, 0) < self.cfg["chat"]["cooldown_min"] * 60:
            return
        self.cooldowns[key] = time.time()
        title = f"{'🔴 ' if a.category == 'TRADE' else ''}{a.title}"
        body = a.body + (f'\n“{a.evidence}”' if a.evidence and a.evidence not in a.body else "")
        notify.send(title, body, priority=pr, category=a.category, click=self.live.url)
        self.state["alerts_sent"] += 1
        self.state["recent_titles"] = (self.state["recent_titles"] + [f"{_fmt(self.cfg, time.time())} {a.title}"])[-25:]
        self._append("alerts.jsonl", {"time": _fmt(self.cfg, time.time()), "source": source, **a.model_dump()})
        log.info("PING p%d [%s] %s", pr, a.category, a.title)

    def _window_text(self, new: list[Line], ctx: list[Line], chat: dict, spike: tuple) -> str:
        now = self.cfg.now()
        segs = ", ".join(f"{k}:{v}m" for k, v in sorted(self.state["segments"].items(), key=lambda x: -x[1])[:5])
        parts = [
            f"Now: {now:%A %Y-%m-%d %H:%M} {self.cfg['stream']['timezone']}. Stream: {self.live.title}",
            f"Self-dev segment already pinged today: {self.state['self_dev_pinged']}. Segment minutes so far: {segs or 'n/a'}",
            "ALREADY ALERTED (latest last):\n" + ("\n".join(self.state["recent_titles"][-15:]) or "(nothing yet)"),
            "TRANSCRIPT — CONTEXT (already analyzed):\n" + ("\n".join(f"[{_fmt(self.cfg, l.ts)}] {l.text}" for l in ctx) or "(none)"),
            "TRANSCRIPT — NEW:\n" + ("\n".join(f"[{_fmt(self.cfg, l.ts)}] {l.text}" for l in new) or "(no speech captured this window)"),
            f"CHAT: {chat['count']} msgs this window; last minute {chat['rate_last_min']:.0f}/min vs typical {chat['rate_baseline']:.0f}/min."
            + (f" ⚠ CHAT SPIKE DETECTED ({spike[1]:.0f}/min vs {spike[2]:.0f}) — explain what it's about." if spike[0] else ""),
            "Tickers chat is naming (unique authors): " + (", ".join(f"{t}({n})" for t, n in chat["hot_tickers"]) or "none"),
            "Streamer/moderator/superchat messages:\n" + ("\n".join(chat["owner_mod"]) or "(none)"),
            "Chat messages that look like plays:\n" + ("\n".join(chat["plays"]) or "(none)"),
            "Recent chat sample:\n" + ("\n".join(chat["sample"][-25:]) or "(none)"),
        ]
        return "\n\n".join(parts)

    def analyze_once(self, spike=(False, 0, 0)):
        with self.lock:
            new = self.lines[self.cursor:]
            self.cursor = len(self.lines)
            ctx_from = time.time() - self.cfg["analysis"]["context_seconds"] - self.cfg["analysis"]["interval_seconds"]
            ctx = [l for l in self.lines[: self.cursor - len(new)] if l.ts >= ctx_from]
        chat = self.stats.take_window()
        if not new and chat["count"] == 0:
            return
        res = self.brain.analyze(self._window_text(new, ctx, chat, spike))
        source = "llm"
        if res is None:
            source = "heuristic"
            res = brain_mod.heuristic([l.text for l in new], chat, self.cfg.now(), self.known, self.cfg["stream"]["self_dev_hint"])
            if spike[0]:
                hot = ", ".join(t for t, _ in chat["hot_tickers"][:4])
                res.alerts.append(brain_mod.Alert(category="CHAT_SPIKE", priority=3, title=f"Chat spike{' — ' + hot if hot else ''}",
                                                  body=f"{spike[1]:.0f} msgs/min vs ~{spike[2]:.0f} normal.", tickers=[], evidence=""))
            for t in chat["crowd_tickers"][:2]:
                res.alerts.append(brain_mod.Alert(category="PLAY", priority=2, title=f"{t} — chat piling in",
                                                  body=f"Many chatters naming {t} right now.", tickers=[t], evidence=""))
        mins = round(self.cfg["analysis"]["interval_seconds"] / 60, 1)
        self.state["segments"][res.segment] = round(self.state["segments"].get(res.segment, 0) + mins, 1)
        if res.segment == "self_development" and not self.state["self_dev_pinged"]:
            self.state["self_dev_pinged"] = True
            if not any(a.category == "SELF_DEV" for a in res.alerts):
                res.alerts.append(brain_mod.Alert(category="SELF_DEV", priority=4, title="Self-development hour is on",
                                                  body="The philosophy / self-development segment just started.", tickers=[], evidence=""))
        elif any(a.category == "SELF_DEV" for a in res.alerts):
            if self.state["self_dev_pinged"]:
                res.alerts = [a for a in res.alerts if a.category != "SELF_DEV" or a.priority >= 5]
            self.state["self_dev_pinged"] = True
        for a in res.alerts:
            self.ping(a, source)
        for n in res.notes:
            self._append("notes.jsonl", {"time": f"{self.cfg.now():%H:%M}", "segment": res.segment, "note": n})
        if res.look_at_screen and time.time() - self.last_screen > self.cfg["analysis"]["screen_look_cooldown_min"] * 60:
            self.last_screen = time.time()
            self._look(new, res.screen_reason)

    def _look(self, new, reason):
        img = grab_frame(self.live.url, WORK / "frame.jpg")
        if not img:
            return
        ctx = "\n".join(l.text for l in new[-6:])
        sr = self.brain.read_screen(img, f"{ctx}\n(reason to look: {reason})")
        if sr and (sr.position_details or sr.tickers):
            self.ping(brain_mod.Alert(category="TRADE" if sr.position_details else "MARKET_MOVE", priority=4,
                                      title=f"{' '.join(sr.tickers[:3]) + ' — ' if sr.tickers else ''}on screen",
                                      body=f"{sr.summary} {sr.position_details}".strip(), tickers=sr.tickers, evidence=""), "screen")
            self._append("notes.jsonl", {"time": f"{self.cfg.now():%H:%M}", "segment": "screen", "note": sr.summary})

    # ---- main loop ----
    def run(self) -> str:
        """Returns 'ended' (stream over) or 'continue' (runtime budget spent, stream still up)."""
        self.state["llm_usd_prev"] = self.state.get("llm_usd", 0.0)
        self.brain.spent_usd = 0.0
        if self.state["llm_usd_prev"] >= self.cfg["analysis"]["daily_budget_usd"]:
            self.brain.client = None
        self.audio.start()
        self.tx.start()
        self.chat.start()
        interval = self.cfg["analysis"]["interval_seconds"]
        hard_stop = self.cfg.at(self.cfg["stream"]["hard_stop"]).timestamp()
        grace = self.cfg["stream"]["offline_grace_min"] * 60
        next_win, next_ckpt, next_live_check = time.time() + interval, time.time() + 1200, time.time() + 300
        offline_since: float | None = None
        outcome = "ended"
        try:
            while True:
                time.sleep(5)
                now = time.time()
                spike = self.stats.spike()
                cd_ok = now - self.cooldowns.get("spike", 0) > self.cfg["chat"]["cooldown_min"] * 60
                if spike[0] and cd_ok:
                    self.cooldowns["spike"] = now
                    self.analyze_once(spike)
                    next_win = now + interval
                elif now >= next_win:
                    self.analyze_once()
                    next_win = now + interval
                if now >= next_live_check:
                    next_live_check = now + 300
                    quiet = (self.audio.alive_since is None) and (not self.chat.last_ok or now - self.chat.last_ok > 300)
                    cur = resolve_live(self.cfg.channel_live_url) if quiet else self.live
                    if cur is not None and cur.video_id != self.live.video_id:
                        log.info("stream moved to a new video (%s); restarting on it", cur.video_id)
                        outcome = "continue"
                        break
                    if cur is None:
                        offline_since = offline_since or now
                        log.info("stream looks offline (%.0f min)", (now - offline_since) / 60)
                    else:
                        offline_since = None
                if offline_since and now - offline_since > grace:
                    log.info("stream ended")
                    break
                if now >= hard_stop:
                    log.info("hard stop reached")
                    break
                if now >= self.deadline:
                    outcome = "continue"
                    log.info("runtime budget reached; handing off to the next run")
                    break
                if now >= next_ckpt:
                    next_ckpt = now + 1200
                    self.save()
                    git_checkpoint(f"watch {self.day}: checkpoint")
        finally:
            self.analyze_once()
            for x in (self.audio, self.tx, self.chat):
                x.close()
            self.state["live_seconds"] += time.time() - self.started
            self.save()
            self.chatlog.close()
        return outcome


def cmd_run(cfg) -> int:
    max_min = float(os.environ.get("WATCHER_MAX_MINUTES", "340"))
    deadline = time.time() + max_min * 60
    now = cfg.now()
    day = cfg.today()
    d = day_dir(day)
    if (d / "WRAPPED").exists():
        log.info("day %s already wrapped; nothing to do", day)
        return 0
    if os.environ.get("GITHUB_EVENT_NAME") == "schedule" and now < cfg.at(cfg["stream"]["earliest_start"]):
        log.info("scheduled run before %s %s — the other cron will cover today", cfg["stream"]["earliest_start"], cfg["stream"]["timezone"])
        return 0
    give_up = cfg.at(cfg["stream"]["give_up_if_not_live_by"]).timestamp()
    live = resolve_live(cfg.channel_live_url)
    while live is None:
        if time.time() > give_up or time.time() > deadline:
            st = json.loads((d / "state.json").read_text()) if (d / "state.json").exists() else {}
            if st.get("runs"):  # we already watched earlier today; it has simply ended
                log.info("stream over; wrapping")
            else:
                log.info("no stream found today by %s", cfg["stream"]["give_up_if_not_live_by"])
                (d / "metrics.json").write_text(json.dumps({"no_stream": True, "checked_until": f"{cfg.now():%H:%M}"}))
            wrap_day(cfg, day, REPO_URL)
            return 0
        log.info("not live yet; checking again in 2 min")
        time.sleep(120)
        live = resolve_live(cfg.channel_live_url)
    log.info("LIVE: %s (%s)", live.title, live.url)
    first_run = not (d / "state.json").exists()
    if first_run and cfg["notify"]["announce_start"]:
        mode = "Claude" if cfg.has_llm else "keyword mode (add ANTHROPIC_API_KEY for full judgment)"
        notify.send("👀 Watching the stream", f"{live.title}\nListening + reading chat — {mode}.", priority=2, category="SYSTEM", click=live.url)
    outcome = Session(cfg, live, deadline).run()
    if outcome == "continue":
        (ROOT / ".continue").write_text("1")
        return 0
    wrap_day(cfg, day, REPO_URL)
    return 0


def cmd_probe(cfg) -> int:
    """End-to-end smoke test on whatever is live right now (or the latest VOD if nothing is)."""
    ok = True
    live = resolve_live(cfg.channel_live_url)
    print("live:", live)
    target_url = live.url if live else None
    if not live:
        import yt_dlp

        from .youtube import _ydl_opts

        with yt_dlp.YoutubeDL(_ydl_opts(extract_flat=True, playlistend=3)) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/{cfg['stream']['channel']}/streams", download=False)
        entries = info.get("entries") or []
        print("recent streams:", [(e.get("id"), e.get("title"), e.get("live_status")) for e in entries[:3]])
        if entries:
            target_url = f"https://www.youtube.com/watch?v={entries[0]['id']}"
    if not target_url:
        print("FAIL: cannot see the channel at all")
        return 1
    chunks = WORK / "probe"
    pipe = AudioPipe(target_url, chunks, 15)
    pipe.start()
    got = collections.Counter()

    def on_chat(msgs):
        got["n"] += len(msgs)
        for m in msgs[:3]:
            print("  chat:", m.role, m.author, "|", m.text[:100])

    poller = ChatPoller(live.video_id, on_chat) if live else None
    if poller:
        poller.start()
    time.sleep(50)
    pipe.close()
    if poller:
        poller.close()
    wavs = sorted(chunks.glob("*.wav"))
    print("audio chunks:", len(wavs), "restarts:", pipe.restarts)
    if wavs:
        from faster_whisper import WhisperModel

        m = WhisperModel(cfg["audio"]["whisper_model"], device="cpu", compute_type="int8")
        t0 = time.time()
        segs, _ = m.transcribe(str(wavs[0]), language="en", beam_size=1, vad_filter=True)
        text = " ".join(s.text for s in segs)
        print(f"transcript ({time.time() - t0:.1f}s for 15s audio):", text[:500])
    else:
        ok = False
        print("FAIL: no audio captured")
    if poller:
        print("chat messages in ~50s:", got["n"], "errors:", poller.errors)
        ok &= poller.errors == 0 or got["n"] > 0
    sent = notify.send("✅ Watcher probe", f"Live: {bool(live)}. Audio chunks: {len(wavs)}. Chat msgs: {got['n']}.",
                       priority=2, category="SYSTEM", click=target_url)
    print("notify sent:", sent)
    print("llm:", "configured" if cfg.has_llm else "NOT configured (heuristic mode)")
    return 0 if ok else 1


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", stream=sys.stdout)
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "run"
    cfg = load()
    if cmd == "run":
        return cmd_run(cfg)
    if cmd == "probe":
        return cmd_probe(cfg)
    if cmd == "wrap":
        wrap_day(cfg, argv[1] if len(argv) > 1 else cfg.today(), REPO_URL)
        return 0
    if cmd == "test-notify":
        return 0 if notify.send("🔔 Test ping", "Stock Market Live watcher can reach your phone.", priority=3, category="SYSTEM") else 1
    print(__doc__)
    return 2

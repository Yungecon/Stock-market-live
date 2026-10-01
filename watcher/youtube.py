"""YouTube access: find today's live video, pipe its audio, poll its live chat, grab a frame.

Chat uses the same innertube endpoint the web player's chat popout uses, set to
"Live chat" (every message) rather than "Top chat", so volume stats are real."""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests
import yt_dlp

from .config import WORK

log = logging.getLogger("youtube")
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


def cookiefile() -> str | None:
    """YT_COOKIES (Netscape cookies.txt contents) unlocks YouTube when it bot-checks runner IPs."""
    txt = os.environ.get("YT_COOKIES")
    if not txt:
        return None
    WORK.mkdir(parents=True, exist_ok=True)
    p = WORK / "cookies.txt"
    p.write_text(txt)
    return str(p)


# YouTube bot-walls datacenter IPs per player client; some clients get through when others don't.
# PLAYER_CLIENT is set once a working client is found (see pick_client).
PLAYER_CLIENTS = ["default", "tv", "web_safari", "mweb", "android_vr", "ios", "tv_simply", "web_embedded"]
PLAYER_CLIENT: str | None = None


def _ydl_opts(client: str | None = None, **extra):
    o = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    if cf := cookiefile():
        o["cookiefile"] = cf
    client = client or PLAYER_CLIENT
    if client and client != "default":
        o["extractor_args"] = {"youtube": {"player_client": [client]}}
    o.update(extra)
    return o


def _cli_client_args() -> list[str]:
    args = []
    if cf := cookiefile():
        args += ["--cookies", cf]
    if PLAYER_CLIENT and PLAYER_CLIENT != "default":
        args += ["--extractor-args", f"youtube:player_client={PLAYER_CLIENT}"]
    return args


def pick_client(video_url: str, fmt: str = "bestaudio/best[height<=480]/best") -> str | None:
    """Find a player client that can actually see formats for this video from this IP."""
    global PLAYER_CLIENT
    for c in ([PLAYER_CLIENT] if PLAYER_CLIENT else []) + PLAYER_CLIENTS:
        try:
            with yt_dlp.YoutubeDL(_ydl_opts(client=c, format=fmt)) as ydl:
                info = ydl.extract_info(video_url, download=False)
            if info.get("url"):
                PLAYER_CLIENT = c
                log.info("player client '%s' works", c)
                return c
        except Exception as e:  # noqa: BLE001
            log.info("player client '%s' blocked: %s", c, str(e).split(":")[-1][:120].strip())
    return None


@dataclass
class LiveInfo:
    video_id: str
    title: str
    is_live: bool
    live_status: str
    started: float | None

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


def resolve_live(channel_live_url: str) -> LiveInfo | None:
    """Returns the channel's current broadcast, or None when nothing is live."""
    try:
        return _resolve_live_html(channel_live_url)
    except Exception as e:  # noqa: BLE001
        log.info("html resolve failed (%s); trying yt-dlp", str(e)[:150])
    return _resolve_live_ytdlp(channel_live_url)


def _resolve_live_html(channel_live_url: str) -> LiveInfo | None:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
    s.cookies.set("CONSENT", "YES+1", domain=".youtube.com")
    html = s.get(channel_live_url, timeout=20).text
    m = re.search(r'<link rel="canonical" href="https://www\.youtube\.com/watch\?v=([\w-]{11})"', html)
    if not m:
        if "youtube.com/channel/" in html or "/@" in html:
            return None  # landed on the channel page: nothing live
        raise RuntimeError("unrecognized /live page")
    vid = m.group(1)
    live_now = '"isLiveNow":true' in html or '"isLive":true' in html
    upcoming = '"isUpcoming":true' in html
    t = re.search(r'<meta name="title" content="([^"]*)"', html)
    title = (t.group(1) if t else "").replace("&amp;", "&").replace("&#39;", "'").replace("&quot;", '"')
    if not live_now or upcoming:
        return None
    return LiveInfo(video_id=vid, title=title, is_live=True, live_status="is_live", started=None)


def _resolve_live_ytdlp(channel_live_url: str) -> LiveInfo | None:
    try:
        with yt_dlp.YoutubeDL(_ydl_opts()) as ydl:
            info = ydl.extract_info(channel_live_url, download=False, process=False)
        if info.get("_type") == "url" and info.get("url"):  # channel/live redirect to watch page
            with yt_dlp.YoutubeDL(_ydl_opts()) as ydl:
                info = ydl.extract_info(info["url"], download=False, process=False)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "not currently live" not in msg and "will begin" not in msg:
            log.warning("resolve_live (yt-dlp): %s", msg[:300])
        return None
    status = info.get("live_status") or ("is_live" if info.get("is_live") else "none")
    li = LiveInfo(
        video_id=info["id"],
        title=info.get("title") or "",
        is_live=status == "is_live",
        live_status=status,
        started=info.get("release_timestamp") or info.get("timestamp"),
    )
    return li if li.is_live else None


def media_url(video_url: str, fmt: str) -> tuple[str, dict]:
    with yt_dlp.YoutubeDL(_ydl_opts(format=fmt)) as ydl:
        info = ydl.extract_info(video_url, download=False)
    return info["url"], info.get("http_headers", {})


class AudioPipe:
    """yt-dlp -> ffmpeg -> fixed-length 16 kHz mono WAV chunks in out_dir.
    Restarts itself if the pipe dies while the stream is still up."""

    def __init__(self, video_url: str, out_dir: Path, chunk_seconds: int):
        self.video_url, self.out_dir, self.chunk_seconds = video_url, out_dir, chunk_seconds
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.restarts = 0
        self.alive_since: float | None = None
        self.downtime_s = 0.0
        self.thread = threading.Thread(target=self._run, daemon=True, name="audio")

    def start(self):
        self.thread.start()

    def _run(self):
        seq = 0
        while not self.stop.is_set():
            t_down = time.time()
            if self.restarts and self.restarts % 3 == 0:
                pick_client(self.video_url)  # the client that worked may have been walled since
            cmd_dl = ["yt-dlp", "-q", "--no-warnings", *_cli_client_args(), "-f", "bestaudio/best[height<=480]/best",
                      "-o", "-", self.video_url]
            pattern = str(self.out_dir / f"r{seq:02d}_%06d.wav")
            cmd_ff = [
                "ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-vn", "-ac", "1", "-ar", "16000",
                "-f", "segment", "-segment_time", str(self.chunk_seconds), "-reset_timestamps", "1", pattern,
            ]
            dl = subprocess.Popen(cmd_dl, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ff = subprocess.Popen(cmd_ff, stdin=dl.stdout, stderr=subprocess.PIPE)
            dl.stdout.close()
            self.alive_since = time.time()
            if seq:
                self.downtime_s += self.alive_since - t_down
            while not self.stop.is_set() and ff.poll() is None:
                time.sleep(1)
            for p in (dl, ff):
                if p.poll() is None:
                    p.terminate()
            err = (dl.stderr.read() or b"").decode(errors="ignore")[-400:]
            if self.stop.is_set():
                break
            self.restarts += 1
            seq += 1
            self.alive_since = None
            log.warning("audio pipe ended (restart %d): %s", self.restarts, err.strip())
            time.sleep(min(60, 5 * self.restarts))

    def close(self):
        self.stop.set()


@dataclass
class ChatMsg:
    ts: float
    author: str
    author_id: str
    text: str
    role: str  # owner | moderator | member | viewer
    paid: str = ""

    def to_json(self):
        return self.__dict__


def _runs_text(runs) -> str:
    out = []
    for r in runs or []:
        if "text" in r:
            out.append(r["text"])
        elif "emoji" in r:
            sc = r["emoji"].get("shortcuts") or []
            out.append(sc[0] if sc else "")
    return "".join(out)


def _role(renderer) -> str:
    role = "viewer"
    for b in renderer.get("authorBadges", []):
        br = b.get("liveChatAuthorBadgeRenderer", {})
        icon = br.get("icon", {}).get("iconType", "")
        tip = br.get("tooltip", "").lower()
        if icon == "OWNER" or tip == "owner":
            return "owner"
        if icon == "MODERATOR" or "moderator" in tip:
            role = "moderator"
        elif "member" in tip and role == "viewer":
            role = "member"
    return role


def parse_actions(actions) -> list[ChatMsg]:
    msgs = []
    for a in actions or []:
        item = a.get("addChatItemAction", {}).get("item") or a.get("replayChatItemAction", {})
        if not item:
            continue
        for kind in ("liveChatTextMessageRenderer", "liveChatPaidMessageRenderer"):
            r = item.get(kind)
            if not r:
                continue
            msgs.append(
                ChatMsg(
                    ts=int(r.get("timestampUsec", time.time() * 1e6)) / 1e6,
                    author=r.get("authorName", {}).get("simpleText", ""),
                    author_id=r.get("authorExternalChannelId", ""),
                    text=_runs_text(r.get("message", {}).get("runs")),
                    role=_role(r),
                    paid=r.get("purchaseAmountText", {}).get("simpleText", "") if kind.endswith("PaidMessageRenderer") else "",
                )
            )
    return msgs


def _next_cont(conts) -> tuple[str | None, float]:
    for c in conts or []:
        for k in ("invalidationContinuationData", "timedContinuationData", "reloadContinuationData"):
            if k in c:
                return c[k].get("continuation"), c[k].get("timeoutMs", 5000) / 1000
    return None, 5.0


class ChatPoller:
    def __init__(self, video_id: str, on_msgs):
        self.video_id, self.on_msgs = video_id, on_msgs
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"})
        if cf := cookiefile():
            import http.cookiejar

            jar = http.cookiejar.MozillaCookieJar(cf)
            try:
                jar.load(ignore_discard=True, ignore_expires=True)
                self.s.cookies = jar  # type: ignore[assignment]
            except Exception as e:  # noqa: BLE001
                log.warning("cookie load failed: %s", e)
        self.stop = threading.Event()
        self.errors = 0
        self.total = 0
        self.last_ok: float | None = None
        self.thread = threading.Thread(target=self._run, daemon=True, name="chat")

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()

    def _bootstrap(self):
        html = self.s.get(
            "https://www.youtube.com/live_chat", params={"is_popout": 1, "v": self.video_id}, timeout=20
        ).text
        key = re.search(r'"INNERTUBE_API_KEY":"([^"]+)"', html)
        ver = re.search(r'"INNERTUBE_CLIENT_VERSION":"([^"]+)"', html) or re.search(r'"clientVersion":"([^"]+)"', html)
        m = re.search(r'(?:window\["ytInitialData"\]|var ytInitialData)\s*=\s*(\{.+?\});\s*</script>', html, re.S)
        if not (key and ver and m):
            raise RuntimeError("live_chat bootstrap failed (no innertube config in page; bot check?)")
        data = json.loads(m.group(1))
        lcr = data["contents"]["liveChatRenderer"]
        cont, _ = _next_cont(lcr.get("continuations"))
        # Switch "Top chat" -> "Live chat" (all messages) when the selector exists.
        try:
            items = lcr["header"]["liveChatHeaderRenderer"]["viewSelector"]["sortFilterSubMenuRenderer"]["subMenuItems"]
            for it in items:
                if "live chat" in it.get("title", "").lower() and "top" not in it.get("title", "").lower():
                    cont = it["continuation"]["reloadContinuationData"]["continuation"]
        except (KeyError, TypeError):
            pass
        return key.group(1), ver.group(1), cont, parse_actions(lcr.get("actions"))

    def _run(self):
        while not self.stop.is_set():
            try:
                key, ver, cont, first = self._bootstrap()
                if first:
                    self._emit(first)
                while cont and not self.stop.is_set():
                    r = self.s.post(
                        "https://www.youtube.com/youtubei/v1/live_chat/get_live_chat",
                        params={"key": key, "prettyPrint": "false"},
                        json={"context": {"client": {"clientName": "WEB", "clientVersion": ver, "hl": "en", "gl": "US"}},
                              "continuation": cont},
                        timeout=20,
                    )
                    r.raise_for_status()
                    lcc = r.json().get("continuationContents", {}).get("liveChatContinuation")
                    if not lcc:
                        raise RuntimeError("chat continuation ended")
                    self._emit(parse_actions(lcc.get("actions")))
                    cont, wait = _next_cont(lcc.get("continuations"))
                    self.last_ok = time.time()
                    self.errors = 0
                    self.stop.wait(max(1.0, min(wait, 10.0)))
            except Exception as e:  # noqa: BLE001 — keep polling through any hiccup
                self.errors += 1
                log.warning("chat poll error #%d: %s", self.errors, str(e)[:200])
                self.stop.wait(min(120, 5 * self.errors))

    def _emit(self, msgs):
        if msgs:
            self.total += len(msgs)
            self.on_msgs(msgs)


def grab_frame(video_url: str, out: Path) -> Path | None:
    """One JPEG of what's on screen right now (for when the transcript points at the screen)."""
    try:
        url, headers = media_url(video_url, "best[height<=720]/best")
        hdr = "".join(f"{k}: {v}\r\n" for k, v in headers.items())
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-headers", hdr, "-i", url, "-frames:v", "1", "-q:v", "3", str(out)],
            timeout=60, check=True,
        )
        return out if out.exists() else None
    except Exception as e:  # noqa: BLE001
        log.warning("grab_frame failed: %s", e)
        return None

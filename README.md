# Stock Market Live — stream watcher

An agent that listens to the [@thestockmarket](https://youtube.com/@thestockmarket) livestream every market day,
reads the live chat, and pings your phone when something matters: **his trades, market-moving news, sharp
moves, plays the crowd piles into, chat eruptions, and the start of the ~10:30 ET self-development hour.**
Jokes, games and banter get filtered out.

At the end of each day it writes:
- `briefs/YYYY-MM-DD-market.md`: your brief (trades, news, plays, the self-dev lessons, what to watch tomorrow)
- `briefs/YYYY-MM-DD-capture.md`: its own QA on how well it captured the day
- `HANDOFF.md`: the living document it reads before every analysis window the next day

## How it works

```
GitHub Actions (weekdays 8:35 ET) ──> waits for the stream to go live
   yt-dlp ─ audio ─> ffmpeg 30s chunks ─> faster-whisper ─> transcript.md
   YouTube live-chat poller ─> chat.jsonl.gz + rate / ticker / owner-mod stats
   every 90s (or right away on a chat spike): transcript + chat ─> Claude ─> pings via ntfy
   on screen-references ("look at my position") ─> grabs one frame ─> Claude vision
stream ends ─> daily wrap (briefs + HANDOFF.md rewrite) ─> committed to this repo
```

GitHub-hosted jobs cap at 6 h, so a run hands off to a fresh one after ~5h40m while the stream is still up.

## Setup

1. **Get pings:** install the **ntfy** app (iOS / Android), tap +, subscribe to topic `sml-yung-03f916aeb1`.
   (Or set a `NTFY_TOPIC` secret to use your own, or `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`, or `DISCORD_WEBHOOK_URL`.)
2. **Give it judgment:** add an `ANTHROPIC_API_KEY` repo secret (Settings → Secrets and variables → Actions).
   Without it the watcher runs on keyword rules: it still works, but it's noisier and the briefs are plain lists.
3. If YouTube starts bot-checking the runner (`Sign in to confirm you're not a bot` in the logs), export
   youtube.com cookies (Netscape cookies.txt format) into a `YT_COOKIES` secret.

## Run it by hand

Actions → **watch** → Run workflow → pick a mode:
- `probe`: 60-second smoke test (live check, audio, transcript, chat, one ping)
- `run`: watch now
- `wrap`: write today's briefs
- `test-notify`: one test ping

Locally: `pip install -r requirements.txt` (plus ffmpeg), then `python -m watcher probe`.

## Tuning

Everything is in [`config.toml`](config.toml): times, chat-spike thresholds, models, daily LLM budget
(default $6/day), minimum ping priority, daily ping cap. Edit `HANDOFF.md` to teach it something directly.

## Cost

- **LLM:** the live classifier runs ~250 calls per trading day on `claude-opus-5-5` at low effort, capped by
  `daily_budget_usd`. Switch `live_model` to `claude-sonnet-5-5` to roughly halve that.
- **Actions minutes:** a full stream day is about 7–8 runner hours. Private repos only get 2,000 free minutes a month
  (3,000 on Pro), which covers about a week of streams. Making the repo public makes standard runners free, and so
  does running it on your own machine as a self-hosted runner (change `runs-on`).

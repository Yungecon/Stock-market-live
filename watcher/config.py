"""Config + paths. Everything tunable lives in config.toml; secrets come from env."""
from __future__ import annotations

import datetime as dt
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BRIEFS = ROOT / "briefs"
HANDOFF = ROOT / "HANDOFF.md"
WORK = Path(os.environ.get("WATCHER_WORK", ROOT / "work"))


@dataclass
class Config:
    raw: dict

    def __getitem__(self, k):
        return self.raw[k]

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.raw["stream"]["timezone"])

    def now(self) -> dt.datetime:
        return dt.datetime.now(self.tz)

    def today(self) -> str:
        return self.now().strftime("%Y-%m-%d")

    def at(self, hhmm: str, day: dt.datetime | None = None) -> dt.datetime:
        day = day or self.now()
        h, m = (int(x) for x in hhmm.split(":"))
        return day.replace(hour=h, minute=m, second=0, microsecond=0)

    @property
    def channel_live_url(self) -> str:
        ch = self.raw["stream"]["channel"]
        return f"https://www.youtube.com/{ch}/live"

    @property
    def has_llm(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))


def load() -> Config:
    with open(ROOT / "config.toml", "rb") as f:
        return Config(tomllib.load(f))


def day_dir(day: str) -> Path:
    p = DATA / day
    p.mkdir(parents=True, exist_ok=True)
    return p

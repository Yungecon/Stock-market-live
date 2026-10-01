"""Turns audio chunks into timestamped transcript lines with faster-whisper (CPU, int8)."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("transcribe")


@dataclass
class Line:
    ts: float  # wall-clock start of the chunk (epoch seconds)
    text: str


class Transcriber:
    def __init__(self, chunk_dir: Path, chunk_seconds: int, model: str, max_backlog: int, on_line):
        self.chunk_dir, self.chunk_seconds = chunk_dir, chunk_seconds
        self.model_name, self.max_backlog, self.on_line = model, max_backlog, on_line
        self.stop = threading.Event()
        self.done_chunks = 0
        self.dropped_chunks = 0
        self.lag_s = 0.0
        self.downgraded = False
        self.thread = threading.Thread(target=self._run, daemon=True, name="transcribe")
        self._model = None

    def _load(self, name):
        from faster_whisper import WhisperModel

        log.info("loading whisper model %s", name)
        self._model = WhisperModel(name, device="cpu", compute_type="int8")
        self.model_name = name

    def start(self):
        self._load(self.model_name)
        self.thread.start()

    def close(self):
        self.stop.set()

    def _ready_chunks(self) -> list[Path]:
        files = sorted(self.chunk_dir.glob("r*_*.wav"))
        if not files:
            return []
        # ffmpeg is still writing the newest file of the current run; everything older is complete.
        newest_run = files[-1].name.split("_")[0]
        done = [f for f in files if f.name.split("_")[0] != newest_run]
        cur = [f for f in files if f.name.split("_")[0] == newest_run]
        done += cur[:-1]
        if cur and time.time() - cur[-1].stat().st_mtime > self.chunk_seconds + 20:
            done.append(cur[-1])  # stale: the pipe died after writing it
        return done

    def _run(self):
        while not self.stop.is_set():
            ready = self._ready_chunks()
            if not ready:
                self.stop.wait(2)
                continue
            if len(ready) > self.max_backlog:
                if not self.downgraded and self.model_name != "tiny.en":
                    log.warning("transcription lagging (%d chunks) — switching to tiny.en", len(ready))
                    self._load("tiny.en")
                    self.downgraded = True
                else:
                    for f in ready[: len(ready) - self.max_backlog]:
                        f.unlink(missing_ok=True)
                        self.dropped_chunks += 1
                    ready = ready[len(ready) - self.max_backlog :]
            f = ready[0]
            start_ts = f.stat().st_mtime - self.chunk_seconds  # mtime ~= when ffmpeg closed the chunk
            try:
                segs, _ = self._model.transcribe(
                    str(f), language="en", beam_size=1, vad_filter=True, condition_on_previous_text=False
                )
                text = " ".join(s.text.strip() for s in segs).strip()
            except Exception as e:  # noqa: BLE001
                log.warning("transcribe failed on %s: %s", f.name, e)
                text = ""
            f.unlink(missing_ok=True)
            self.done_chunks += 1
            self.lag_s = max(0.0, time.time() - (start_ts + self.chunk_seconds))
            if text:
                self.on_line(Line(ts=start_ts, text=text))

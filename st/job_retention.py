"""Expire local ACD job directories without touching active jobs or other files."""

import logging
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time

logger = logging.getLogger(__name__)
JOB_NAME = re.compile(r"(?:dock|scan)_[A-Za-z0-9_-]+")


def retention_days():
    try:
        days = float(os.getenv("ACD_JOB_RETENTION_DAYS", "7"))
        if not math.isfinite(days) or days <= 0:
            raise ValueError
        return days
    except ValueError:
        logger.warning("Invalid ACD_JOB_RETENTION_DAYS; using 7 days")
        return 7.0


def mark_created(directory):
    """Write once; reading a job must not extend its retention period."""
    marker = Path(directory) / ".acd_created_at"
    try:
        with marker.open("x", encoding="ascii") as handle:
            handle.write(str(time.time()))
    except FileExistsError:
        pass


def purge_expired_jobs(root, jobs, *, days=None, now=None):
    root = Path(root).resolve()
    cutoff = (time.time() if now is None else now) - (
        retention_days() if days is None else days
    ) * 86400
    removed = []
    for directory in root.iterdir():
        if not JOB_NAME.fullmatch(directory.name):
            continue
        if directory.is_symlink() or not directory.is_dir():
            continue
        if directory.resolve().parent != root:
            continue
        # In-memory states identify work active in this process. Stale on-disk
        # running states after a restart must not keep abandoned jobs forever.
        if (jobs.get(directory.name) or {}).get("status") in {"queued", "running"}:
            continue
        try:
            marker = directory / ".acd_created_at"
            if marker.is_symlink():
                continue
            created = directory.stat().st_mtime  # legacy jobs have no marker
            if marker.is_file():
                try:
                    stamp = float(marker.read_text(encoding="ascii"))
                    if math.isfinite(stamp):
                        created = stamp
                except (ValueError, UnicodeError):
                    pass
            if created >= cutoff:
                continue
            shutil.rmtree(directory)
            jobs.pop(directory.name, None)
            removed.append(directory.name)
        except FileNotFoundError:
            continue  # already removed by DELETE or a concurrent cleanup
        except OSError:
            logger.exception("Could not purge expired job %s", directory.name)
    return removed


class JobRetention:
    def __init__(self, root, jobs, interval=3600):
        self.root = root
        self.jobs = jobs
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = None

    def run_once(self):
        try:
            removed = purge_expired_jobs(self.root, self.jobs)
            if removed:
                logger.info("Purged %d expired ACD job directories", len(removed))
        except OSError:
            logger.exception("ACD job retention sweep failed")

    def start(self):
        self.run_once()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="acd-retention")
        self.thread.start()

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self.run_once()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

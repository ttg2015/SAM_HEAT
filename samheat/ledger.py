"""ledger.json: one entry per run, with a full history. File-locked, because with
AUTO_ADVANCE several jobs may finish (and write) at the same time."""
import fcntl
import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix('.lock')

    def load(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    @contextmanager
    def edit(self):
        """with ledger.edit() as state: ...  (exclusive lock; saved on exit)"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_path, 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                state = self.load()
                yield state
                tmp = self.path.with_suffix('.tmp')
                tmp.write_text(json.dumps(state, indent=2, default=str))
                tmp.replace(self.path)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)


def note(entry, text):
    entry.setdefault('history', []).append(f'{datetime.now():%Y-%m-%d %H:%M} {text}')

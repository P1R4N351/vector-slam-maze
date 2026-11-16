"""Advisory POSIX flock + intra-process RLock reentrant locking for NavMapAtlas files."""

from __future__ import annotations

from contextlib import contextmanager
import errno
import fcntl
import os
from pathlib import Path
import threading
import time
from typing import Dict, Iterator

DEFAULT_ATLAS_LOCK_TIMEOUT_S = float(os.environ.get("VECTOR_ATLAS_LOCK_TIMEOUT_S", "10.0"))
LOCK_POLL_INTERVAL_S = 0.02

_MAP_GUARD = threading.Lock()
_HELD_COUNTS: Dict[Path, int] = {}
_PATH_MUTEXES: Dict[Path, threading.RLock] = {}


class AtlasLockTimeoutError(TimeoutError):
    """Raised when waiting past timeout_s to acquire an atlas advisory/thread lock."""


def _get_path_mutex(resolved: Path) -> threading.RLock:
    """Return process-local reentrant mutex bound to a resolved atlas path."""
    assert isinstance(resolved, Path) and str(resolved), "valid resolved path required"
    with _MAP_GUARD:
        lock = _PATH_MUTEXES.get(resolved)
        if lock is None:
            lock = threading.RLock()
            _PATH_MUTEXES[resolved] = lock
        return lock


def _open_flock_fd(lock_file: Path, deadline: float, timeout_s: float) -> int:
    """Open lock file and acquire non-blocking exclusive POSIX flock within deadline."""
    assert lock_file.parent.is_dir(), f"lock parent missing: {lock_file.parent}"
    assert timeout_s > 0.0, "timeout_s must be positive"
    fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR, 0o644)
    max_polls = max(1, int(timeout_s / LOCK_POLL_INTERVAL_S) + 8)
    try:
        for _ in range(max_polls):
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                rem = deadline - time.monotonic()
                if rem <= 0.0:
                    raise AtlasLockTimeoutError(f"timed out after {timeout_s:.2f}s waiting for atlas lock: {lock_file}") from exc
                time.sleep(min(LOCK_POLL_INTERVAL_S, rem))
        raise AtlasLockTimeoutError(f"timed out after {timeout_s:.2f}s waiting for atlas lock: {lock_file}")
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def atlas_file_lock(atlas_path: Path, timeout_s: float = DEFAULT_ATLAS_LOCK_TIMEOUT_S) -> Iterator[Path]:
    """Serialize read-modify-write operations on atlas_path across threads and processes."""
    assert atlas_path is not None and str(atlas_path), "atlas_path required"
    assert timeout_s > 0.0, "timeout_s must be positive"
    resolved = Path(atlas_path).resolve()
    mutex = _get_path_mutex(resolved)
    lock_file = resolved.with_suffix(f"{resolved.suffix}.lock")
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + float(timeout_s)
    if not mutex.acquire(timeout=float(timeout_s)):
        raise AtlasLockTimeoutError(f"timed out after {timeout_s:.2f}s waiting for thread mutex: {lock_file}")
    fd: int | None = None
    try:
        if _HELD_COUNTS.get(resolved, 0) == 0:
            fd = _open_flock_fd(lock_file, deadline, float(timeout_s))
        _HELD_COUNTS[resolved] = _HELD_COUNTS.get(resolved, 0) + 1
        try:
            yield lock_file
        finally:
            _HELD_COUNTS[resolved] -= 1
            if _HELD_COUNTS[resolved] == 0:
                _HELD_COUNTS.pop(resolved, None)
                if fd is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                    os.close(fd)
    finally:
        mutex.release()

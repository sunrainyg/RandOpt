"""Timestamped experiment-directory and console logging helpers."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import atexit
import io
from pathlib import Path
import re
import sys
from typing import Iterator, TextIO


class _Tee(io.TextIOBase):
    """Write to both the original stream and a UTF-8 log file."""

    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self.primary = primary
        self.secondary = secondary

    def write(self, text: str) -> int:
        self.primary.write(text)
        self.secondary.write(text)
        self.secondary.flush()
        return len(text)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.primary, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:
        return getattr(self.primary, "encoding", "utf-8") or "utf-8"


_ACTIVE_LOG_FILE: TextIO | None = None
_ORIGINAL_STDOUT: TextIO | None = None
_ORIGINAL_STDERR: TextIO | None = None


def _safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-_") or "run"


def make_timestamped_run_dir(
    log_root: str,
    dataset: str,
    perturbation_method: str,
    resume: bool = False,
) -> str:
    """Create and return ``logs/<dataset>_<method>_<timestamp>/``."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [_safe_component(dataset), _safe_component(perturbation_method)]
    if resume:
        parts.append("resume")
    run_name = "_".join(parts) + f"_{timestamp}"
    run_dir = Path(log_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return str(run_dir)


def stop_console_tee() -> None:
    """Restore stdout/stderr and close the active run log, if any."""

    global _ACTIVE_LOG_FILE, _ORIGINAL_STDOUT, _ORIGINAL_STDERR
    if _ACTIVE_LOG_FILE is None:
        return

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        if _ORIGINAL_STDOUT is not None:
            sys.stdout = _ORIGINAL_STDOUT
        if _ORIGINAL_STDERR is not None:
            sys.stderr = _ORIGINAL_STDERR
        _ACTIVE_LOG_FILE.close()
        _ACTIVE_LOG_FILE = None
        _ORIGINAL_STDOUT = None
        _ORIGINAL_STDERR = None


def start_console_tee(run_dir: str) -> str:
    """Tee subsequent stdout/stderr to ``<run_dir>/run.log``.

    The function is idempotent for a process. Ray worker logs remain managed by
    Ray; this captures the main RandOpt driver output and exceptions.
    """

    global _ACTIVE_LOG_FILE, _ORIGINAL_STDOUT, _ORIGINAL_STDERR
    if _ACTIVE_LOG_FILE is not None:
        return str(Path(run_dir) / "run.log")

    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_path = path / "run.log"

    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr
    _ACTIVE_LOG_FILE = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(_ORIGINAL_STDOUT, _ACTIVE_LOG_FILE)
    sys.stderr = _Tee(_ORIGINAL_STDERR, _ACTIVE_LOG_FILE)
    atexit.register(stop_console_tee)
    print(f"Console log: {log_path}")
    return str(log_path)


@contextmanager
def timestamped_experiment_logging(
    log_root: str,
    dataset: str,
    perturbation_method: str,
    resume: bool = False,
) -> Iterator[str]:
    """Context-manager variant for standalone experiment scripts."""

    run_dir = make_timestamped_run_dir(
        log_root=log_root,
        dataset=dataset,
        perturbation_method=perturbation_method,
        resume=resume,
    )
    start_console_tee(run_dir)
    try:
        yield run_dir
    finally:
        stop_console_tee()

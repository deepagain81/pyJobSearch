# all debug constants
from __future__ import annotations
import os, sys, time, traceback
from contextlib import contextmanager
from typing import Optional

# --- Read-only config from env (override in .env) ---
LOG_TS_FORMAT = os.getenv("LOG_TS_FORMAT", "%H:%M:%S")
LOG_PREFIX = os.getenv("LOG_PREFIX", "")
STEP_ICON_START = os.getenv("STEP_ICON_START", "▶")
STEP_ICON_DONE  = os.getenv("STEP_ICON_DONE", "✔")
STEP_ICON_ERR   = os.getenv("STEP_ICON_ERR", "✖")

# Modes (set at runtime via set_modes; defaults can come from env)
_PROGRESS = True
_DEBUG = False

def _ts() -> str:
    return time.strftime(LOG_TS_FORMAT)

def set_modes(progress: bool, debug: bool) -> None:
    """Called by main script after parsing CLI flags."""
    global _PROGRESS, _DEBUG
    _PROGRESS, _DEBUG = progress, debug

def log(msg: str) -> None:
    """Milestone logger to stderr."""
    if _PROGRESS or _DEBUG:
        prefix = f"{LOG_PREFIX} " if LOG_PREFIX else ""
        print(f"[{_ts()}] {prefix}{msg}", file=sys.stderr)

def dlog(msg: str) -> None:
    """Verbose debug logger."""
    if _DEBUG:
        prefix = f"{LOG_PREFIX} [DEBUG]" if LOG_PREFIX else "[DEBUG]"
        print(f"[{_ts()}] {prefix} {msg}", file=sys.stderr)

@contextmanager
def step(title: str):
    """Step timing + errors. Prints traceback if debug mode is on."""
    log(f"{STEP_ICON_START} {title} — start")
    t0 = time.time()
    try:
        yield
    except SystemExit:
        raise
    except Exception as e:
        log(f"{STEP_ICON_ERR} {title} — ERROR: {e}")
        if _DEBUG:
            traceback.print_exc(file=sys.stderr)
        raise
    else:
        dt = time.time() - t0
        log(f"{STEP_ICON_DONE} {title} — done in {dt:.2f}s")

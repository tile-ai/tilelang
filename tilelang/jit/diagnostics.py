from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import logging
import time

from tilelang.env import env

logger = logging.getLogger("tilelang.jit.diagnostics")


def diagnostics_enabled(*, verbose: bool = False) -> bool:
    return bool(verbose) or env.is_jit_diagnostics_enabled()


def _format_context(context: dict[str, object]) -> str:
    if not context:
        return ""
    parts = [f"{key}={value!r}" for key, value in sorted(context.items())]
    return " " + ", ".join(parts)


@contextmanager
def jit_phase(name: str, *, enabled: bool | None = None, verbose: bool = False, **context: object) -> Iterator[None]:
    if enabled is None:
        enabled = diagnostics_enabled(verbose=verbose)
    if not enabled:
        yield
        return

    context_msg = _format_context(context)
    start = time.monotonic()
    logger.info("TileLang JIT phase start: %s%s", name, context_msg)
    try:
        yield
    except Exception:
        elapsed = time.monotonic() - start
        logger.exception("TileLang JIT phase failed: %s elapsed=%.3fs%s", name, elapsed, context_msg)
        raise
    else:
        elapsed = time.monotonic() - start
        logger.info("TileLang JIT phase done: %s elapsed=%.3fs%s", name, elapsed, context_msg)

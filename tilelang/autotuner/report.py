"""Experimental, opt-in autotuning observations; not a persistent tuning cache."""

from collections.abc import Callable
import json
import logging


class TrialReporter:
    """Invoke a caller-owned sink synchronously on the run's aggregation thread.

    Records contain detached JSON representations of configs, not arbitrary objects.
    The sink must not mutate/re-enter the tuner and should not perform slow work.
    An ordinary reporting error disables the sink for this run, not tuning.
    """

    def __init__(self, sink: Callable[[dict], None] | None, logger: logging.Logger):
        self.sink = sink
        self.logger = logger

    def emit(self, status, index=None, config=None, latency=None, validation="not_run", error=None):
        if self.sink is None:
            return
        try:
            record = {
                "status": status,
                "index": index,
                "config": json.loads(json.dumps(config, allow_nan=False)),
                "latency_ms": latency,
                "validation": validation,
                "error": str(error)[:2048] if error is not None else None,
            }
            self.sink(record)
        except Exception:
            self.sink = None
            self.logger.warning("Trial reporting failed; disabling the sink for this run.", exc_info=True)

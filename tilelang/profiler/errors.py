"""Errors that distinguish missing profiling evidence from device failures."""


class ProfileUnavailableError(RuntimeError):
    """Profiling completed without a usable measurement of the workload."""

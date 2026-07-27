# ruff: noqa
"""CUDA-aware pytest-xdist scheduler.

This plugin ensures that each xdist worker is pinned to a unique CUDA device.
The controller process assigns devices ahead of worker start-up so each worker
sees its dedicated GPU via ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Any, Deque
from collections.abc import Iterable, MutableMapping

import pytest
from _pytest.config import Config
from _pytest.stash import StashKey
from xdist.scheduler.load import LoadScheduling

ENV_DEVICE_LIST = "PYTEST_XDIST_CUDA_DEVICES"
ENV_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
WORKER_ENV_DEVICE = "PYTEST_XDIST_WORKER_CUDA_DEVICE"
# Optional override: how many workers per selected GPU.
ENV_WORKERS_PER_DEVICE = "PYTEST_XDIST_CUDA_WORKERS_PER_DEVICE"

ALLOCATOR_KEY: StashKey[CudaDeviceAllocator] = StashKey()
NUM_WORKERS_PER_DEVICE = 4  # default if ENV_WORKERS_PER_DEVICE is unset/invalid


def _workers_per_device() -> int:
    """Return number of workers per GPU from env or default."""
    raw = os.environ.get(ENV_WORKERS_PER_DEVICE)
    if not raw:
        return NUM_WORKERS_PER_DEVICE
    try:
        val = int(str(raw).strip())
        if val > 0:
            return val
    except Exception:
        pass
    return NUM_WORKERS_PER_DEVICE


class CudaDeviceAllocator:
    """Allocate one CUDA device per execnet spec/worker."""

    def __init__(self, devices: Iterable[str]) -> None:
        # cleaned = [str(device).strip() for device in devices if str(device).strip()]
        cleaned = []
        workers_per_device = _workers_per_device()
        for device in devices:
            dev_str = str(device).strip()
            if dev_str:
                cleaned += [dev_str] * workers_per_device

        self._total = len(cleaned)
        self._queue: Deque[str] = deque(cleaned)
        self._assignments: dict[str, str] = {}

    @property
    def count(self) -> int:
        return self._total

    @property
    def assigned(self) -> dict[str, str]:
        return dict(self._assignments)

    def assign(self, spec: object) -> str:
        key = self._spec_key(spec)
        if key in self._assignments:
            return self._assignments[key]
        if not self._queue:
            raise pytest.UsageError(
                f"Not enough CUDA devices for pytest-xdist workers; need at least {len(self._assignments) + 1}, available {self._total}."
            )
        device = self._queue.popleft()
        self._assignments[key] = device
        return device

    @staticmethod
    def _spec_key(spec: object) -> str:
        identifier = getattr(spec, "id", None)
        if identifier is None:
            identifier = repr(spec)
        return str(identifier)


class CudaDeviceScheduler(LoadScheduling):
    """Load scheduler that records device assignments for logging."""

    def __init__(self, config: Config, log, allocator: CudaDeviceAllocator) -> None:
        super().__init__(config, log)
        self._allocator = allocator
        if allocator.count < self.numnodes:
            raise pytest.UsageError(f"Not enough CUDA devices for pytest-xdist workers: need {self.numnodes}, found {allocator.count}.")

    def add_node(self, node) -> None:  # WorkerController (runtime import avoidance)
        device = self._allocator.assign(node.gateway.spec)
        node.workerinput.setdefault("cuda_device", device)
        self.log("assigned", node.gateway.id, "to CUDA device", device)
        super().add_node(node)


def _parse_device_list(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _discover_cuda_devices() -> list[str]:
    """
    Discover available CUDA devices, then select only ceil(n/2) devices.

    - Priority of discovery:
      1) ENV: PYTEST_XDIST_CUDA_DEVICES
      2) ENV: CUDA_VISIBLE_DEVICES
      3) torch.cuda device count (if available)

    After discovery, if there are n devices, we keep ceil(n/2) devices.
    """
    devices: list[str] = []

    primary = os.environ.get(ENV_DEVICE_LIST)
    if primary:
        devices = _parse_device_list(primary)
    else:
        visible = os.environ.get(ENV_VISIBLE_DEVICES)
        if visible:
            devices = _parse_device_list(visible)
        else:
            try:  # Lazy import; torch is optional.
                import torch  # type: ignore
            except Exception:
                torch = None  # type: ignore
            if torch and torch.cuda.is_available():  # type: ignore[truthy-function]
                devices = [str(idx) for idx in range(torch.cuda.device_count())]

    if devices:
        # Keep only ceil(n/2) devices
        n = len(devices)
        limit = (n + 1) // 2
        devices = devices[:limit]

    return devices


def _xdist_enabled(config: Config) -> bool:
    try:
        dist_mode = config.getoption("dist")
        tx_specs = config.getoption("tx")
    except (AttributeError, ValueError):
        return False
    return dist_mode != "no" and bool(tx_specs)


def _ensure_allocator(config: Config) -> CudaDeviceAllocator:
    try:
        return config.stash[ALLOCATOR_KEY]
    except KeyError:
        devices = _discover_cuda_devices()
        if not devices:
            raise pytest.UsageError(f"Cannot auto-discover CUDA devices. Set CUDA_VISIBLE_DEVICES or {ENV_DEVICE_LIST}.")
        allocator = CudaDeviceAllocator(devices)
        config.stash[ALLOCATOR_KEY] = allocator
        return allocator


@pytest.hookimpl
def pytest_configure(config: Config) -> None:
    if getattr(config, "workerinput", None) is not None:
        device = config.workerinput.get("cuda_device")  # type: ignore[arg-type]
        if device:
            os.environ[ENV_VISIBLE_DEVICES] = str(device)
            os.environ[WORKER_ENV_DEVICE] = str(device)
        _set_worker_process_title(config, device, None)
        return
    if not _xdist_enabled(config):
        return
    _ensure_allocator(config)


@pytest.hookimpl
def pytest_xdist_setupnodes(config: Config, specs: Iterable[Any]) -> None:
    if not _xdist_enabled(config):
        return
    allocator = _ensure_allocator(config)
    for spec in specs:
        device = allocator.assign(spec)
        env: MutableMapping[str, str]
        env = getattr(spec, "env", None) or {}
        env = dict(env)
        env[ENV_VISIBLE_DEVICES] = device
        env[WORKER_ENV_DEVICE] = device
        spec.env = env  # type: ignore[attr-defined]


@pytest.hookimpl
def pytest_configure_node(node) -> None:  # WorkerController
    allocator = _ensure_allocator(node.config)
    device = allocator.assign(node.gateway.spec)
    node.workerinput["cuda_device"] = device


@pytest.hookimpl(tryfirst=True)
def pytest_xdist_make_scheduler(config: Config, log):
    if getattr(config, "workerinput", None) is not None:
        return None
    if not _xdist_enabled(config):
        return None
    allocator = _ensure_allocator(config)
    return CudaDeviceScheduler(config, log, allocator)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    # Only act on workers; controller has no workerinput/device.
    if getattr(item.config, "workerinput", None) is not None:
        device = os.environ.get(WORKER_ENV_DEVICE)
        _set_worker_process_title(item.config, device, item.nodeid)
    return None


def _set_worker_process_title(config: Config, device: str | None, test_name: str | None) -> None:
    """Optionally label worker processes for easier inspection."""
    try:
        import setproctitle
    except Exception:
        return

    workerid = None
    workerinput = getattr(config, "workerinput", None)
    if isinstance(workerinput, dict):
        workerid = workerinput.get("workerid")

    title_parts = ["pytest-xdist-worker"]
    if workerid:
        title_parts.append(str(workerid))
    if device:
        title_parts.append(f"cuda{device}")
    if test_name:
        title_parts.append(test_name)

    setproctitle.setproctitle(" ".join(title_parts))

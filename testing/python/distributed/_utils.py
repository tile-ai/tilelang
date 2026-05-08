from __future__ import annotations

import os
from collections.abc import Callable
import importlib

import pytest
import torch
import torch.multiprocessing

import tilelang.testing

_DISTRIBUTED_WORKERS: dict[tuple[str, str], Callable[[int, int], None]] = {}


def distributed_test(
    *,
    nprocs: int | None = 2,
    require_fabric: bool = False,
    require_multicast: bool = False,
):
    """Decorate a worker(local_rank, num_ranks) as a distributed pytest case."""

    def decorator(worker: Callable[[int, int], None]):
        worker_key = (worker.__module__, worker.__qualname__)
        _DISTRIBUTED_WORKERS[worker_key] = worker

        @tilelang.testing.requires_cuda
        def test_wrapper():
            _skip_if_distributed_disabled()
            resolved_nprocs = _resolve_nprocs(nprocs)
            _skip_if_not_enough_gpus(resolved_nprocs)
            if require_fabric:
                _skip_if_no_fabric()
            if require_multicast:
                _skip_if_no_multicast()
            torch.multiprocessing.spawn(_distributed_worker_entry, args=(worker_key, resolved_nprocs), nprocs=resolved_nprocs)

        test_wrapper.__name__ = worker.__name__
        test_wrapper.__doc__ = worker.__doc__
        return test_wrapper

    return decorator


def _distributed_worker_entry(local_rank: int, worker_key: tuple[str, str], num_ranks: int):
    module_name, _ = worker_key
    importlib.import_module(module_name)
    worker = _DISTRIBUTED_WORKERS[worker_key]
    worker(local_rank, num_ranks)


def _resolve_nprocs(nprocs: int | None) -> int:
    if nprocs is None:
        return torch.cuda.device_count()
    return nprocs


def _skip_if_distributed_disabled():
    enabled = os.environ.get("TILELANG_USE_DISTRIBUTED", "0").lower() in (
        "1",
        "true",
        "on",
    )
    if not enabled:
        pytest.skip("Requires TILELANG_USE_DISTRIBUTED=1")


def _skip_if_not_enough_gpus(nprocs: int):
    num_gpus = torch.cuda.device_count()
    if num_gpus < nprocs:
        pytest.skip(f"Need >= {nprocs} GPUs, found {num_gpus}")


def _skip_if_no_fabric():
    from tilelang.distributed.shared_memory import _supports_vmm_fabric

    if not _supports_vmm_fabric():
        pytest.skip("VMM fabric not supported on this hardware")


def _skip_if_no_multicast():
    from tilelang.distributed.shared_memory import _supports_multicast

    if not _supports_multicast():
        pytest.skip("NVSwitch multicast not supported on this hardware")

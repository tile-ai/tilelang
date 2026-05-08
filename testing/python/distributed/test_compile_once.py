"""Tests for distributed compile_once coordination."""
from __future__ import annotations

import os
import importlib
import time

import pytest
import torch
import torch.distributed as dist

import tilelang
import tilelang.language as T
import tilelang.testing
from testing.python.distributed._utils import distributed_test

os.environ.setdefault("NCCL_DEBUG", "WARN")


def _identity_kernel():
    @T.prim_func
    def main(src: T.Tensor((16,), T.float32), dst: T.Tensor((16,), T.float32)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if tx < 16:
                dst[tx] = src[tx]

    return main


@tilelang.jit(compile_once=True)
def _jit_identity_kernel():
    return _identity_kernel()


@distributed_test()
def test_compile_once_root_runs_before_non_root(local_rank: int, num_ranks: int):
    from tilelang.distributed import init_dist

    _, _, group = init_dist(local_rank, num_ranks)
    tilelang_jit = importlib.import_module("tilelang.jit")

    compile_time = None

    def fake_cached(**_kwargs):
        nonlocal compile_time
        compile_time = time.monotonic()
        return f"kernel:{local_rank}"

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(tilelang_jit, "cached", fake_cached)

        result = tilelang.compile(_identity_kernel(), compile_once=True, compile_group=group)
        assert result == f"kernel:{local_rank}"

        compile_times = [None for _ in range(num_ranks)]
        dist.all_gather_object(compile_times, compile_time, group=group)
        assert compile_times[0] is not None
        assert compile_times[1] is not None
        assert compile_times[0] <= compile_times[1]
    finally:
        monkeypatch.undo()
        dist.destroy_process_group()


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test()
def test_compile_once_compile_api(local_rank: int, num_ranks: int):
    from tilelang.distributed import init_dist

    _, _, group = init_dist(local_rank, num_ranks)
    kernel = tilelang.compile(_identity_kernel(), compile_once=True, compile_group=group)

    src = torch.arange(16, dtype=torch.float32, device=f"cuda:{local_rank}") + local_rank
    dst = torch.empty_like(src)
    kernel(src, dst)
    torch.cuda.synchronize()
    assert torch.equal(src, dst)

    dist.destroy_process_group()


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@distributed_test()
def test_compile_once_jit_api(local_rank: int, num_ranks: int):
    from tilelang.distributed import init_dist

    _, _, group = init_dist(local_rank, num_ranks)
    _jit_identity_kernel.compile_group = group
    try:
        kernel = _jit_identity_kernel()

        src = torch.arange(16, dtype=torch.float32, device=f"cuda:{local_rank}") + local_rank
        dst = torch.empty_like(src)
        kernel(src, dst)
        torch.cuda.synchronize()
        assert torch.equal(src, dst)
    finally:
        _jit_identity_kernel.compile_group = None
        dist.destroy_process_group()


@distributed_test()
def test_compile_once_nonzero_root(local_rank: int, num_ranks: int):
    from tilelang.distributed import init_dist

    _, _, group = init_dist(local_rank, num_ranks)
    tilelang_jit = importlib.import_module("tilelang.jit")

    compile_time = None

    def fake_cached(**_kwargs):
        nonlocal compile_time
        compile_time = time.monotonic()
        return f"kernel:{local_rank}"

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(tilelang_jit, "cached", fake_cached)

        result = tilelang.compile(
            _identity_kernel(),
            compile_once=True,
            compile_group=group,
            compile_root=1,
        )
        assert result == f"kernel:{local_rank}"

        compile_times = [None for _ in range(num_ranks)]
        dist.all_gather_object(compile_times, compile_time, group=group)
        assert compile_times[0] is not None
        assert compile_times[1] is not None
        assert compile_times[1] <= compile_times[0]
    finally:
        monkeypatch.undo()
        dist.destroy_process_group()


@distributed_test()
def test_compile_once_root_failure_reaches_all_ranks(local_rank: int, num_ranks: int):
    from tilelang.distributed import init_dist

    _, _, group = init_dist(local_rank, num_ranks)
    tilelang_jit = importlib.import_module("tilelang.jit")
    compile_calls = 0

    def fake_cached(**_kwargs):
        nonlocal compile_calls
        compile_calls += 1
        if local_rank == 0:
            raise ValueError("root compile failed")
        return f"kernel:{local_rank}"

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(tilelang_jit, "cached", fake_cached)

        with pytest.raises(RuntimeError, match="root compile failed"):
            tilelang.compile(_identity_kernel(), compile_once=True, compile_group=group)

        assert compile_calls == (1 if local_rank == 0 else 0)
    finally:
        monkeypatch.undo()
        dist.destroy_process_group()


if __name__ == "__main__":
    tilelang.testing.main()

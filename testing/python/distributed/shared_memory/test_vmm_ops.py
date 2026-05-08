"""
VMM (Virtual Memory Management) and IPC shared-memory operations.

Single-GPU tests run directly under pytest.
Multi-GPU tests use torch.multiprocessing.spawn and require >= 2 GPUs.
"""

from __future__ import annotations

import ctypes
import ctypes.util

import pytest
import torch
import torch.distributed as dist

import tilelang.testing
import tilelang.utils.allocator as allocator_mod
from testing.python.distributed._utils import distributed_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _libcudart():
    return ctypes.CDLL(ctypes.util.find_library("cudart") or "libcudart.so")


def _skip_if_no_fabric():
    from tilelang.distributed.shared_memory import _supports_vmm_fabric

    if not _supports_vmm_fabric():
        pytest.skip("VMM fabric not supported on this hardware")


# ---------------------------------------------------------------------------
# Single-GPU tests
# ---------------------------------------------------------------------------


@tilelang.testing.requires_cuda
def test_supports_fabric():
    from tilelang.distributed.shared_memory import _supports_vmm_fabric

    result = _supports_vmm_fabric()
    assert isinstance(result, bool)


def test_resolve_use_vmm_defaults_to_supported_fabric(monkeypatch):
    monkeypatch.delenv("TILESCALE_USE_VMM", raising=False)
    monkeypatch.setattr(allocator_mod, "_supports_vmm_fabric", lambda: True)

    assert allocator_mod._resolve_use_vmm(None, is_distributed=True)
    assert not allocator_mod._resolve_use_vmm(None, is_distributed=False)
    assert not allocator_mod._resolve_use_vmm(False, is_distributed=True)


def test_resolve_use_vmm_falls_back_without_fabric(monkeypatch):
    monkeypatch.delenv("TILESCALE_USE_VMM", raising=False)
    monkeypatch.setattr(allocator_mod, "_supports_vmm_fabric", lambda: False)

    assert not allocator_mod._resolve_use_vmm(None, is_distributed=True)
    assert allocator_mod._resolve_use_vmm(True, is_distributed=True)


def test_resolve_use_vmm_env_override(monkeypatch):
    monkeypatch.setenv("TILESCALE_USE_VMM", "0")
    assert not allocator_mod._resolve_use_vmm(None, is_distributed=True)

    monkeypatch.setenv("TILESCALE_USE_VMM", "1")
    assert allocator_mod._resolve_use_vmm(None, is_distributed=False)


@tilelang.testing.requires_cuda
def test_vmm_malloc_free():
    torch.cuda.set_device(0)
    _skip_if_no_fabric()
    from tilelang.distributed.shared_memory import _vmm_free, _vmm_malloc

    size = 1024 * 1024  # 1 MB
    ptr = _vmm_malloc(size)
    assert ptr != 0, "vmm_malloc returned null"

    lib = _libcudart()
    rc = lib.cudaMemset(ctypes.c_void_p(ptr), 0, ctypes.c_size_t(size))
    assert rc == 0, f"cudaMemset on VMM pointer failed: {rc}"

    _vmm_free(ptr)


@tilelang.testing.requires_cuda
def test_vmm_handle_export_import():
    torch.cuda.set_device(0)
    _skip_if_no_fabric()
    from tilelang.distributed.shared_memory import (
        _close_vmm_handle,
        _create_vmm_handle,
        _open_vmm_handle,
        _vmm_free,
        _vmm_malloc,
    )

    size = 4096
    ptr = _vmm_malloc(size)
    assert ptr != 0

    lib = _libcudart()
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int

    pattern = (ctypes.c_uint8 * size)(*([0xAB] * size))
    rc = lib.cudaMemcpy(ctypes.c_void_p(ptr), ctypes.byref(pattern), size, 1)
    assert rc == 0, f"cudaMemcpy H2D failed: {rc}"

    handle = _create_vmm_handle(ptr)
    assert len(handle) > 0, "handle is empty"

    ptr2 = _open_vmm_handle(handle)
    assert ptr2 != 0, "open_vmm_handle returned null"

    readback = (ctypes.c_uint8 * size)()
    rc = lib.cudaMemcpy(ctypes.byref(readback), ctypes.c_void_p(ptr2), size, 2)
    assert rc == 0, f"cudaMemcpy D2H failed: {rc}"
    assert all(b == 0xAB for b in readback), "Data mismatch after handle export/import"

    _close_vmm_handle(ptr2)
    _vmm_free(ptr)


# ---------------------------------------------------------------------------
# Multi-GPU worker functions (called by spawn)
# ---------------------------------------------------------------------------


@distributed_test(require_fabric=True)
def test_distributed_vmm(local_rank: int, num_ranks: int):
    from tilelang.distributed.utils import init_dist
    from tilelang.utils.allocator import BaseAllocator

    _, _, group = init_dist(local_rank, num_ranks)

    allocator = BaseAllocator(
        size=1024 * 1024,
        device="cuda",
        is_distributed=True,
        local_rank=local_rank,
        num_local_ranks=num_ranks,
        group=group,
    )

    assert allocator.initialized()
    assert allocator._use_vmm
    assert allocator._buffer_ptrs is not None
    assert allocator._buffer_ptrs.shape[0] == num_ranks
    assert allocator._buffer_ptrs[local_rank].item() != 0

    t = allocator._allocate_tensor((256,), torch.float32)
    t.fill_(float(local_rank + 1))
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


@distributed_test()
def test_distributed_ipc_fallback(local_rank: int, num_ranks: int):
    from tilelang.distributed.utils import create_dist_tensor, create_tensor, init_dist

    rank, _, group = init_dist(local_rank, num_ranks)

    data = create_tensor([1024], torch.float32)
    data.fill_(float(rank + 1))

    buffer_ptrs = create_dist_tensor(local_rank, num_ranks, data, rank, group, use_vmm=False)

    assert buffer_ptrs.shape[0] == num_ranks
    remote_rank = (local_rank + 1) % num_ranks
    assert buffer_ptrs[remote_rank].item() != 0, f"Remote rank {remote_rank} pointer is zero"

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    tilelang.testing.main()

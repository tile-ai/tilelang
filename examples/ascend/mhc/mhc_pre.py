# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Q4Kernels contributors.
"""Shifted mHC Pre for contiguous BF16 residuals on Ascend 950.

``mhc_pre`` accepts new input objects on each call. ``bind_mhc_pre`` binds
stable input objects for repeated execution with fresh outputs each time.
"""

import functools
import hashlib
import os
from pathlib import Path
import tempfile

__all__ = ["bind_mhc_pre", "mhc_pre"]


def _ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def _roundup(value, alignment):
    return _ceildiv(value, alignment) * alignment


@functools.lru_cache(maxsize=512, typed=True)
def _select_kernel(n, hidden, num_aic_cores=28, num_vec_cores=56):
    """Select the measured MK/M partition without importing a device runtime."""
    if type(n) is not int or not 0 < n <= 2**31 - 4096:
        raise ValueError("N must be a positive integer no greater than 2**31 - 4096")
    if type(hidden) is not int or hidden not in (4096, 5120, 7168):
        raise ValueError("H must be 4096, 5120 or 7168")
    if type(num_aic_cores) is not int or type(num_vec_cores) is not int or num_aic_cores <= 0 or num_vec_cores < 2:
        raise ValueError("Expected positive AIC capacity and at least two AIVs")
    cores = min(num_aic_cores, num_vec_cores // 2)
    m_groups = min(cores, _ceildiv(n, 256))
    nominal_k_groups = cores // m_groups
    mk_m = min(256, _roundup(_ceildiv(n, m_groups), 16))
    use_large = nominal_k_groups == 1 or ((num_aic_cores, num_vec_cores) == (28, 56) and nominal_k_groups == 2 and n % mk_m != 0)

    if use_large:
        block_m, best_cost = 256, float("inf")
        for candidate in range(256, 15, -16):
            k_l1 = min(32768 // candidate, 1024) // 128 * 128
            if k_l1 // 2 >= 256 or (4 * hidden) % k_l1:
                continue
            cost = _ceildiv(_ceildiv(n, candidate), cores) * candidate
            if cost < best_cost:
                block_m, best_cost = candidate, cost
        block_m = min(block_m, 208)
        k_l1 = min(32768 // block_m, 1024) // 128 * 128
        kwargs = dict(
            hidden=hidden,
            block_m=block_m,
            block_k_l1=k_l1,
            mad_k=min(64, 256 // max(block_m, 32) * 32),
            num_aic_cores=cores,
            full_m_tiles=n % block_m == 0,
        )
        return dict(route="large", block_m=block_m, num_aic_cores=cores, kernel_kwargs=kwargs, workspace_shapes=())

    split_d = _roundup(_ceildiv(4 * hidden, nominal_k_groups), 256) // 4
    k_groups = _ceildiv(hidden, split_d)
    rows = mk_m // 2
    k_l1 = min(32768 // mk_m, 1024) // 128 * 128
    block_k = min(k_l1, split_d)
    ub_k = min(block_k, k_l1 // 2)
    mad_k = min(block_k, 256 // max(mk_m, 32) * 32)
    target_rows = rows if mk_m <= 32 else min(rows, 64, _roundup(_ceildiv(n, 2 * cores), 8))
    y_acc_bytes = 4 * rows * block_k
    other_bytes = 8 * rows * ub_k + 8 * (rows + 1) * ub_k + 160 * rows + 4 * _roundup(rows, 64)
    for coeff_rows in range(target_rows, 7, -8):
        phase2_bytes = 128 * k_groups * coeff_rows + 256 * coeff_rows + 4 * (k_groups + 1) * _roundup(coeff_rows, 64) + 352
        if other_bytes + max(y_acc_bytes, phase2_bytes) <= 245760:
            break
    else:
        raise ValueError("MK buffers exceed the Ascend 950 UB budget")
    kwargs = dict(
        hidden=hidden,
        block_m=mk_m,
        block_k=block_k,
        num_aic_cores=cores,
        m_groups=m_groups,
        k_groups=k_groups,
        split_d=split_d,
        coeff_rows=coeff_rows,
        ub_k=ub_k,
        mad_k=mad_k,
        full_m_tiles=n % mk_m == 0,
    )
    tiles = _ceildiv(n, mk_m)
    return dict(
        route="mk",
        block_m=mk_m,
        num_aic_cores=cores,
        kernel_kwargs=kwargs,
        workspace_shapes=((k_groups, 2, tiles, rows, 32), (k_groups, 2, tiles, rows)),
    )


@functools.lru_cache(maxsize=512)
def _compile_kernel(route, parameters):
    if route == "mk":
        from .mhc_pre_mk import mhc_pre_mk

        return mhc_pre_mk.compile(**dict(parameters))
    from .mhc_pre_large import mhc_pre_large

    return mhc_pre_large.compile(**dict(parameters))


# Capture only tensor options and shapes: cached allocators never retain NPU
# output templates or input tensors. All four or six allocations stay inside
# one native call, including on the small-N path.
_ALLOCATION_SOURCE = r"""
#include <ATen/ops/empty.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <tuple>
#include <utility>
#include <vector>

pybind11::cpp_function bind_allocate(const at::Tensor& like,
                                   const std::vector<std::vector<int64_t>>& shapes) {
    const auto bf16 = like.options().dtype(at::kBFloat16);
    const auto fp32 = like.options().dtype(at::kFloat);
    TORCH_CHECK(shapes.size() == 4 || shapes.size() == 6,
                "mHC Pre requires four outputs and optionally two workspaces");
    if (shapes.size() == 4) {
        return pybind11::cpp_function([shapes, bf16, fp32]() {
            auto y = at::empty(shapes[0], bf16);
            auto pre = at::empty(shapes[1], fp32);
            auto post = at::empty(shapes[2], fp32);
            auto comb = at::empty(shapes[3], fp32);
            return std::make_tuple(std::move(y), std::move(pre),
                                   std::move(post), std::move(comb));
        }, pybind11::call_guard<pybind11::gil_scoped_release>());
    }
    return pybind11::cpp_function([shapes, bf16, fp32]() {
        auto y = at::empty(shapes[0], bf16);
        auto pre = at::empty(shapes[1], fp32);
        auto post = at::empty(shapes[2], fp32);
        auto comb = at::empty(shapes[3], fp32);
        auto raw = at::empty(shapes[4], fp32);
        auto sq = at::empty(shapes[5], fp32);
        return std::make_tuple(std::move(y), std::move(pre), std::move(post),
                               std::move(comb), std::move(raw), std::move(sq));
    }, pybind11::call_guard<pybind11::gil_scoped_release>());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("bind_allocate", &bind_allocate);
}
"""


@functools.lru_cache(maxsize=1)
def _allocation_module():
    from torch.utils.cpp_extension import get_default_build_root, load

    digest = hashlib.sha256(_ALLOCATION_SOURCE.encode()).hexdigest()
    directory = Path(os.environ.get("TORCH_EXTENSIONS_DIR") or get_default_build_root()) / "mhc_pre" / digest
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "allocate.cpp"
    if not source.is_file() or source.read_text() != _ALLOCATION_SOURCE:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
            temporary = Path(output.name)
            output.write(_ALLOCATION_SOURCE.encode())
        try:
            temporary.replace(source)
        finally:
            temporary.unlink(missing_ok=True)
    return load(name="mhc_pre_allocate_" + digest[:16], sources=[str(source)], extra_cflags=["-O2", "-g0"], with_cuda=False)


class _Executor:
    def __init__(self, kernel, allocate, launch, split_k):
        self._kernel = kernel
        self._allocate = allocate
        self._launch = launch
        self._split_k = split_k

    def __call__(self, x, previous, phi, alpha, bias):
        outputs = self._allocate()
        if self._split_k:
            y, pre, post, comb, raw, sq = outputs
            # Separate read formals preserve the GM publication boundary.
            # Both pairs deliberately alias the same freshly allocated storage.
            self._launch(x, previous, phi, alpha, bias, y, pre, post, comb, raw, sq, raw, sq)
        else:
            self._launch(x, previous, phi, alpha, bias, *outputs)
        return outputs[:4]


@functools.lru_cache(maxsize=128)
def _get_executor(n, hidden, num_aic_cores, num_vec_cores, device_index):
    import torch

    config = _select_kernel(n, hidden, num_aic_cores, num_vec_cores)
    kernel = _compile_kernel(config["route"], tuple(sorted(config["kernel_kwargs"].items())))
    device = torch.device("npu", device_index)
    shapes = ((n, hidden), (n, 4, 1), (n, 4, 1), (n, 4, 4)) + config["workspace_shapes"]
    options_source = torch.empty((), dtype=torch.bfloat16, device=device)
    allocate = _allocation_module().bind_allocate(options_source, shapes)
    kernel.adapter._prepare_torch_device(device)
    launch = kernel.adapter.get_exportable_executable()["main"]
    return _Executor(kernel, allocate, launch, config["route"] == "mk")


def _validate_inputs(x, prev_pre, phi, alpha, bias, num_aic_cores, num_vec_cores):
    import torch

    if x.ndim != 3 or x.shape[1] != 4:
        raise ValueError("Expected residual[N,4,H]")
    n, _, hidden = x.shape
    _select_kernel(n, hidden, num_aic_cores, num_vec_cores)
    if tuple(prev_pre.shape) not in ((n, 4), (n, 4, 1)):
        raise ValueError("Expected prev_pre[N,4] or [N,4,1]")
    for name, tensor, shape, dtype in (
        ("residual", x, (n, 4, hidden), torch.bfloat16),
        ("prev_pre", prev_pre, tuple(prev_pre.shape), torch.float32),
        ("phi", phi, (24, 4 * hidden), torch.float32),
        ("alpha", alpha, (3,), torch.float32),
        ("bias", bias, (24,), torch.float32),
    ):
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"{name}: expected shape={shape}, dtype={dtype}")
        if tensor.device.type != "npu" or tensor.device != x.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on the same NPU")
    return n, hidden, prev_pre.view(n, 4)


def bind_mhc_pre(x, prev_pre, phi, alpha, bias, *, num_aic_cores=28, num_vec_cores=56):
    """Bind stable input objects; each run allocates and returns four outputs.

    Input contents may change in place. Shape, dtype, strides and device must
    remain fixed. The native launcher reads the current NPU stream each call.
    """
    n, hidden, previous = _validate_inputs(x, prev_pre, phi, alpha, bias, num_aic_cores, num_vec_cores)
    execute = _get_executor(n, hidden, num_aic_cores, num_vec_cores, x.device.index)

    def run():
        return execute(x, previous, phi, alpha, bias)

    return run


def mhc_pre(x, prev_pre, phi, alpha, bias, *, num_aic_cores=28, num_vec_cores=56):
    """Return layer_input, next_pre_mix, post_mix and comb_mix.

    Residuals are contiguous BF16 [N,4,H]; previous mix and projection
    parameters are FP32. All inputs must be on the same NPU. Compilation and
    executor setup are cached by shape, capacity and device, never by inputs.
    """
    n, hidden, previous = _validate_inputs(x, prev_pre, phi, alpha, bias, num_aic_cores, num_vec_cores)
    execute = _get_executor(n, hidden, num_aic_cores, num_vec_cores, x.device.index)
    return execute(x, previous, phi, alpha, bias)

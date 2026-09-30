"""Shared source-language validation at the TileIR adapter boundary."""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.jit.adapter.tileir.adapter import TileIRKernelAdapter

from tileir_jit_test_utils import cuda_target_for_test, skip_if_tileir_toolchain_unavailable


def _local_index_kernel(access):
    @T.prim_func
    def main(out: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            local = T.alloc_local((32,), "float32")
            for j in T.serial(32):
                local[j] = T.float32(j)
            for i in T.Parallel(32):
                if access == "store":
                    local[i] = T.float32(i)
                elif access == "load":
                    out[i] = local[i]
                else:
                    out[i] = local[0]

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("backend", ["tvm_ffi", "tileir"])
@pytest.mark.parametrize("access", ["load", "store"])
def test_backends_reject_parallel_local_index(monkeypatch, backend, access):
    if backend == "tileir":
        skip_if_tileir_toolchain_unavailable()
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    with pytest.raises(ValueError, match="Local buffer.*is indexed by T.Parallel loop variable"):
        tilelang.compile(_local_index_kernel(access), target=cuda_target_for_test(), execution_backend=backend)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("as_module", [False, True])
def test_tileir_preparation_accepts_constant_local_index(as_module):
    func = _local_index_kernel("constant")
    source = tvm.IRModule({"main": func}) if as_module else func
    _, prepared = TileIRKernelAdapter._prepare_device_module(source, cuda_target_for_test())
    assert TileIRKernelAdapter._count_kernel_launches(prepared) == 1


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("ambient_disabled,explicit_disabled", [(False, True), (True, None), (True, False)])
def test_tileir_preparation_honors_semantic_check_config(ambient_disabled, explicit_disabled):
    key = tilelang.PassConfigKey.TL_DISABLE_PRELOWER_SEMANTIC_CHECK
    config = None if explicit_disabled is None else {key: explicit_disabled}
    func = _local_index_kernel("load")
    with tvm.transform.PassContext(config={key: ambient_disabled}):
        if explicit_disabled is False:
            with pytest.raises(ValueError, match="Local buffer.*is indexed by T.Parallel loop variable"):
                TileIRKernelAdapter._prepare_device_module(func, cuda_target_for_test(), config)
        else:
            TileIRKernelAdapter._prepare_device_module(func, cuda_target_for_test(), config)
        assert bool(tvm.transform.PassContext.current().config[key]) == ambient_disabled


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("opt_level", [0, 3])
def test_tileir_semantic_checks_accept_backend_options(monkeypatch, opt_level):
    skip_if_tileir_toolchain_unavailable()
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    options = {tilelang.PassConfigKey.TL_TILEIR_OPT_LEVEL: opt_level}
    tilelang.compile(
        _local_index_kernel("constant"),
        target=cuda_target_for_test(),
        execution_backend="tileir",
        pass_configs=options,
    )
    with pytest.raises(ValueError, match="Local buffer.*is indexed by T.Parallel loop variable"):
        tilelang.compile(
            _local_index_kernel("load"),
            target=cuda_target_for_test(),
            execution_backend="tileir",
            pass_configs=options,
        )

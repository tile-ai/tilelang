import torch
import pytest
import tilelang
from examples.norm.rms_norm import rms_norm, rms_norm_splitk, ref_program

def test_rms_norm():
    M, N, blk_m = 1024, 1024, 128
    program = rms_norm(M, N, blk_m)
    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target="cuda",
        execution_backend="cython",
        pass_configs={"tl.disable_tma_lower": True}
    )
    profiler = kernel.get_profiler()
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)

def test_rms_norm_splitk():
    M, N, blk_m, blk_k = 1024, 1024, 128, 128
    program = rms_norm_splitk(M, N, blk_m, blk_k)
    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target="cuda",
        execution_backend="cython",
        pass_configs={"tl.disable_tma_lower": True}
    )
    profiler = kernel.get_profiler()
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)

def test_rms_norm_edge_cases():
    # Test with small dimensions
    M, N, blk_m = 32, 32, 16
    program = rms_norm(M, N, blk_m)
    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target="cuda",
        execution_backend="cython",
        pass_configs={"tl.disable_tma_lower": True}
    )
    profiler = kernel.get_profiler()
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01)

    # Test with non-power-of-2 dimensions
    M, N, blk_m = 1000, 1000, 128
    program = rms_norm(M, N, blk_m)
    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target="cuda",
        execution_backend="cython",
        pass_configs={"tl.disable_tma_lower": True}
    )
    profiler = kernel.get_profiler()
    profiler.assert_allclose(ref_program, rtol=0.01, atol=0.01) 
import pytest

import tilelang.language as T
from tilelang import tvm
from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter, TensorCoreIntrinEmitterWithLadderTransform
from tilelang.cuda.op.gemm.gemm_mma import GemmMMA
from tilelang.cuda.op.gemm.gemm_mma_sm70 import GemmMMASm70
from tilelang.tileop.gemm.registry import resolve_gemm_impl


def test_sm75_uses_regular_mma_gemm_impl():
    Target = tvm.target.Target

    assert resolve_gemm_impl("cuda.mma", Target("cuda -arch=sm_70")) is GemmMMASm70
    assert resolve_gemm_impl("cuda.mma", Target("cuda -arch=sm_75")) is GemmMMA


def test_turing_fp16_emitter_uses_sm75_m16n8k8_shape():
    emitter = TensorCoreIntrinEmitter(
        a_dtype=T.float16,
        b_dtype=T.float16,
        accum_dtype=T.float32,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=32,
        warp_col_tiles=32,
        chunk=32,
        is_turing=True,
    )

    assert emitter.mma_prefix == "m16n8k8"
    assert emitter.micro_size_x == 16
    assert emitter.micro_size_y == 8
    assert emitter.micro_size_k == 8
    assert emitter.local_size_a == 4
    assert emitter.local_size_b == 2
    assert emitter.local_size_out == 4
    assert emitter.warp_rows == 2
    assert emitter.warp_cols == 4


def test_turing_int8_emitter_uses_sm75_m8n8k16_shape():
    emitter = TensorCoreIntrinEmitter(
        a_dtype=T.int8,
        b_dtype=T.int8,
        accum_dtype=T.int32,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=32,
        warp_col_tiles=32,
        chunk=64,
        is_turing=True,
    )

    assert emitter.mma_prefix == "m8n8k16"
    assert emitter.micro_size_x == 8
    assert emitter.micro_size_y == 8
    assert emitter.micro_size_k == 16
    assert emitter.local_size_a == 4
    assert emitter.local_size_b == 4
    assert emitter.local_size_out == 2
    assert emitter.warp_rows == 4
    assert emitter.warp_cols == 4
    assert emitter.get_store_index_map() is not None


def test_turing_int4_emitter_uses_sm75_m8n8k32_shape():
    emitter = TensorCoreIntrinEmitter(
        a_dtype=T.int4,
        b_dtype=T.int4,
        accum_dtype=T.int32,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=32,
        warp_col_tiles=32,
        chunk=64,
        is_turing=True,
    )

    assert emitter.mma_prefix == "m8n8k32"
    assert emitter.micro_size_x == 8
    assert emitter.micro_size_y == 8
    assert emitter.micro_size_k == 32
    assert emitter.local_size_a == 8
    assert emitter.local_size_b == 8
    assert emitter.local_size_out == 2
    assert emitter.warp_rows == 4
    assert emitter.warp_cols == 4
    assert emitter.get_store_index_map() is not None


def test_turing_m8n8_integer_emitter_uses_matching_store_lane_map():
    emitter = TensorCoreIntrinEmitter(
        a_dtype=T.int8,
        b_dtype=T.int8,
        accum_dtype=T.int32,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=32,
        warp_col_tiles=32,
        chunk=64,
        is_turing=True,
    )

    assert emitter.local_size_out == 2
    assert emitter._use_fp64_store_index_map()


def test_non_turing_int8_emitter_keeps_sm80_m16n8k32_shape():
    emitter = TensorCoreIntrinEmitter(
        a_dtype=T.int8,
        b_dtype=T.int8,
        accum_dtype=T.int32,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=32,
        warp_col_tiles=32,
        chunk=64,
        is_turing=False,
    )

    assert emitter.mma_prefix == "m16n8k32"
    assert emitter.micro_size_x == 16
    assert emitter.micro_size_k == 32
    assert emitter.micro_size_y == 16
    assert emitter.local_size_a == 16
    assert emitter.local_size_b == 16
    assert emitter.local_size_out == 8


def test_uint4_mma_emitter_is_not_advertised_until_fragment_layouts_support_it():
    with pytest.raises(ValueError, match="Unsupported dtype: uint4"):
        TensorCoreIntrinEmitter(
            a_dtype="uint4",
            b_dtype="uint4",
            accum_dtype=T.int32,
            a_transposed=False,
            b_transposed=True,
            block_row_warps=2,
            block_col_warps=2,
            warp_row_tiles=32,
            warp_col_tiles=32,
            chunk=64,
            is_turing=True,
        )


def test_ladder_transform_int8_micro_size_uses_actual_k_dim():
    emitter = TensorCoreIntrinEmitterWithLadderTransform(
        a_dtype=T.int8,
        b_dtype=T.int8,
        accum_dtype=T.int32,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=2,
        block_col_warps=2,
        warp_row_tiles=32,
        warp_col_tiles=32,
        chunk=64,
    )

    assert emitter.mma_prefix == "m16n8k32"
    assert emitter.micro_size_x == 16
    assert emitter.micro_size_y == 16
    assert emitter.micro_size_k == 32
    assert emitter.local_size_a == 16
    assert emitter.local_size_b == 16
    assert emitter.local_size_out == 8

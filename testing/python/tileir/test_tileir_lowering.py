"""Structured CUDA Tile IR lowering tests for representative kernels."""

from __future__ import annotations


import pytest

import tilelang
import tilelang.testing
from tilelang import language as T
from tilelang import tvm as tvm
from tilelang.tileir import checks
from tilelang.tileir.errors import _UnsupportedTileIRNode
from tilelang.jit.adapter.tileir.adapter import TileIRKernelAdapter
from tilelang.tileir.artifact import TileIRLaunchMetadata, TileIRLoweringResult
from tilelang.tileir import lowering as tileir_lowering
from tilelang.tileir.lowering import lower_primfunc_to_tileir
from tilelang.tileir.semantic import extract_semantic_program
from tilelang.backend.target import determine_target

Range = tvm.ir.Range
structural_equal = tvm.ir.structural_equal
Target = tvm.target.Target


from tileir_test_utils import (
    _build_tileir_module_for_test,
    _cuda_target_for_test,
    _semantic_kinds,
    _semantic_tile_ops,
    _load_mla_paged_example,
    _load_mla_decode_example,
    _load_mla_kv_fp8_example,
    _load_mla_persistent_example,
    _load_mhc_pre_example,
    _load_gemv_example,
    _load_dequant_gemv_example,
    _load_deepseek_v4_act_quant_example,
    _load_minference_example,
    _lower_tileir_primfunc_for_test,
    _prepared_tileir_kernel_for_test,
    _tileir_source_for_test,
)
from tilelang.transform import PassConfigKey


def test_tileir_hint_default_arch_is_canonicalized_last():
    hints = tileir_lowering._validated_hints(
        {
            "default": {"occupancy": 2},
            "sm_120": {"num_cta_in_cga": 4},
            "sm_100": {"num_cta_in_cga": 2},
        }
    )

    assert tuple(arch for arch, _ in hints) == ("sm_100", "sm_120", "default")


def test_tileir_lowers_scalar_and_guard(monkeypatch, tmp_path):
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def guarded_kernel(n: int):
        @T.prim_func
        def main(A: T.Tensor([n], T.float32)):
            with T.Kernel(1, 1, threads=1) as (bx, by):
                if bx < n and by < n:
                    A[0] = 1.0

        return main

    prim_func = guarded_kernel.get_tir(4)
    toolchain = checks.TileIRToolchain(
        cuda_tile_ir_module=checks.CUDA_TILE_IR_MLIR_MODULE,
        tileiras_path=tmp_path / "tileiras",
        tileiras_version="tileiras 13.3",
    )

    def fake_assemble(tileir_module, *, kernel_name, target, toolchain, launch_metadata=None, argument_names=(), opt_level=3):
        del tileir_module, target, toolchain
        return TileIRLoweringResult(
            kernel_name=kernel_name,
            cubin=b"cubin",
            tileir_source=f"module @{kernel_name}",
            launch_metadata=launch_metadata or TileIRLaunchMetadata(),
            argument_names=argument_names,
        )

    monkeypatch.setattr(tileir_lowering, "assemble_tileir_module", fake_assemble)

    result = lower_primfunc_to_tileir(prim_func, _cuda_target_for_test(), toolchain)

    assert result.kernel_name == "main"
    assert result.cubin


@pytest.mark.xfail(
    raises=_UnsupportedTileIRNode,
    strict=True,
    reason="naive_gemv is a SIMT (per-thread) kernel: it indexes shared/global buffers "
    "by threadIdx (B_shared[tn, tk], C[bn*BLOCK_N + tn]). A per-thread scatter into a "
    "tile is not expressible in cuTile's collective tile model, so lowering raises "
    "_UnsupportedTileIRNode by design. Supporting it would require recognizing the "
    "per-thread reduction as a collective tile reduction, which is outside the "
    "structured lowering contract.",
)
def test_tileir_lowers_gemv_dynamic_shared_scalar_store():
    example = _load_gemv_example()
    prim_func = example.naive_gemv.get_tir(None, None, 128, 128, T.float16, T.float32, N=128, K=128)

    result = _lower_tileir_primfunc_for_test(prim_func)

    assert result.kernel_name == "naive_gemv"
    assert result.launch_metadata.grid == (1, 1, 1)
    assert result.cubin


def _vec_add_with_hints(n: int, num_worker_warps: int | None = None):
    @tilelang.jit
    def vec_add(n: int, num_ctas: int | None = None, occupancy: int | None = None, num_worker_warps: int | None = num_worker_warps):
        @T.prim_func
        def main(A: T.Tensor([n], T.float32), B: T.Tensor([n], T.float32), C: T.Tensor([n], T.float32)):
            with T.Kernel(T.ceildiv(n, 128), threads=128, num_ctas=num_ctas, occupancy=occupancy, num_worker_warps=num_worker_warps) as bx:
                for i in T.Parallel(128):
                    idx = bx * 128 + i
                    if idx < n:
                        C[idx] = A[idx] + B[idx]

        return main

    return vec_add


def test_tileir_lowers_kernel_entry_hints():
    """Verify that num_ctas/occupancy annotation hints are forwarded to the MLIR entry.

    The pipeline generates correct num_cta_in_cga/occupancy hints in the MLIR
    entry function. The test checks the MLIR text directly via
    ``build_tileir_module`` rather than checking ``result.tileir_source``
    after assembly, because assembly is unrelated to hint propagation.
    """
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.semantic import materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc

    vec_add = _vec_add_with_hints(256)

    # sm_120 matches the test target used in _cuda_target_for_test().
    _arch = "sm_120"

    # Hinted: num_ctas=4, occupancy=3 → must appear in MLIR entry attributes.
    pf_hinted = vec_add.get_tir(256, num_ctas=4, occupancy=3)
    pf_hinted = materialize_launch_nest(pf_hinted)
    pf_hinted = _split_grid_sync_primfunc(pf_hinted)
    mlir_hinted = str(build_tileir_module(pf_hinted, arch=_arch, num_cta=4, occupancy=3))
    assert "num_cta_in_cga = 4" in mlir_hinted or "num_cta_in_cga=4" in mlir_hinted
    assert "occupancy = 3" in mlir_hinted or "occupancy=3" in mlir_hinted

    # Baseline (no hints): optimization_hints attribute must NOT appear.
    pf_base = vec_add.get_tir(256)
    pf_base = materialize_launch_nest(pf_base)
    pf_base = _split_grid_sync_primfunc(pf_base)
    mlir_base = str(build_tileir_module(pf_base, arch=_arch))
    assert "optimization_hints" not in mlir_base


def test_tileir_lowers_num_worker_warps_hint():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.semantic import materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc

    vec_add = _vec_add_with_hints(256)
    pf = vec_add.get_tir(256, num_worker_warps=8)
    pf = materialize_launch_nest(pf)
    pf = _split_grid_sync_primfunc(pf)
    mlir = str(build_tileir_module(pf, arch="sm_120", num_worker_warps=8))
    assert "num_worker_warps" in mlir

    # Invalid value must raise at options validation.
    from tilelang.tileir.errors import TileIRLoweringError
    from tilelang.tileir.lowering import _lowering_options

    pf_bad = vec_add.get_tir(256, num_worker_warps=6)
    with pytest.raises(TileIRLoweringError, match="num_worker_warps"):
        _lowering_options(pf_bad, None)


def test_tileir_fast_math_emits_flush_to_zero_division():
    """TL_ENABLE_FAST_MATH must reach the division emit: fast_math=True emits the
    flush_to_zero (approximate-reciprocal) ``divf`` matching what cuTile emits;
    fast_math=False keeps the precise operation."""
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.semantic import materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc

    @tilelang.jit
    def div_kernel(M: int, N: int):
        @T.prim_func
        def main(A: T.Tensor((M, N), "float32"), B: T.Tensor((M, N), "float32"), C: T.Tensor((M, N), "float32")):
            with T.Kernel(1, threads=128):
                for i, j in T.Parallel(M, N):
                    C[i, j] = A[i, j] / B[i, j]

        return main

    pf = _split_grid_sync_primfunc(materialize_launch_nest(div_kernel.get_tir(64, 64)))
    mlir_fast = str(build_tileir_module(pf, arch="sm_120", fast_math=True))
    mlir_prec = str(build_tileir_module(pf, arch="sm_120", fast_math=False))
    assert "divf" in mlir_prec, "expected a divf op in the lowered division kernel"
    assert "flush_to_zero" in mlir_fast, "fast_math=True must emit flush_to_zero divf (else it is inert)"
    assert "flush_to_zero" not in mlir_prec, "fast_math=False must keep the precise divf"


def test_tileir_fast_math_emits_approx_tanh_and_rsqrt():
    """fast_math must reach the per-element tanh (gelu) and the rsqrt (norm)
    emit: tanh -> hardware APPROX (matches torch approximate='tanh'), rsqrt ->
    flush_to_zero."""
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)
    from tilelang.tileir.pipeline import build_tileir_module
    from tilelang.tileir.semantic import materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc

    @tilelang.jit
    def trans_kernel(M: int, N: int):
        @T.prim_func
        def main(A: T.Tensor((M, N), "float32"), B: T.Tensor((M, N), "float32"), C: T.Tensor((M, N), "float32")):
            with T.Kernel(1, threads=128):
                for i, j in T.Parallel(M, N):
                    B[i, j] = T.tanh(A[i, j])
                    C[i, j] = T.rsqrt(A[i, j])

        return main

    pf = _split_grid_sync_primfunc(materialize_launch_nest(trans_kernel.get_tir(64, 64)))
    mlir_fast = str(build_tileir_module(pf, arch="sm_120", fast_math=True))
    mlir_prec = str(build_tileir_module(pf, arch="sm_120", fast_math=False))
    assert "tanh" in mlir_prec and "rsqrt" in mlir_prec, "expected tanh + rsqrt ops"
    assert "rounding<approx>" in mlir_fast, "fast_math must emit APPROX tanh (else inert)"
    assert "flush_to_zero" in mlir_fast, "fast_math must emit flush_to_zero rsqrt (else inert)"
    assert "rounding<approx>" not in mlir_prec, "fast_math=False must keep precise tanh"
    assert "flush_to_zero" not in mlir_prec, "fast_math=False must keep precise rsqrt"


@pytest.mark.parametrize(
    "num_ctas, occupancy",
    [
        (3, None),  # not a power of two
        (32, None),  # exceeds [1, 16]
        (1, 0),  # occupancy below [1, 32]
        (1, 33),  # occupancy above [1, 32]
    ],
)
def test_tileir_rejects_invalid_entry_hints(num_ctas, occupancy):
    from tilelang.tileir.errors import TileIRLoweringError

    vec_add = _vec_add_with_hints(256)
    with pytest.raises(TileIRLoweringError):
        _lower_tileir_primfunc_for_test(vec_add.get_tir(256, num_ctas=num_ctas, occupancy=occupancy))


@pytest.mark.parametrize("order", ["row", "column"])
def test_tileir_ignores_threadblock_swizzle_hint(order):
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def swizzled_store():
        @T.prim_func
        def main(C: T.Tensor((4, 4), T.int32)):
            with T.Kernel(4, 4, threads=1) as (bx, by):
                T.use_swizzle(panel_size=2, order=order)
                C[by, bx] = by * 10 + bx

        return main

    source = _tileir_source_for_test(swizzled_store)
    store_lines = [line for line in source.splitlines() if "store_ptr_tko" in line or "store_view_tko" in line]

    assert store_lines
    assert all("[%blockId_y, %blockId_x]" in line for line in store_lines)
    assert "select " not in source
    assert "rem" not in source
    assert "divi " not in source


def _copy_kernel_with_hints():
    @tilelang.jit
    def copy_kernel(M: int, N: int, latency: int | None = None, disable_tma: bool = False):
        @T.prim_func
        def main(A: T.Tensor([M, N], T.float32), C: T.Tensor([M, N], T.float32)):
            with T.Kernel(1, 1, threads=128) as (bx, by):
                A_shared = T.alloc_shared((M, N), T.float32)
                T.copy(A[0:M, 0:N], A_shared, latency=latency, disable_tma=disable_tma)
                T.copy(A_shared, C[0:M, 0:N])

        return main

    return copy_kernel


def test_tileir_lowers_copy_load_store_hints():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)
    copy_kernel = _copy_kernel_with_hints()

    # latency -> cuTile load/store latency hint; disable_tma -> allow_tma=false.
    src = _tileir_source_for_test(copy_kernel, 128, 128, 8, True)
    assert "optimization_hints" in src
    assert "latency = 8" in src
    assert "allow_tma = false" in src

    # A plain T.copy emits no load/store hint (unchanged lowering).
    baseline = _tileir_source_for_test(copy_kernel, 128, 128)
    assert "optimization_hints" not in baseline


def test_tileir_copy_omits_padding_only_for_statically_in_bounds_tiles():
    """Static bounds affect padding without changing the view-based copy path."""
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    @tilelang.jit
    def tiled_copy(n: int, block_n: int = 32, latency: int | None = None):
        @T.prim_func
        def main(A: T.Tensor((n,), T.float16), B: T.Tensor((n,), T.float16)):
            with T.Kernel(T.ceildiv(n, block_n), threads=128) as bx:
                tile = T.alloc_shared((block_n,), T.float16)
                T.copy(A[bx * block_n], tile, latency=latency)
                T.copy(tile, B[bx * block_n])

        return main

    exact = _tileir_source_for_test(tiled_copy, 256)
    tail = _tileir_source_for_test(tiled_copy, 250)
    large_exact = _tileir_source_for_test(tiled_copy, 8192, 8192)
    hinted = _tileir_source_for_test(tiled_copy, 256, 32, 4)

    assert "padding_value = zero" not in exact
    assert "load_view_tko" in exact
    assert "store_view_tko" in exact
    assert "load_ptr_tko" not in exact
    assert "store_ptr_tko" not in exact
    assert "padding_value = zero" in tail
    assert "load_view_tko" in tail
    assert "store_view_tko" in tail
    assert "padding_value = zero" not in large_exact
    assert "load_view_tko" in large_exact
    assert "store_view_tko" in large_exact
    assert "padding_value = zero" not in hinted
    assert "load_view_tko" in hinted
    assert "store_view_tko" in hinted
    assert "latency = 4" in hinted


def test_tileir_global_disable_tma_lower_forces_allow_tma_false():
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)
    copy_kernel = _copy_kernel_with_hints()
    target, prepared, semantic_program = _prepared_tileir_kernel_for_test(copy_kernel, 128, 128)
    semantic_program.kernels[0]

    # The native `tl.disable_tma_lower` pass config flows into the lowering options...
    options = tileir_lowering._lowering_options(prepared, {PassConfigKey.TL_DISABLE_TMA_LOWER: True})
    assert options.disable_tma is True

    # ...and forces allow_tma=false on copies that carry no per-copy hint, matching how
    # the CUDA backend honors the global pass config (the two are OR-ed).
    src = str(_build_tileir_module_for_test(prepared, target, options=options))
    assert src.count("allow_tma = false") == 2

    # Without the config, a plain copy leaves the TMA decision to the assembler.
    baseline = str(_build_tileir_module_for_test(prepared, target))
    assert "allow_tma = false" not in baseline


@pytest.mark.parametrize("latency", [0, 11])
def test_tileir_rejects_invalid_copy_latency(latency):
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)
    from tilelang.tileir.errors import TileIRLoweringError

    copy_kernel = _copy_kernel_with_hints()
    with pytest.raises(TileIRLoweringError):
        _tileir_source_for_test(copy_kernel, 128, 128, latency)


# Each kernel uses a *literal* element-dtype annotation. This file uses PEP 563 string
# annotations, and the JIT resolves a prim_func's annotations via get_type_hints, which
# evaluates them against module globals only — so a literal `T.<dtype>` resolves (T is a
# module global) while a closure/arg-supplied dtype would not.
@tilelang.jit
def _copy_int4():
    @T.prim_func
    def main(A: T.Tensor([16], T.int4), C: T.Tensor([16], T.int4)):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(16):
                C[i] = A[i]

    return main


@tilelang.jit
def _copy_int16():
    @T.prim_func
    def main(A: T.Tensor([16], T.int16), C: T.Tensor([16], T.int16)):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(16):
                C[i] = A[i]

    return main


@tilelang.jit
def _copy_uint16():
    @T.prim_func
    def main(A: T.Tensor([16], T.uint16), C: T.Tensor([16], T.uint16)):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(16):
                C[i] = A[i]

    return main


@tilelang.jit
def _copy_float8_e8m0fnu():
    @T.prim_func
    def main(A: T.Tensor([16], T.float8_e8m0fnu), C: T.Tensor([16], T.float8_e8m0fnu)):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(16):
                C[i] = A[i]

    return main


@tilelang.jit
def _copy_float4_e2m1fn():
    @T.prim_func
    def main(A: T.Tensor([16], T.float4_e2m1fn), C: T.Tensor([16], T.float4_e2m1fn)):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(16):
                C[i] = A[i]

    return main


@pytest.mark.parametrize(
    "kernel, tileir_type",
    [
        (_copy_int4, "i4"),
        (_copy_int16, "i16"),
        (_copy_uint16, "i16"),
        (_copy_float8_e8m0fnu, "f8E8M0FNU"),
        (_copy_float4_e2m1fn, "f4E2M1FN"),
    ],
)
def test_tileir_lowers_public_cuda_tile_element_dtypes(kernel, tileir_type):
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    src = _tileir_source_for_test(kernel)

    assert tileir_type in src


def test_tileir_rejects_dequant_gemv_thread_allreduce():
    # This fp16xint4 dequant GEMV is a SIMT kernel: a 2D thread block (reduce_thread x
    # n_partition) partitions outputs across threadIdx.y and reduces per-thread partials
    # across threadIdx.x via `tvm_thread_allreduce`. The tile backend executes tiles
    # collectively and cannot represent per-thread lanes, so it must reject this kernel.
    #
    # The pipeline rejects the kernel for "decode_i4u_to_f16 not yet lowered"
    # before reaching tvm_thread_allreduce —
    # the key safety contract is that the kernel is REJECTED (not silently miscompiled),
    # regardless of which unsupported construct is caught first.
    from tilelang.tileir.errors import TileIRLoweringNotImplementedError

    example = _load_dequant_gemv_example()
    prim_func = example.dequantize_gemv.get_tir(
        1,  # M
        1024,  # N
        1024,  # K
        T.float16,  # in_dtype
        T.float16,  # out_dtype
        T.float16,  # accum_dtype
        4,  # num_bits
        T.int8,  # storage_dtype
        "uint",  # source_format
        4,  # n_partition
        32,  # reduce_thread
        True,  # fast_decoding
        False,  # trans_A
        True,  # trans_B
        -1,  # group_size
        False,  # with_scaling
    )
    program = extract_semantic_program(prim_func)
    assert "reduce_scope" in _semantic_kinds(program.kernels[0].body)

    # Must raise TileIRLoweringNotImplementedError (kernel is rejected, not miscompiled).
    # The kernel may be rejected at a different unsupported construct than
    # tvm_thread_allreduce, but the safety guarantee is the same: hard rejection.
    with pytest.raises(TileIRLoweringNotImplementedError):
        _lower_tileir_primfunc_for_test(prim_func)


@pytest.mark.parametrize("round_scale", [False, True])
def test_tileir_lowers_deepseek_v4_fp8_act_quant(round_scale):
    torch = pytest.importorskip("torch")
    example = _load_deepseek_v4_act_quant_example()
    x = torch.empty((32, 128), dtype=torch.bfloat16)
    prim_func = example.fp8_quant_kernel.get_tir(x, 128, round_scale)

    result = _lower_tileir_primfunc_for_test(prim_func)

    assert result.kernel_name == "fp8_quant_kernel"
    assert result.cubin


@pytest.mark.xfail(
    reason="data-dependent global gather K[bz, by, column_index[k+i], j] is not "
    "expressible in cuTile's collective tile model: make_partition_view and "
    "load_view_tko take one scalar partition index per dimension and cannot "
    "represent a per-row indirect gather.",
    strict=True,
    raises=_UnsupportedTileIRNode,
)
def test_tileir_lowers_minference_sparse_attention_register_barriers():
    example = _load_minference_example()
    prim_func = example._tl_vs_sparse_flashattn.get_tir(
        1,  # batch
        1,  # heads
        128,  # seq_len
        64,  # dim
        1024,  # vertical_size keeps the original example's default padded index contract
        8,  # slash_size
    )
    program = extract_semantic_program(prim_func)

    assert "register_control" in _semantic_kinds(program.kernels[0].body)
    assert "tl.tileop.tma_copy" in _semantic_tile_ops(program.kernels[0].body)
    result = _lower_tileir_primfunc_for_test(prim_func)

    assert result.kernel_name == "vs_sparse_flashattn_ws"
    assert result.launch_metadata.grid == (2, 1, 1)
    assert result.cubin


def test_tileir_lowers_mla_paged_decode_multi_kernel_artifact():
    try:
        toolchain = checks.check_tileir_available()
    except checks.TileIRDependencyError as exc:
        pytest.skip(f"CUDA TileIR toolchain unavailable: {exc}")

    example = _load_mla_paged_example()
    prim_func = example.mla_decode_tilelang.get_tir(
        1,  # batch
        32,  # h_q
        1,  # h_kv
        1024,  # max_seqlen_pad
        512,  # dv
        64,  # dpe
        64,  # block_N
        16,  # block_H, must divide h_q // h_kv
        4,  # num_split
        64,  # block_size
        None,
    )

    result = lower_primfunc_to_tileir(prim_func, _cuda_target_for_test(), toolchain)

    assert result.kernel_name == "main_split"
    assert result.temporary_buffers == ()
    assert len(result.kernels) == 2
    assert result.kernels[0].launch_metadata.grid == (1, 2, 4)
    assert result.kernels[1].launch_metadata.grid == (32, 1, 1)
    assert result.kernels[0].argument_names == (
        "Q",
        "Q_pe",
        "KV",
        "K_pe",
        "block_table",
        "cache_seqlens",
        "glse",
        "Output_partial",
    )
    assert result.kernels[1].argument_names == ("glse", "Output_partial", "Output")
    assert all(kernel.cubin for kernel in result.kernels)


def test_tileir_lowers_mla_paged_decode_no_split_ifthenelse():
    try:
        toolchain = checks.check_tileir_available()
    except checks.TileIRDependencyError as exc:
        pytest.skip(f"CUDA TileIR toolchain unavailable: {exc}")

    example = _load_mla_paged_example()
    prim_func = example.mla_decode_tilelang.get_tir(
        1,  # batch
        32,  # h_q
        1,  # h_kv
        1024,  # max_seqlen_pad
        512,  # dv
        64,  # dpe
        64,  # block_N
        16,  # block_H, must divide h_q // h_kv
        1,  # num_split selects main_no_split
        64,  # block_size
        None,
    )

    result = lower_primfunc_to_tileir(prim_func, _cuda_target_for_test(), toolchain)

    assert result.kernel_name == "main_no_split"
    assert result.launch_metadata.grid == (1, 2, 1)
    assert result.argument_names == (
        "Q",
        "Q_pe",
        "KV",
        "K_pe",
        "block_table",
        "cache_seqlens",
        "glse",
        "Output_partial",
        "Output",
    )
    assert result.cubin


def test_tileir_lowers_mla_decode_split_let_bound_region_extents(monkeypatch, tmp_path):
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    example = _load_mla_decode_example()
    softmax_scale = (64 + 64) ** -0.5
    prim_func = example.flashattn.get_tir(
        1,  # batch
        16,  # heads
        1,  # kv_heads
        64,  # kv_ctx
        64,  # dim
        64,  # pe_dim
        64,  # block_N
        16,  # block_H
        2,  # num_split
        softmax_scale,
    )
    toolchain = checks.TileIRToolchain(
        cuda_tile_ir_module=checks.CUDA_TILE_IR_MLIR_MODULE,
        tileiras_path=tmp_path / "tileiras",
        tileiras_version="tileiras 13.3",
    )

    def fake_assemble(tileir_module, *, kernel_name, target, toolchain, launch_metadata=None, argument_names=(), opt_level=3):
        del tileir_module, target, toolchain
        return TileIRLoweringResult(
            kernel_name=kernel_name,
            cubin=b"cubin",
            tileir_source=f"module @{kernel_name}",
            launch_metadata=launch_metadata or TileIRLaunchMetadata(),
            argument_names=argument_names,
        )

    monkeypatch.setattr(tileir_lowering, "assemble_tileir_module", fake_assemble)

    result = lower_primfunc_to_tileir(prim_func, _cuda_target_for_test(), toolchain)

    assert result.kernel_name == "main_split"
    assert [kernel.kernel_name for kernel in result.kernels] == ["main_split_0", "main_split_1"]
    assert result.kernels[0].launch_metadata.grid == (1, 1, 2)


def test_tileir_lowers_mla_kv_fp8_decode_register_control():
    try:
        toolchain = checks.check_tileir_available()
    except checks.TileIRDependencyError as exc:
        pytest.skip(f"CUDA TileIR toolchain unavailable: {exc}")

    example = _load_mla_kv_fp8_example()
    prim_func = example.flashattn.get_tir(
        1,  # batch
        16,  # heads
        1,  # kv_heads
        64,  # kv_ctx
        64,  # dim
        64,  # pe_dim
        64,  # block_N
        16,  # block_H
    )

    program = extract_semantic_program(prim_func)
    assert "register_control" in _semantic_kinds(program.kernels[0].body)

    result = lower_primfunc_to_tileir(prim_func, _cuda_target_for_test(), toolchain)

    assert result.kernel_name == "main_no_split"
    assert result.launch_metadata.grid == (1, 1, 1)
    assert result.cubin


def test_tileir_lowers_mla_persistent_split_grid_sync(monkeypatch, tmp_path):
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    example = _load_mla_persistent_example()
    monkeypatch.setattr(example.driver, "get_num_sms", lambda: 2)
    prim_func = example.flashattn.get_tir(
        1,  # batch
        16,  # heads
        1,  # kv_heads
        64,  # kv_ctx
        64,  # dim
        64,  # pe_dim
        64,  # block_N
        16,  # block_H
        2,  # num_split
    )
    toolchain = checks.TileIRToolchain(
        cuda_tile_ir_module=checks.CUDA_TILE_IR_MLIR_MODULE,
        tileiras_path=tmp_path / "tileiras",
        tileiras_version="tileiras 13.3",
    )

    def fake_assemble(tileir_module, *, kernel_name, target, toolchain, launch_metadata=None, argument_names=(), opt_level=3):
        del tileir_module, target, toolchain
        return TileIRLoweringResult(
            kernel_name=kernel_name,
            cubin=b"cubin",
            tileir_source=f"module @{kernel_name}",
            launch_metadata=launch_metadata or TileIRLaunchMetadata(),
            argument_names=argument_names,
        )

    monkeypatch.setattr(tileir_lowering, "assemble_tileir_module", fake_assemble)

    result = lower_primfunc_to_tileir(prim_func, _cuda_target_for_test(), toolchain)

    assert result.kernel_name == "main_split_persistent"
    assert [kernel.kernel_name for kernel in result.kernels] == ["main_split_persistent_0", "main_split_persistent_1"]
    assert result.kernels[0].launch_metadata.grid == (2, 1, 1)
    assert result.kernels[1].launch_metadata.grid == (2, 1, 1)
    assert result.temporary_buffers == ()


def test_tileir_lowers_mhc_gemm_sqrsum_with_bf16_and_tfloat32_views():
    # Exercises the affine sub-tile fragment read ``x_frag[i, jj*4 + j]``
    # (serial jj, parallel j): lowered as a tile-granular ct.extract of the
    # (token_block, 4) sub-tile at index jj along the hidden axis.
    pytest.importorskip(checks.CUDA_TILE_IR_MLIR_MODULE)

    example = _load_mhc_pre_example()
    target = determine_target("tileir -arch=sm_120", return_object=True)
    prim_func = example.mhc_pre_gemm_sqrsum_tilelang.get_tir(None, None, None, None, 24, 16384)
    _, prepared = TileIRKernelAdapter._prepare_device_module(prim_func, target)
    tileir_lowering._extract_semantic_program_for_lowering(prepared, target)
    module = _build_tileir_module_for_test(prepared, target)

    source = str(module)

    assert "tensor_view<?x?xbf16" in source
    # tfloat32 buffers use float32 STORAGE (tf32 is a compute-only format —
    # cuda_tile's elementwise ops reject tile<…xtf32>); the Gemm emit
    # down-casts f32 operands to tf32 per the f32×f32→tf32 convention.
    assert "tensor_view<?x?xf32" in source
    assert "tensor_view<?x?xtf32" not in source
    assert "load_view_tko" in source
    assert source.count("mma") >= 1


@pytest.mark.parametrize(
    "arch, expected",
    [
        ("sm_90a", "sm_90"),  # Hopper, nvcc arch-specific suffix
        ("sm_100a", "sm_100"),  # Blackwell, arch-specific suffix
        ("sm_100f", "sm_100"),  # Blackwell, family-specific suffix
        ("sm_90", "sm_90"),  # already a base SM token — unchanged
        ("sm_120", "sm_120"),
        ("sm_80", "sm_80"),
    ],
)
def test_tileir_target_arch_strips_feature_suffix_for_tileiras(arch, expected):
    # tileiras --gpu-name accepts only base SM tokens (sm_80..sm_124, no a/f
    # suffix); an auto-detected target on Hopper/Blackwell carries sm_90a/sm_100a
    # and would fail assembly with "Cannot find option named 'sm_90a'". The
    # TileIR assembly boundary must normalize to the base SM name.
    from tilelang.tileir.assembly import target_arch

    target = determine_target({"kind": "tileir", "arch": arch}, return_object=True)
    assert target_arch(target) == expected

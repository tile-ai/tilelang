"""Native tensor contracts for architecture-specific CUDA matrix macros."""

from itertools import product

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx

import tilelang.language as T
from tilelang.cuda.intrinsics.macro.mma_sm70_macro_generator import TensorCoreIntrinEmitter as SM70Emitter
from tilelang.cuda.intrinsics.macro.mma_sm75_macro_generator import TensorCoreIntrinEmitterSM75
from tilelang.cuda.intrinsics.macro.mma_sm120_macro_generator import TensorCoreIntrinEmitterSM120
from tilelang.cuda.intrinsics.macro.mma_sp_macro_generator import SparseTensorCoreIntrinEmitter
from tilelang.cuda.intrinsics.macro.wgmma_macro_generator import TensorCoreIntrinEmitter as WGMMAEmitter
from tilelang.cuda.intrinsics.macro.wgmma_sp_macro_generator import WGSparseTensorCoreIntrinEmitter
from tilelang.cuda.intrinsics.macro.tcgen05_macro_generator import TensorCoreIntrinEmitter as TCGEN05Emitter
from tilelang.layout import make_full_bank_swizzled_layout
from tilelang.tileop._lowering import close_lowered_function


def _tensor(name, shape, dtype="float16", scope="local"):
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope, layout="default"))


def _region(value, begins, extents):
    return ir.TensorRegion(value, [ir.Range.from_min_extent(i, n) for i, n in zip(begins, extents, strict=True)], ir.TensorRegionType())


def _integer(value):
    return int(sym.Analyzer().simplify(value)) if isinstance(value, ir.Expr) else int(value)


def _options(**kwargs):
    values = dict(
        a_dtype=tvm_ffi.dtype("float16"),
        b_dtype=tvm_ffi.dtype("float16"),
        accum_dtype=tvm_ffi.dtype("float32"),
        block_row_warps=1,
        block_col_warps=1,
        warp_row_tiles=16,
        warp_col_tiles=16,
        chunk=16,
        thread_var=ir.Var("thread", "int32"),
    )
    values.update(kwargs)
    return values


def _close(body, inputs, before, tmp_path):
    closed, captures = close_lowered_function(body)
    assert [ir.save_json(value) for value in inputs] == before
    assert tirx.analysis.check_well_formed(closed)
    assert not tirx.analysis.undefined_vars(closed)
    assert all(any(value.same_as(capture) for capture in captures) for value in inputs)
    calls = []
    tvm_ffi.structural_walk(closed.body, (ir.Call, calls.append))
    (tmp_path / "function.json").write_text(ir.save_json(closed))
    (tmp_path / "function.py").write_text(closed.script())
    return closed, calls


@pytest.mark.parametrize("dtype,expected", [("float16", (2, 0)), ("float32", (0, 2))])
@pytest.mark.parametrize("dtype_object", [False, True])
def test_sm70_accumulator_kind_selects_distinct_lane_map(dtype, expected, dtype_object):
    # SM70 lane 2, register 0 has a different coordinate for f16 vs f32 C.
    emitter = SM70Emitter(**_options(accum_dtype=tvm_ffi.dtype(dtype) if dtype_object else dtype))
    mapped = emitter.get_store_index_map().map_indices([2, 0])
    assert tuple(map(_integer, mapped)) == expected


@pytest.mark.parametrize("transpose_a,transpose_b", list(product([False, True], repeat=2)))
def test_sm70_macros_keep_native_pipeline_slice_and_captures(transpose_a, transpose_b, tmp_path):
    emitter = SM70Emitter(**_options(a_transposed=transpose_a, b_transposed=transpose_b))
    a, b = _tensor("A", (2, 64, 64), scope="shared"), _tensor("B", (2, 64, 64), scope="shared")
    ar, br = _region(a, (1, 3, 5), (1, 16, 16)), _region(b, (1, 7, 11), (1, 16, 16))
    la, lb = _tensor("la", (emitter.local_size_a,)), _tensor("lb", (emitter.local_size_b,))
    lc = _tensor("lc", (emitter.local_size_out,), "float32")
    inputs = [a, b, la, lb, lc, emitter.thread_var]
    before = [ir.save_json(value) for value in inputs]

    def build():
        @T.prim_func
        def body():
            emitter.ldmatrix_a(la, ar, 0)
            emitter.ldmatrix_b(lb, br, 0)
            emitter.mma(la, lb, lc)

        return body

    if transpose_a:
        # The original SM70 shared-A load explicitly rejects this orientation.
        with pytest.raises(AssertionError, match="A must be not transposed"):
            build()
        assert [ir.save_json(value) for value in inputs] == before
        return

    body = build()

    closed, calls = _close(body, inputs, before, tmp_path)
    mma = [call for call in calls if isinstance(call.op, ir.Op) and call.op.name == "tl.ptx_mma_sm70"]
    assert len(mma) == 1
    assert mma[0].args[0].value == "m16n16k4"
    assert mma[0].args[1].value == ("col" if transpose_a else "row")
    assert mma[0].args[2].value == ("col" if transpose_b else "row")
    loads = []
    tvm_ffi.structural_walk(closed.body, (ir.TensorLoad, loads.append))
    shared = [load for load in loads if load.source.same_as(a) or load.source.same_as(b)]
    assert len(shared) == 2
    assert all(_integer(load.indices[0]) == 1 for load in shared)


@pytest.mark.parametrize("dtype,prefix", [("float16", "m16n8k8"), ("int8", "m8n8k16"), ("uint8", "m8n8k16"), ("int4", "m8n8k32")])
def test_sm75_native_mma_dtype_shape_and_register_calls(dtype, prefix, tmp_path):
    output_dtype = "float32" if dtype == "float16" else "int32"
    emitter = TensorCoreIntrinEmitterSM75(
        **_options(a_dtype=tvm_ffi.dtype(dtype), b_dtype=tvm_ffi.dtype(dtype), accum_dtype=output_dtype, chunk=32)
    )
    la = _tensor("la", (emitter.warp_rows * emitter.local_size_a,), dtype)
    lb = _tensor("lb", (emitter.warp_cols * emitter.local_size_b,), dtype)
    lc = _tensor("lc", (emitter.warp_rows * emitter.warp_cols * emitter.local_size_out,), output_dtype)
    inputs = [la, lb, lc]
    before = [ir.save_json(value) for value in inputs]

    @T.prim_func
    def body():
        emitter.mma(la, lb, lc)

    _, calls = _close(body, inputs, before, tmp_path)
    mma = [call for call in calls if isinstance(call.op, ir.Op) and call.op.name == "tirx.ptx_legacy.mma"]
    assert mma
    assert emitter.mma_prefix == prefix
    assert all(call.args[0].value == prefix for call in mma)


@pytest.mark.parametrize("register_a", [False, True])
def test_wgmma_native_descriptor_macro_closes_ss_and_rs(register_a, tmp_path):
    emitter = WGMMAEmitter(**_options(block_row_warps=4, warp_col_tiles=64, chunk=64, b_transposed=True))
    a = _tensor("A", (64, 64), scope="local.fragment" if register_a else "shared")
    b, c = _tensor("B", (64, 64), scope="shared"), _tensor("C", (64, 64), "float32", "local.fragment")
    if not register_a:
        emitter._assign_a_shared_layout(make_full_bank_swizzled_layout(a))
    emitter._assign_b_shared_layout(make_full_bank_swizzled_layout(b))
    ar, br, cr = [_region(value, (0, 0), (64, 64)) for value in (a, b, c)]
    inputs = [a, b, c]
    before = [ir.save_json(value) for value in inputs]

    @T.prim_func
    def body():
        emitter.wgmma(ar, br, cr, clear_accum=True)

    _, calls = _close(body, inputs, before, tmp_path)
    names = [call.op.name for call in calls if isinstance(call.op, ir.Op)]
    assert any("wgmma" in name and ("rs" if register_a else "ss") in name for name in names)
    assert any("warpgroup_commit" in name for name in names)
    assert any("warpgroup_wait" in name for name in names)


@pytest.mark.parametrize("tmem_a", [False, True])
def test_tcgen05_native_full_macros_keep_descriptor_and_tmem_contract(tmem_a, tmp_path):
    emitter = TCGEN05Emitter(**_options(warp_row_tiles=128, warp_col_tiles=64, chunk=64, b_transposed=True))
    emitter.meta = (128, 64, 16, 0, 0)
    a = _tensor("A", (128, 64), scope="shared.tmem" if tmem_a else "shared")
    b, c = _tensor("B", (64, 64), scope="shared"), _tensor("C", (128, 64), "float32", "shared.tmem")
    if tmem_a:
        emitter._assign_a_tmem_layout(emitter.make_mma_store_layout(a, operand="A"))
    else:
        emitter._assign_a_shared_layout(make_full_bank_swizzled_layout(a))
    emitter._assign_b_shared_layout(make_full_bank_swizzled_layout(b))
    emitter._assign_c_tmem_layout(emitter.make_mma_store_layout(c, is_ts=tmem_a))
    ar, br, cr = [_region(value, (0, 0), tuple(value.ty.shape)) for value in (a, b, c)]
    inputs = [a, b, c]
    before = [ir.save_json(value) for value in inputs]

    @T.prim_func
    def body():
        emitter.tcgen05mma(ar, br, cr, None, clear_accum=True)

    _, calls = _close(body, inputs, before, tmp_path)
    names = [call.op.name for call in calls if isinstance(call.op, ir.Op)]
    assert any("tcgen05_mma_" + ("ts" if tmem_a else "ss") in name for name in names)


def test_sm120_blockscale_fp4_macros_keep_shared_and_scale_slice(tmp_path):
    emitter = TensorCoreIntrinEmitterSM120(
        **_options(
            a_dtype="float4_e2m1fn",
            b_dtype="float4_e2m1fn",
            accum_dtype="float32",
            chunk=64,
            b_transposed=True,
            is_blockscaled=True,
        )
    )
    a, b = _tensor("A", (2, 64, 128), "float4_e2m1fn", "shared"), _tensor("B", (2, 64, 128), "float4_e2m1fn", "shared")
    ar, br = _region(a, (1, 3, 8), (1, 16, 64)), _region(b, (1, 5, 16), (1, 16, 64))
    la, lb = _tensor("la", (emitter.local_size_a,), "float4_e2m1fn"), _tensor("lb", (emitter.local_size_b,), "float4_e2m1fn")
    lc = _tensor("lc", (emitter.local_size_out,), "float32")
    sa, sb = _tensor("SFA", (2, 32, 8), "uint32", "shared"), _tensor("SFB", (2, 32, 8), "uint32", "shared")
    sar, sbr = _region(sa, (1, 3, 1), (1, 16, 1)), _region(sb, (1, 5, 2), (1, 16, 1))
    inputs = [a, b, la, lb, lc, sa, sb, emitter.thread_var]
    before = [ir.save_json(value) for value in inputs]

    @T.prim_func
    def body():
        emitter.ldmatrix_a(la, ar, 0)
        emitter.ldmatrix_b(lb, br, 0)
        emitter.mma(la, lb, lc, SFA_buf=sar, SFB_buf=sbr)

    closed, calls = _close(body, inputs, before, tmp_path)
    names = [call.op.name for call in calls if isinstance(call.op, ir.Op)]
    assert names.count("tl.ptx_ldmatrix") == 2
    assert any("mma" in name and "block" in name for name in names)
    loads = []
    tvm_ffi.structural_walk(closed.body, (ir.TensorLoad, loads.append))
    shared_loads = [load for load in loads if any(load.source.same_as(value) for value in [a, b, sa, sb])]
    assert shared_loads
    assert all(_integer(load.indices[0]) == 1 for load in shared_loads)


@pytest.mark.parametrize("dtype,metadata_dtype,prefix", [("float16", "uint8", "m16n8k32"), ("int8", "uint16", "m16n8k64")])
def test_sparse_mma_preserves_metadata_and_register_operands(dtype, metadata_dtype, prefix, tmp_path):
    options = _options(a_dtype=tvm_ffi.dtype(dtype), b_dtype=tvm_ffi.dtype(dtype), accum_dtype="float32" if dtype == "float16" else "int32")
    options.pop("chunk")
    emitter = SparseTensorCoreIntrinEmitter(**options, e_dtype=metadata_dtype, warp_k=64)
    la, lb = _tensor("la", (emitter.local_size_a,), dtype), _tensor("lb", (emitter.local_size_b,), dtype)
    le, lc = _tensor("le", (emitter.local_size_e,), metadata_dtype), _tensor("lc", (emitter.local_size_out,), emitter.accum_dtype)
    inputs = [la, le, lb, lc]
    before = [ir.save_json(value) for value in inputs]

    @T.prim_func
    def body():
        emitter.mma_sp(la, le, lb, lc)

    _, calls = _close(body, inputs, before, tmp_path)
    mma = [call for call in calls if isinstance(call.op, ir.Op) and "mma_sp" in call.op.name]
    assert len(mma) == 2
    assert all(call.args[0].value == prefix for call in mma)


@pytest.mark.parametrize("dtype,metadata_dtype", [("float16", "uint8"), ("int8", "uint16")])
def test_sparse_mma_loads_keep_pinned_leading_indices(dtype, metadata_dtype, tmp_path):
    options = _options(a_dtype=dtype, b_dtype=dtype, accum_dtype="float32" if dtype == "float16" else "int32", b_transposed=True)
    options.pop("chunk")
    emitter = SparseTensorCoreIntrinEmitter(**options, e_dtype=metadata_dtype, warp_k=64)
    a, b = _tensor("A", (2, 64, 64), dtype, "shared"), _tensor("B", (2, 64, 128), dtype, "shared")
    e = _tensor("E", (2, 64, 64), metadata_dtype, "shared")
    ar = _region(a, (1, 3, 5), (1, 16, 32))
    br = _region(b, (1, 7, 11), (1, 16, 64))
    er = _region(e, (1, 3, 2), (1, 16, 64 // emitter.e_factor))
    la, lb = _tensor("la", (emitter.local_size_a,), dtype), _tensor("lb", (emitter.local_size_b,), dtype)
    le, lc = _tensor("le", (emitter.local_size_e,), metadata_dtype), _tensor("lc", (emitter.local_size_out,), emitter.accum_dtype)
    inputs = [a, b, e, la, lb, le, lc, emitter.thread_var]
    before = [ir.save_json(value) for value in inputs]

    @T.prim_func
    def body():
        emitter.ldmatrix_a(la, ar, 0)
        emitter.ldmatrix_b(lb, br, 0)
        emitter.ldmatrix_e(le, er, 0)
        emitter.mma_sp(la, le, lb, lc)

    closed, _ = _close(body, inputs, before, tmp_path)
    loads = []
    tvm_ffi.structural_walk(closed.body, (ir.TensorLoad, loads.append))
    shared = [load for load in loads if any(load.source.same_as(value) for value in (a, b, e))]
    assert len(shared) >= 3
    assert all(_integer(load.indices[0]) == 1 for load in shared)


@pytest.mark.parametrize("register_a", [False, True])
def test_sparse_wgmma_native_macro_retains_metadata_input(register_a, tmp_path):
    options = _options(block_row_warps=4, warp_col_tiles=64, b_transposed=True)
    options.pop("chunk")
    emitter = WGSparseTensorCoreIntrinEmitter(**options, e_dtype="uint8", warp_k=64)
    a = _tensor("A", (64, 32), scope="local.fragment" if register_a else "shared")
    b = _tensor("B", (64, 64), scope="shared")
    e = _tensor("E", (64, 64 // emitter.e_factor), "uint8", "shared")
    c = _tensor("C", (64, 64), "float32", "local.fragment")
    if not register_a:
        from tilelang.layout import make_half_bank_swizzled_layout

        emitter._assign_a_shared_layout(make_half_bank_swizzled_layout(a))
    emitter._assign_b_shared_layout(make_full_bank_swizzled_layout(b))
    ar, er, br, cr = [_region(value, (0, 0), tuple(value.ty.shape)) for value in (a, e, b, c)]
    inputs = [a, e, b, c]
    before = [ir.save_json(value) for value in inputs]

    operation = emitter.wgmma_rs if register_a else emitter.wgmma_ss

    @T.prim_func
    def body():
        operation(ar, er, br, cr, clear_accum=True)

    _, calls = _close(body, inputs, before, tmp_path)
    assert any(isinstance(call.op, ir.Op) and "wgmma_sp" in call.op.name for call in calls)


@pytest.mark.parametrize("kind,indices", [("tensor", (0, 0)), ("element", (3, 5)), ("region", (7, 11))])
def test_tcgen05_access_pointer_preserves_native_tensor_coordinates(kind, indices):
    tensor = _tensor("input", (32, 64), scope="shared")
    before = ir.save_json(tensor)
    if kind == "tensor":
        operand = tensor
    elif kind == "element":
        operand = ir.TensorLoad(tensor, list(indices))
    else:
        operand = _region(tensor, indices, (8, 16))
    pointer = TCGEN05Emitter._access_ptr_from(operand, "rw")
    assert pointer.op.name == "tl.access_ptr"
    assert isinstance(pointer.args[0], ir.TensorLoad)
    assert pointer.args[0].source.same_as(tensor)
    assert tuple(map(_integer, pointer.args[0].indices)) == indices
    assert _integer(pointer.args[2]) == 3
    assert _integer(pointer.args[1]) == {"tensor": 32 * 64, "element": 1, "region": 8 * 16}[kind]
    assert ir.save_json(tensor) == before


@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "float4_e2m1fn"])
def test_tcgen05_blockscaled_macro_captures_tmem_scale_pointers_and_barrier(dtype, tmp_path):
    emitter = TCGEN05Emitter(
        **_options(
            a_dtype=dtype,
            b_dtype=dtype,
            warp_row_tiles=128,
            warp_col_tiles=64,
            chunk=256,
            b_transposed=True,
        )
    )
    assert emitter.k_dim == 32
    emitter.meta = (128, 64, 32, 0, 0)
    a, b = _tensor("A", (128, 256), dtype, "shared"), _tensor("B", (64, 256), dtype, "shared")
    c = _tensor("C", (128, 64), "float32", "shared.tmem")
    sfa, sfb = _tensor("SFA", (128, 4), "uint32", "shared.tmem"), _tensor("SFB", (64, 4), "uint32", "shared.tmem")
    barrier = _tensor("barrier", (2,), "uint64", "shared.barrier")
    emitter._assign_a_shared_layout(make_full_bank_swizzled_layout(a))
    emitter._assign_b_shared_layout(make_full_bank_swizzled_layout(b))
    emitter._assign_c_tmem_layout(emitter.make_mma_store_layout(c))
    ar, br, cr = [_region(value, (0, 0), tuple(value.ty.shape)) for value in (a, b, c)]
    inputs = [a, b, c, sfa, sfb, barrier]
    before = [ir.save_json(value) for value in inputs]

    @T.prim_func
    def body():
        emitter.tcgen05mma_blockscaled(ar, br, cr, sfa, sfb, barrier[1], 64, 32, 32, clear_accum=True)

    closed, calls = _close(body, inputs, before, tmp_path)
    bindings = {}

    def record_binding(binding):
        bindings[binding.var] = binding.value

    tvm_ffi.structural_walk(closed.body, (ir.Bind, record_binding))

    def resolve(value):
        while isinstance(value, ir.Var) and value in bindings:
            value = bindings[value]
        return value

    instructions = [call for call in calls if isinstance(call.op, ir.Op) and "tcgen05_mma_blockscaled_ss" in call.op.name]
    assert len(instructions) == 1
    for index, tensor in [(9, sfa), (11, sfb)]:
        pointer = resolve(instructions[0].args[index])
        assert pointer.op.name == "tirx.tensor_data_ptr"
        assert pointer.args[0].same_as(tensor)
    arrivals = [call for call in calls if isinstance(call.op, ir.Op) and "tcgen05_mma_arrive" in call.op.name]
    assert len(arrivals) == 1
    pointer = resolve(arrivals[0].args[0])
    assert isinstance(pointer, ir.Call), "Barrier arrival needs a slot address, not the slot contents"
    assert pointer.op.name == "tl.access_ptr"
    assert pointer.args[0].source.same_as(barrier)
    assert _integer(pointer.args[0].indices[0]) == 1
    assert _integer(pointer.args[2]) == 3

"""Tests for packed x2 intrinsics (add2, sub2, mul2, fma2, max2, min2, abs2).

Each operation is tested for all three supported dtype families:
  - float32   (float32x2)
  - bfloat16  (bfloat16x2)
  - float16   (float16x2)

Three kinds of tests:
  1. Codegen tests  -- verify that the CUDA source contains ``tl::<op>``
     and that bfloat16x2/float16x2 emit proper native-type casts
     (__nv_bfloat162 / __half2) instead of the ambiguous uint1 overload.
  2. Correctness tests -- compile, run, and compare against PyTorch reference.
  3. Auto-vectorization tests -- verify SM100 auto-vectorization behaviour.
"""

import tilelang
from tilelang import tvm as tvm
import tilelang.language as T
import tilelang.testing
import pytest
import torch

SM100_TARGET = {"kind": "cuda", "arch": "sm_100"}
SM80_TARGET = {"kind": "cuda", "arch": "sm_80"}

M = 128  # number of threads / element-pairs

# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP = {"float32": (T.float32, torch.float32), "bfloat16": (T.bfloat16, torch.bfloat16), "float16": (T.float16, torch.float16)}

# ---------------------------------------------------------------------------
# Generic kernel builders using T.Ramp for packed x2 access
# ---------------------------------------------------------------------------


def _make_binary_kernel(op_func, dtype_tl):
    """Build a kernel: C[idx] = op(A[idx], B[idx])."""

    @T.prim_func
    def main(
        A: T.Tensor((M * 2,), dtype=dtype_tl),
        B: T.Tensor((M * 2,), dtype=dtype_tl),
        C: T.Tensor((M * 2,), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            idx = T.Ramp(tid * 2, 1, 2)
            C[idx] = op_func(A[idx], B[idx])

    return main


def _make_ternary_kernel(op_func, dtype_tl):
    """Build a kernel: D[idx] = op(A[idx], B[idx], C[idx])."""

    @T.prim_func
    def main(
        A: T.Tensor((M * 2,), dtype=dtype_tl),
        B: T.Tensor((M * 2,), dtype=dtype_tl),
        C: T.Tensor((M * 2,), dtype=dtype_tl),
        D: T.Tensor((M * 2,), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            idx = T.Ramp(tid * 2, 1, 2)
            D[idx] = op_func(A[idx], B[idx], C[idx])

    return main


def _make_unary_kernel(op_func, dtype_tl):
    """Build a kernel: C[idx] = op(A[idx])."""

    @T.prim_func
    def main(
        A: T.Tensor((M * 2,), dtype=dtype_tl),
        C: T.Tensor((M * 2,), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            idx = T.Ramp(tid * 2, 1, 2)
            C[idx] = op_func(A[idx])

    return main


# ---------------------------------------------------------------------------
# Helper: lower to CUDA source
# ---------------------------------------------------------------------------


def _lower_to_cuda_source(func, target=SM80_TARGET, pass_configs=None) -> str:
    with tvm.transform.PassContext(config=pass_configs), tvm.target.Target(target):
        artifact = tilelang.lower(func, target=target)
    assert artifact.kernel_source is not None
    return artifact.kernel_source


# ---------------------------------------------------------------------------
# Auto-vectorization kernels via T.Parallel
# ---------------------------------------------------------------------------

# Map from Python operator string to (lambda, tl_func_name)
_AUTO_VEC_OPS = {"add": (lambda a, b: a + b, "add2"), "sub": (lambda a, b: a - b, "sub2"), "mul": (lambda a, b: a * b, "mul2")}


def _make_auto_vec_binary_kernel(py_op, dtype_tl, width: int = 4):
    """Build a kernel that uses T.Parallel to let the vectoriser emit tl::<op>2."""

    @T.prim_func
    def main(
        A: T.Tensor((M, width), dtype=dtype_tl),
        B: T.Tensor((M, width), dtype=dtype_tl),
        C: T.Tensor((M, width), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            for i, v in T.Parallel(M, width):
                C[i, v] = py_op(A[i, v], B[i, v])

    return main


def _make_auto_vec_fma_kernel(dtype_tl, width: int = 4):
    """Build a kernel that lets CUDA codegen fuse mul + add into tl::fma2."""

    @T.prim_func
    def main(
        A: T.Tensor((M, width), dtype=dtype_tl),
        B: T.Tensor((M, width), dtype=dtype_tl),
        C: T.Tensor((M, width), dtype=dtype_tl),
        D: T.Tensor((M, width), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            for i, v in T.Parallel(M, width):
                D[i, v] = A[i, v] * B[i, v] + C[i, v]

    return main


def _make_auto_vec_reduce_kernel(reduce_func, *, nan_propagate=False):
    """Build a row reduction whose local fragment is contiguous per thread."""

    @T.prim_func
    def main(
        A: T.Tensor((M, 128), dtype=T.float32),
        C: T.Tensor((M,), dtype=T.float32),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            src = T.alloc_fragment((M, 128), T.float32)
            dst = T.alloc_fragment((M,), T.float32)
            T.copy(A, src)
            if nan_propagate:
                reduce_func(src, dst, dim=1, nan_propagate=True)
            else:
                reduce_func(src, dst, dim=1)
            T.copy(dst, C)

    return main


def _make_auto_vec_scalar_reduction_kernel(width: int = 8, *, unit_loop: bool = False):
    """Build an MQA-epilogue-like scalar accumulator reduction."""

    @T.prim_func
    def main(
        Scores: T.Tensor((M, width), dtype=T.float32),
        Weights: T.Tensor((M, width), dtype=T.float32),
        Scale: T.Tensor((M,), dtype=T.float32),
        Out: T.Tensor((M,), dtype=T.float32),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            acc = T.alloc_local((1,), T.float32)
            acc[0] = T.float32(0)
            for h in T.vectorized(width):
                if unit_loop:
                    for _ in T.serial(1):
                        acc[0] += T.max(Scores[tid, h] * Scale[tid], T.float32(0)) * Weights[tid, h]
                else:
                    acc[0] += T.max(Scores[tid, h] * Scale[tid], T.float32(0)) * Weights[tid, h]
            Out[tid] = acc[0]

    return main


def _make_auto_vec_batched_reduce_kernel(reduce_func, *, rows=M, width=64, threads=256):
    """Build a reduction that shuffles packed values between threads."""

    @T.prim_func
    def main(
        A: T.Tensor((rows, width), dtype=T.float32),
        C: T.Tensor((rows,), dtype=T.float32),
    ):
        with T.Kernel(1, threads=threads):
            src = T.alloc_shared((rows, width), T.float32)
            dst = T.alloc_fragment((rows,), T.float32)
            T.copy(A, src, disable_tma=True)
            reduce_func(src, dst, dim=1, batch=2)
            T.copy(dst, C)

    return main


def _make_auto_vec_batched_reduce_workspace_kernel():
    """Build a packed reduction whose cross-warp step uses shared workspace."""
    rows = 4
    width = 512
    threads = 128
    vec_size = 8

    def fragment_layout(i, j):
        linear = i * width + j
        thread_id = linear // vec_size % threads
        local_id = linear // (threads * vec_size) * vec_size + linear % vec_size
        return thread_id, local_id

    @T.prim_func
    def main(
        A: T.Tensor((rows, width), dtype=T.float32),
        C: T.Tensor((rows,), dtype=T.float32),
    ):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((rows, width), T.float32)
            dst = T.alloc_fragment((rows,), T.float32)
            T.annotate_layout({src: T.Fragment(src.shape, forward_fn=fragment_layout)})
            T.copy(A, src)
            T.reduce_sum(src, dst, dim=1, batch=2)
            T.copy(dst, C)

    return main


def _make_auto_vec_scalar_reduction_chunks_kernel(chunks: int = 4, width: int = 8):
    """Build a chunked MQA-epilogue-like scalar accumulator reduction."""

    @T.prim_func
    def main(
        Scores: T.Tensor((M, chunks, width), dtype=T.float32),
        Weights: T.Tensor((M, chunks, width), dtype=T.float32),
        Scale: T.Tensor((M,), dtype=T.float32),
        Out: T.Tensor((M,), dtype=T.float32),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            acc = T.alloc_local((1,), T.float32)
            acc[0] = T.float32(0)
            for c in T.unroll(chunks):
                for h in T.vectorized(width):
                    acc[0] += T.max(Scores[tid, c, h] * Scale[tid], T.float32(0)) * Weights[tid, c, h]
            Out[tid] = acc[0]

    return main


def _make_cross_loop_vector_reduction_ir(*, accumulating: bool, observe=False, self_dependent_rhs=False):
    f32 = tvm.DataType("float32")
    f32x4 = f32.with_lanes(4)
    zero = tvm.tirx.FloatImm("float32", 0)
    acc = tvm.tirx.decl_buffer((1,), f32, name="acc", scope="local")
    vec = tvm.tirx.decl_buffer((1,), f32x4, name="vec", scope="local")
    observed = tvm.tirx.decl_buffer((4,), f32, name="observed", scope="local")
    c = tvm.tirx.Var("c", "int32")
    vec_load = tvm.tirx.BufferLoad(vec, (0,))

    horizontal_sum = tvm.tirx.Shuffle((vec_load,), (0,))
    for lane in range(1, 4):
        horizontal_sum += tvm.tirx.Shuffle((vec_load,), (lane,))

    update = tvm.tirx.Broadcast(tvm.tirx.Cast("float32", c), 4)
    if self_dependent_rhs:
        update = vec_load + update
    if accumulating:
        update = vec_load + update
    loop_body = [
        tvm.tirx.AllocBuffer(vec),
        tvm.tirx.BufferStore(vec, tvm.tirx.Broadcast(zero, 4), (0,)),
        tvm.tirx.BufferStore(vec, update, (0,)),
    ]
    if observe:
        loop_body.append(tvm.tirx.BufferStore(observed, tvm.tirx.Shuffle((vec_load,), (0,)), (c,)))
    loop_body.append(tvm.tirx.BufferStore(acc, tvm.tirx.BufferLoad(acc, (0,)) + horizontal_sum, (0,)))
    loop = tvm.tirx.For(
        c,
        0,
        4,
        tvm.tirx.ForKind.SERIAL,
        tvm.tirx.SeqStmt(loop_body),
    )
    target = tvm.target.Target(SM100_TARGET)
    outer_body = [
        tvm.tirx.AllocBuffer(acc),
        tvm.tirx.BufferStore(acc, zero, (0,)),
    ]
    if observe:
        outer_body.append(tvm.tirx.AllocBuffer(observed))
    outer_body.append(loop)
    func = tvm.tirx.PrimFunc(
        (),
        tvm.tirx.SeqStmt(outer_body),
    ).with_attr("target", target)
    return tvm.IRModule({"main": func}), target


# ===================================================================
# Parametrised op / dtype lists
# ===================================================================

# Binary ops: (name, func)
_BINARY_OPS = [
    ("add2", T.add2),
    ("sub2", T.sub2),
    ("mul2", T.mul2),
    ("max2", T.max2),
    ("min2", T.min2),
]

# All 3 dtype families
_DTYPES = ["float32", "bfloat16", "float16"]

# Native cast types expected in codegen for 16-bit packed types
_NATIVE_CAST_TYPE = {"bfloat16": "__nv_bfloat162", "float16": "__half2"}

# Torch reference functions
_TORCH_REFS = {
    "add2": lambda a, b: a + b,
    "sub2": lambda a, b: a - b,
    "mul2": lambda a, b: a * b,
    "max2": lambda a, b: torch.maximum(a, b),
    "min2": lambda a, b: torch.minimum(a, b),
    "fma2": lambda a, b, c: a * b + c,
    "abs2": lambda a: torch.abs(a),
}

_REDUCE_OPS = [
    ("sum", T.reduce_sum, ("add2",)),
    ("abssum", T.reduce_abssum, ("add2", "abs2")),
    ("max", T.reduce_max, ("max2",)),
    ("min", T.reduce_min, ("min2",)),
    ("absmax", T.reduce_absmax, ("max2", "abs2")),
]


def _torch_reduce(a, op_name):
    if op_name == "sum":
        return torch.sum(a, dim=1)
    if op_name == "abssum":
        return torch.sum(torch.abs(a), dim=1)
    if op_name == "max":
        return torch.max(a, dim=1).values
    if op_name == "min":
        return torch.min(a, dim=1).values
    if op_name == "absmax":
        return torch.max(torch.abs(a), dim=1).values
    raise ValueError(f"Unsupported reduction: {op_name}")


# ===================================================================
# Codegen tests
# ===================================================================


@pytest.mark.parametrize("op_name,op_func", _BINARY_OPS, ids=[n for n, _ in _BINARY_OPS])
def test_binary_rejects_mixed_packed_dtypes(op_name, op_func):
    x = tvm.tirx.Var("x", "float16x2")
    y = tvm.tirx.Var("y", "bfloat16x2")

    with pytest.raises(ValueError, match="same dtype"):
        op_func(x, y)


@pytest.mark.parametrize("mixed_index", [0, 1, 2])
def test_fma2_rejects_mixed_packed_dtypes(mixed_index):
    x = tvm.tirx.Var("x", "float16x2")
    y = tvm.tirx.Var("y", "bfloat16x2")
    args = [x, x, x]
    args[mixed_index] = y

    with pytest.raises(ValueError, match="same dtype"):
        T.fma2(*args)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype_name", _DTYPES)
@pytest.mark.parametrize("op_name,op_func", _BINARY_OPS, ids=[n for n, _ in _BINARY_OPS])
def test_codegen_binary(op_name, op_func, dtype_name):
    """Binary ops emit tl::<op> with correct native-type casts."""
    dtype_tl, _ = _DTYPE_MAP[dtype_name]
    func = _make_binary_kernel(op_func, dtype_tl)
    src = _lower_to_cuda_source(func, target=SM80_TARGET)
    assert f"tl::{op_name}" in src, f"Expected tl::{op_name} in generated CUDA source"
    # For 16-bit types, verify that the codegen emits casts to the correct
    # native type instead of the ambiguous uint1 overload.
    if dtype_name in _NATIVE_CAST_TYPE:
        assert _NATIVE_CAST_TYPE[dtype_name] in src, f"Expected {_NATIVE_CAST_TYPE[dtype_name]} cast in CUDA source for {dtype_name}"


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype_name", _DTYPES)
def test_codegen_fma2(dtype_name):
    """fma2 emits tl::fma2 with correct native-type casts."""
    dtype_tl, _ = _DTYPE_MAP[dtype_name]
    func = _make_ternary_kernel(T.fma2, dtype_tl)
    src = _lower_to_cuda_source(func, target=SM80_TARGET)
    assert "tl::fma2" in src
    if dtype_name in _NATIVE_CAST_TYPE:
        assert _NATIVE_CAST_TYPE[dtype_name] in src, f"Expected {_NATIVE_CAST_TYPE[dtype_name]} cast in CUDA source for {dtype_name}"


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype_name", _DTYPES)
def test_codegen_abs2(dtype_name):
    """abs2 emits tl::abs2 with correct native-type casts."""
    dtype_tl, _ = _DTYPE_MAP[dtype_name]
    func = _make_unary_kernel(T.abs2, dtype_tl)
    src = _lower_to_cuda_source(func, target=SM80_TARGET)
    assert "tl::abs2" in src
    if dtype_name in _NATIVE_CAST_TYPE:
        assert _NATIVE_CAST_TYPE[dtype_name] in src, f"Expected {_NATIVE_CAST_TYPE[dtype_name]} cast in CUDA source for {dtype_name}"


# ---------------------------------------------------------------------------
# Auto-vectorization codegen tests (T.Parallel -> tl::*2)
# ---------------------------------------------------------------------------

_AUTO_VEC_OP_NAMES = list(_AUTO_VEC_OPS.keys())  # ["add", "sub", "mul"]


# float32: auto-vectorization should emit tl::<op>2 on SM100+
@tilelang.testing.requires_cuda_compute_version(10)
@pytest.mark.parametrize("op_key", _AUTO_VEC_OP_NAMES)
def test_codegen_auto_vec_f32(op_key):
    py_op, tl_func = _AUTO_VEC_OPS[op_key]
    func = _make_auto_vec_binary_kernel(py_op, T.float32)
    src = _lower_to_cuda_source(func, target=SM100_TARGET)
    assert f"tl::{tl_func}" in src, f"Expected tl::{tl_func} in SM100 auto-vectorised CUDA source for float32 {op_key}"


@tilelang.testing.requires_cuda_compute_version(10)
@pytest.mark.parametrize("op_key", _AUTO_VEC_OP_NAMES)
def test_codegen_auto_vec_f32_width8(op_key):
    py_op, tl_func = _AUTO_VEC_OPS[op_key]
    func = _make_auto_vec_binary_kernel(py_op, T.float32, width=8)
    src = _lower_to_cuda_source(func, target=SM100_TARGET)
    assert "\x00" not in src, "Generated CUDA source should not contain embedded NUL bytes"
    for field in "xyzw":
        assert f".{field})) = tl::{tl_func}(" in src, (
            f"Expected {field}-field packed tl::{tl_func} emission in width-8 float32 auto-vectorised source"
        )


# float32: auto-vectorization should NOT emit tl::<op>2 before SM100
@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op_key", _AUTO_VEC_OP_NAMES)
def test_codegen_auto_vec_f32_no_sm80(op_key):
    py_op, tl_func = _AUTO_VEC_OPS[op_key]
    func = _make_auto_vec_binary_kernel(py_op, T.float32)
    src = _lower_to_cuda_source(func, target=SM80_TARGET)
    assert f"tl::{tl_func}" not in src, f"tl::{tl_func} should NOT appear in SM80 auto-vectorised CUDA source for float32 {op_key}"


@tilelang.testing.requires_cuda_compute_version(10)
def test_codegen_auto_vec_fma_f32():
    func = _make_auto_vec_fma_kernel(T.float32)
    src = _lower_to_cuda_source(func, target=SM100_TARGET)
    assert "tl::fma2" in src, "Expected tl::fma2 in SM100 auto-vectorised CUDA source for float32 mul+add"


@tilelang.testing.requires_cuda_compute_version(10)
@pytest.mark.parametrize("op_name,reduce_func,packed_ops", _REDUCE_OPS, ids=[op[0] for op in _REDUCE_OPS])
def test_codegen_auto_vec_reduce_f32_sm100(op_name, reduce_func, packed_ops):
    func = _make_auto_vec_reduce_kernel(reduce_func)
    src = _lower_to_cuda_source(func, target=SM100_TARGET)
    for packed_op in packed_ops:
        assert f"tl::{packed_op}" in src, f"Expected tl::{packed_op} in SM100 float32 {op_name} reduction"


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op_name,reduce_func,packed_ops", _REDUCE_OPS, ids=[op[0] for op in _REDUCE_OPS])
def test_codegen_auto_vec_reduce_f32_no_sm80(op_name, reduce_func, packed_ops):
    func = _make_auto_vec_reduce_kernel(reduce_func)
    src = _lower_to_cuda_source(func, target=SM80_TARGET)
    for packed_op in packed_ops:
        assert f"tl::{packed_op}" not in src, f"tl::{packed_op} should not appear in pre-SM100 float32 {op_name} reduction"


@tilelang.testing.requires_cuda_compute_version(10)
@pytest.mark.parametrize(
    "reduce_func,packed_op",
    [(T.reduce_max, "max2"), (T.reduce_min, "min2"), (T.reduce_absmax, "max2")],
)
def test_codegen_auto_vec_reduce_f32_ignores_half_nan_mode(reduce_func, packed_op):
    func = _make_auto_vec_reduce_kernel(reduce_func, nan_propagate=True)
    src = _lower_to_cuda_source(func, target=SM100_TARGET)
    assert f"tl::{packed_op}" in src
    assert f"tl::{packed_op}_nan" not in src


@tilelang.testing.requires_cuda_compute_version(10)
@pytest.mark.parametrize("unit_loop", [False, True], ids=["direct", "unit-loop"])
def test_codegen_auto_vec_scalar_reduction_f32_sm100(unit_loop):
    func = _make_auto_vec_scalar_reduction_kernel(unit_loop=unit_loop)
    src = _lower_to_cuda_source(func, target=SM100_TARGET)
    assert all(f"tl::{op}" in src for op in ("max2", "mul2", "fma2"))
    assert src.count("tl::fma2") == 4
    assert "for (int h = 0; h < 8; ++h)" not in src


@tilelang.testing.requires_cuda_compute_version(10)
def test_codegen_auto_vec_scalar_reduction_chunk_accumulator_f32_sm100():
    func = _make_auto_vec_scalar_reduction_chunks_kernel()
    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    src = _lower_to_cuda_source(func, target=SM100_TARGET, pass_configs=pass_configs)
    scalar_updates = [line for line in src.splitlines() if "acc[0] = (acc[0] +" in line]
    assert src.count("tl::fma2") == 4
    assert src.count("tl::add2") == 1
    assert len(scalar_updates) == 1


@tilelang.testing.requires_cuda_compute_version(10)
def test_codegen_auto_vec_scalar_reduction_preserves_chunk_order_without_fast_math():
    func = _make_auto_vec_scalar_reduction_chunks_kernel()
    src = _lower_to_cuda_source(func, target=SM100_TARGET)
    scalar_updates = [line for line in src.splitlines() if "acc[0] = (acc[0] +" in line]
    assert src.count("tl::fma2") == 4
    assert "acc_chunk_acc_vec" not in src
    assert "tl::add2" not in src
    # The single source update remains inside the runtime chunk loop.
    assert len(scalar_updates) == 1


def test_cross_loop_vector_reduction_rejects_overwrite_buffer_reuse():
    mod, target = _make_cross_loop_vector_reduction_ir(accumulating=False)
    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    with target, tvm.transform.PassContext(config=pass_configs):
        lowered = tilelang.transform.VectorizeLoop()(mod)

    allocations = []
    tvm.tirx.stmt_functor.post_order_visit(
        lowered["main"].body,
        lambda node: allocations.append(node.buffer.name) if isinstance(node, tvm.tirx.AllocBuffer) else None,
    )
    assert "acc_chunk_acc_vec" in allocations


@pytest.mark.parametrize(
    "kwargs",
    (
        {"accumulating": True, "observe": True},
        {"accumulating": True, "self_dependent_rhs": True},
    ),
)
def test_cross_loop_vector_reduction_rejects_nonprivate_accumulator(kwargs):
    mod, target = _make_cross_loop_vector_reduction_ir(**kwargs)
    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    with target, tvm.transform.PassContext(config=pass_configs):
        lowered = tilelang.transform.VectorizeLoop()(mod)

    allocations = []
    tvm.tirx.stmt_functor.post_order_visit(
        lowered["main"].body,
        lambda node: allocations.append(node.buffer.name) if isinstance(node, tvm.tirx.AllocBuffer) else None,
    )
    assert "acc_chunk_acc_vec" in allocations


def test_cross_loop_vector_reduction_reuses_additive_accumulator():
    mod, target = _make_cross_loop_vector_reduction_ir(accumulating=True)
    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    with target, tvm.transform.PassContext(config=pass_configs):
        lowered = tilelang.transform.VectorizeLoop()(mod)

    loops = []
    allocations = []

    def collect(node):
        if isinstance(node, tvm.tirx.For):
            loops.append(node)
        elif isinstance(node, tvm.tirx.AllocBuffer):
            allocations.append(node.buffer.name)

    tvm.tirx.stmt_functor.post_order_visit(lowered["main"].body, collect)
    assert "acc_chunk_acc_vec" not in allocations
    assert len(loops) == 1
    assert isinstance(loops[0].body, tvm.tirx.BufferStore)


@tilelang.testing.requires_cuda_compute_version(10)
def test_correctness_auto_vec_scalar_reduction_chunk_accumulator_f32_sm100():
    func = _make_auto_vec_scalar_reduction_chunks_kernel()
    pass_configs = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    kernel = tilelang.compile(func, out_idx=[3], target=SM100_TARGET, pass_configs=pass_configs)
    scores = torch.randn(M, 4, 8, device="cuda", dtype=torch.float32)
    weights = torch.randn(M, 4, 8, device="cuda", dtype=torch.float32)
    scale = torch.randn(M, device="cuda", dtype=torch.float32)
    out = kernel(scores, weights, scale)
    ref = (torch.maximum(scores * scale[:, None, None], torch.zeros_like(scores)) * weights).sum(dim=(1, 2))
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
def test_codegen_auto_vec_fma_half_types(dtype_name):
    dtype_tl, _ = _DTYPE_MAP[dtype_name]
    func = _make_auto_vec_fma_kernel(dtype_tl, width=8)
    src = _lower_to_cuda_source(func, target=SM80_TARGET)
    assert "tl::fma2" in src, f"Expected tl::fma2 in CUDA source for {dtype_name} mul+add"
    assert _NATIVE_CAST_TYPE[dtype_name] in src, f"Expected {_NATIVE_CAST_TYPE[dtype_name]} cast in CUDA source for {dtype_name}"


# bfloat16 / float16: auto-vectorization should emit tl::<op>2 on any target
# (the C++ helpers have compile-time arch fallbacks).
@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype_name", ["bfloat16", "float16"])
@pytest.mark.parametrize("op_key", _AUTO_VEC_OP_NAMES)
def test_codegen_auto_vec_half_types(op_key, dtype_name):
    py_op, tl_func = _AUTO_VEC_OPS[op_key]
    dtype_tl, _ = _DTYPE_MAP[dtype_name]
    func = _make_auto_vec_binary_kernel(py_op, dtype_tl)
    src = _lower_to_cuda_source(func, target=SM80_TARGET)
    assert f"tl::{tl_func}" in src, f"Expected tl::{tl_func} in auto-vectorised CUDA source for {dtype_name} {op_key}"
    # Verify correct native-type cast
    assert _NATIVE_CAST_TYPE[dtype_name] in src, (
        f"Expected {_NATIVE_CAST_TYPE[dtype_name]} cast in auto-vectorised CUDA source for {dtype_name} {op_key}"
    )


# ===================================================================
# Numerical correctness tests
# ===================================================================


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype_name", _DTYPES)
@pytest.mark.parametrize("op_name,op_func", _BINARY_OPS, ids=[n for n, _ in _BINARY_OPS])
def test_correctness_binary(op_name, op_func, dtype_name):
    """Binary ops produce correct results for all dtypes."""
    dtype_tl, dtype_torch = _DTYPE_MAP[dtype_name]
    func = _make_binary_kernel(op_func, dtype_tl)
    kernel = tilelang.compile(func, out_idx=[2], target="cuda")
    a = torch.randn(M * 2, device="cuda", dtype=dtype_torch)
    b = torch.randn(M * 2, device="cuda", dtype=dtype_torch)
    c = kernel(a, b)
    ref = _TORCH_REFS[op_name](a, b)
    torch.testing.assert_close(c, ref)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype_name", _DTYPES)
def test_correctness_fma2(dtype_name):
    """fma2 produces correct results for all dtypes."""
    dtype_tl, dtype_torch = _DTYPE_MAP[dtype_name]
    func = _make_ternary_kernel(T.fma2, dtype_tl)
    kernel = tilelang.compile(func, out_idx=[3], target="cuda")
    a = torch.randn(M * 2, device="cuda", dtype=dtype_torch)
    b = torch.randn(M * 2, device="cuda", dtype=dtype_torch)
    c = torch.randn(M * 2, device="cuda", dtype=dtype_torch)
    d = kernel(a, b, c)
    ref = _TORCH_REFS["fma2"](a, b, c)
    # Hardware FMA fuses multiply-add into a single rounding step, so it can
    # differ from the separate mul+add reference by up to 1 ULP.  Use relaxed
    # tolerances for 16-bit types.
    if dtype_name == "float32":
        torch.testing.assert_close(d, ref)
    else:
        torch.testing.assert_close(d, ref, atol=1e-2, rtol=1e-1)


@tilelang.testing.requires_cuda_compute_version(10)
@pytest.mark.parametrize("op_name,reduce_func", [(op_name, reduce_func) for op_name, reduce_func, _ in _REDUCE_OPS])
def test_correctness_auto_vec_reduce_f32(op_name, reduce_func):
    func = _make_auto_vec_reduce_kernel(reduce_func)
    kernel = tilelang.compile(func, out_idx=[1], target="cuda")
    a = torch.randn((M, 128), device="cuda", dtype=torch.float32)
    result = kernel(a)
    reference = _torch_reduce(a, op_name)
    torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda_compute_version(10)
def test_correctness_auto_vec_batched_reduce_f32():
    func = _make_auto_vec_batched_reduce_kernel(T.reduce_sum)
    kernel = tilelang.compile(func, out_idx=[1], target="cuda")
    a = torch.randn((M, 64), device="cuda", dtype=torch.float32)
    result = kernel(a)
    torch.testing.assert_close(result, torch.sum(a, dim=1), atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
def test_correctness_auto_vec_batched_reduce_f32_workspace():
    func = _make_auto_vec_batched_reduce_workspace_kernel()
    kernel = tilelang.compile(func, out_idx=[1], target="cuda")
    source = kernel.get_kernel_source()
    allreduce_call = next(line for line in source.splitlines() if "AllReduce<" in line)
    assert "make_int2" not in allreduce_call
    assert "(&(workspace[0]))" in allreduce_call

    a = torch.randn((4, 512), device="cuda", dtype=torch.float32)
    result = kernel(a)
    torch.testing.assert_close(result, torch.sum(a, dim=1), atol=1e-4, rtol=1e-4)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype_name", _DTYPES)
def test_correctness_abs2(dtype_name):
    """abs2 produces correct results for all dtypes."""
    dtype_tl, dtype_torch = _DTYPE_MAP[dtype_name]
    func = _make_unary_kernel(T.abs2, dtype_tl)
    kernel = tilelang.compile(func, out_idx=[1], target="cuda")
    a = torch.randn(M * 2, device="cuda", dtype=dtype_torch)
    c = kernel(a)
    ref = _TORCH_REFS["abs2"](a)
    torch.testing.assert_close(c, ref)


if __name__ == "__main__":
    tilelang.testing.main()

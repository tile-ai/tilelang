import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.layout import make_swizzled_layout
from tilelang.utils.language import to_tile_region


def _kernel(
    op, dtype="float32", shape=(8, 128), dim=-1, src_scope="local.fragment", dst_scope="local.fragment", clear=True, keepdim=False, batch=1
):
    axis = dim if dim is None or dim >= 0 else dim + len(shape)
    out_shape = (1,) if axis is None or len(shape) == 1 else shape[:axis] + ((1,) if keepdim else ()) + shape[axis + 1 :]
    reduce_op = T.reduce_any if op == "any" else T.reduce_all

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Seed: T.Tensor(out_shape, "bool"), Out: T.Tensor(out_shape, "bool")):
        with T.Kernel(1, threads=128):
            src = T.alloc_buffer(shape, dtype, scope=src_scope)
            dst = T.alloc_buffer(out_shape, "bool", scope=dst_scope)
            T.copy(A, src)
            if not clear:
                T.copy(Seed, dst)
            reduce_op(src, dst, dim=dim, clear=clear, batch=batch)
            T.copy(dst, Out)

    return main, out_shape


def _run(op, data, **kwargs):
    program, out_shape = _kernel(op, str(data.dtype).removeprefix("torch."), tuple(data.shape), **kwargs)
    seed = (torch.arange(torch.Size(out_shape).numel(), device="cuda") % 2 == 0).reshape(out_shape)
    result = tilelang.compile(program, out_idx=-1, target="cuda")(data.cuda(), seed)
    predicate = data.cuda() != 0
    dim = kwargs.get("dim", -1)
    reducer = torch.any if op == "any" else torch.all
    expected = reducer(predicate).reshape(1) if dim is None else reducer(predicate, dim=dim, keepdim=kwargs.get("keepdim", False))
    expected = expected.reshape(out_shape)
    if not kwargs.get("clear", True):
        expected = (seed | expected) if op == "any" else (seed & expected)
    torch.testing.assert_close(result, expected)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op", ["any", "all"])
@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.int8, torch.uint8, torch.int32, torch.int64, torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_logical_reduce_truth_conversion(op, dtype):
    data = torch.full((8, 128), 2, dtype=dtype)
    data[0] = 0
    data[2, 63] = 0
    data[3] = 0
    data[3, 127] = 1
    if dtype.is_floating_point:
        data[4] = float("nan")
        data[5] = 0
        data[5, 31] = float("nan")
        data[6] = -0.0
        data[7] = 0.25
    elif dtype == torch.int64:
        data[4] = 1 << 32
        data[5] = -(1 << 40)
    elif dtype not in (torch.bool, torch.uint8):
        data[4] = -2
    _run(op, data)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "op,src_scope,dst_scope,dim,keepdim",
    [
        ("any", "shared", "shared", 0, False),
        ("all", "shared", "local.fragment", 1, True),
        ("any", "local.fragment", "shared", None, False),
        ("all", "local.fragment", "local.fragment", None, False),
        ("all", "shared.dyn", "shared.dyn", 0, True),
    ],
)
def test_logical_reduce_scopes_shapes_and_accumulation(op, src_scope, dst_scope, dim, keepdim):
    data = torch.ones((16, 64), dtype=torch.float32)
    data[::2] = 0
    _run(op, data, src_scope=src_scope, dst_scope=dst_scope, dim=dim, keepdim=keepdim, clear=False)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op", ["any", "all"])
def test_logical_reduce_whole_region(op):
    data = torch.ones((2, 4, 64), dtype=torch.int32)
    data[1, 2, 41] = 0
    _run(op, data, dim=None)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op", ["any", "all"])
def test_logical_reduce_one_dimensional_and_singleton_axis(op):
    data = torch.tensor([0, 2, 0, 2], dtype=torch.int32)
    _run(op, data)
    _run(op, data.reshape(4, 1))


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op", ["any", "all"])
def test_logical_reduce_batch(op):
    data = torch.ones((128, 64), dtype=torch.int32)
    data[::2] = 0
    _run(op, data, batch=4, clear=False)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op", ["any", "all"])
@pytest.mark.parametrize(
    "threads,width,scale,batch",
    [(32, 32, 1, 1), (128, 128, 1, 1), (32, 32, 1, 4), (128, 128, 1, 4), (32, 16, 1, 1), (32, 16, 2, 1)],
    ids=["warp", "cross-warp", "warp-batch", "cross-warp-batch", "subwarp", "interleaved"],
)
def test_logical_reduce_warp_vote(op, threads, width, scale, batch):
    groups = threads // width
    rows, cols = groups * 4, width * 4
    reduce_op = T.reduce_any if op == "any" else T.reduce_all

    @T.prim_func
    def main(A: T.Tensor((rows, cols), "int32"), Out: T.Tensor((rows,), "bool")):
        with T.Kernel(1, threads=threads):
            src = T.alloc_fragment((rows, cols), "int32")
            dst = T.alloc_fragment((rows,), "bool")
            T.annotate_layout(
                {
                    src: T.Fragment(
                        src.shape,
                        forward_fn=lambda i, j: (
                            (i % groups // scale) * width * scale + (j % width) * scale + i % scale,
                            i // groups * 4 + j // width,
                        ),
                    )
                }
            )
            T.copy(A, src)
            reduce_op(src, dst, batch=batch)
            T.copy(dst, Out)

    data = torch.full((rows, cols), 0 if op == "any" else 2, dtype=torch.int32, device="cuda")
    # Opposite results in adjacent groups expose accidental whole-warp votes
    # for subwarps/interleaved layouts. The last element also tests cross-warp
    # communication before the vote leaf and separate outputs in run_batch.
    data[1::2, -1] = 2 if op == "any" else 0
    kernel = tilelang.compile(main, out_idx=-1, target="cuda")
    result = kernel(data)
    expected = torch.any(data != 0, dim=1) if op == "any" else torch.all(data != 0, dim=1)
    torch.testing.assert_close(result, expected)

    ptx = kernel._get_ptx()
    if width >= 32 and scale == 1:
        assert f"vote.sync.{op}.pred" in ptx
        assert "shfl.sync.bfly" not in ptx
        if width > 32:
            assert "bar.sync" in ptx
    else:
        assert "vote.sync." not in ptx
        assert "shfl.sync.bfly" in ptx


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op,row", [("any", 7), ("all", 15)])
def test_logical_reduce_swizzled_subregions(op, row):
    reduce_op = T.reduce_any if op == "any" else T.reduce_all

    @T.prim_func
    def main(A: T.Tensor((16, 64), "float16"), Out: T.Tensor((3,), "bool")):
        with T.Kernel(1, threads=128):
            src = T.alloc_shared((16, 64), "float16")
            dst = T.alloc_shared((3,), "bool")
            T.annotate_layout({src: make_swizzled_layout(src)})
            T.copy(A, src)
            T.fill(dst, op == "any")
            reduce_op(src[row : row + 1, :], dst[1:2], dim=None)
            T.copy(dst, Out)

    data = torch.ones((16, 64), dtype=torch.float16, device="cuda")
    if op == "any":
        data.zero_()
        data[8] = 1
    result = tilelang.compile(main, out_idx=-1, target="cuda")(data)
    expected = torch.tensor([op == "any", op == "all", op == "any"], device="cuda")
    torch.testing.assert_close(result, expected)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op", ["any", "all"])
def test_logical_reduce_shared_subregion_nonoverlap(op):
    reduce_op = T.reduce_any if op == "any" else T.reduce_all

    @T.prim_func
    def main(Out: T.Tensor((32,), "bool")):
        with T.Kernel(1, threads=128):
            values = T.alloc_shared((32,), "bool")
            T.fill(values, True)
            T.fill(values[16:17], False)
            reduce_op(values[0:16], values[16:17], dim=None)
            T.copy(values, Out)

    result = tilelang.compile(main, out_idx=-1, target="cuda")()
    torch.testing.assert_close(result, torch.ones((32,), dtype=torch.bool, device="cuda"))


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op", ["any", "all"])
@pytest.mark.parametrize("clear", [True, False])
@pytest.mark.parametrize("dst_scope", ["local.fragment", "shared"])
def test_logical_reduce_empty_region(op, clear, dst_scope):
    reduce_op = T.reduce_any if op == "any" else T.reduce_all

    @T.prim_func
    def main(Seed: T.Tensor((4,), "bool"), Out: T.Tensor((4,), "bool")):
        with T.Kernel(1, threads=128):
            src = T.alloc_shared((4, 8), "int32")
            dst = T.alloc_buffer((4,), "bool", scope=dst_scope)
            T.copy(Seed, dst)
            reduce_op(src[:, 0:0], dst, dim=1, clear=clear)
            T.copy(dst, Out)

    seed = torch.tensor([True, False, True, False], device="cuda")
    result = tilelang.compile(main, out_idx=-1, target="cuda")(seed)
    expected = torch.full_like(seed, op == "all") if clear else seed
    torch.testing.assert_close(result, expected)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op", ["any", "all"])
def test_logical_reduce_non_aligned_replicated_fragment(op):
    reduce_op = T.reduce_any if op == "any" else T.reduce_all

    @T.prim_func
    def main(A: T.Tensor((4, 61), "float32"), Out: T.Tensor((4,), "bool")):
        with T.Kernel(1, threads=128):
            src = T.alloc_fragment((4, 61), "float32")
            dst = T.alloc_fragment((4,), "bool")
            T.annotate_layout({src: T.Fragment((4, 61), forward_fn=lambda i, j, rep: (i * 32 + rep, j), replicate=32)})
            T.copy(A, src)
            reduce_op(src, dst, dim=1)
            T.copy(dst, Out)

    data = torch.full((4, 61), 0.0 if op == "any" else 1.0, device="cuda")
    result = tilelang.compile(main, out_idx=-1, target="cuda")(data)
    torch.testing.assert_close(result, torch.full((4,), op == "all", device="cuda"))


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op", ["any", "all"])
def test_logical_reduce_symbolic_shared_offset(op):
    reduce_op = T.reduce_any if op == "any" else T.reduce_all

    @T.prim_func
    def main(A: T.Tensor((16, 64), "int32"), Out: T.Tensor((16,), "bool")):
        with T.Kernel(16, threads=128) as row:
            src = T.alloc_shared((16, 64), "int32")
            dst = T.alloc_fragment((1,), "bool")
            T.copy(A, src)
            reduce_op(src[row : row + 1, :], dst, dim=None)
            T.copy(dst, Out[row : row + 1])

    data = torch.ones((16, 64), dtype=torch.int32, device="cuda")
    data[::2] = 0
    result = tilelang.compile(main, out_idx=-1, target="cuda")(data)
    torch.testing.assert_close(result, torch.arange(16, device="cuda") % 2 == 1)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op", ["any", "all"])
def test_logical_reduce_partial_thread_range(op):
    reduce_op = T.reduce_any if op == "any" else T.reduce_all

    @T.prim_func
    def main(A: T.Tensor((2, 512), "int32"), Out: T.Tensor((2, 512), "bool")):
        with T.Kernel(1, threads=128):
            src = T.alloc_fragment((2, 512), "int32")
            dst = T.alloc_fragment((2,), "bool")
            T.annotate_layout(
                {
                    src: T.Fragment(src.shape, forward_fn=lambda i, j: (32 + j // 8, i * 8 + j % 8)),
                    dst: T.Fragment(dst.shape, forward_fn=lambda i, rep: (32 + rep, i), replicate=64),
                }
            )
            T.copy(A, src)
            reduce_op(src, dst)
            for i, j in T.Parallel(2, 512):
                Out[i, j] = dst[i]

    data = torch.full((2, 512), 0 if op == "any" else 2, dtype=torch.int32, device="cuda")
    data[1, -1] = 2 if op == "any" else 0
    result = tilelang.compile(main, out_idx=-1, target="cuda")(data)
    expected = torch.tensor([op == "all", op == "any"], device="cuda")[:, None].expand(2, 512)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("op", [T.reduce_any, T.reduce_all])
@pytest.mark.parametrize("scope", ["local", "global"])
def test_logical_reduce_rejects_scopes(op, scope):
    src = tvm.tirx.decl_buffer((4, 8), "float32", scope=scope)
    dst = tvm.tirx.decl_buffer((4,), "bool", scope="local.fragment")
    with pytest.raises(ValueError, match=f"shared or fragment input buffer, got {scope}"):
        op(src, dst)
    src = tvm.tirx.decl_buffer((4, 8), "float32", scope="local.fragment")
    dst = tvm.tirx.decl_buffer((4,), "bool", scope=scope)
    with pytest.raises(ValueError, match=f"shared or fragment output buffer, got {scope}"):
        op(src, dst)


@pytest.mark.parametrize("op", [T.reduce_any, T.reduce_all])
def test_logical_reduce_rejects_axis_shape_and_overlap(op):
    src = tvm.tirx.decl_buffer((4, 8), "bool", scope="local.fragment")
    dst = tvm.tirx.decl_buffer((4,), "bool", scope="local.fragment")
    with pytest.raises(ValueError, match="out of range"):
        op(src, dst, dim=2)
    with pytest.raises(ValueError, match="out of range"):
        op(src, dst, dim=-3)
    with pytest.raises(ValueError, match="Invalid reduce output shape"):
        op(src, dst, dim=None)
    with pytest.raises(ValueError, match="non-overlapping"):
        op(src, src[:, 0:1], dim=1)
    with pytest.raises(ValueError, match="full fragment input"):
        op(src[:, 0:4], dst)
    with pytest.raises(ValueError, match="full fragment output"):
        op(src, src[:, 0:0], dim=1)
    shared = tvm.tirx.decl_buffer((4, 8), "bool", scope="shared")
    with pytest.raises(ValueError, match="non-overlapping"):
        op(shared, shared[:, 0:1], dim=1)
    scalar = tvm.tirx.decl_buffer((1,), "bool", scope="shared")
    with pytest.raises(ValueError, match="Invalid reduce output shape"):
        op(shared[:, 0:0], scalar, dim=1)


@pytest.mark.parametrize("op", [T.reduce_any, T.reduce_all])
@pytest.mark.parametrize("empty,clear", [(False, True), (True, True), (True, False)])
@pytest.mark.parametrize(
    "source_dtype,output_dtype,message",
    [
        ("float32", "int32", "requires a bool output"),
        ("float8_e4m3fn", "bool", "requires a boolean, integer, or floating-point input"),
        ("int32x2", "bool", "requires a boolean, integer, or floating-point input"),
    ],
)
def test_logical_reduce_rejects_dtype_before_empty_folding(op, empty, clear, source_dtype, output_dtype, message):
    src = tvm.tirx.decl_buffer((4, 8), source_dtype, scope="shared")
    dst = tvm.tirx.decl_buffer((4,), output_dtype, scope="shared")
    with pytest.raises(TypeError, match=message):
        op(src[:, 0:0] if empty else src, dst, clear=clear)


@pytest.mark.parametrize("op", [T.reduce_any, T.reduce_all])
def test_logical_reduce_rejects_dynamic_region_at_creation(op):
    n = T.Var("n", "int32")
    src = tvm.tirx.decl_buffer((4, 8), "float32", scope="shared")
    dst = tvm.tirx.decl_buffer((4,), "bool", scope="local.fragment")
    with pytest.raises(ValueError, match="requires non-negative static region extents in native lowering"):
        op(src[:, 0:n], dst)


@pytest.mark.parametrize("op", ["any", "all"])
@pytest.mark.parametrize("scope", ["local.fragment", "shared"])
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("hip", marks=tilelang.testing.requires_rocm.marks(support_required="compile-only")),
        pytest.param("cutedsl", marks=tilelang.testing.requires_cuda.marks(support_required="compile-only")),
    ],
)
def test_logical_reduce_backend_codegen(op, scope, backend):
    from tilelang.engine.lower import lower

    build_func = f"target.build.tilelang_{backend}_without_compile"
    build = tvm.ffi.get_global_func(build_func, allow_missing=True)
    if build is None:
        pytest.skip(f"{backend} codegen is not enabled in this build")
    program, _ = _kernel(op, "int64", shape=(1, 512), dim=None, src_scope=scope, dst_scope=scope)
    if backend == "cutedsl":
        from tilelang.backend.module import BackendContext
        from tilelang.cuda.cutedsl_backend import BACKEND
        from tilelang.cuda.target import normalize_cutedsl_target
        from tilelang.engine.lower import lower_with_context

        target = normalize_cutedsl_target({"kind": "cutedsl", "arch": "sm_90"})
        host = tvm.target.Target("llvm")
        target = tvm.target.Target(target, host)
        # Exercise CuTeDSL target lowering without requiring its runtime package.
        context = BackendContext(BACKEND, target, host, BACKEND.execution_backends[0])
        with target:
            source = lower_with_context(program, context).kernel_source
    else:
        target = tvm.target.Target({"kind": "hip", "mcpu": "gfx942"})
        with target:
            source = lower(program, target=target).kernel_source
    assert "BitOr" in source if op == "any" else "BitAnd" in source
    assert "!= " in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "source_scope,output_dtype,message",
    [("local", "bool", "requires a shared or fragment"), ("local.fragment", "int32", "requires a bool output")],
)
def test_logical_reduce_native_validation(source_scope, output_dtype, message):
    @T.prim_func
    def main(Out: T.Tensor((4,), output_dtype)):
        with T.Kernel(1, threads=128):
            src = T.alloc_buffer((4, 8), "int32", scope=source_scope)
            dst = T.alloc_fragment((4,), output_dtype)
            T.fill(src, 1)
            T.call_intrin(
                "handle", tvm.tirx.op.Op.get("tl.tileop.reduce"), to_tile_region(src, "r"), to_tile_region(dst, "w"), "any", 1, True
            )
            T.copy(dst, Out)

    with pytest.raises(Exception, match=message):
        tilelang.compile(main, out_idx=-1, target="cuda")


if __name__ == "__main__":
    tilelang.testing.main()

import numpy as np
import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import _ffi_api, tvm
from tilelang.layout import make_swizzled_layout
from tilelang.utils.language import to_tile_region


def _kernel(
    kind,
    shape,
    dtype="float32",
    index_dtype="int32",
    dim=-1,
    keepdim=False,
    src_scope="local.fragment",
    dst_scope="local.fragment",
    threads=128,
    swizzle=False,
):
    axis = dim % len(shape)
    out_shape = (1,) if len(shape) == 1 else shape[:axis] + ((1,) if keepdim else ()) + shape[axis + 1 :]
    reduce = T.reduce_argmax if kind == "max" else T.reduce_argmin

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Out: T.Tensor(out_shape, index_dtype)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_buffer(shape, dtype, scope=src_scope)
            dst = T.alloc_buffer(out_shape, index_dtype, scope=dst_scope)
            if swizzle:
                T.annotate_layout({src: make_swizzled_layout(src)})
            T.copy(A, src)
            reduce(src, dst, dim)
            T.copy(dst, Out)

    return main, out_shape


def _run(kind, data, **kwargs):
    program, shape = _kernel(kind, tuple(data.shape), str(data.dtype).removeprefix("torch."), **kwargs)
    kernel = tilelang.compile(program, out_idx=-1, target="cuda")
    source = data.cuda()
    original = source.clone()
    result = kernel(source)
    array = data.float().numpy() if data.dtype == torch.bfloat16 else data.numpy()
    expected = (np.argmax if kind == "max" else np.argmin)(array, axis=kwargs.get("dim", -1)).reshape(shape)
    np.testing.assert_array_equal(result.cpu().numpy(), expected)
    np.testing.assert_array_equal(source.cpu().view(torch.uint8).numpy(), original.cpu().view(torch.uint8).numpy())
    return kernel


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("kind", ["min", "max"])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.uint16,
        torch.int32,
        torch.uint32,
        torch.int64,
        torch.uint64,
    ],
)
def test_arg_reduce_dtypes_and_extrema(kind, dtype):
    data = torch.randint(0, 100, (5, 128)).to(dtype)
    if dtype.is_floating_point:
        data[0, 7] = data[0, 95] = float("nan")
        data[1] = float("-inf")
        data[2] = float("inf")
        data[3] = -0.0
        data[3, 99] = 0.0
        data[4] = float("nan")
    else:
        # Adjacent large integers must remain distinct through the comparator
        # and shuffle, including unsigned values above the signed range.
        array = data.numpy()
        limits = np.iinfo(array.dtype)
        array[0] = limits.max - 1
        array[0, 97] = limits.max
        array[1] = limits.min + 1
        array[1, 65] = limits.min
        array[2] = limits.max
        array[3] = limits.min
    _run(kind, data, index_dtype="int64")


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("kind", ["min", "max"])
@pytest.mark.parametrize(
    "shape,dim,keepdim,src_scope,dst_scope",
    [
        ((3, 5), 0, False, "local.fragment", "local.fragment"),
        ((3, 5), 0, True, "shared", "shared"),
        ((2, 3, 7), 1, False, "local.fragment", "shared"),
        ((2, 3, 7), -2, True, "shared", "local.fragment"),
        ((137,), -1, False, "local.fragment", "local.fragment"),
        ((4, 1), -1, True, "shared.dyn", "shared.dyn"),
    ],
)
def test_arg_reduce_ragged_shapes(kind, shape, dim, keepdim, src_scope, dst_scope):
    data = torch.randn(shape)
    if shape == (3, 5):
        data = torch.tensor([[1, 10, 3, 40, 5], [2, 9, 9, 30, 4], [0, 8, 6, 20, 3]], dtype=torch.float32)
    _run(kind, data, dim=dim, keepdim=keepdim, src_scope=src_scope, dst_scope=dst_scope)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("kind", ["min", "max"])
@pytest.mark.parametrize("width,scale,replicas", [(16, 2, 1), (32, 1, 1), (64, 2, 1), (128, 1, 1), (32, 1, 4)])
def test_arg_reduce_thread_topology(kind, width, scale, replicas):
    groups = 128 // width // replicas
    shape = (groups * 2, width * 3)
    reduce = T.reduce_argmax if kind == "max" else T.reduce_argmin

    @T.prim_func
    def main(A: T.Tensor(shape, "float32"), Out: T.Tensor((shape[0],), "int32")):
        with T.Kernel(1, threads=128):
            src = T.alloc_fragment(shape, "float32")
            dst = T.alloc_fragment((shape[0],), "int32")
            T.annotate_layout(
                {
                    src: T.Fragment(
                        shape,
                        forward_fn=lambda i, j, rep=0: (
                            rep * (128 // replicas) + (i % groups // scale) * width * scale + j % width * scale + i % scale,
                            i // groups * 3 + j // width,
                        ),
                        replicate=replicas,
                    )
                }
            )
            T.copy(A, src)
            reduce(src, dst)
            T.copy(dst, Out)

    kernel = tilelang.compile(main, out_idx=-1, target="cuda")
    data = torch.randn(shape, device="cuda")
    data[:, 11] = data[:, width * 2 + 7] = float("nan")
    expected = (torch.argmax if kind == "max" else torch.argmin)(data, dim=1).int()
    torch.testing.assert_close(kernel(data), expected)
    assert "ArgAllReduce<" in kernel.get_kernel_source()


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("n", [96, 128, 192])
def test_arg_reduce_gemm_layout(n):
    @T.prim_func
    def main(
        A: T.Tensor((32, 64), "float16"), B: T.Tensor((64, n), "float16"), Min: T.Tensor((32,), "int32"), Max: T.Tensor((32,), "int64")
    ):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((32, 64), "float16")
            b = T.alloc_shared((64, n), "float16")
            scores = T.alloc_fragment((32, n), "float32")
            imin = T.alloc_fragment((32,), "int32")
            imax = T.alloc_fragment((32,), "int64")
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, scores, clear_accum=True)
            T.reduce_argmin(scores, imin)
            T.reduce_argmax(scores, imax)
            T.copy(imin, Min)
            T.copy(imax, Max)

    a = torch.randint(-4, 5, (32, 64), device="cuda").half()
    b = torch.randint(-4, 5, (64, n), device="cuda").half()
    b[:, n // 2 :] = b[:, : n // 2]  # Equal values across distinct MMA registers/lanes.
    expected = a.float() @ b.float()
    kernel = tilelang.compile(main, out_idx=[2, 3], target="cuda")
    imin, imax = kernel(a, b)
    torch.testing.assert_close(imin, expected.argmin(1).int())
    torch.testing.assert_close(imax, expected.argmax(1))
    assert "arg_staging" not in kernel.get_kernel_source()


@tilelang.testing.requires_cuda
def test_arg_reduce_swizzled_shared():
    data = torch.randn((8, 64), dtype=torch.float16)
    data[0, 7] = data[0, 49] = float("nan")
    _run("min", data, src_scope="shared", swizzle=True)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("stages", [0, 2])
def test_arg_reduce_shared_region_and_repeated_calls(stages):
    @T.prim_func
    def main(A: T.Tensor((6, 17), "float32"), Out: T.Tensor((2, 4), "int32")):
        with T.Kernel(1, threads=128):
            src = T.alloc_shared((6, 17), "float32")
            dst = T.alloc_shared((6,), "int32")
            T.copy(A, src)
            for i in T.Pipelined(2, num_stages=stages):
                T.reduce_argmax(src[1:5, 3 + i : 16 + i], dst[1:5], dim=1)
                for j in T.Parallel(4):
                    Out[i, j] = dst[j + 1]

    data = torch.randn((6, 17), device="cuda")
    data[1:5, 11] = 100
    kernel = tilelang.compile(main, out_idx=-1, target="cuda")
    expected = torch.stack([data[1:5, 3 + i : 16 + i].argmax(1).int() for i in range(2)])
    torch.testing.assert_close(kernel(data), expected)


@tilelang.testing.requires_cuda
def test_arg_reduce_partial_and_replicated_layouts():
    @T.prim_func
    def main(A: T.Tensor((2, 512), "int64"), Out: T.Tensor((2, 512), "int32")):
        with T.Kernel(1, threads=128):
            src = T.alloc_fragment((2, 512), "int64")
            dst = T.alloc_fragment((2,), "int32")
            T.annotate_layout(
                {
                    src: T.Fragment(src.shape, forward_fn=lambda i, j: (32 + j // 8, i * 8 + j % 8)),
                    dst: T.Fragment(dst.shape, forward_fn=lambda i, rep: (32 + rep, i), replicate=64),
                }
            )
            T.copy(A, src)
            T.reduce_argmax(src, dst)
            for i, j in T.Parallel(2, 512):
                Out[i, j] = dst[i]

    data = torch.full((2, 512), 2**60, device="cuda", dtype=torch.int64)
    data[1, 301] += 1
    data[1, 477] += 1
    kernel = tilelang.compile(main, out_idx=-1, target="cuda")
    torch.testing.assert_close(kernel(data), data.argmax(1).int()[:, None].expand(2, 512))


@tilelang.testing.requires_cuda
def test_arg_reduce_destination_redistribution():
    @T.prim_func
    def main(A: T.Tensor((8, 96), "float32"), Out: T.Tensor((8,), "int32")):
        with T.Kernel(1, threads=128):
            src = T.alloc_fragment((8, 96), "float32")
            dst = T.alloc_fragment((8,), "int32")
            T.annotate_layout(
                {
                    src: T.Fragment(src.shape, forward_fn=lambda i, j: (i * 16 + j % 16, j // 16)),
                    dst: T.Fragment(dst.shape, forward_fn=lambda i, rep: (rep * 8 + i, 0), replicate=16),
                }
            )
            T.copy(A, src)
            T.reduce_argmin(src, dst)
            T.copy(dst, Out)

    data = torch.randn((8, 96), device="cuda")
    kernel = tilelang.compile(main, out_idx=-1, target="cuda")
    torch.testing.assert_close(kernel(data), data.argmin(1).int())


def test_arg_reduce_stays_a_tileop():
    program, _ = _kernel("max", (4, 128))
    calls = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and node.op == tvm.ir.Op.get("tl.tileop.arg_reduce"):
            calls.append(node)

    tvm.tirx.stmt_functor.post_order_visit(program.body, visit)
    assert len(calls) == 1


@tilelang.testing.requires_rocm
@pytest.mark.parametrize("arch", ["gfx942", "gfx1100"])
@pytest.mark.parametrize("shape,dim,dtype", [((4, 128), 1, "uint64"), ((3, 5), 0, "bfloat16")])
def test_arg_reduce_rocm_codegen(arch, shape, dim, dtype):
    if tvm.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True) is None:
        pytest.skip("ROCm codegen is not enabled")
    from tilelang.engine.lower import lower

    program, _ = _kernel("max", shape, dtype=dtype, dim=dim, index_dtype="int64")
    target = tvm.target.Target({"kind": "hip", "mcpu": arch})
    with target:
        artifact = lower(program, target=target)
    assert "tl::arg_reduce_update<" in artifact.kernel_source
    assert "tl::ArgAllReduce<" in artifact.kernel_source
    assert "tl_templates/hip/reduce.h" in artifact.kernel_source


@tilelang.testing.requires_cuda
def test_arg_reduce_rejects_cutedsl():
    from tilelang.cuda.target import normalize_cutedsl_target
    from tilelang.engine.lower import lower
    from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

    if tvm.get_global_func("target.build.tilelang_cutedsl_without_compile", allow_missing=True) is None:
        pytest.skip("CuTeDSL codegen is not enabled")
    try:
        check_cutedsl_available()
    except ImportError as error:
        pytest.skip(str(error))
    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": "sm_90"})
    with target, pytest.raises(ValueError, match="arg_reduce has no implementation"):
        lower(_kernel("max", (4, 128))[0], target=target)


def _partial_kernel():
    @T.prim_func
    def main(A: T.Tensor((2, 128), "float32"), Out: T.Tensor((2,), "int32")):
        with T.Kernel(1, threads=256):
            if T.get_thread_binding() >= 64 and T.get_thread_binding() < 192:
                src = T.alloc_fragment((2, 128), "float32")
                dst = T.alloc_fragment((2,), "int32")
                T.copy(A, src)
                T.reduce_argmax(src, dst)
                T.copy(dst, Out)

    return main


@tilelang.testing.requires_cuda
def test_arg_reduce_partial_block():
    data = torch.randn((2, 128), device="cuda")
    data[0, 71] = data[0, 125] = float("nan")
    kernel = tilelang.compile(_partial_kernel(), target="cuda", out_idx=-1)
    torch.testing.assert_close(kernel(data), data.argmax(1).int())


@tilelang.testing.requires_rocm
def test_arg_reduce_rocm_rejects_partial_block():
    if tvm.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True) is None:
        pytest.skip("ROCm codegen is not enabled")
    from tilelang.engine.lower import lower

    target = tvm.target.Target({"kind": "hip", "mcpu": "gfx942"})
    with target, pytest.raises(ValueError, match="requires all threads"):
        lower(_partial_kernel(), target=target)


def test_arg_reduce_native_contract():
    def native(shape=(4, 8), out_shape=(4,), dtype="float32", index_dtype="int32", dim=-1, scope="local.fragment", kind="max"):
        src = tvm.tirx.decl_buffer(shape, dtype, scope=scope)
        dst = tvm.tirx.decl_buffer(out_shape, index_dtype, scope="local.fragment")
        return _ffi_api.ArgReduceOp(
            [to_tile_region(src, "r"), to_tile_region(dst, "w"), tvm.tirx.StringImm(kind), tvm.tirx.IntImm("int32", dim)], {}
        )

    native()
    for kwargs, message in [
        ({"dim": 2}, "axis"),
        ({"dim": -3}, "axis"),
        ({"index_dtype": "float32"}, "int32 or int64"),
        ({"dtype": "bool"}, "scalar integer"),
        ({"shape": (4, 0)}, "positive static"),
        ({"out_shape": ()}, "output shape"),
        ({"out_shape": (4, 2)}, "output shape"),
        ({"scope": "global"}, "shared or fragment"),
        ({"kind": "sum"}, "kind"),
    ]:
        with pytest.raises((ValueError, TypeError), match=message):
            native(**kwargs)

    with pytest.raises(ValueError, match="expects source"):
        _ffi_api.ArgReduceOp([], {})

    buf = tvm.tirx.decl_buffer((2, 8), "int32", scope="shared")

    def region(row, width):
        return tvm.tirx.BufferRegion(buf, [tvm.ir.Range.from_min_extent(row, 1), tvm.ir.Range.from_min_extent(0, width)])

    src = to_tile_region(region(0, 8), "r")
    kind, dim = tvm.tirx.StringImm("max"), tvm.tirx.IntImm("int32", 1)
    _ffi_api.ArgReduceOp([src, to_tile_region(region(1, 1), "w"), kind, dim], {})
    with pytest.raises(ValueError, match="must not overlap"):
        _ffi_api.ArgReduceOp([src, to_tile_region(region(0, 1), "w"), kind, dim], {})


@pytest.mark.parametrize("initialized", [False, True])
def test_arg_reduce_native_access_effects(initialized, capfd):
    op = tvm.ir.Op.get("tl.tileop.arg_reduce")

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            src = T.alloc_fragment((4, 8), "float32")
            dst = T.alloc_fragment((4,), "int32")
            if initialized:
                T.fill(src, 1)
            T.evaluate(T.call_intrin("handle", op, to_tile_region(src, "w"), to_tile_region(dst, "r"), "max", 1))

    tilelang.transform.VerifyBufferInit()(tvm.IRModule.from_expr(main))
    warnings = capfd.readouterr().err
    assert ("Buffer read before initialization" in warnings) != initialized


if __name__ == "__main__":
    tilelang.testing.main()

import numpy as np
import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit(out_idx=[1])
def arg_reduce_kernel(
    shape, dim=-1, kind="max", dtype="float32", index_dtype="int32", threads=128, src_shared=False, dst_shared=False, keepdim=False
):
    axis = dim % len(shape)
    out_shape = list(shape)
    if keepdim:
        out_shape[axis] = 1
    else:
        out_shape.pop(axis)
    op = T.reduce_argmax if kind == "max" else T.reduce_argmin

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), B: T.Tensor(out_shape, index_dtype)):
        with T.Kernel(1, threads=threads):
            src = T.alloc_shared(shape, dtype) if src_shared else T.alloc_fragment(shape, dtype)
            dst = T.alloc_shared(out_shape, index_dtype) if dst_shared else T.alloc_fragment(out_shape, index_dtype)
            T.copy(A, src)
            op(src, dst, dim)
            if len(out_shape) == 0:
                T.copy(dst[()], B[()])
            else:
                T.copy(dst, B)

    return main


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize(
    "shape,dim,threads", [((128,), -1, 32), ((4, 128), 1, 128), ((128, 4), 0, 128), ((4, 16, 8), 1, 128), ((2, 4, 64), -1, 256)]
)
@pytest.mark.parametrize("index_dtype,keepdim", [("int32", False), ("int64", True)])
def test_arg_reduce_dimensions(kind, shape, dim, threads, index_dtype, keepdim):
    torch.manual_seed(0)
    x = torch.randint(-8, 9, shape, device="cuda").float()
    kernel = arg_reduce_kernel(shape, dim, kind, index_dtype=index_dtype, threads=threads, keepdim=keepdim)
    expected = getattr(torch, "arg" + kind)(x, dim=dim, keepdim=keepdim)
    torch.testing.assert_close(kernel(x), expected.to(getattr(torch, index_dtype)), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize(
    "dtype", ["float16", "bfloat16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
)
def test_arg_reduce_dtypes(kind, dtype):
    torch.manual_seed(1)
    x = torch.randint(0, 32, (4, 128), device="cuda").to(getattr(torch, dtype))
    kernel = arg_reduce_kernel(tuple(x.shape), 1, kind, dtype=dtype)
    if dtype in ("uint16", "uint32", "uint64"):
        expected = torch.from_numpy(getattr(np, "arg" + kind)(x.cpu().numpy(), axis=1)).cuda().int()
    else:
        expected = getattr(torch, "arg" + kind)(x, dim=1).int()
    torch.testing.assert_close(kernel(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("src_shared,dst_shared", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_arg_reduce_special_values(kind, src_shared, dst_shared, dtype):
    x = torch.full((8, 256), -3.0, device="cuda", dtype=getattr(torch, dtype))
    extreme = float("inf") if kind == "max" else -float("inf")
    x[0, 200] = extreme
    x[0, 33] = extreme
    x[1, 0] = extreme
    x[1, 130] = float("nan")
    x[1, 63] = float("nan")
    x[2] = float("nan")
    x[3] = extreme
    x[4] = -extreme
    x[5] = 0.0
    x[5, 0] = -0.0
    x[6, -1] = float("nan")
    x[7, 255] = extreme
    kernel = arg_reduce_kernel(tuple(x.shape), 1, kind, dtype=dtype, src_shared=src_shared, dst_shared=dst_shared)
    expected = getattr(torch, "arg" + kind)(x, dim=1).int()
    for _ in range(3):
        torch.testing.assert_close(kernel(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("extent", [1, 7, 31, 33, 129])
def test_arg_reduce_tail(kind, extent):
    x = torch.full((4, extent), -2.0, device="cuda")
    x[:, -1] = 7.0 if kind == "max" else -7.0
    torch.testing.assert_close(
        arg_reduce_kernel(tuple(x.shape), 1, kind)(x), torch.full((4,), extent - 1, device="cuda", dtype=torch.int32), rtol=0, atol=0
    )


def test_arg_reduce_integer_precision():
    x = torch.full((4, 128), 2**60, dtype=torch.int64, device="cuda")
    x[:, 97] += 1
    x[:, 3] -= 1
    for kind in ("min", "max"):
        expected = getattr(torch, "arg" + kind)(x, dim=1).int()
        torch.testing.assert_close(arg_reduce_kernel(tuple(x.shape), 1, kind, dtype="int64")(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize(
    "shape,dtype,out_shape,index_dtype,dim,scope,message",
    [
        ((4, 128), "float32", (4,), "int32", 2, "local.fragment", "out of bounds"),
        ((4, 128), "float32", (4,), "int32", -3, "local.fragment", "out of bounds"),
        ((4, 128), "float32", (4,), "float32", 1, "local.fragment", "output dtype"),
        ((4, 128), "float32", (3,), "int32", 1, "local.fragment", "output shape"),
        ((4, 128), "float32", (4,), "int32", 1, "global", "fragment or shared"),
        ((4, 0), "float32", (4,), "int32", 1, "local.fragment", "positive constant"),
        ((4, 2**31), "float32", (4,), "int32", 1, "local.fragment", "cannot be represented"),
        ((4, 128), "bool", (4,), "int32", 1, "local.fragment", "input dtype"),
    ],
)
def test_arg_reduce_invalid_arguments(kind, shape, dtype, out_shape, index_dtype, dim, scope, message):
    from tvm import tirx

    src = tirx.decl_buffer(shape, dtype, scope=scope)
    dst = tirx.decl_buffer(out_shape, index_dtype, scope="local.fragment")
    with pytest.raises(ValueError, match=message):
        getattr(T, "reduce_arg" + kind)(src, dst, dim)


@pytest.mark.parametrize("kind", ["max", "min"])
def test_arg_reduce_operand_type(kind):
    from tvm import tirx

    src = tirx.decl_buffer((4, 128), "float32", scope="local.fragment")
    dst = tirx.decl_buffer((4,), "int32", scope="local.fragment")
    op = getattr(T, "reduce_arg" + kind)
    with pytest.raises(TypeError, match="Buffers"):
        op(src[0, 0], dst)
    with pytest.raises(TypeError, match="dim must be an integer"):
        op(src, dst, 1.5)


@pytest.mark.parametrize("kind", ["max", "min"])
def test_arg_reduce_gemm_accumulator(kind):
    op = T.reduce_argmax if kind == "max" else T.reduce_argmin

    @tilelang.jit(out_idx=[2])
    def make_kernel():
        @T.prim_func
        def main(A: T.Tensor((32, 64), "float16"), B: T.Tensor((64, 128), "float16"), I: T.Tensor((32,), "int32")):
            with T.Kernel(1, threads=128):
                a = T.alloc_shared((32, 64), "float16")
                b = T.alloc_shared((64, 128), "float16")
                scores = T.alloc_fragment((32, 128), "float32")
                indices = T.alloc_fragment((32,), "int32")
                T.copy(A, a)
                T.copy(B, b)
                T.gemm(a, b, scores, clear_accum=True)
                op(scores, indices, dim=1)
                T.copy(indices, I)

        return main

    torch.manual_seed(7)
    a = torch.randn((32, 64), dtype=torch.float16, device="cuda")
    b = torch.randn((64, 128), dtype=torch.float16, device="cuda")
    # FP64 reference avoids precision-mode differences in the reference GEMM.
    expected = getattr(torch, "arg" + kind)(a.double() @ b.double(), dim=1).int()
    torch.testing.assert_close(make_kernel()(a, b), expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64", "int8", "int32", "int64", "uint64"])
@pytest.mark.parametrize("shape,dim", [((7, 4), 0), ((2, 7, 4), 1)])
def test_arg_reduce_padding_identities(kind, dtype, shape, dim):
    is_float = dtype.startswith("float") or dtype == "bfloat16"
    if is_float:
        identity = -float("inf") if kind == "max" else float("inf")
        x = torch.full(shape, identity, device="cuda", dtype=getattr(torch, dtype))
        # Equal identity values must choose a real input, never the padding.
        expected = torch.zeros(shape[:dim] + shape[dim + 1 :], device="cuda", dtype=torch.int64)
        if shape[0] == 2:
            x.select(dim, shape[dim] - 1).fill_(float("nan"))
            expected.fill_(shape[dim] - 1)
    else:
        info = np.iinfo(getattr(np, dtype))
        x = torch.from_numpy(np.full(shape, info.min if kind == "max" else info.max, dtype=getattr(np, dtype))).cuda()
        expected = torch.zeros(shape[:dim] + shape[dim + 1 :], device="cuda", dtype=torch.int64)
    kernel = arg_reduce_kernel(shape, dim, kind, dtype=dtype, index_dtype="int64", src_shared=True, dst_shared=True)
    torch.testing.assert_close(kernel(x), expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()

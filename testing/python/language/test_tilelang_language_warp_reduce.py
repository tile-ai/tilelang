import torch
import pytest

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tvm import tirx


@tilelang.jit
def get_kernel(reduce_op: str, dtype: str, threads=(32, 1, 1)):
    assert reduce_op in ["sum", "max", "min", "bitand", "bitor"]
    N = threads[0] * threads[1] * threads[2]

    @T.prim_func
    def main(x: T.Tensor((N,), dtype)):
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding(0) + threads[0] * (T.get_thread_binding(1) + threads[1] * T.get_thread_binding(2))
            local_val = T.alloc_local([1], dtype)
            local_val[0] = x[tx]
            reduced_val = T.alloc_local([1], dtype)
            if reduce_op == "sum":
                reduced_val[0] = T.warp_reduce_sum(local_val[0])
            elif reduce_op == "max":
                reduced_val[0] = T.warp_reduce_max(local_val[0])
            elif reduce_op == "min":
                reduced_val[0] = T.warp_reduce_min(local_val[0])
            elif reduce_op == "bitand":
                reduced_val[0] = T.warp_reduce_bitand(local_val[0])
            elif reduce_op == "bitor":
                reduced_val[0] = T.warp_reduce_bitor(local_val[0])
            x[tx] = reduced_val[0]

    return main


def test_warp_reduce_sum():
    a = torch.randn((32,), dtype=torch.float32, device="cuda")
    kernel = get_kernel("sum", T.float32)
    ref = torch.full_like(a, a.sum())
    kernel(a)
    torch.testing.assert_close(a, ref)


def test_warp_reduce_max():
    a = torch.randn((32,), dtype=torch.float32, device="cuda")
    kernel = get_kernel("max", T.float32)
    print(kernel.get_kernel_source())
    ref = torch.full_like(a, a.max())
    kernel(a)
    torch.testing.assert_close(a, ref)


def test_warp_reduce_min():
    a = torch.randn((32,), dtype=torch.float32, device="cuda")
    kernel = get_kernel("min", T.float32)
    ref = torch.full_like(a, a.min())
    kernel(a)
    torch.testing.assert_close(a, ref)


def test_warp_reduce_bitand():
    a = torch.randint(0, 100, size=(32,), dtype=torch.int32, device="cuda")
    kernel = get_kernel("bitand", T.int32)
    ref_val = a[0]
    for i in range(1, a.shape[0]):
        ref_val = ref_val & a[i]
    ref = torch.full_like(a, ref_val)
    kernel(a)
    torch.testing.assert_close(a, ref)


def test_warp_reduce_bitor():
    a = torch.randint(0, 100, size=(32,), dtype=torch.int32, device="cuda")
    kernel = get_kernel("bitor", T.int32)
    ref_val = a[0]
    for i in range(1, a.shape[0]):
        ref_val = ref_val | a[i]
    ref = torch.full_like(a, ref_val)
    kernel(a)
    torch.testing.assert_close(a, ref)


WARP_REDUCE_CASES_64 = [
    # (op, dtype, N)
    ("sum", "int64", 32),
    ("max", "int64", 32),
    ("min", "int64", 32),
    ("bitand", "int64", 32),
    ("bitor", "int64", 32),
]


@pytest.mark.parametrize(
    ("op", "dtype", "N"),
    WARP_REDUCE_CASES_64,
)
def test_warp_reduce_64(op, dtype, N):
    def warp_reduce_ref(a):
        if op == "sum":
            return torch.full_like(a, a.sum())
        elif op == "max":
            return torch.full_like(a, a.max())
        elif op == "min":
            return torch.full_like(a, a.min())
        elif op == "bitand":
            ref_val = a[0]
            for i in range(1, a.shape[0]):
                ref_val = ref_val & a[i]
            return torch.full_like(a, ref_val)
        elif op == "bitor":
            ref_val = a[0]
            for i in range(1, a.shape[0]):
                ref_val = ref_val | a[i]
            return torch.full_like(a, ref_val)
        raise AssertionError(f"Unknown op: {op}")

    torch_dtype = getattr(torch, dtype)
    tl_dtype = getattr(T, dtype)

    a = torch.randint(1 << 32, (1 << 63) - 1, (N,), dtype=torch_dtype, device="cuda")
    ref = warp_reduce_ref(a)

    kernel = get_kernel(op, tl_dtype)
    kernel(a)

    torch.testing.assert_close(a, ref)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "threads",
    [
        (1, 1, 1),
        (3, 1, 1),
        (7, 1, 1),
        (31, 1, 1),
        (33, 1, 1),
        (48, 1, 1),
        (100, 1, 1),
        (7, 7, 1),
        (3, 3, 5),
        (32, 1, 1),
        (8, 8, 1),
        (4, 8, 2),
    ],
)
@pytest.mark.parametrize(
    "op,dtype",
    [(op, dtype) for op in ("sum", "min", "max") for dtype in ("float32", "float64", "float16", "bfloat16", "int32", "int64")]
    + [(op, dtype) for op in ("bitand", "bitor") for dtype in ("int32", "int64")],
)
def test_warp_reduce_partial(op, dtype, threads):
    N = threads[0] * threads[1] * threads[2]
    indices = torch.arange(N, dtype=torch.int64, device="cuda")
    if op in ("bitand", "bitor"):
        # Distinct bits expose missing lanes and int64 high-word shuffles.
        lane = indices % 32
        a = torch.ones_like(indices) << (lane + 31 if dtype == "int64" else lane % 31)
        if dtype == "int32":
            a[lane == 31] = -(1 << 31)
        if op == "bitand":
            a = ~a
        a = a.to(getattr(torch, dtype))
    else:
        a = (indices // 32 + indices % 2 + 2).to(getattr(torch, dtype))
        if op in ("min", "max"):
            tail = (indices % 32 == 31) | (indices == N - 1)
            a[tail] = (indices[tail] // 32 + 1).to(a.dtype)
        if dtype == "int64":
            a += 1 << 40
        if op == "max":
            a = -a

    ref = torch.empty_like(a)
    for start in range(0, N, 32):
        warp = a[start : start + 32]
        if op == "sum":
            value = warp.sum()
        elif op == "min":
            value = warp.min()
        elif op == "max":
            value = warp.max()
        else:
            value = warp[0]
            for element in warp[1:]:
                value = value & element if op == "bitand" else value | element
        ref[start : start + 32] = value

    kernel = get_kernel(op, dtype, threads)
    kernel(a)
    torch.testing.assert_close(a, ref, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("all_nan", [False, True])
def test_warp_reduce_partial_nan(op, all_nan):
    a = torch.full((7,), float("nan"), dtype=torch.float32, device="cuda")
    expected = float("nan")
    if not all_nan:
        a[1], a[5] = 2, -3
        expected = -3 if op == "min" else 2
    ref = torch.full_like(a, expected)
    kernel = get_kernel(op, "float32", (7, 1, 1))
    kernel(a)
    torch.testing.assert_close(a, ref, rtol=0, atol=0, equal_nan=True)


@tilelang.testing.requires_cuda
def test_warp_reduce_codegen_launch_extent():
    build = tvm.get_global_func("target.build.tilelang_cuda_without_compile")
    dynamic = tirx.Var("threads", "int32")
    cases = [
        ({"threadIdx.x": 32}, 32),
        ({"threadIdx.x": 8, "threadIdx.y": 8}, 64),
        ({"threadIdx.x": 4, "threadIdx.y": 8, "threadIdx.z": 2}, 64),
        ({"threadIdx.x": 48}, None),
        ({"threadIdx.x": 32, "threadIdx.y": dynamic}, None),
        ({"blockIdx.x": 32}, None),
        (None, None),
        ({"threadIdx.x": 0}, None),
        ({"threadIdx.x": -32}, None),
        ({"threadIdx.x": 32, "threadIdx.y": 1 << 62}, None),
    ]
    operations = ("sum", "min", "max", "bitand", "bitor")
    functions = {}
    for index, (extents, _) in enumerate(cases):
        value = tirx.Var(f"value_{index}", "int64")
        body = tirx.SeqStmt([tirx.Evaluate(tirx.call_intrin("int64", f"tl.warp_reduce_{op}", value)) for op in operations])
        # A launch-bounds maximum alone must not authorize specialization.
        axis = tvm.te.thread_axis("threadIdx.x")
        body = tirx.AttrStmt(axis, "thread_extent", 32, body)
        name = f"reduce_{index}"
        func = tirx.PrimFunc([value, dynamic], body).with_attr("global_symbol", name)
        func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
        if extents is not None:
            func = func.with_attr(
                "thread_extent",
                {tag: tirx.const(extent, "int64") if isinstance(extent, int) else extent for tag, extent in extents.items()},
            )
        functions[name] = func
    mod = tvm.IRModule(functions)
    # Preserve the map's keys but force full kernels to precede fallbacks.
    for key, func in zip(list(mod.functions), functions.values()):
        mod[key] = func
    assert [str(func.attrs["global_symbol"]) for func in mod.functions.values()] == list(functions)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    with target:
        source = build(mod, target).inspect_source()
    for index, (_, extent) in enumerate(cases):
        for op in operations:
            specialization = f"<int64_t, {extent}>" if extent is not None else ""
            assert f"tl::warp_reduce_{op}{specialization}(value_{index})" in source


if __name__ == "__main__":
    tilelang.testing.main()

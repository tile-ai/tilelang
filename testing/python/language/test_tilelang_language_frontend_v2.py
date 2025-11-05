import tilelang
import tilelang.language as T
import torch
import tilelang.testing
import tvm
from tvm.script.ir_builder.base import IRBuilderFrame
from tvm.tir.expr import IntImm, Var


def test_argument():

    @T.prim_func
    def test_argument(
        t_1: T.bool,
        t_2: T.short,
        t_3: T.int,
        t_4: T.long,
        t_5: T.half,
        t_6: T.float,
        t_7: T.long,
        t_8: T.int8,
        t_9: T.int16,
        t_10: T.int32,
        t_11: T.int64,
        t_12: T.uint8,
        t_13: T.uint16,
        t_14: T.uint32,
        t_15: T.uint64,
        t_16: T.float8_e4m3fn,
        t_17: T.float8_e4m3fnuz,
        t_18: T.float8_e5m2,
        t_19: T.float8_e5m2fnuz,
        t_20: T.float8_e8m0fnu,
        t_21: T.float16,
        t_22: T.bfloat16,
        t_23: T.float32,
        t_24: T.float64,
    ):
        pass


def test_expr():
    from tilelang.language.v2.dtypes import _all_dtypes
    errors = []
    for name in _all_dtypes:
        dtype = getattr(T, name)
        assert isinstance(dtype, tvm.DataType), f"{dtype} is not tvm.DataType"
        try:
            dtype(1.0)
            dtype()
        except TypeError:
            pass
        except Exception:
            errors.append(name)
    assert not errors


# def test_var_decl_sugar():

#     @T.prim_func
#     def test_var_decl_sugar():
#         with T.Kernel(128, 128) as (bx, by):
#             var_1: T.bool = 1.0
#             var_2: T.short = 1.0
#             var_3: T.int = 1.0
#             var_4: T.long = 1.0
#             var_5: T.half = 1.0
#             var_6: T.float = 1.0
#             var_7: T.long = 1.0
#             var_8: T.int8 = 1.0
#             var_9: T.int16 = 1.0
#             var_10: T.int32 = 1.0
#             var_11: T.int64 = 1.0
#             var_12: T.uint8 = 1.0
#             var_13: T.uint16 = 1.0
#             var_14: T.uint32 = 1.0
#             var_15: T.uint64 = 1.0
#             var_16: T.float8_e4m3fn = 1.0
#             var_17: T.float8_e4m3fnuz = 1.0
#             var_18: T.float8_e5m2 = 1.0
#             var_19: T.float8_e5m2fnuz = 1.0
#             var_20: T.float8_e8m0fnu = 1.0
#             var_21: T.float16 = 1.0
#             var_22: T.bfloat16 = 1.0
#             var_23: T.float32 = 1.0
#             var_24: T.float64 = 1.0
#             var_1: T.bool = var_1
#             var_2: T.short = var_2
#             var_3: T.int = var_3
#             var_4: T.long = var_4
#             var_5: T.half = var_5
#             var_6: T.float = var_6
#             var_7: T.long = var_7
#             var_8: T.int8 = var_8
#             var_9: T.int16 = var_9
#             var_10: T.int32 = var_10
#             var_11: T.int64 = var_11
#             var_12: T.uint8 = var_12
#             var_13: T.uint16 = var_13
#             var_14: T.uint32 = var_14
#             var_15: T.uint64 = var_15
#             var_16: T.float8_e4m3fn = var_16
#             var_17: T.float8_e4m3fnuz = var_17
#             var_18: T.float8_e5m2 = var_18
#             var_19: T.float8_e5m2fnuz = var_19
#             var_20: T.float8_e8m0fnu = var_20
#             var_21: T.float16 = var_21
#             var_22: T.bfloat16 = var_22
#             var_23: T.float32 = var_23
#             var_24: T.float64 = var_24

#     s = test_var_decl_sugar.script()
#     for i in range(1, 25):
#         assert f'var_{i}_1' in s
#         assert 'tl.local_var_init' in s


def test_dtype_str_repr():

    @T.prim_func
    def test_str_repr():
        buf_1 = T.alloc_buffer((1,), dtype=T.bool, scope='shared')  # noqa F841
        buf_2 = T.alloc_buffer((1,), dtype=T.short, scope='shared')  # noqa F841
        buf_3 = T.alloc_buffer((1,), dtype=T.int, scope='shared')  # noqa F841
        buf_4 = T.alloc_buffer((1,), dtype=T.long, scope='shared')  # noqa F841
        buf_5 = T.alloc_buffer((1,), dtype=T.half, scope='shared')  # noqa F841
        buf_6 = T.alloc_buffer((1,), dtype=T.float, scope='shared')  # noqa F841
        buf_7 = T.alloc_buffer((1,), dtype=T.long, scope='shared')  # noqa F841
        buf_8 = T.alloc_buffer((1,), dtype=T.int8, scope='shared')  # noqa F841
        buf_9 = T.alloc_buffer((1,), dtype=T.int16, scope='shared')  # noqa F841
        buf_10 = T.alloc_buffer((1,), dtype=T.int32, scope='shared')  # noqa F841
        buf_11 = T.alloc_buffer((1,), dtype=T.int64, scope='shared')  # noqa F841
        buf_12 = T.alloc_buffer((1,), dtype=T.uint8, scope='shared')  # noqa F841
        buf_13 = T.alloc_buffer((1,), dtype=T.uint16, scope='shared')  # noqa F841
        buf_14 = T.alloc_buffer((1,), dtype=T.uint32, scope='shared')  # noqa F841
        buf_15 = T.alloc_buffer((1,), dtype=T.uint64, scope='shared')  # noqa F841
        buf_16 = T.alloc_buffer((1,), dtype=T.float8_e4m3fn, scope='shared')  # noqa F841
        buf_17 = T.alloc_buffer((1,), dtype=T.float8_e4m3fnuz, scope='shared')  # noqa F841
        buf_18 = T.alloc_buffer((1,), dtype=T.float8_e5m2, scope='shared')  # noqa F841
        buf_19 = T.alloc_buffer((1,), dtype=T.float8_e5m2fnuz, scope='shared')  # noqa F841
        buf_20 = T.alloc_buffer((1,), dtype=T.float8_e8m0fnu, scope='shared')  # noqa F841
        buf_21 = T.alloc_buffer((1,), dtype=T.float16, scope='shared')  # noqa F841
        buf_22 = T.alloc_buffer((1,), dtype=T.bfloat16, scope='shared')  # noqa F841
        buf_23 = T.alloc_buffer((1,), dtype=T.float32, scope='shared')  # noqa F841
        buf_24 = T.alloc_buffer((1,), dtype=T.float64, scope='shared')  # noqa F841


def test_torch_eq():
    dtypes = [
        T.bool,
        T.short,
        T.int,
        T.long,
        T.half,
        T.float,
        T.long,
        T.int8,
        T.int16,
        T.int32,
        T.int64,
        T.uint8,
        T.uint16,
        T.uint32,
        T.uint64,
        T.float8_e4m3fn,
        T.float8_e4m3fnuz,
        T.float8_e5m2,
        T.float8_e5m2fnuz,
        T.float8_e8m0fnu,
        T.float16,
        T.bfloat16,
        T.float32,
        T.float64,
    ]
    torch_dtypes = [
        torch.bool,
        torch.short,
        torch.int,
        torch.long,
        torch.half,
        torch.float,
        torch.long,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
        torch.float8_e8m0fnu,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ]
    for a, b in zip(dtypes, torch_dtypes):
        assert a == b, f"{a} and {b} are not equal"
        assert T.dtype(b) == a, "dtype conversion error"


def test_var_assign():

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def test_var_assign(A: T.Tensor((2,), T.int32)):
        with T.Kernel(1) as _:
            a: T.int32 = 1
            b: T.int32 = a
            a = 2
            d: T.int32 = a
            A[0] = b
            A[1] = d

    res = test_var_assign()()
    assert res[0] == 1
    assert res[1] == 2


def test_marco_return():

    @T.macro
    def macro_return_constant():
        return 0

    @T.macro
    def macro_return_frame(x):
        return T.alloc_var(T.float32, init=x)

    @T.macro
    def macro_return_expr(x):
        y = x + 1.0
        return y

    @T.macro
    def macro_apply_func(x, fn):
        return fn(x)

    def check(x, ty):
        assert isinstance(x, ty)

    @T.prim_func
    def test_macro_return():
        with T.Kernel(1) as _:
            a = macro_return_constant()
            b = macro_return_frame(3.0)
            c = macro_return_expr(4.0)
            d = macro_apply_func(5.0, lambda x: x * 2.0)
            check(a, (int, float, T.PrimExpr))
            check(b, T.PrimExpr)
            check(c, T.PrimExpr)
            check(d, T.PrimExpr)


def test_prim_func_generator():

    @T.prim_func(generator=True)
    def prim_func_gen(
            A=T.Tensor((128,), T.float32),  # noqa: B008
            B=T.Tensor((128,), T.float32),  # noqa: B008
    ):
        with T.Kernel(128) as (tx,):
            T.copy(A[tx], B[tx])

    prim_func_gen()

    @T.prim_func
    def foo() -> T.Tensor((128,), T.float32):
        pass

    assert isinstance(foo, T.PrimFunc)


def test_serial_for_with_step():

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def test_stepped_serial(A: T.Tensor((10,), T.int32)):
        with T.Kernel(1) as _:
            for i in range(0, 10, 2):
                T.device_assert(0 <= i < 10 and i % 2 == 0, "i out of range")
                A[i] = 1.0
            for i in range(1, 10, 2):
                T.device_assert(1 <= i < 10 and i % 2 == 1, "i out of range")
                A[i] = 2.0

    ker = test_stepped_serial()
    res = ker()
    ref = torch.tensor([1, 2, 1, 2, 1, 2, 1, 2, 1, 2], dtype=torch.int32, device='cuda')
    assert torch.all(res == ref), f"Expected {ref}, but got {res}"

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def test_serial_step_neg(A: T.Tensor((10,), T.int32)):
        with T.Kernel(1) as _:
            for i in range(10, 0, -1):
                T.device_assert(0 < i <= 10, "i out of range")
                A[10 - i] = i

    ker = test_serial_step_neg()
    res = ker()
    ref = torch.tensor([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=torch.int32, device='cuda')
    assert torch.all(res == ref), f"Expected {ref}, but got {res}"

    assert isinstance(T.serial(1, 10, 1), IRBuilderFrame)
    assert isinstance(T.serial(1, 10, IntImm('int32', 1)), IRBuilderFrame)
    assert not isinstance(T.serial(1, 10, Var('tmp', 'int32')), IRBuilderFrame)
    assert not isinstance(T.serial(10, -1, -1), IRBuilderFrame)


def test_swap_logic():

    @tilelang.jit
    @T.prim_func
    def swap_var(A: T.Tensor[(2,), T.float32]):
        with T.Kernel(1, threads=1) as _:
            a = T.alloc_var(T.float32, A[0])
            b = T.alloc_var(T.float32, A[1])
            a, b = b, a
            A[0], A[1] = a, b

    @tilelang.jit
    @T.prim_func
    def swap_idx(A: T.Tensor[(2,), T.float32]):
        with T.Kernel(1, threads=1) as _:
            A[0], A[1] = A[1], A[0]

    k_swap_var = swap_var()
    data = torch.tensor([1.0, 2.0], dtype=torch.float32).cuda()
    k_swap_var(data)
    ref = torch.tensor([2.0, 1.0], dtype=torch.float32).cuda()
    torch.testing.assert_close(data, ref)

    k_swap_idx = swap_idx()
    data = torch.tensor([1.0, 2.0], dtype=torch.float32).cuda()
    k_swap_idx(data)
    ref = torch.tensor([2.0, 1.0], dtype=torch.float32).cuda()
    torch.testing.assert_close(data, ref)


if __name__ == '__main__':
    tilelang.testing.main()

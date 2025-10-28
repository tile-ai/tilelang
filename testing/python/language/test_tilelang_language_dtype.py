import tilelang
import tilelang.language as T
import torch
import tilelang.testing
import tvm

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
        except TypeError as e:
            pass
        except Exception as e:
            errors.append(name)
    assert not errors


def test_var_decl_sugar():
    @T.prim_func
    def test_var_decl_sugar():
        with T.Kernel(128, 128) as (bx, by):
            var_1: T.bool = 1.0
            var_2: T.short = 1.0
            var_3: T.int = 1.0
            var_4: T.long = 1.0
            var_5: T.half = 1.0
            var_6: T.float = 1.0
            var_7: T.long = 1.0
            var_8: T.int8 = 1.0
            var_9: T.int16 = 1.0
            var_10: T.int32 = 1.0
            var_11: T.int64 = 1.0
            var_12: T.uint8 = 1.0
            var_13: T.uint16 = 1.0
            var_14: T.uint32 = 1.0
            var_15: T.uint64 = 1.0
            var_16: T.float8_e4m3fn = 1.0
            var_17: T.float8_e4m3fnuz = 1.0
            var_18: T.float8_e5m2 = 1.0
            var_19: T.float8_e5m2fnuz = 1.0
            var_20: T.float8_e8m0fnu = 1.0
            var_21: T.float16 = 1.0
            var_22: T.bfloat16 = 1.0
            var_23: T.float32 = 1.0
            var_24: T.float64 = 1.0
            var_1: T.bool = var_1
            var_2: T.short = var_2
            var_3: T.int = var_3
            var_4: T.long = var_4
            var_5: T.half = var_5
            var_6: T.float = var_6
            var_7: T.long = var_7
            var_8: T.int8 = var_8
            var_9: T.int16 = var_9
            var_10: T.int32 = var_10
            var_11: T.int64 = var_11
            var_12: T.uint8 = var_12
            var_13: T.uint16 = var_13
            var_14: T.uint32 = var_14
            var_15: T.uint64 = var_15
            var_16: T.float8_e4m3fn = var_16
            var_17: T.float8_e4m3fnuz = var_17
            var_18: T.float8_e5m2 = var_18
            var_19: T.float8_e5m2fnuz = var_19
            var_20: T.float8_e8m0fnu = var_20
            var_21: T.float16 = var_21
            var_22: T.bfloat16 = var_22
            var_23: T.float32 = var_23
            var_24: T.float64 = var_24

    s = test_var_decl_sugar.script()
    for i in range(1, 25):
        assert f'var_{i}_1' in s
        assert f'tl.local_var_init' in s

def test_dtype_str_repr():
    @T.prim_func
    def test_str_repr():
        buf_1 = T.alloc_buffer((1,), dtype=T.bool, scope='shared')
        buf_2 = T.alloc_buffer((1,), dtype=T.short, scope='shared')
        buf_3 = T.alloc_buffer((1,), dtype=T.int, scope='shared')
        buf_4 = T.alloc_buffer((1,), dtype=T.long, scope='shared')
        buf_5 = T.alloc_buffer((1,), dtype=T.half, scope='shared')
        buf_6 = T.alloc_buffer((1,), dtype=T.float, scope='shared')
        buf_7 = T.alloc_buffer((1,), dtype=T.long, scope='shared')
        buf_8 = T.alloc_buffer((1,), dtype=T.int8, scope='shared')
        buf_9 = T.alloc_buffer((1,), dtype=T.int16, scope='shared')
        buf_10 = T.alloc_buffer((1,), dtype=T.int32, scope='shared')
        buf_11 = T.alloc_buffer((1,), dtype=T.int64, scope='shared')
        buf_12 = T.alloc_buffer((1,), dtype=T.uint8, scope='shared')
        buf_13 = T.alloc_buffer((1,), dtype=T.uint16, scope='shared')
        buf_14 = T.alloc_buffer((1,), dtype=T.uint32, scope='shared')
        buf_15 = T.alloc_buffer((1,), dtype=T.uint64, scope='shared')
        buf_16 = T.alloc_buffer((1,), dtype=T.float8_e4m3fn, scope='shared')
        buf_17 = T.alloc_buffer((1,), dtype=T.float8_e4m3fnuz, scope='shared')
        buf_18 = T.alloc_buffer((1,), dtype=T.float8_e5m2, scope='shared')
        buf_19 = T.alloc_buffer((1,), dtype=T.float8_e5m2fnuz, scope='shared')
        buf_20 = T.alloc_buffer((1,), dtype=T.float8_e8m0fnu, scope='shared')
        buf_21 = T.alloc_buffer((1,), dtype=T.float16, scope='shared')
        buf_22 = T.alloc_buffer((1,), dtype=T.bfloat16, scope='shared')
        buf_23 = T.alloc_buffer((1,), dtype=T.float32, scope='shared')
        buf_24 = T.alloc_buffer((1,), dtype=T.float64, scope='shared')

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


if __name__ == '__main__':
    tilelang.testing.main()

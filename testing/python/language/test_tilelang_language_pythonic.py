import builtins

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tvm import tirx


def stores(func):
    result = []
    tirx.stmt_functor.post_order_visit(func.body, lambda node: result.append(node) if isinstance(node, tirx.BufferStore) else None)
    return result


def test_comprehensions():
    @T.prim_func
    def main(B: T.Tensor((4,), "int32")):
        table = [((j >> 4) & 7) | ((j >> 7) << 3) for j in range(256)]
        pairs = [(i, j) for i in range(3) for j in range(i) if j != 1]
        mapping = {i: i * i for i in range(4)}
        unique = {i % 2 for i in range(4)}
        B[0] = table[144]
        B[1] = len(pairs)
        B[2] = mapping[3]
        B[3] = len(unique)

    assert [int(s.value) for s in stores(main)] == [9, 2, 9, 2]


@pytest.mark.parametrize("iterable", [[1, 3, 5], (1, 3, 5), zip((0, 1, 2), (1, 3, 5)), enumerate((1, 3, 5))])
def test_python_iterables(iterable):
    @T.prim_func
    def main(B: T.Tensor((8,), "int32")):
        for value in iterable:
            if isinstance(value, tuple):
                B[value[0]] = value[1]
            else:
                B[value] = value

    assert [int(s.value) for s in stores(main)] == [1, 3, 5]


def test_unpack_and_generator():
    @T.prim_func
    def main(B: T.Tensor((3,), "int32")):
        for i, value in enumerate(x * 2 for x in range(3)):
            B[i] = value

    assert [int(s.value) for s in stores(main)] == [0, 2, 4]


@pytest.mark.parametrize("loop", [range, builtins.range, T.serial])
def test_dynamic_serial(loop):
    n = T.dynamic("n")

    @T.prim_func
    def main(B: T.Tensor((n,), "int32")):
        for i in loop(n):
            B[i] = i

    loops = []
    tirx.stmt_functor.post_order_visit(main.body, lambda node: loops.append(node) if isinstance(node, tirx.For) else None)
    assert len(loops) == 1
    assert loops[0].extent.same_as(main.buffer_map[main.params[0]].shape[0])


def test_shadowed_range():
    def range(n):
        return [n, n + 1]

    @T.prim_func
    def main(B: T.Tensor((2,), "int32")):
        for i in range(0):
            B[i] = i + 10

    assert [int(s.value) for s in stores(main)] == [10, 11]


def test_static_break_continue():
    @T.prim_func
    def main(B: T.Tensor((8,), "int32")):
        for i in [0, 1, 2, 3, 4]:
            if i == 1:
                continue
            if i == 3:
                break
            B[i] = i + 10
        B[7] = 99

    assert [int(s.value) for s in stores(main)] == [10, 12, 99]


def test_nested_static_loops():
    @T.prim_func
    def main(B: T.Tensor((4,), "int32")):
        for i in [0, 1]:
            for j in [0, 1, 2]:
                if j == 1:
                    break
                B[i] = i + j

    assert [int(s.value) for s in stores(main)] == [0, 1]


@pytest.mark.parametrize("control", ["break", "continue"])
def test_runtime_break_in_static_loop(control):
    with pytest.raises(NotImplementedError, match="expanded loop has no runtime control-flow target"):

        @T.prim_func
        def main(B: T.Tensor((4,), "int32")):
            for i in [0, 1]:
                if B[0] > 0:
                    if control == "break":
                        break
                    else:
                        continue
                B[i] = i


def test_comprehension_scope():
    @T.prim_func
    def main(B: T.Tensor((4,), "int32")):
        for i in range(4):
            B[i] = i
        values = [i for i in range(4)]
        B[0] = values[3]

    assert int(stores(main)[-1].value) == 3


@tilelang.testing.requires_cuda
def test_pythonic_kernel():
    @T.prim_func
    def main(B: T.Tensor((256,), "int32")):
        table = [((j >> 4) & 7) | ((j >> 7) << 3) for j in range(256)]
        with T.Kernel(1, threads=32):
            for i in range(8):
                for offset, value in enumerate(table[:4]):
                    B[i * 32 + T.get_thread_binding(0)] = i * 10 + value + offset

    kernel = tilelang.compile(main, out_idx=[0], target="cuda")
    actual = kernel()
    expected = torch.arange(8, device="cuda", dtype=torch.int32).repeat_interleave(32) * 10 + 3
    torch.testing.assert_close(actual, expected)


def test_python_generator_resumes_after_break():
    values = (i for i in range(4))

    @T.prim_func
    def main(B: T.Tensor((4,), "int32")):
        for i in values:
            B[i] = i
            break
        for i in values:
            B[i] = i

    assert [int(s.value) for s in stores(main)] == [0, 1, 2, 3]


def test_base_builder_control_flow():
    from tilelang.language.eager.ast import BaseBuilder, mutate

    class TestBuilder(BaseBuilder):
        def set_fileline(self, filename, lineno, name):
            pass

    def function():
        values = []
        for i in range(5):
            if i == 1:
                continue
            if i == 3:
                break
            values.append(i)
        i = 0
        while i < 5:
            i += 1
            if i == 1:
                continue
            if i == 3:
                break
            values.append(i)
        return values

    assert mutate(function).gen(TestBuilder())() == function() == [0, 2, 2]


def test_serial_inside_python_loop():
    @T.prim_func
    def main(B: T.Tensor((4,), "int32")):
        for offset in [0, 1]:
            for i in range(4):
                if i == 2:
                    break
                B[i] = offset
            B[3] = offset

    loops = []
    tirx.stmt_functor.post_order_visit(main.body, lambda node: loops.append(node) if isinstance(node, tirx.For) else None)
    assert len(loops) == 2
    assert all("break" in loop.script() for loop in loops)


def test_empty_iterables():
    @T.prim_func
    def main(B: T.Tensor((1,), "int32")):
        for _i in []:
            B[0] = 9
        for _i in range(0):
            B[0] = 8
        B[0] = 7

    assert int(stores(main)[-1].value) == 7


def test_comprehension_keeps_outer_value():
    @T.prim_func
    def main(B: T.Tensor((2,), "int32")):
        i = 99
        values = [[i + j for j in range(i)] for i in range(3)]
        B[0] = i
        B[1] = values[2][1]

    assert [int(s.value) for s in stores(main)] == [99, 3]


def test_runtime_comprehension_filter():
    with pytest.raises(TypeError, match="Comprehension filters must be evaluable"):

        @T.prim_func
        def main(A: T.Tensor((1,), "int32"), B: T.Tensor((1,), "int32")):
            values = [i for i in range(4) if A[0] > i]
            B[0] = len(values)


def test_symbolic_comprehension_elements():
    @T.prim_func
    def main(A: T.Tensor((1,), "int32"), B: T.Tensor((3,), "int32")):
        values = [A[0] + i for i in range(3)]
        for i, value in enumerate(values):
            B[i] = value

    assert len(stores(main)) == 3


def test_comprehension_unwraps_mutable_values():
    @T.prim_func
    def main(B: T.Tensor((1,), "int32")):
        value = T.alloc_var("int32", init=3)
        values = [x + 1 for x in [value]]
        B[0] = values[0]

    assert len(stores(main)) >= 1


@pytest.mark.parametrize("stop_early", [False, True])
def test_python_loop_preserves_value_scope(stop_early):
    @T.prim_func
    def main(A: T.Tensor((2,), "int32"), B: T.Tensor((1,), "int32")):
        for i in [0, 1]:
            value = A[i] + 1
            if stop_early:
                break
        B[0] = value

    defined = [buffer.data for buffer in main.buffer_map.values()]
    assert not tirx.analysis.undefined_vars(main.body, defined)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("loop", [T.serial, range])
@pytest.mark.parametrize("control", ["break", "continue"])
def test_device_loop_control_inside_python_loop(loop, control):
    @T.prim_func
    def main(A: T.Tensor((1,), "int32"), B: T.Tensor((2,), "int32")):
        with T.Kernel(1, threads=1):
            for offset in [0, 1]:
                total = T.alloc_var("int32", init=0)
                for i in loop(8):
                    if i == A[0]:
                        if control == "break":
                            break
                        else:
                            continue
                    total += i
                B[offset] = total

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    for stop in [0, 3, 7]:
        a = torch.tensor([stop], dtype=torch.int32, device="cuda")
        expected = sum(range(stop)) if control == "break" else sum(range(8)) - stop
        assert kernel(a).cpu().tolist() == [expected, expected]


def test_tir_var_is_not_a_loop_iterable():
    # Var has an __iter__ shim for single-binding unpacking; a bare symbolic
    # extent in a for header must not silently expand once with `i` aliased to it.
    n = T.dynamic("n")

    with pytest.raises(TypeError, match="not iterable"):

        @T.prim_func
        def main(B: T.Tensor((n,), "int32")):
            for i in n:
                B[i] = i


def test_buffer_is_not_a_loop_iterable():
    # Buffer.__getitem__ never raises IndexError, so iterating one would never end.
    with pytest.raises(TypeError, match="not iterable"):

        @T.prim_func
        def main(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
            for x in A:
                B[0] = x


def test_non_iterable_loop_target_keeps_diagnostic():
    with pytest.raises(TypeError, match="Invalid for loop"):

        @T.prim_func
        def main(B: T.Tensor((4,), "int32")):
            for i in 4:
                B[i] = i


if __name__ == "__main__":
    tilelang.testing.main()

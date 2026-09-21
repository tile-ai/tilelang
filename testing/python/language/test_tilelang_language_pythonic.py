import builtins

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tvm import IRModule, arith, tirx


def stores(func):
    result = []
    tirx.stmt_functor.post_order_visit(func.body, lambda node: result.append(node) if isinstance(node, tirx.BufferStore) else None)
    return result


@pytest.fixture(params=["ir", pytest.param("cuda", marks=tilelang.testing.requires_cuda.marks())])
def check_kernel(request):
    """Retain CPU-only frontend coverage and execute the same PrimFunc on CUDA."""

    def check(func, expected, *inputs, initial=None):
        if request.param == "ir":
            return
        expected = torch.tensor(expected, device="cuda", dtype=torch.int32)
        # Detect missing stores and writes to positions that should remain untouched.
        output = torch.full_like(expected, -9999) if initial is None else torch.tensor(initial, device="cuda", dtype=torch.int32)
        inputs = [torch.tensor(value, device="cuda", dtype=torch.int32) for value in inputs]
        kernel = tilelang.compile(func, target="cuda")
        kernel(*inputs, output)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    return check


@pytest.fixture
def forbid_device_iteration(monkeypatch):
    # Fail promptly if a guard regresses, instead of hanging on Buffer.__getitem__.
    def unexpected_iteration(self):
        pytest.fail("Device values must be rejected before acquiring a Python iterator")

    monkeypatch.setattr(tirx.Buffer, "__iter__", unexpected_iteration, raising=False)
    monkeypatch.setattr(tirx.Var, "__iter__", unexpected_iteration)


def test_comprehensions(check_kernel):
    @T.prim_func
    def main(B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=1):
            table = [((j >> 4) & 7) | ((j >> 7) << 3) for j in range(256)]
            pairs = [(i, j) for i in range(3) for j in range(i) if j != 1]
            mapping = {i: i * i for i in range(4)}
            unique = {i % 2 for i in range(4)}
            B[0] = table[144]
            B[1] = len(pairs)
            B[2] = mapping[3]
            B[3] = len(unique)

    assert [int(s.value) for s in stores(main)] == [9, 2, 9, 2]
    check_kernel(main, [9, 2, 9, 2])


@pytest.mark.parametrize("bounds, start", [((4,), 0), ((1, 7, 2), 3), ((5, -1, -2), -2), ((0,), 0)])
def test_enumerated_range_comprehension(bounds, start, check_kernel):
    @T.prim_func
    def main(B: T.Tensor((8,), "int32")):
        with T.Kernel(1, threads=1):
            values = [i + x for i, x in enumerate(range(*bounds), start=start)]
            for i, value in enumerate(values):
                B[i] = value
            B[7] = len(values)

    expected = [i + x for i, x in enumerate(range(*bounds), start=start)]
    assert [int(s.value) for s in stores(main)] == expected + [len(expected)]
    check_kernel(main, expected + [-9999] * (7 - len(expected)) + [len(expected)])


@pytest.mark.parametrize(
    "iterable_factory",
    [lambda: [1, 3, 5], lambda: (1, 3, 5), lambda: zip((0, 1, 2), (1, 3, 5)), lambda: enumerate((1, 3, 5))],
    ids=["list", "tuple", "zip", "enumerate"],
)
def test_python_iterables(iterable_factory, check_kernel):
    iterable = iterable_factory()

    @T.prim_func
    def main(B: T.Tensor((8,), "int32")):
        with T.Kernel(1, threads=1):
            for value in iterable:
                if isinstance(value, tuple):
                    B[value[0]] = value[1]
                else:
                    B[value] = value

    assert [int(s.value) for s in stores(main)] == [1, 3, 5]
    expected = [-9999] * 8
    for value in iterable_factory():
        if isinstance(value, tuple):
            expected[value[0]] = value[1]
        else:
            expected[value] = value
    check_kernel(main, expected)


def test_unpack_and_generator(check_kernel):
    @T.prim_func
    def main(B: T.Tensor((3,), "int32")):
        with T.Kernel(1, threads=1):
            for i, value in enumerate(x * 2 for x in range(3)):
                B[i] = value

    assert [int(s.value) for s in stores(main)] == [0, 2, 4]
    check_kernel(main, [0, 2, 4])


@pytest.mark.parametrize("loop", [range, builtins.range, T.serial])
def test_dynamic_serial(loop, check_kernel):
    n = T.dynamic("n")

    @T.prim_func
    def main(B: T.Tensor((n,), "int32")):
        with T.Kernel(1, threads=1):
            for i in loop(n):
                B[i] = i

    loops = []
    tirx.stmt_functor.post_order_visit(main.body, lambda node: loops.append(node) if isinstance(node, tirx.For) else None)
    loops = [loop for loop in loops if loop.thread_binding is None]
    assert len(loops) == 1
    assert loops[0].extent.same_as(main.buffer_map[main.params[0]].shape[0])
    for size in [1, 7, 33]:
        check_kernel(main, list(range(size)))


def test_shadowed_range(check_kernel):
    def range(n):
        return [n, n + 1]

    @T.prim_func
    def main(B: T.Tensor((2,), "int32")):
        with T.Kernel(1, threads=1):
            for i in range(0):
                B[i] = i + 10

    assert [int(s.value) for s in stores(main)] == [10, 11]
    check_kernel(main, [10, 11])


def test_static_break_continue(check_kernel):
    @T.prim_func
    def main(B: T.Tensor((8,), "int32")):
        with T.Kernel(1, threads=1):
            for i in [0, 1, 2, 3, 4]:
                if i == 1:
                    continue
                if i == 3:
                    break
                B[i] = i + 10
            B[7] = 99

    assert [int(s.value) for s in stores(main)] == [10, 12, 99]
    check_kernel(main, [10, -9999, 12, -9999, -9999, -9999, -9999, 99])


def test_nested_static_loops(check_kernel):
    @T.prim_func
    def main(B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=1):
            for i in [0, 1]:
                for j in [0, 1, 2]:
                    if j == 1:
                        break
                    B[i] = i + j

    assert [int(s.value) for s in stores(main)] == [0, 1]
    check_kernel(main, [0, 1, -9999, -9999])


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


def test_comprehension_scope(check_kernel):
    @T.prim_func
    def main(B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=1):
            for i in range(4):
                B[i] = i
            values = [i for i in range(4)]
            B[0] = values[3]

    assert int(stores(main)[-1].value) == 3
    check_kernel(main, [3, 1, 2, 3])


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


def test_python_generator_resumes_after_break(check_kernel):
    consumed = []

    def source():
        for i in range(4):
            consumed.append(i)
            yield i

    values = source()

    @T.prim_func
    def main(B: T.Tensor((6,), "int32")):
        with T.Kernel(1, threads=1):
            lazy = (i for i in values)
            B[4] = len(consumed)
            for i in lazy:
                B[i] = i
                break
            B[5] = len(consumed)
            for i in lazy:
                B[i] = i

    assert [int(s.value) for s in stores(main)] == [0, 0, 1, 1, 2, 3]
    assert consumed == [0, 1, 2, 3]
    check_kernel(main, [0, 1, 2, 3, 0, 1])
    assert consumed == [0, 1, 2, 3]


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


def test_serial_inside_python_loop(check_kernel):
    @T.prim_func
    def main(B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=1):
            for offset in [0, 1]:
                for i in range(4):
                    if i == 2:
                        break
                    B[i] = offset
                B[3] = offset

    loops = []
    tirx.stmt_functor.post_order_visit(main.body, lambda node: loops.append(node) if isinstance(node, tirx.For) else None)
    loops = [loop for loop in loops if loop.thread_binding is None]
    assert len(loops) == 2
    assert all("break" in loop.script() for loop in loops)
    check_kernel(main, [1, 1, -9999, 1])


def test_empty_iterables(check_kernel):
    @T.prim_func
    def main(B: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            for _i in []:
                B[0] = 9
            for _i in range(0):
                B[0] = 8
            B[0] = 7

    assert int(stores(main)[-1].value) == 7
    check_kernel(main, [7])


def test_comprehension_keeps_outer_value(check_kernel):
    @T.prim_func
    def main(B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=1):
            i = 99
            values = [[i + j for j in range(i)] for i in range(3)]
            B[0] = i
            B[1] = values[2][1]
            items = [1, 2]
            values = [items + 1 for items in items]
            B[2] = items[0]
            B[3] = values[1]

    assert [int(s.value) for s in stores(main)] == [99, 3, 1, 3]
    check_kernel(main, [99, 3, 1, 3])


def test_comprehension_target_does_not_leak():
    with pytest.raises(NameError, match="item"):

        @T.prim_func
        def main(B: T.Tensor((1,), "int32")):
            values = [item for item in range(3)]
            B[0] = item + len(values)  # noqa: F821


@pytest.mark.parametrize("form", ["list", "nested", "generator"])
def test_runtime_comprehension_filter(form):
    with pytest.raises(TypeError, match="Comprehension filters must be evaluable"):

        @T.prim_func
        def main(A: T.Tensor((1,), "int32"), B: T.Tensor((1,), "int32")):
            if form == "list":
                values = [i for i in range(4) if A[0] > i]
            elif form == "nested":
                values = [i + j for i in range(2) for j in range(2) if A[0] > j]
            else:
                values = list(i for i in range(4) if A[0] > i)
            B[0] = len(values)


@pytest.mark.parametrize("condition", [T.int32(0), T.int32(1)])
def test_constant_tir_comprehension_filter(condition, check_kernel):
    @T.prim_func
    def main(B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=1):
            values = [i for i in range(3) if condition]
            for i, value in enumerate(values):
                B[i] = value
            B[3] = len(values)

    expected = [0, 1, 2, 3] if int(condition) else [0]
    assert [int(s.value) for s in stores(main)] == expected
    check_kernel(main, [0, 1, 2, 3] if int(condition) else [-9999, -9999, -9999, 0])


def test_symbolic_comprehension_elements(check_kernel):
    @T.prim_func
    def main(A: T.Tensor((1,), "int32"), B: T.Tensor((3,), "int32")):
        with T.Kernel(1, threads=1):
            values = [A[0] + i + x for i, x in enumerate(range(0, 6, 2))]
            for i, value in enumerate(values):
                B[i] = value

    inlined = tilelang.transform.LetInline()(IRModule.from_expr(main))["main"]
    results = stores(inlined)
    assert len(results) == 3
    a = inlined.buffer_map[inlined.params[0]]
    analyzer = arith.Analyzer()
    assert all(analyzer.can_prove_equal(store.value, a[0] + 3 * i) for i, store in enumerate(results))
    for value in [-7, 13]:
        check_kernel(main, [value + 3 * i for i in range(3)], [value])


def test_comprehensions_accept_shape_and_buffer_containers(check_kernel):
    n = T.dynamic("n")

    @T.prim_func
    def main(A: T.Tensor((n,), "int32"), B: T.Tensor((n,), "int32"), C: T.Tensor((3,), "int32")):
        with T.Kernel(1, threads=1):
            dimensions = [dim for dim in A.shape]
            values = dimensions + [buf[0] for buf in (A, B)]
            for i, value in enumerate(values):
                C[i] = value

    inlined = tilelang.transform.LetInline()(IRModule.from_expr(main))["main"]
    a, b = (inlined.buffer_map[p] for p in inlined.params[:2])
    expected = [a.shape[0], a[0], b[0]]
    results = stores(inlined)
    assert len(results) == len(expected)
    analyzer = arith.Analyzer()
    assert all(analyzer.can_prove_equal(store.value, value) for store, value in zip(results, expected))
    for size in [1, 7, 33]:
        check_kernel(main, [size, -3, 11], [-3] * size, [11] * size)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("bounds, start", [((4,), 0), ((1, 7, 2), 3), ((5, -1, -2), -2), ((0,), 0)])
def test_symbolic_comprehension_kernel(bounds, start):
    n = T.dynamic("n")

    @T.prim_func
    def main(A: T.Tensor((n,), "int32"), B: T.Tensor((n,), "int32")):
        with T.Kernel(T.ceildiv(n, 32), threads=32) as bx:
            index = bx * 32 + T.get_thread_binding(0)
            if index < n:
                values = [A[index] + i + x for i, x in enumerate(range(*bounds), start=start) if i != 1]
                B[index] = 0
                for value in values:
                    B[index] += value

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    for size in [1, 32, 65]:
        a = torch.arange(size, device="cuda", dtype=torch.int32)
        expected = torch.zeros_like(a)
        for i, x in enumerate(range(*bounds), start=start):
            if i != 1:
                expected += a + i + x
        torch.testing.assert_close(kernel(a), expected)


def test_comprehension_unwraps_mutable_values(check_kernel):
    @T.prim_func
    def main(B: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            value = T.alloc_var("int32", init=3)
            values = [x + 1 for x in [value]]
            B[0] = values[0]

    initialization, output = stores(main)
    assert int(initialization.value) == 3
    assert arith.Analyzer().can_prove_equal(output.value, initialization.buffer[0] + 1)
    check_kernel(main, [4])


@pytest.mark.parametrize("stop_early", [False, True])
def test_python_loop_preserves_value_scope(stop_early, check_kernel):
    @T.prim_func
    def main(A: T.Tensor((2,), "int32"), B: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            for i in [0, 1]:
                value = A[i] + 1
                if stop_early:
                    break
            B[0] = value

    defined = [buffer.data for buffer in main.buffer_map.values()]
    assert not tirx.analysis.undefined_vars(main.body, defined)
    check_kernel(main, [8 if stop_early else 14], [7, 13])


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


@pytest.mark.usefixtures("forbid_device_iteration")
def test_tir_var_is_not_a_loop_iterable():
    # Var has an __iter__ shim for single-binding unpacking; a bare symbolic
    # extent in a for header must not silently expand once with `i` aliased to it.
    n = T.dynamic("n")

    with pytest.raises(TypeError, match="not iterable"):

        @T.prim_func
        def main(B: T.Tensor((n,), "int32")):
            for i in n:
                B[i] = i


@pytest.mark.usefixtures("forbid_device_iteration")
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


@pytest.mark.parametrize(
    "consumer, args, kwargs",
    [
        (enumerate, (), {"iterable": [1, 3], "start": 2}),
        (zip, ([0, 1], [1, 3]), {"strict": True}),
        (map, (int, [1, 3]), {}),
        (filter, (None, [0, 1, 3]), {}),
        (iter, ([1, 3],), {}),
        (list, ((1, 3),), {}),
        (tuple, ([1, 3],), {}),
        (set, ([1, 3, 1],), {}),
        (frozenset, ([1, 3, 1],), {}),
        (dict, ([(1, 2), (3, 4)],), {}),
        (sorted, ([3, 1],), {"reverse": True}),
        (reversed, ([1, 3],), {}),
    ],
    ids=lambda value: value.__name__ if callable(value) else None,
)
def test_python_container_construction(consumer, args, kwargs, check_kernel):
    expected = [v for x in consumer(*args, **kwargs) for v in (x if isinstance(x, tuple) else (x,))]

    @T.prim_func
    def main(B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=1):
            values = [v for x in consumer(*args, **kwargs) for v in (x if isinstance(x, tuple) else (x,))]
            for i, value in enumerate(values):
                B[i] = value

    assert [int(s.value) for s in stores(main)] == expected
    device_nodes = []
    tirx.stmt_functor.post_order_visit(
        main.body,
        lambda node: (
            device_nodes.append(node)
            if isinstance(node, tirx.AllocBuffer) or (isinstance(node, tirx.For) and node.thread_binding is None)
            else None
        ),
    )
    assert not device_nodes, "Python container construction must not allocate storage or emit a device loop"
    check_kernel(main, expected + [-9999] * (4 - len(expected)))


@pytest.mark.usefixtures("forbid_device_iteration")
@pytest.mark.parametrize(
    "consumer, operands",
    [
        (enumerate, lambda n, a: ((n,), {})),
        (enumerate, lambda n, a: ((a,), {})),
        (enumerate, lambda n, a: ((), {"iterable": a})),
        (zip, lambda n, a: (([], a), {"strict": True})),
        (map, lambda n, a: ((int, [], a), {})),
        (filter, lambda n, a: ((None, a), {})),
        (iter, lambda n, a: ((n,), {})),
        (list, lambda n, a: ((n,), {})),
        (tuple, lambda n, a: ((a,), {})),
        (set, lambda n, a: ((a,), {})),
        (frozenset, lambda n, a: ((a,), {})),
        (dict, lambda n, a: ((a,), {})),
        (sorted, lambda n, a: ((a,), {})),
        (reversed, lambda n, a: ((a,), {})),
        (enumerate, lambda n, a: ((a[0],), {})),
        (enumerate, lambda n, a: ((a[:],), {})),
        (enumerate, lambda n, a: ((T.int32(4),), {})),
        (enumerate, lambda n, a: ((T.serial(4),), {})),
    ],
)
def test_python_builtin_iterable_checks(consumer, operands):
    # Cover argument conventions and rejected types, not their Cartesian product.
    # Calls do not consume lazy results, so a regression cannot loop over a Buffer.
    args, kwargs = operands(T.dynamic("n"), tirx.decl_buffer((4,), "int32"))
    with pytest.raises(TypeError, match=rf"{consumer.__name__}\(\):.*not .*iterable"):

        @T.prim_func
        def main():
            consumer(*args, **kwargs)


@pytest.mark.usefixtures("forbid_device_iteration")
def test_nested_python_iterable_calls(check_kernel):
    def make_main(invalid):
        @T.prim_func
        def main(A: T.Tensor((2,), "int32"), B: T.Tensor((2,), "int32")):
            with T.Kernel(1, threads=1):
                for i, (x, y) in enumerate(zip([1, 2], iter(A if invalid else [3, 4]))):
                    B[i] = x + y

        return main

    with pytest.raises(TypeError, match=r"iter\(\):.*not iterable"):
        make_main(True)
    main = make_main(False)
    assert [int(s.value) for s in stores(main)] == [4, 6]
    check_kernel(main, [4, 6], [7, 13])


@pytest.mark.parametrize("use_buffer", [False, True], ids=["symbolic-extent", "device-buffer"])
@pytest.mark.usefixtures("forbid_device_iteration")
def test_enumerate_rejects_device_loop_source(use_buffer):
    n = T.dynamic("n")
    with pytest.raises(TypeError, match=r"enumerate\(\):.*not iterable.*Use range\(n\)"):

        @T.prim_func
        def main(A: T.Tensor((n,), "int32"), B: T.Tensor((1,), "int32")):
            for i, _value in enumerate(A if use_buffer else n):
                B[0] = i


@pytest.mark.parametrize("form", ["loop", "list", "generator"])
def test_zip_strict_preserves_errors_and_builder_state(form, check_kernel):
    from tilelang.language.eager.builder import Builder
    from tilelang.language.kernel import KernelLaunchFrame
    from tvm.script.ir_builder import IRBuilder

    def make_main(mismatched):
        @T.prim_func
        def main(B: T.Tensor((1,), "int32")):
            with T.Kernel(1, threads=1):
                value = B[0] + 1
                pairs = zip([1], [2, 3] if mismatched else [2], strict=True)
                if form == "list":
                    pairs = [(x, y) for x, y in pairs]
                elif form == "generator":
                    pairs = ((x, y) for x, y in pairs)
                for x, y in pairs:
                    B[0] = value + x + y

        return main

    # zip fails on its second iteration, after entering the kernel and emitting IR.
    with pytest.raises(ValueError, match="zip.*longer"):
        make_main(True)

    assert Builder.current() is None
    assert not IRBuilder.is_in_scope()
    assert KernelLaunchFrame.Current() is None
    main = make_main(False)
    inlined = tilelang.transform.LetInline()(IRModule.from_expr(main))["main"]
    (output,) = stores(inlined)
    assert arith.Analyzer().can_prove_equal(output.value, output.buffer[0] + 4)
    check_kernel(main, [11], initial=[7])


@pytest.mark.parametrize("enumerated", [False, True], ids=["range", "enumerate-range"])
def test_comprehension_requires_compile_time_bounds(enumerated):
    n = T.dynamic("n")
    # Unlike a for statement, a comprehension cannot build a dynamic-length Python list.
    with pytest.raises(TypeError, match="cannot be interpreted as an integer"):

        @T.prim_func
        def main(B: T.Tensor((n,), "int32")):
            if enumerated:
                values = [x for _i, x in enumerate(range(n))]
            else:
                values = [x for x in range(n)]
            B[0] = len(values)


@pytest.mark.parametrize("value", [None, 4])
def test_comprehension_rejects_non_iterable_python_value(value):
    with pytest.raises(TypeError, match="not iterable"):

        @T.prim_func
        def main(B: T.Tensor((1,), "int32")):
            values = [x for x in value]
            B[0] = len(values)


@pytest.mark.parametrize("loop", [T.serial, T.unroll, T.Parallel])
def test_device_loop_frames_are_not_comprehension_iterables(loop, check_kernel):
    @T.prim_func
    def valid(B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=1):
            for i in loop(4):
                B[i] = i

    assert len(stores(valid)) == 1
    check_kernel(valid, [0, 1, 2, 3])
    with pytest.raises(TypeError, match="comprehension:.*not a compile-time Python iterable"):

        @T.prim_func
        def invalid(B: T.Tensor((1,), "int32")):
            values = [i for i in loop(4)]
            B[0] = len(values)


@pytest.mark.parametrize("form", ["list", "set", "dict", "generator", "nested"])
@pytest.mark.usefixtures("forbid_device_iteration")
def test_comprehension_iterable_checks(form):
    n = T.dynamic("n")
    with pytest.raises(TypeError, match="comprehension:.*not iterable"):

        @T.prim_func
        def main(A: T.Tensor((n,), "int32")):
            if form == "list":
                [x for x in n]
            elif form == "set":
                {x for x in n}
            elif form == "dict":
                {x: 0 for x in n}
            elif form == "generator":
                (x for x in n)
            else:
                [x for seq in [A] for x in seq]


def test_checked_builtin_argument_semantics(check_kernel):
    values = iter([2, 3, None])

    @T.prim_func
    def main(B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=1):
            for i, x in builtins.enumerate(iterable=list(iter(values.__next__, None)), start=1):
                B[i] = x
            B[3] = len(list()) + dict(func=3, self=4)["func"]
            for key in dict(func=3, self=4):
                B[0] = len(key)

    assert [int(s.value) for s in stores(main)] == [2, 3, 3, 4, 4]
    check_kernel(main, [4, 2, 3, 3])


def test_shadowed_iterable_builtin(check_kernel):
    class Callable:
        __hash__ = None

        def __call__(self, value):
            return [value]

        def __eq__(self, other):
            raise AssertionError("Dispatch must not compare user callables")

    enumerate = Callable()

    @T.prim_func
    def main(A: T.Tensor((1,), "int32"), B: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=1):
            for buf in enumerate(A):
                B[0] = buf[0]

    assert len(stores(main)) == 1
    check_kernel(main, [-7], [-7])


if __name__ == "__main__":
    tilelang.testing.main()

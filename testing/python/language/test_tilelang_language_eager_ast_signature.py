from tilelang.language.eager.ast import BaseBuilder, mutate
import tilelang.language as T
from tilelang import tvm
from contextlib import nullcontext
from types import FunctionType, SimpleNamespace
import gc
import importlib.util
import pytest
import weakref


class _TestBuilder(BaseBuilder):
    def set_fileline(self, filename: str, lineno: int, name: str):
        pass


def test_mutate_accepts_varargs_parameter():
    def first_arg(*args):
        return args[0]

    ir_gen = mutate(first_arg)

    assert ir_gen.gen(_TestBuilder())("sentinel", "ignored") == "sentinel"


def test_mutate_preserves_kwargs_parameter():
    def get_kwarg(**kwargs):
        return kwargs["key"]

    ir_gen = mutate(get_kwarg)

    assert ir_gen.gen(_TestBuilder())(key="value") == "value"


def test_macro_rebinds_closure_values():
    def make_macro(state, bias):
        @T.macro
        def add(x):
            return x + state["offset"] + bias

        return add

    state = {"offset": 2}
    first = make_macro(state, 3)
    second = make_macro({"offset": 7}, 11)
    assert first.ir_gen.gen(_TestBuilder())(1) == 6
    assert second.ir_gen.gen(_TestBuilder())(1) == 19
    state["offset"] = 5
    assert first.ir_gen.gen(_TestBuilder())(1) == 9


def test_macro_rebinds_globals():
    def count(x):
        return len(x)

    first = T.macro(count)
    second = T.macro(FunctionType(count.__code__, dict(count.__globals__, len=lambda x: 9)))
    assert first.ir_gen.gen(_TestBuilder())([1, 2]) == 2
    assert second.ir_gen.gen(_TestBuilder())([1, 2]) == 9
    assert first.ir_gen.gen(_TestBuilder())([1, 2]) == 2


def test_macro_rebinds_captured_buffers():
    def store(buffer, value):
        @T.macro
        def write():
            buffer[0] = value

        write()

    @T.prim_func
    def kernel(A: T.Tensor((1,), T.int32), B: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=1):
            store(A, 3)
            store(B, 7)

    stores = []
    tvm.tirx.stmt_functor.post_order_visit(
        kernel.body, lambda node: stores.append(node) if isinstance(node, tvm.tirx.BufferStore) else None
    )
    assert len(stores) == 2
    for store, parameter, value in zip(stores, kernel.params, (3, 7)):
        assert store.buffer.same_as(kernel.buffer_map[parameter])
        assert int(store.value) == value


def test_macro_does_not_retain_closure():
    def make_macro(state):
        @T.macro
        def read():
            return state()

        return read

    def state():
        return 3

    ref = weakref.ref(state)
    macro = make_macro(state)
    assert macro.ir_gen.gen(_TestBuilder())() == 3
    del macro, state
    gc.collect()
    assert ref() is None


def test_macro_resolves_current_launch_context():
    class SkippingBuilder(_TestBuilder):
        def skip_kernel_ctx(self):
            return True

    def make_macro(scope):
        @T.macro
        def launch():
            with scope():
                return 17

        return launch

    # Only launch contexts are skipped during phase-1 tracing. The same code
    # can capture an ordinary context manager on a later invocation.
    for scope, expected in [(T.Kernel, None), (nullcontext, 17), (T.Kernel, None)]:
        macro = make_macro(scope)
        assert macro.ir_gen.gen(SkippingBuilder())() == expected


def test_macro_resolves_current_body_annotations():
    def make_macro(dialect):
        @T.macro
        def identity(x):
            x: dialect.float32
            return x

        return identity

    for dtype in (T.float32, T.int32):
        macro = make_macro(SimpleNamespace(float32=dtype))
        assert macro.ir_gen.extra_type_hints == {"x": dtype}


def test_macro_resolves_current_parameter_annotations():
    def make_macro(annotation):
        @T.macro
        def identity(x: annotation):
            return x

        return identity

    for annotation in (T.Ref, T.float32):
        macro = make_macro(annotation)
        assert macro.annotations == {"x": annotation}


@pytest.mark.parametrize("control", [None, "break", "continue", "return 7"])
def test_deep_loops_do_not_exhaust_python_block_stack(tmp_path, control):
    # The original function fits CPython's block stack. Frontend-generated
    # control-flow scaffolding must not triple its nesting depth.
    lines = ["def nested():"]
    depth = 14 if control != "return 7" else 8
    for i in range(depth):
        lines.append("    " * (i + 1) + f"for i{i} in range(1):")
    lines.append("    " * (depth + 1) + (control or "pass"))
    lines.append("    return 7")
    path = tmp_path / "nested_loops.py"
    path.write_text("\n".join(lines) + "\n")
    spec = importlib.util.spec_from_file_location("nested_loops", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ir_gen = mutate(module.nested)
    assert ir_gen.gen(_TestBuilder())() == 7

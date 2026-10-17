from tilelang.language.eager.ast import BaseBuilder, mutate
import importlib.util
import pytest


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


def test_mutate_maps_generated_lines_to_kernel():
    import inspect
    import sys
    import types

    import tilelang.language as T

    # Keep context and iterable expressions separate from generated wrapper calls.
    # fmt: off
    def kernel(x: T.Tensor((8,), T.float32)):
        with (
            T.Kernel(1, threads=8)
        ):
            for j in (
                T.serial(8)
            ):
                if x[j] > 0:
                    x[j] = 0.0
                else:
                    x[j] = 1.0
    # fmt: on

    def code_lines(code):
        lines = {line for _, _, line in code.co_lines() if line is not None}
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                lines |= code_lines(const)
        return lines

    first = inspect.getsourcelines(kernel)[1]
    assert code_lines(mutate(kernel).gen.__code__) == {first + offset for offset in (0, 1, 2, 4, 5, 7, 8, 10)}
    executed = []

    def trace(frame, event, arg):
        if frame.f_code.co_name == "kernel" and frame.f_code.co_varnames[0] == "x" and event == "line":
            executed.append(frame.f_lineno)
        return trace

    # The real builder visits both device branches. Its Python branch arcs
    # must connect the original for/if headers to their bodies and exit.
    previous_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        T.prim_func(kernel)
    finally:
        sys.settrace(previous_trace)
    arcs = set(zip(executed, executed[1:]))
    assert {(first + a, first + b) for a, b in ((4, 7), (7, 8), (7, 10), (4, 1))} <= arcs


@pytest.mark.parametrize(
    "method, offset",
    [
        (None, 18),
        ("ctx_with", 1),
        ("context", 2),
        ("context_arg", 3),
        ("ctx_for", 6),
        ("iter_call", 6),
        ("iterable", 7),
        ("ctx_if", 9),
        ("unwrap_value", 10),
    ],
)
def test_mutate_preserves_multiline_traceback(monkeypatch, method, offset):
    import inspect
    import traceback
    from contextlib import nullcontext
    from types import SimpleNamespace

    sequence = SimpleNamespace(iterable=range, context=nullcontext, context_arg=None)

    # Keep distinct expression lines to check traceback locations after rewriting.
    # fmt: off
    def kernel():
        with (
            sequence.context(
                sequence.context_arg
            )
        ):
            for i in (
                sequence.iterable(1)
            ):
                if i == 0:
                    left, right = (
                        1,
                        2,
                    )
                    value = other = left + right
                    pass
                    return (
                        value
                        + other.attribute
                    )
    # fmt: on

    first = inspect.getsourcelines(kernel)[1]
    builder = _TestBuilder()
    if method in ("iterable", "context", "context_arg"):
        monkeypatch.delattr(sequence, method)
    elif method is not None:

        def fail(*args, **kwargs):
            raise RuntimeError("builder failure")

        monkeypatch.setattr(builder, method, fail)
    with pytest.raises(AttributeError if method in (None, "iterable", "context", "context_arg") else RuntimeError) as excinfo:
        mutate(kernel).gen(builder)()
    frame = next(frame for frame in traceback.extract_tb(excinfo.value.__traceback__) if frame.name == "kernel")
    assert frame.lineno == first + offset

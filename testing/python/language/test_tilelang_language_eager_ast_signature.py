from tilelang.language.eager.ast import BaseBuilder, mutate
import importlib.util
import pytest
from pathlib import Path


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


def _load_function(tmp_path, source, name="kernel"):
    path = tmp_path / "generated_name_kernel.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("generated_name_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


@pytest.mark.parametrize("name", ["__tb", "__tb_fl", "__tb_fn", "__kwargs", "__0", "__0_iter", "__1", "_", "__tb_1"])
def test_generated_names_do_not_shadow_parameters(tmp_path, name):
    """Loop and branch scaffolding must preserve a legal parameter name."""
    function = _load_function(
        tmp_path,
        f"def kernel({name}):\n    for i in range(1):\n        pass\n    if True:\n        pass\n    return {name}\n",
    )
    transformed = mutate(function).gen(_TestBuilder())
    assert transformed(123) == function(123)
    assert transformed(**{name: 123}) == function(**{name: 123})


def test_generated_names_preserve_all_parameter_kinds():
    """Hygiene must not rename public positional, variadic or keyword arguments."""

    def kernel(__tb, /, __tb_fl=2, *__kwargs, __tb_fn=3, **__0):
        return __tb + __tb_fl + __kwargs[0] + __tb_fn + __0["extra"]

    transformed = mutate(kernel).gen(_TestBuilder())
    assert transformed(1, 2, 4, __tb_fn=8, extra=16) == kernel(1, 2, 4, __tb_fn=8, extra=16)


def test_generated_names_preserve_tuple_and_chained_assignments():
    """User assignment targets stay untouched while generated temporaries change."""

    def kernel(__tb, __tb_fl, __0):
        __tb, __tb_fl = __tb_fl, __tb
        first = second = __0
        return __tb, __tb_fl, first, second

    assert mutate(kernel).gen(_TestBuilder())(1, 2, 3) == kernel(1, 2, 3)


def test_generated_names_preserve_captured_values():
    """A generated loop local must not hide a free variable from its closure."""
    __0 = 123
    __tb = 456

    def kernel():
        for _i in range(1):
            pass
        return __0, __tb

    assert mutate(kernel).gen(_TestBuilder())() == kernel()


def test_generated_names_preserve_globals(tmp_path):
    """Generated builder/span names cannot hide user globals either."""
    function = _load_function(tmp_path, "__tb = 123\n__tb_fl = 456\ndef kernel():\n    return __tb, __tb_fl\n")
    assert mutate(function).gen(_TestBuilder())() == function()


@pytest.mark.parametrize("name", ["__tb", "__tb_fl", "__tb_fn", "__kwargs"])
def test_generated_names_preserve_function_names(tmp_path, name):
    """Returning the generated closure must not resolve to an internal local."""
    function = _load_function(tmp_path, f"def {name}(value):\n    return value\n", name=name)
    transformed = mutate(function).gen(_TestBuilder())
    assert transformed.__name__ == function.__name__
    assert transformed(123) == function(123)


def test_generated_names_preserve_loop_control_iterator(tmp_path):
    """The companion iterator local used for break must be fresh as well."""
    function = _load_function(
        tmp_path,
        "def kernel(__0_iter, _):\n    for i in range(2):\n        break\n    while False:\n        pass\n    return __0_iter, _\n",
    )
    assert mutate(function).gen(_TestBuilder())(123, 456) == function(123, 456)


def test_generated_names_preserve_source_spans():
    """Metadata names remain filenames/function names, not user argument values."""

    class SpanBuilder(_TestBuilder):
        def __init__(self):
            self.locations = []

        def set_fileline(self, filename, lineno, name):
            self.locations.append((filename, lineno, name))

    def kernel(__tb_fl, __tb_fn):
        return __tb_fl, __tb_fn

    builder = SpanBuilder()
    assert mutate(kernel).gen(builder)(123, 456) == (123, 456)
    assert builder.locations
    assert all(filename == str(Path(__file__).absolute()) and name == "kernel" for filename, _, name in builder.locations)


@pytest.mark.parametrize("name", ["__tb", "__tb_fl", "__tb_fn", "__kwargs", "__0", "_"])
def test_generated_names_preserve_prim_func_tensor_parameters(tmp_path, name):
    """The public eager decorator must still load the original tensor parameter."""
    from tvm import tirx

    function = _load_function(
        tmp_path,
        "import tilelang.language as T\n"
        "@T.prim_func\n"
        f"def kernel({name}: T.Tensor((1,), T.int32), B: T.Tensor((1,), T.int32)):\n"
        "    with T.Kernel(1, threads=1):\n"
        "        for i in range(1):\n"
        "            if True:\n"
        f"                B[0] = {name}[0]\n",
    )
    original = function.buffer_map[function.params[0]]
    loads = []
    tirx.stmt_functor.post_order_visit(function.body, lambda node: loads.append(node) if isinstance(node, tirx.BufferLoad) else None)
    assert loads
    assert all(load.buffer.same_as(original) for load in loads)

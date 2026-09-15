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

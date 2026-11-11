"""Tests for opt-in #line directive emission in C-family codegen.

tl.emit_line_directives is a level: 0 is off, 1 anchors statements, and 2
re-anchors every emitted source line so multi-line statements and closing
braces cannot drift. The tests cover the CPU c target (no GPU needed), the
device source paths, and the environment -> pass-config bridge.
"""

import re

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.transform.pass_config import normalize_pass_configs

N = 128
TILE = 256
LEVEL = tilelang.PassConfigKey.TL_EMIT_LINE_DIRECTIVES


@pytest.fixture(autouse=True)
def _clear_line_directive_env(monkeypatch):
    """Keep the suite hermetic: the environment also supplies this default."""
    monkeypatch.delenv("TILELANG_EMIT_LINE_DIRECTIVES", raising=False)


@T.prim_func
def vec_add(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
    for i in T.Parallel(N):
        B[i] = A[i] + 1.0  # line_marker_store


@T.prim_func
def vec_add_device(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
    with T.Kernel(1, threads=N) as bx:
        for i in T.Parallel(N):
            B[bx * N + i] = A[bx * N + i] + 1.0  # line_marker_store_device


@T.prim_func
def tiled_add(A: T.Tensor((4 * TILE,), "float32"), B: T.Tensor((4 * TILE,), "float32"), C: T.Tensor((4 * TILE,), "float32")):
    with T.Kernel(4, threads=N) as bx:
        a_local = T.alloc_fragment((TILE,), "float32")
        b_local = T.alloc_fragment((TILE,), "float32")
        c_local = T.alloc_fragment((TILE,), "float32")
        T.copy(A[bx * TILE : bx * TILE + TILE], a_local)
        T.copy(B[bx * TILE : bx * TILE + TILE], b_local)
        for i in T.Parallel(TILE):
            c_local[i] = a_local[i] + b_local[i]
        T.copy(c_local, C[bx * TILE : bx * TILE + TILE])


@T.prim_func
def cond_add(A: T.Tensor((4 * TILE,), "float32"), B: T.Tensor((4 * TILE,), "float32"), C: T.Tensor((4 * TILE,), "float32")):
    with T.Kernel(4, threads=N) as bx:
        a_local = T.alloc_fragment((TILE,), "float32")
        b_local = T.alloc_fragment((TILE,), "float32")
        c_local = T.alloc_fragment((TILE,), "float32")
        T.copy(A[bx * TILE : bx * TILE + TILE], a_local)
        T.copy(B[bx * TILE : bx * TILE + TILE], b_local)
        for i in T.Parallel(TILE):
            c_local[i] = T.if_then_else(a_local[i] > b_local[i], a_local[i] + b_local[i], a_local[i] - b_local[i])
        T.copy(c_local, C[bx * TILE : bx * TILE + TILE])


def _lowered_source(func, level: int) -> str:
    with tvm.target.Target("c"), tvm.transform.PassContext(opt_level=3, config={LEVEL: level}):
        artifact = tilelang.lower(func)
    assert artifact.kernel_source is not None, "CPU C codegen produced no kernel source"
    return artifact.kernel_source


def _host_source(func_or_module, config) -> str:
    """Lower with the C host codegen and return the generated source."""
    build = tvm.get_global_func("target.build.tilelang_c_host", allow_missing=True)
    if build is None:
        pytest.skip("c_host codegen is not built")
    module = func_or_module if isinstance(func_or_module, tvm.IRModule) else tvm.IRModule.from_expr(func_or_module)
    with tvm.transform.PassContext(config=config):
        return build(module, tvm.target.Target("c")).inspect_source()


def _marker_line(marker: str) -> int:
    with open(__file__) as f:
        for i, line in enumerate(f, 1):
            if marker in line:
                return i
    raise ValueError(f"marker not found: {marker}")


def _line_directives(source: str) -> list[tuple[int, str]]:
    return [(int(num), fname) for num, fname in re.findall(r'^#line (\d+) "(.*)"$', source, re.M)]


def _effective_locations(source: str) -> list[tuple[str, int, str]]:
    """Resolve each physical line under C #line rules to (file, line, text)."""
    locations: list[tuple[str, int, str]] = []
    file = ""
    line = 0
    for text in source.splitlines():
        directive = re.match(r'^#line (\d+) "(.*)"$', text)
        if directive:
            line = int(directive.group(1))
            file = directive.group(2)
            continue
        locations.append((file, line, text))
        line += 1
    return locations


def _location_of(source: str, needle: str) -> tuple[str, int]:
    return next((file, line) for file, line, text in _effective_locations(source) if needle in text)


def _spanless_tail_source(config) -> str:
    """Host source whose second and third statements have no span."""
    source_name = tvm.ir.SourceName(__file__)
    entry_span = tvm.ir.Span(source_name, 10, 10, 0, 1)
    stmt_span = tvm.ir.Span(source_name, 20, 20, 0, 1)
    body = tvm.tirx.SeqStmt(
        [
            tvm.tirx.Evaluate(tvm.tirx.call_extern("int32", "consume", 1), span=stmt_span),
            tvm.tirx.Evaluate(tvm.tirx.call_extern("int32", "consume", 2)),
            tvm.tirx.Evaluate(tvm.tirx.call_extern("int32", "consume", 3)),
        ]
    )
    func = tvm.tirx.PrimFunc([], body, span=entry_span).with_attr("global_symbol", "line_kernel")
    return _host_source(func, config)


def _unanchored_statement_lines(source: str) -> list[str]:
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if "__global__" in line and line.rstrip().endswith("{"))
    end = max(i for i, line in enumerate(lines) if line.rstrip() == "}")
    return [lines[i] for i in range(start + 1, end) if lines[i].strip().endswith(";") and not lines[i - 1].startswith("#line ")]


def test_statements_are_mapped():
    """Level 1 maps the function entry and each spanned statement, and re-emits
    the same line for every statement lowered from it (no dedup)."""
    source = _lowered_source(vec_add, 1)
    directives = _line_directives(source)
    assert directives and {fname for _, fname in directives} == {__file__}, source
    assert (_marker_line("def vec_add"), __file__) in directives, source
    store_line = _marker_line("line_marker_store")
    assert directives.count((store_line, __file__)) >= 2, source


def test_line_directives_disabled():
    """Level 0 and an absent config key both emit nothing."""
    assert "#line" not in _lowered_source(vec_add, 0)
    with tvm.target.Target("c"), tvm.transform.PassContext(opt_level=3):
        source = tilelang.lower(vec_add).kernel_source
    assert source is not None and "#line" not in source, source


def _device_directives(target) -> list[tuple[int, str]]:
    with tvm.transform.PassContext(opt_level=3, config={LEVEL: 1}), tvm.target.Target(target):
        artifact = tilelang.lower(vec_add_device, target=target)
    assert artifact.kernel_source is not None, f"{target} codegen produced no kernel source"
    return _line_directives(artifact.kernel_source)


@tilelang.testing.requires_cuda
def test_cuda_source_maps_statements():
    assert (_marker_line("line_marker_store_device"), __file__) in _device_directives({"kind": "cuda"})


@tilelang.testing.requires_rocm
def test_hip_source_maps_statements():
    assert (_marker_line("line_marker_store_device"), __file__) in _device_directives({"kind": "hip"})


@pytest.mark.parametrize(("level", "expected"), [(1, 1), (2, 3)], ids=["statement", "per_line"])
def test_spanless_statements_follow_the_level(level, expected):
    """Level 1 lets span-less statements inherit the previous directive; level 2
    re-anchors each one."""
    source = _spanless_tail_source({LEVEL: level})
    assert _line_directives(source).count((20, __file__)) == expected, source


def test_every_line_anchors_synthesized_declarations():
    """Lowering emits span-less declarations; level 2 re-anchors each one."""
    default_source = _lowered_source(tiled_add, 1)
    every_source = _lowered_source(tiled_add, 2)
    assert len(_line_directives(every_source)) > len(_line_directives(default_source))

    lines = every_source.splitlines()
    declarations = [i for i, line in enumerate(lines) if line.strip().startswith("float ") and line.rstrip().endswith(";")]
    assert len(declarations) >= 3, every_source
    for index in declarations:
        assert lines[index - 1].startswith("#line "), every_source


@pytest.mark.parametrize(("value", "expected"), [("2", 2), (" 2 ", 2), ("true", 1)])
def test_environment_defaults_reach_codegen(monkeypatch, value, expected):
    """Environment defaults are parsed once and reach codegen through
    normalize_pass_configs, the bridge the JIT and compile-only paths use."""
    monkeypatch.setenv("TILELANG_EMIT_LINE_DIRECTIVES", value)
    config = normalize_pass_configs(None)
    assert config[LEVEL.value] == expected
    source = _spanless_tail_source(config)
    assert _line_directives(source).count((20, __file__)) == (3 if expected >= 2 else 1)


def test_pass_config_normalization(monkeypatch):
    key = LEVEL.value
    assert normalize_pass_configs({LEVEL: True})[key] == 1
    assert normalize_pass_configs({LEVEL: False})[key] == 0
    monkeypatch.setenv("TILELANG_EMIT_LINE_DIRECTIVES", "2")
    assert normalize_pass_configs(None)[key] == 2

    # An explicit pass config still beats the environment default.
    monkeypatch.setenv("TILELANG_EMIT_LINE_DIRECTIVES", "0")
    assert normalize_pass_configs({LEVEL: 2})[key] == 2


def test_invalid_level_is_rejected(monkeypatch):
    key = LEVEL.value
    with pytest.raises(ValueError, match="integer level"):
        normalize_pass_configs({LEVEL: "per-line"})

    # A malformed environment default is only parsed when the key is absent.
    monkeypatch.setenv("TILELANG_EMIT_LINE_DIRECTIVES", "invalid")
    assert normalize_pass_configs({LEVEL: 0})[key] == 0
    with pytest.raises(ValueError, match="integer level"):
        normalize_pass_configs(None)


def test_functions_do_not_inherit_anchors():
    """A function without a span must be re-anchored explicitly: printing no
    directive leaves the last one in effect through the next function."""
    source_name = tvm.ir.SourceName("/tmp/first.py")
    entry_span = tvm.ir.Span(source_name, 10, 10, 0, 1)
    stmt_span = tvm.ir.Span(source_name, 23, 23, 0, 1)
    first = tvm.tirx.PrimFunc(
        [], tvm.tirx.Evaluate(tvm.tirx.call_extern("int32", "consume", 1), span=stmt_span), span=entry_span
    ).with_attr("global_symbol", "first_kernel")
    second = tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(tvm.tirx.call_extern("int32", "consume", 2))).with_attr(
        "global_symbol", "second_kernel"
    )
    source = _host_source(tvm.IRModule({"first_kernel": first, "second_kernel": second}), {LEVEL: 2})

    assert _location_of(source, "consume(1);") == ("/tmp/first.py", 23), source
    tail = source[source.index("first_kernel() {") :]
    tail = tail[tail.index("}\n") :]
    assert _location_of(tail, "consume(2);")[0] == "<tilelang-generated>", source


def test_spanless_function_returns_to_generated_source():
    """A function without a span remaps lines after a spanned statement to the
    generated pseudo-source, including its closing brace."""
    source_name = tvm.ir.SourceName("/tmp/body.py")
    body_span = tvm.ir.Span(source_name, 23, 23, 0, 1)
    body = tvm.tirx.SeqStmt(
        [
            tvm.tirx.Evaluate(tvm.tirx.call_extern("int32", "consume", 1), span=body_span),
            tvm.tirx.Evaluate(tvm.tirx.call_extern("int32", "consume", 2)),
        ]
    )
    func = tvm.tirx.PrimFunc([], body).with_attr("global_symbol", "no_span_kernel")
    source = _host_source(func, {LEVEL: 2})

    locations = _effective_locations(source)
    closing = max(i for i, (_, _, text) in enumerate(locations) if text.strip() == "}")
    assert _location_of(source, "consume(1);") == ("/tmp/body.py", 23), source
    assert _location_of(source, "consume(2);")[0] == "<tilelang-generated>", source
    assert locations[closing][0] == "<tilelang-generated>", source


def test_closing_braces_keep_their_owning_block():
    """Closing lines keep their block's position: a loop's back edge must not
    map one line past the body, and the function brace keeps its span."""
    source_name = tvm.ir.SourceName("/tmp/loop.py")
    entry_span = tvm.ir.Span(source_name, 5, 5, 0, 1)
    loop_span = tvm.ir.Span(source_name, 10, 10, 0, 1)
    body_span = tvm.ir.Span(source_name, 23, 23, 0, 1)
    index = tvm.tirx.Var("i", "int32")
    body = tvm.tirx.Evaluate(tvm.tirx.call_extern("int32", "consume", index), span=body_span)
    loop = tvm.tirx.For(
        index,
        tvm.tirx.IntImm("int32", 0),
        tvm.tirx.IntImm("int32", 4),
        tvm.tirx.ForKind.SERIAL,
        body,
        span=loop_span,
    )
    func = tvm.tirx.PrimFunc([], loop, span=entry_span).with_attr("global_symbol", "loop_kernel")
    source = _host_source(func, {LEVEL: 2})

    locations = _effective_locations(source)
    header = next(i for i, (_, _, text) in enumerate(locations) if "for (" in text)
    body_index = next(i for i in range(header + 1, len(locations)) if "consume(" in locations[i][2])
    closing = next(i for i in range(body_index + 1, len(locations)) if locations[i][2].strip() == "}")
    function_close = max(i for i, (_, _, text) in enumerate(locations) if text.strip() == "}")
    assert locations[header][:2] == ("/tmp/loop.py", 10), source
    assert locations[body_index][:2] == ("/tmp/loop.py", 23), source
    assert locations[closing][:2] == ("/tmp/loop.py", 10), source
    assert locations[function_close][:2] == ("/tmp/loop.py", 5), source


@pytest.mark.parametrize("codegen", ["c_host", "metal_without_compile"])
def test_function_span_anchors_signature_and_closing_brace(codegen):
    """The function span anchors the lines outside statements. Backends that
    override InitFuncState must chain to the TileLang base or this is lost."""
    build = tvm.get_global_func(f"target.build.tilelang_{codegen}", allow_missing=True)
    if build is None:
        pytest.skip(f"{codegen} codegen is not built")
    target = "c" if codegen == "c_host" else "metal"
    source_name = tvm.ir.SourceName("/tmp/one.py")
    entry_span = tvm.ir.Span(source_name, 10, 10, 0, 1)
    stmt_span = tvm.ir.Span(source_name, 20, 20, 0, 1)
    body = tvm.tirx.Evaluate(tvm.tirx.call_extern("int32", "consume", 1), span=stmt_span)
    func = tvm.tirx.PrimFunc([], body, span=entry_span).with_attr("global_symbol", "line_kernel")
    if target == "metal":
        func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
        func = func.with_attr("tirx.kernel_launch_params", [])
    with tvm.transform.PassContext(config={LEVEL: 2}):
        source = build(tvm.IRModule.from_expr(func), tvm.target.Target(target)).inspect_source()

    locations = _effective_locations(source)
    signature = next(i for i, (_, _, text) in enumerate(locations) if "line_kernel(" in text and ";" not in text)
    body_index = next(i for i, (_, _, text) in enumerate(locations) if "consume(1);" in text)
    closing = max(i for i, (_, _, text) in enumerate(locations) if text.strip() == "}")
    assert locations[signature][:2] == ("/tmp/one.py", 10), source
    assert locations[body_index][:2] == ("/tmp/one.py", 20), source
    assert locations[closing][:2] == ("/tmp/one.py", 10), source


@pytest.mark.parametrize("kernel", [tiled_add, cond_add], ids=["vectorized_add", "conditional_select"])
@tilelang.testing.requires_cuda
def test_every_line_anchors_multiline_cuda_statements(kernel):
    """One TIR statement can print several CUDA lines (a packed binary op or a
    lowered conditional store); each line needs its own directive."""
    target = {"kind": "cuda"}
    with tvm.transform.PassContext(opt_level=3, config={LEVEL: 2}), tvm.target.Target(target):
        artifact = tilelang.lower(kernel, target=target)
    assert artifact.kernel_source is not None, "CUDA codegen produced no kernel source"
    unanchored = _unanchored_statement_lines(artifact.kernel_source)
    assert not unanchored, f"statement lines without a #line directive: {unanchored}\n{artifact.kernel_source}"


@pytest.mark.parametrize(
    ("codegen", "target"),
    [
        ("c_host", "c"),
        # The FFI registration check below also supports Metal source codegen
        # on hosts without a Metal runtime.
        pytest.param(
            "metal_without_compile",
            "metal",
            marks=tilelang.testing.requires_metal.marks(support_required="optional"),
        ),
    ],
)
@pytest.mark.parametrize("emit", [False, True], ids=["disabled", "enabled"])
def test_host_and_metal_line_directives(codegen, target, emit):
    build = tvm.get_global_func(f"target.build.tilelang_{codegen}", allow_missing=True)
    if build is None:
        pytest.skip(f"{codegen} codegen is not built")
    source_name = tvm.ir.SourceName(__file__)
    entry_span = tvm.ir.Span(source_name, 10, 10, 0, 1)
    stmt_span = tvm.ir.Span(source_name, 20, 20, 0, 1)
    body = tvm.tirx.SeqStmt([tvm.tirx.Evaluate(tvm.tirx.call_extern("int32", "consume", value), span=stmt_span) for value in (1, 2)])
    func = tvm.tirx.PrimFunc([], body, span=entry_span).with_attr("global_symbol", "line_kernel")
    if target == "metal":
        func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
        func = func.with_attr("tirx.kernel_launch_params", [])
    with tvm.transform.PassContext(config={LEVEL: emit}):
        source = build(tvm.IRModule.from_expr(func), tvm.target.Target(target)).inspect_source()

    directives = _line_directives(source)
    if emit:
        assert (10, __file__) in directives, source
        assert directives.count((20, __file__)) == 2, source
    else:
        assert not directives, source


if __name__ == "__main__":
    tilelang.testing.main()

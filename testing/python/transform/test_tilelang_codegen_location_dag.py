"""C-family source mappings preserve callee-first provenance in location DAGs."""

import re

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

import tilelang


@pytest.fixture
def location_graph(request):
    unknown = ir.UnknownLoc()
    source = ir.SourceLoc(ir.SourceName("deep.py"), 101, 8, 102, 19)
    outer = ir.SourceLoc(ir.SourceName("outer.py"), 203, 4, 203, 12)
    nodes = []
    edges = []

    def call(callee, caller):
        loc = ir.CallSiteLoc(callee, caller)
        nodes.append(loc)
        edges.append((loc, callee, caller))
        return loc

    kind = request.param
    expected = (101, "deep.py")
    if kind == "source":
        root = source
    elif kind == "unknown":
        root, expected = unknown, None
    elif kind == "callee_first":
        root = call(source, outer)
    elif kind == "caller_fallback":
        root, expected = call(unknown, outer), (203, "outer.py")
    elif kind == "nested_priority":
        root = call(call(unknown, source), outer)
    elif kind in ("shared_unknown", "shared_fallback"):
        root = unknown
        for _ in range(128):
            root = call(root, root)
        root, expected = (call(root, outer), (203, "outer.py")) if kind == "shared_fallback" else (root, None)
    else:
        root = source
        for _ in range(8192):
            root = call(root, unknown) if kind == "deep_callee" else call(unknown, root)
    result = [root, expected, edges]
    try:
        yield result
        assert (source.source_name.name, source.start_line, source.start_column, source.end_line, source.end_column) == (
            "deep.py",
            101,
            8,
            102,
            19,
        )
        for loc, callee, caller in edges:
            assert loc.callee.same_as(callee)
            assert loc.caller.same_as(caller)
    finally:
        result[0] = None
        root = None
        loc = callee = caller = None
        edges.clear()
        while nodes:
            nodes.pop()


@pytest.mark.parametrize(
    "location_graph",
    [
        "source",
        "unknown",
        "callee_first",
        "caller_fallback",
        "nested_priority",
        "shared_unknown",
        "shared_fallback",
        "deep_callee",
        "deep_caller",
    ],
    indirect=True,
)
@pytest.mark.parametrize("emit", [False, True])
def test_location_graph_line_directives(location_graph, emit, tmp_path):
    location, expected, _ = location_graph
    statement = ir.Evaluate(tirx.call_extern("consume", ir.IntImm("int32", 1), ty="int32"), loc=location)
    entry = ir.SourceLoc(ir.SourceName("entry.py"), 13, 2, 13, 9)
    func = tirx.Function([], [statement], ir.PrimType("void"), loc=entry)
    func = func.with_attr("global_symbol", "location_probe")
    build = tvm_ffi.get_global_func("target.build.tilelang_c_host")
    config = {tilelang.PassConfigKey.TL_EMIT_LINE_DIRECTIVES: emit}
    with tvm_tirx.transform.PassContext(config=config):
        source = build(tvm_tirx.IRModule({"location_probe": func}), tvm_tirx.target.Target("c")).inspect_source()
    (tmp_path / "location.c").write_text(source)
    directives = [(int(line), name) for line, name in re.findall(r'^#line (\d+) "(.*)"$', source, re.M)]
    if not emit:
        assert directives == [], source
    elif expected is None:
        assert directives == [(13, "entry.py")], source
    else:
        assert directives == [(13, "entry.py"), expected], source
        # The selected directive must immediately precede this statement, not
        # merely occur elsewhere in a function or a declaration.
        call = re.search(r'(?m)^#line (\d+) "(.*)"\n\s*consume\(1\);', source)
        assert call is not None, source
        assert (int(call.group(1)), call.group(2)) == expected
    assert func.loc.same_as(entry)
    assert statement.loc.same_as(location)

"""PTO physical launch groups for kernels with explicit AIV subcores."""

import ast

import pytest

import tilelang
from tilelang import tvm
from tilelang.ascend import language as T
from tilelang.backend.target import determine_target


tirx = tvm.tirx


def _calls(node, name):
    return [call for call in ast.walk(node) if isinstance(call, ast.Call) and ast.unparse(call.func) == name]


def _sections(func, role):
    return [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Call)
            and ast.unparse(item.context_expr.func) == "tl.mixed_kernel_section"
            and ast.literal_eval(item.context_expr.args[0]) == role
            for item in node.items
        )
    ]


def _functions(source):
    return {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}


def _assert_mixed(source, func):
    decorator = func.decorator_list[0]
    assert ast.unparse(decorator.func) == "pto.jit"
    assert "kernel_kind" not in {keyword.arg for keyword in decorator.keywords}
    assert not _calls(ast.parse(source), "tl.finalize_mixed_kernel")


def _assert_idle_cube(func):
    cubes = _sections(func, "cube")
    assert len(cubes) == 1
    assert len(cubes[0].body) == 1
    assert ast.unparse(cubes[0].body[0]) == "pto.init_core()"


def _assert_vector_guard(section, sids):
    assert ast.unparse(section.body[0]) == "pto.init_core()"
    guards = [node for node in section.body if isinstance(node, ast.If)]
    if sids == 1:
        assert len(section.body) == 2
        assert len(guards) == 1
        assert ast.unparse(guards[0].test) == "pto.get_subblock_idx() == 0"
    else:
        assert guards == []


@pytest.mark.parametrize("blocks", [1, 3])
@pytest.mark.parametrize("sids", [1, 2])
def test_vector_only_mixed_kernel_keeps_physical_launch_group(blocks, sids):
    @T.prim_func
    def main(src: T.Tensor((blocks * 128,), "float32"), dst: T.Tensor((blocks * 128,), "float32")):
        with T.MixedKernel(blocks, sids=sids) as (bx, sid):
            ub = T.alloc_shared((64,), "float32")
            begin = bx * 128 + sid * 64
            T.copy(src[begin : begin + 64], ub)
            T.copy(ub, dst[begin : begin + 64])

    with determine_target("pto", return_object=True):
        artifact = tilelang.lower(main, target="pto", enable_device_compile=False)
    source = artifact.kernel_source
    funcs = _functions(source)
    assert len(funcs) == 1
    func = next(iter(funcs.values()))
    _assert_mixed(source, func)
    _assert_idle_cube(func)
    vectors = _sections(func, "vector")
    assert len(vectors) == 1
    _assert_vector_guard(vectors[0], sids)
    assert _calls(func, "pto.get_block_idx")
    assert _calls(func, "pto.get_subblock_idx")
    device_func = next(iter(artifact.device_mod.functions.values()))
    assert int(device_func.attrs["thread_extent"]["blockIdx.x"]) == blocks


def _device_func(name, roles):
    """Keep physical sections intact at the codegen boundary, without scheduling."""
    src = tirx.decl_buffer((256,), "float16", name="src")
    out = tirx.decl_buffer((len(roles),), "int32", name="out")
    sections = []
    for index, (role, vector_count) in enumerate(roles):
        annotations = {} if vector_count is None else {"vector_count": tirx.IntImm("int32", vector_count)}
        if role == "CUBE":
            l1 = tirx.decl_buffer((256,), "float16", name="l1", scope="shared.l1")
            copy = tirx.Evaluate(
                tirx.call_intrin(
                    "handle",
                    tvm.ir.Op.get("tl.ascend_copy_gm_to_cbuf"),
                    tirx.address_of(l1[0]),
                    tirx.address_of(src[0]),
                    0,
                    32,
                    0,
                    16,
                    16,
                    0,
                    0,
                    0,
                    16,
                    "",
                )
            )
            body = tirx.SeqStmt([tirx.AllocBuffer(l1), copy])
        else:
            body = tirx.BufferStore(out, index + 17, [index])
        sections.append(tirx.SBlock([], [], [], role, body, annotations=annotations))
    body = sections[0] if len(sections) == 1 else tirx.SeqStmt(sections)
    return tirx.PrimFunc([src.data, out.data], body, buffer_map={src.data: src, out.data: out}).with_attr("global_symbol", name)


def _codegen(functions):
    target = tvm.target.Target({"kind": "ascend", "keys": ["pto"]})
    mod = tvm.IRModule({name: func.with_attr("target", target) for name, func in functions.items()})
    return tvm.get_global_func("target.build.tilelang_pto_without_compile")(mod, target).inspect_source()


@pytest.mark.parametrize("role,kind", [("VECTOR", "vector"), ("CUBE", "cube")])
def test_plain_kernel_keeps_its_launch_kind(role, kind):
    source = _codegen({"main": _device_func("main", [(role, None)])})
    func = _functions(source)["main"]
    keywords = {keyword.arg: ast.literal_eval(keyword.value) for keyword in func.decorator_list[0].keywords}
    assert keywords["kernel_kind"] == kind
    assert not _sections(func, "cube") and not _sections(func, "vector")
    assert not _calls(ast.parse(source), "tl.finalize_mixed_kernel")
    assert len(_calls(func, "pto.init_core")) == 1
    assert len(_calls(func, "pto.store" if role == "VECTOR" else "pto.mte_gm_l1_frac")) == 1


@pytest.mark.parametrize("sids", [None, 1, 2])
def test_existing_mixed_kernel_keeps_its_cube_body(sids):
    source = _codegen({"main": _device_func("main", [("CUBE", None), ("VECTOR", sids)])})
    func = _functions(source)["main"]
    _assert_mixed(source, func)
    cubes = _sections(func, "cube")
    vectors = _sections(func, "vector")
    assert len(cubes) == len(vectors) == 1
    assert len(_calls(cubes[0], "pto.init_core")) == 1
    assert len(_calls(cubes[0], "pto.mte_gm_l1_frac")) == 1
    assert len(_calls(vectors[0], "pto.store")) == 1
    _assert_vector_guard(vectors[0], sids or 2)


def test_multiple_vector_sections_share_one_idle_cube():
    source = _codegen({"main": _device_func("main", [("VECTOR", 1), ("VECTOR", 2)])})
    func = _functions(source)["main"]
    _assert_mixed(source, func)
    _assert_idle_cube(func)
    vectors = _sections(func, "vector")
    assert len(vectors) == 2
    for section, sids in zip(vectors, (1, 2)):
        _assert_vector_guard(section, sids)
        assert len(_calls(section, "pto.store")) == 1


def test_launch_group_detection_is_per_function():
    functions = {
        "mixed_vector": _device_func("mixed_vector", [("VECTOR", 2)]),
        "plain_vector": _device_func("plain_vector", [("VECTOR", None)]),
        "real_mixed": _device_func("real_mixed", [("CUBE", None), ("VECTOR", 1)]),
        "plain_cube": _device_func("plain_cube", [("CUBE", None)]),
    }
    source = _codegen(functions)
    funcs = _functions(source)
    _assert_mixed(source, funcs["mixed_vector"])
    _assert_idle_cube(funcs["mixed_vector"])
    _assert_mixed(source, funcs["real_mixed"])
    assert len(_calls(_sections(funcs["real_mixed"], "cube")[0], "pto.mte_gm_l1_frac")) == 1
    for name, kind in (("plain_vector", "vector"), ("plain_cube", "cube")):
        func = funcs[name]
        keywords = {keyword.arg: ast.literal_eval(keyword.value) for keyword in func.decorator_list[0].keywords}
        assert keywords["kernel_kind"] == kind
        assert not _sections(func, "cube") and not _sections(func, "vector")

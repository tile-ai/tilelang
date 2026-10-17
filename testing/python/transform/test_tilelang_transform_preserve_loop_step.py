import pytest

import numpy as np

import tilelang as tl
from tilelang import tvm


def test_lower_opaque_block_preserves_non_unit_loop_step():
    output_buffer = tvm.tirx.decl_buffer((6,), "int32", name="output")
    i = tvm.tirx.Var("i", "int32")
    loop = tvm.tirx.For(
        i,
        1,
        5,
        tvm.tirx.ForKind.SERIAL,
        tvm.tirx.BufferStore(output_buffer, 1, [i]),
        step=tvm.tirx.IntImm("int32", 2),
    )
    before = tvm.tirx.PrimFunc(
        [output_buffer.data],
        loop,
        buffer_map={output_buffer.data: output_buffer},
    ).with_attr("global_symbol", "main")

    mod = tl.transform.LowerOpaqueBlock()(tvm.IRModule.from_expr(before))
    executable = tvm.compile(mod["main"], target="c").jit(options=["-std=c++17"])

    output = tvm.runtime.tensor(np.zeros(6, dtype="int32"))
    executable["main"](output)

    np.testing.assert_array_equal(
        output.numpy(),
        np.array([0, 1, 0, 1, 0, 1], dtype="int32"),
    )


@pytest.mark.parametrize("rng", [(0, 6, 2), (0, 5, 2)], ids=lambda v: f"rng=({v[0]},{v[1]},{v[2]})")
@pytest.mark.parametrize("explicit", [False, True], ids=lambda v: f"explicit={v}")
def test_unroll_loop_preserves_non_unit_loop_step(rng, explicit):
    output_buffer = tvm.tirx.decl_buffer((8,), "int32", name="output")
    i = tvm.tirx.Var("i", "int32")
    start, stop, step = rng
    loop = tvm.tirx.For(
        i,
        start,
        stop,
        tvm.tirx.ForKind.UNROLLED,
        tvm.tirx.BufferStore(output_buffer, i, [i]),
        step=tvm.tirx.IntImm("int32", step),
        annotations={"pragma_unroll_explicit": explicit},
    )
    before = tvm.tirx.PrimFunc(
        [output_buffer.data],
        loop,
        buffer_map={output_buffer.data: output_buffer},
    ).with_attr("global_symbol", "main")

    mod = tl.transform.UnrollLoop()(tvm.IRModule.from_expr(before))
    executable = tvm.compile(mod["main"], target="c").jit(options=["-std=c++17"])

    output = tvm.runtime.tensor(np.zeros(8, dtype="int32"))
    executable["main"](output)

    np.testing.assert_array_equal(
        output.numpy(),
        np.array([0, 0, 2, 0, 4, 0, 0, 0], dtype="int32"),
    )


def _build_loop_source(backend, start, stop, step, kind, unroll_factor=None, lexical_scope=False):
    i = tvm.tirx.Var("i", "int32")
    output = tvm.tirx.decl_buffer((32,), "int32", name="output")
    body = tvm.tirx.BufferStore(output, i, [i])
    if lexical_scope:
        body = tvm.tirx.AttrStmt(0, "lexical_alloc_scope", 1, body)
    # Construct raw For IR because TileLang's frontend normalizes loop steps.
    body = tvm.tirx.For(i, start, stop - start, kind, body, step=step)
    if unroll_factor is not None:
        body = tvm.tirx.AttrStmt(i, "pragma_unroll_factor", unroll_factor, body)
    params = [output.data]
    params.extend(tvm.tirx.analysis.undefined_vars(body, params))
    func = tvm.tirx.PrimFunc(params, body, buffer_map={output.data: output})
    func = func.with_attr("global_symbol", "loop_step")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    mod = tvm.IRModule({"loop_step": func})
    build = tvm.get_global_func(f"target.build.tilelang_{backend}_without_compile", allow_missing=True)
    if build is None:
        pytest.skip(f"TileLang was built without the {backend} code generator")
    target_config = {"kind": "cuda", "arch": "sm_80"} if backend == "cuda" else {"kind": "hip", "mcpu": "gfx942"}
    return build(mod, tvm.target.Target(target_config)).inspect_source()


@pytest.mark.parametrize("backend", ["cuda", "hip"])
@pytest.mark.parametrize("kind", [tvm.tirx.ForKind.SERIAL, tvm.tirx.ForKind.UNROLLED])
@pytest.mark.parametrize(
    "start, stop, step",
    [(0, 8, None), (0, 8, 1), (0, 8, 2), (3, 13, 3)],
)
def test_loop_step_codegen(backend, start, stop, step, kind):
    source = _build_loop_source(backend, start, stop, step, kind)

    increment = "++i" if step is None else f"i += {step}"
    assert f"for (int i = {start}; i < {stop}; {increment}) {{" in source
    assert ("#pragma unroll\n" in source) == (kind == tvm.tirx.ForKind.UNROLLED)


@pytest.mark.parametrize("backend", ["cuda", "hip"])
@pytest.mark.parametrize("start", [0, 3])
def test_loop_codegen_simplifies_expressions(backend, start):
    n = tvm.tirx.Var("n", "int32")
    minimum = tvm.tirx.Add(tvm.tirx.IntImm("int32", start), tvm.tirx.IntImm("int32", 0))
    stop = tvm.tirx.Add(n, tvm.tirx.IntImm("int32", 0))
    step = tvm.tirx.Add(tvm.tirx.IntImm("int32", 1), tvm.tirx.IntImm("int32", 1))
    source = _build_loop_source(backend, minimum, stop, step, tvm.tirx.ForKind.SERIAL)

    assert f"for (int i = {start}; i < n; i += 2) {{" in source


@pytest.mark.parametrize("backend", ["cuda", "hip"])
@pytest.mark.parametrize("kind", [tvm.tirx.ForKind.SERIAL, tvm.tirx.ForKind.UNROLLED])
def test_symbolic_loop_step_codegen(backend, kind):
    step = tvm.tirx.Var("step", "int32")
    source = _build_loop_source(backend, 3, 13, step, kind)

    assert "for (int i = 3; i < 13; i += step) {" in source


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_loop_step_preserves_lexical_scope(backend):
    source = _build_loop_source(backend, 3, 13, 3, tvm.tirx.ForKind.UNROLLED, lexical_scope=True)

    assert "#pragma unroll\n  for (int i = 3; i < 13; i += 3) {\n    output[i] = i;\n  }" in source


def test_cuda_loop_step_preserves_unroll_factor():
    source = _build_loop_source("cuda", 3, 13, 3, tvm.tirx.ForKind.UNROLLED, unroll_factor=2)

    assert "#pragma unroll 2\n  for (int i = 3; i < 13; i += 3) {\n    output[i] = i;\n  }" in source

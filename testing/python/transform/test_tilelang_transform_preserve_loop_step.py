import pytest

import numpy as np
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang as tl
import tvm_tirx
from tilelang.testing.packed import compile_packed_function


def test_lower_opaque_block_preserves_non_unit_loop_step():
    output_buffer = ir.Var("output", tirx.TensorType((6,), "int32"))
    i = ir.Var("i", "int32")
    loop = ir.For(
        i,
        1,
        6,  # Native end = old minimum (1) + extent (5).
        2,
        ir.ForKind.DEFAULT,
        [ir.TensorStore(output_buffer, [i], 1)],
    )
    before = tirx.Function(
        [output_buffer],
        [loop],
    ).with_attr("global_symbol", "main")

    mod = tl.transform.LowerSourceAnnotations()(tvm_tirx.IRModule.from_expr(before))
    executable = compile_packed_function(mod["main"])

    output = tvm_tirx.runtime.tensor(np.zeros(6, dtype="int32"))
    executable(output)

    np.testing.assert_array_equal(
        output.numpy(),
        np.array([0, 1, 0, 1, 0, 1], dtype="int32"),
    )


@pytest.mark.parametrize("rng", [(0, 6, 2), (0, 5, 2)], ids=lambda v: f"rng=({v[0]},{v[1]},{v[2]})")
@pytest.mark.parametrize("explicit", [False, True], ids=lambda v: f"explicit={v}")
def test_unroll_loop_preserves_non_unit_loop_step(rng, explicit):
    output_buffer = ir.Var("output", tirx.TensorType((8,), "int32"))
    i = ir.Var("i", "int32")
    start, stop, step = rng
    loop = ir.For(
        i,
        start,
        stop,
        step,
        ir.ForKind.UNROLLED,
        [ir.TensorStore(output_buffer, [i], i)],
        annotations={"pragma_unroll_explicit": explicit},
    )
    before = tirx.Function(
        [output_buffer],
        [loop],
    ).with_attr("global_symbol", "main")

    mod = tl.transform.UnrollLoop()(tvm_tirx.IRModule.from_expr(before))
    if not explicit:
        loops = []
        tvm_ffi.structural_walk(mod["main"].body, (ir.For, loops.append))
        assert len(loops) == 1
        assert int(loops[0].step) == step
        assert int(loops[0].begin) == start and int(loops[0].end) == stop
    # Test TileLang's pass and emitter without running a second, upstream
    # UnrollLoop as part of tirx.compile's default pipeline.
    executable = compile_packed_function(mod["main"])

    output = tvm_tirx.runtime.tensor(np.zeros(8, dtype="int32"))
    executable(output)

    np.testing.assert_array_equal(
        output.numpy(),
        np.array([0, 0, 2, 0, 4, 0, 0, 0], dtype="int32"),
    )


def test_unroll_loop_preserves_empty_non_unit_loop():
    output_buffer = ir.Var("output", tirx.TensorType((1,), "int32"))
    i = ir.Var("i", "int32")
    loop = ir.For(
        i,
        0,
        0,
        2,
        ir.ForKind.UNROLLED,
        [ir.TensorStore(output_buffer, [0], 1)],
        annotations={"pragma_unroll_explicit": True},
    )
    before = tirx.Function(
        [output_buffer],
        [loop],
    ).with_attr("global_symbol", "main")

    mod = tl.transform.UnrollLoop()(tvm_tirx.IRModule.from_expr(before))
    executable = compile_packed_function(mod["main"])

    output = tvm_tirx.runtime.tensor(np.zeros(1, dtype="int32"))
    executable(output)

    np.testing.assert_array_equal(output.numpy(), np.zeros(1, dtype="int32"))


def _attribute(node, key, value, body):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"node": node, "key": key, "value": value})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, [body])


def _build_loop_source(backend, start, stop, step, kind, unroll_factor=None, lexical_scope=False):
    i = ir.Var("i", "int32")
    output = ir.Var("output", tirx.TensorType((32,), "int32"))
    output_ptr = ir.Var("output", ir.PtrType("int32"))
    body = ir.TensorStore(output, [i], i)
    if lexical_scope:
        body = _attribute(None, "lexical_alloc_scope", 1, body)
    # Construct raw For IR because TileLang's frontend normalizes loop steps.
    annotations = {} if unroll_factor is None else {"pragma_unroll_factor": unroll_factor}
    body = ir.For(i, start, stop, 1 if step is None else step, kind, [body], annotations=annotations)
    # These tests enter the raw device emitter directly. A native device
    # function receives a physical pointer and declares its tensor view in
    # the body; public Tensor parameters belong to the packed host boundary.
    params = [output_ptr]
    params.extend(tirx.analysis.undefined_vars(body, [output]))
    declaration = ir.Bind(output, ir.Call("tirx.decl_tensor", [output_ptr], ty_args=[output.ty]))
    func = tirx.Function(params, [declaration, body])
    func = func.with_attr("global_symbol", "loop_step")
    func = func.with_attr("calling_conv", tvm_tirx.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    assert tirx.analysis.check_well_formed(func)
    mod = tvm_tirx.IRModule({"loop_step": func})
    build = tvm_ffi.get_global_func(f"target.build.tilelang_{backend}_without_compile", allow_missing=True)
    if build is None:
        pytest.skip(f"TileLang was built without the {backend} code generator")
    target_config = {"kind": "cuda", "arch": "sm_80"} if backend == "cuda" else {"kind": "hip", "mcpu": "gfx942"}
    # The native emitter consumes physical pointer operands. Preserve the raw
    # loop while applying only the annotation and tensor-address boundaries.
    target = tvm_tirx.target.Target(target_config)
    with target:
        mod = tl.transform.LowerSourceAnnotations()(mod)
        mod = tirx.transform.LowerTensorLayouts()(mod)
        mod = tirx.transform.FlattenTensor()(mod)
        return build(mod, target).inspect_source()


@pytest.mark.parametrize("backend", ["cuda", "hip"])
@pytest.mark.parametrize("kind", [ir.ForKind.DEFAULT, ir.ForKind.UNROLLED])
@pytest.mark.parametrize(
    "start, stop, step",
    [(0, 8, None), (0, 8, 1), (0, 8, 2), (3, 13, 3)],
)
def test_loop_step_codegen(backend, start, stop, step, kind):
    source = _build_loop_source(backend, start, stop, step, kind)

    # Native For always carries a step. Both spellings of one are canonical.
    increment = "++i" if step in (None, 1) else f"i += {step}"
    assert f"for (int i = {start}; i < {stop}; {increment}) {{" in source
    assert ("#pragma unroll\n" in source) == (kind == ir.ForKind.UNROLLED)


@pytest.mark.parametrize("backend", ["cuda", "hip"])
@pytest.mark.parametrize("start", [0, 3])
def test_loop_codegen_simplifies_expressions(backend, start):
    n = ir.Var("n", "int32")
    minimum = ir.prim.Add(ir.IntImm("int32", start), ir.IntImm("int32", 0))
    stop = ir.prim.Add(n, ir.IntImm("int32", 0))
    step = ir.prim.Add(ir.IntImm("int32", 1), ir.IntImm("int32", 1))
    source = _build_loop_source(backend, minimum, stop, step, ir.ForKind.DEFAULT)

    assert f"for (int i = {start}; i < n; i += 2) {{" in source


@pytest.mark.parametrize("backend", ["cuda", "hip"])
@pytest.mark.parametrize("kind", [ir.ForKind.DEFAULT, ir.ForKind.UNROLLED])
def test_symbolic_loop_step_codegen(backend, kind):
    step = ir.Var("step", "int32")
    source = _build_loop_source(backend, 3, 13, step, kind)

    assert "for (int i = 3; i < 13; i += step) {" in source


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_loop_step_preserves_lexical_scope(backend):
    source = _build_loop_source(backend, 3, 13, 3, ir.ForKind.UNROLLED, lexical_scope=True)

    # HIP keeps the explicit lexical body and an unfolded view offset. Both
    # statements store to precisely i for the complete loop domain [3, 13).
    direct = "#pragma unroll\n  for (int i = 3; i < 13; i += 3) {\n    output[i] = i;\n  }"
    lexical = "#pragma unroll\n  for (int i = 3; i < 13; i += 3) {\n    {\n      output[((i - 3) + 3)] = i;\n    }\n  }"
    assert direct in source or lexical in source
    assert source.count(" = i;") == 1


def test_cuda_loop_step_preserves_unroll_factor():
    source = _build_loop_source("cuda", 3, 13, 3, ir.ForKind.UNROLLED, unroll_factor=2)

    assert "#pragma unroll 2\n  for (int i = 3; i < 13; i += 3) {\n    output[i] = i;\n  }" in source

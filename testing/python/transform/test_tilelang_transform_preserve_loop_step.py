import re

import numpy as np
import pytest
import tvm_ffi
from tvm_tirx import ir, runtime, tirx
from tvm_tirx.ir import prim
from tvm_tirx.target import Target

import tilelang as tl
from tilelang._compat.tensor import TensorView


def _run_loop(output_size, begin, end, step, kind, value, transform, annotations=None):
    output = TensorView.create("output", [output_size], "int32")
    index = ir.Var("index", "int32")
    loop = ir.For(
        index,
        begin,
        end,
        step,
        kind,
        output.store(index, index if value is None else value),
        annotations or {},
    )
    function = tirx.Function([output.tensor], loop).with_attr("global_symbol", "main")
    module = transform(ir.IRModule({"main": function}))
    assert tirx.analysis.check_well_formed(module)
    executable = tirx.compile(module, target="c")
    result = runtime.tensor(np.zeros(output_size, dtype="int32"))
    executable(result)
    return result.numpy()


def test_lower_opaque_block_preserves_non_unit_loop_step():
    result = _run_loop(6, 1, 6, 2, ir.ForKind.DEFAULT, 1, tl.transform.LowerOpaqueBlock())
    np.testing.assert_array_equal(result, np.array([0, 1, 0, 1, 0, 1], dtype="int32"))


@pytest.mark.parametrize("rng", [(0, 6, 2), (0, 5, 2)])
@pytest.mark.parametrize("explicit", [False, True])
def test_unroll_loop_preserves_non_unit_loop_step(rng, explicit):
    begin, end, step = rng
    result = _run_loop(
        8,
        begin,
        end,
        step,
        ir.ForKind.UNROLLED,
        None,
        tl.transform.UnrollLoop(),
        {"pragma_unroll_explicit": explicit},
    )
    np.testing.assert_array_equal(result, np.array([0, 0, 2, 0, 4, 0, 0, 0], dtype="int32"))


def test_unroll_loop_preserves_empty_non_unit_loop():
    result = _run_loop(
        1,
        0,
        0,
        2,
        ir.ForKind.UNROLLED,
        1,
        tl.transform.UnrollLoop(),
        {"pragma_unroll_explicit": True},
    )
    np.testing.assert_array_equal(result, np.zeros(1, dtype="int32"))


def _build_loop_source(backend, start, stop, step, kind, unroll_factor=None, lexical_scope=False):
    index = ir.Var("index", "int32")
    output = TensorView.create("output_view", [32], "int32")
    pointer = ir.Var("output", ir.PtrType("int32"))
    body = output.store(index, index)
    if lexical_scope:
        body = ir.RegionStmt("tirx.scope", [], [], {}, body)
    annotations = {} if unroll_factor is None else {"pragma_unroll_factor": unroll_factor}
    body = ir.SeqStmt(
        [
            output.declare(pointer),
            ir.For(index, start, stop, 1 if step is None else step, kind, body, annotations),
        ]
    )
    parameters = [pointer]
    parameters.extend(tirx.analysis.undefined_vars(body, parameters))
    function = tirx.Function(parameters, body).with_attr("global_symbol", "loop_step")
    function = function.with_attr("tirx.is_global_func", True)
    build = tvm_ffi.get_global_func(f"target.build.tilelang_{backend}_without_compile", allow_missing=True)
    if build is None:
        pytest.skip(f"TileLang was built without the {backend} code generator")
    config = {"kind": "cuda", "arch": "sm_80"} if backend == "cuda" else {"kind": "rocm", "mcpu": "gfx942"}
    target = Target(config)
    module = ir.IRModule({"loop_step": function.with_attr("target", target)})
    with target:
        module = tirx.transform.LowerTensorLayouts()(module)
        module = tirx.transform.FlattenTensor()(module)
        return build(module, target).inspect_source()


def _assert_loop(source, start, stop, step):
    if isinstance(step, str):
        condition = f"(({step}) > 0 && index < {stop}) || (({step}) < 0 && index > {stop})"
    else:
        condition = f"index {'<' if step > 0 else '>'} {stop}"
    assert re.search(
        rf"for \((?:int|int32_t) index = {start}; {re.escape(condition)}; (?:index \+= {step}|\+\+index)\)",
        source,
    )
    if step != 1:
        assert f"index += {step}" in source


@pytest.mark.parametrize("backend", ["cuda", "hip"])
@pytest.mark.parametrize("kind", [ir.ForKind.DEFAULT, ir.ForKind.UNROLLED])
@pytest.mark.parametrize("start,stop,step", [(0, 8, None), (0, 8, 1), (0, 8, 2), (3, 13, 3), (13, 3, -2)])
def test_loop_step_codegen(backend, start, stop, step, kind):
    source = _build_loop_source(backend, start, stop, step, kind)
    _assert_loop(source, start, stop, 1 if step is None else step)
    assert ("#pragma unroll\n" in source) == (kind == ir.ForKind.UNROLLED)


@pytest.mark.parametrize("backend", ["cuda", "hip"])
@pytest.mark.parametrize("start", [0, 3])
def test_loop_codegen_simplifies_expressions(backend, start):
    extent = ir.Var("extent", "int32")
    minimum = prim.Add(ir.const(start), ir.const(0))
    stop = prim.Add(extent, ir.const(0))
    step = prim.Add(ir.const(1), ir.const(1))
    source = _build_loop_source(backend, minimum, stop, step, ir.ForKind.DEFAULT)
    _assert_loop(source, start, "extent", 2)


@pytest.mark.parametrize("backend", ["cuda", "hip"])
@pytest.mark.parametrize("kind", [ir.ForKind.DEFAULT, ir.ForKind.UNROLLED])
def test_symbolic_loop_step_codegen(backend, kind):
    step = ir.Var("step", "int32")
    source = _build_loop_source(backend, 3, 13, step, kind)
    _assert_loop(source, 3, 13, "step")


@pytest.mark.parametrize("backend", ["cuda", "hip"])
def test_loop_step_preserves_lexical_scope(backend):
    source = _build_loop_source(backend, 3, 13, 3, ir.ForKind.UNROLLED, lexical_scope=True)
    _assert_loop(source, 3, 13, 3)
    assert "#pragma unroll\n" in source
    assert "[index] = index;" in source


def test_cuda_loop_step_preserves_unroll_factor():
    source = _build_loop_source("cuda", 3, 13, 3, ir.ForKind.UNROLLED, unroll_factor=2)
    _assert_loop(source, 3, 13, 3)
    assert "#pragma unroll 2\n" in source

"""Graph operator recognition, Ascend data paths, and numerical integration."""

from types import SimpleNamespace

import pytest
import tilelang
import tilelang.testing
from tilelang.ascend.graph.operators import try_schedule_gemm, try_schedule_reduction
from tvm import te, tirx

from testing.ascend._ir import calls, nodes


_BACKENDS = [pytest.param("ascend", "tvm_ffi", id="ascendc"), pytest.param("pto", "pto", id="pto", marks=pytest.mark.pto)]
_TARGETS = ["ascend", pytest.param("pto", marks=pytest.mark.pto)]


def _compile(program, target, execution_backend):
    kernel = tilelang.compile(program, target=target, execution_backend=execution_backend, out_idx=-1)
    assert kernel.execution_backend == execution_backend
    if execution_backend == "pto":
        from tilelang.jit.adapter.pto import PTOKernelAdapter

        assert isinstance(kernel.adapter, PTOKernelAdapter)
    else:
        from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter

        assert isinstance(kernel.adapter, TVMFFIKernelAdapter)
    return kernel


def reduction_program(op="sum", shape=(7, 128), dtype="float16", keepdims=False):
    source = te.placeholder(shape, dtype=dtype, name="input")
    axis = te.reduce_axis((0, shape[-1]), name="k")
    out_shape = (*shape[:-1], 1) if keepdims else shape[:-1]

    def expression(*indices):
        index = indices[:-1] if keepdims else indices
        return (te.max if op == "max" else te.sum)(source[(*index, axis)], axis=axis)

    reduced = te.compute(out_shape, expression, name="reduced")
    output = (
        te.compute(out_shape, lambda *indices: reduced[indices] / tirx.const(shape[-1], dtype), name="mean") if op == "mean" else reduced
    )
    return te.create_prim_func([source, output])


def gemm_program(m=64, n=64, k=64, dtype="float16", out_dtype="float32", transpose_b=False, bias=False, relu=False):
    source = te.placeholder((m, k), dtype=dtype, name="input")
    weight = te.placeholder((n, k) if transpose_b else (k, n), dtype=dtype, name="weight")
    axis = te.reduce_axis((0, k), name="k")
    product = te.compute(
        (m, n),
        lambda i, j: te.sum(
            source[i, axis].astype("float32") * (weight[j, axis] if transpose_b else weight[axis, j]).astype("float32"), axis=axis
        ),
        name="matmul",
    )
    inputs = [source, weight]
    output = product
    if out_dtype != "float32":
        output = te.compute((m, n), lambda i, j: output[i, j].astype(out_dtype), name="cast")
    if bias:
        weight_bias = te.placeholder((n,), dtype=out_dtype, name="bias")
        inputs.append(weight_bias)
        before_bias = output
        output = te.compute((m, n), lambda i, j: before_bias[i, j] + weight_bias[j], name="add")
    if relu:
        before_relu = output
        output = te.compute((m, n), lambda i, j: te.max(before_relu[i, j], tirx.const(0, out_dtype)), name="relu")
    return te.create_prim_func([*inputs, output])


@pytest.mark.parametrize("op", ["sum", "max", "mean"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_last_axis_reduction_has_float32_epoch(op, keepdims):
    scheduled = try_schedule_reduction(reduction_program(op, keepdims=keepdims))
    assert scheduled is not None
    assert calls(scheduled, "tl.reducer_update")
    assert calls(scheduled, "tl.tileop.finalize_reducer_v2")
    updates = calls(scheduled, "tl.reducer_update")
    assert all(str(call.args[1].dtype) == "float32" for call in updates)
    assert any(block.name_hint == "SIMT_VF" for block in nodes(scheduled, tirx.SBlock))
    assert not any(axis.iter_type == tirx.IterVar.CommReduce for block in nodes(scheduled, tirx.SBlock) for axis in block.iter_vars)


@pytest.mark.parametrize("op", ["sum", "max", "mean"])
@pytest.mark.parametrize("target", _TARGETS)
def test_last_axis_reduction_codegen(op, target):
    scheduled = try_schedule_reduction(reduction_program(op))
    source = tilelang.lower(scheduled, target=target).kernel_source

    if target == "pto":
        assert "with pto.simt(" in source
        assert f"tl.simt_allreduce_{'max' if op == 'max' else 'sum'}" in source
    else:
        assert "__simt_vf__" in source
        assert "asc_vf_call" in source


def test_non_last_axis_reduction_is_not_misidentified():
    source = te.placeholder((64, 128), dtype="float32")
    axis = te.reduce_axis((0, 64), "k")
    output = te.compute((128,), lambda i: te.sum(source[axis, i], axis=axis))
    assert try_schedule_reduction(te.create_prim_func([source, output])) is None


def test_large_reduction_uses_fallback():
    assert try_schedule_reduction(reduction_program(shape=(1, 32768))) is None


@pytest.mark.parametrize("kind", ["reduction", "gemm"])
def test_partial_iteration_domain_uses_fallback(kind):
    program = reduction_program() if kind == "reduction" else gemm_program()
    inner = nodes(program, tirx.For)[0]

    def shorten(node):
        if isinstance(node, tirx.For) and node.same_as(inner):
            return tirx.For(node.loop_var, node.min, node.extent // 2, node.kind, node.body)

    partial = program.with_body(tirx.stmt_functor.ir_transform(program.body, None, shorten))
    schedule = try_schedule_reduction if kind == "reduction" else try_schedule_gemm
    assert schedule(partial) is None


def test_nearby_mean_scale_is_not_rewritten():
    source = te.placeholder((7, 128), dtype="float32")
    axis = te.reduce_axis((0, 128), "k")
    reduced = te.compute((7,), lambda i: te.sum(source[i, axis], axis=axis))
    output = te.compute((7,), lambda i: reduced[i] * tirx.const(1.0005 / 128, "float32"))
    assert try_schedule_reduction(te.create_prim_func([source, output])) is None


@pytest.mark.parametrize("transpose_b", [False, True], ids=["logical-nn", "logical-nt"])
def test_gemm_transposes_only_logical_nn_rhs(transpose_b):
    scheduled = try_schedule_gemm(gemm_program(m=32, n=64, k=16, transpose_b=transpose_b))
    assert scheduled is not None
    (gemm,) = calls(scheduled, "tl.tileop.gemm")
    assert int(gemm.args[4]) == 1
    copies = calls(scheduled, "tl.tileop.ascend_copy")
    transposed = [call for call in copies if call.annotations is not None and "transpose" in call.annotations]
    assert len(transposed) == (0 if transpose_b else 1)


@pytest.mark.parametrize("transpose_b", [False, True], ids=["logical-nn", "logical-nt"])
@pytest.mark.parametrize("target", _TARGETS)
def test_gemm_codegen(transpose_b, target):
    scheduled = try_schedule_gemm(gemm_program(m=32, n=64, k=16, transpose_b=transpose_b))
    artifact = tilelang.lower(scheduled, target=target)
    assert any(calls(func, "tl.ascend_gemm_l1") for func in artifact.device_mod.functions.values())
    if target == "pto":
        assert "tl.PTOGemmL1Template(" in artifact.kernel_source
        assert "run_l1_tile(" in artifact.kernel_source
    else:
        assert "ascend_gemm_l1<" in artifact.kernel_source


def test_gemm_epilogue_keeps_accumulator_in_ub():
    scheduled = try_schedule_gemm(gemm_program(dtype="bfloat16", out_dtype="bfloat16", bias=True, relu=True))
    assert scheduled is not None
    assert len(scheduled.params) == 4
    assert str(scheduled.attrs["tl.graph_schedule"]) == "gemm_epilogue"
    assert any(block.name_hint == "SIMT_VF" for block in nodes(scheduled, tirx.SBlock))
    # Only the original input, weight, bias and final output are global ABI
    # buffers; the accumulator handoff is an on-chip UB allocation.
    assert len(scheduled.buffer_map) == 4
    assert any(str(buffer.scope()) == "shared.dyn" for block in nodes(scheduled, tirx.SBlock) for buffer in block.alloc_buffers)


@pytest.mark.parametrize("target", _TARGETS)
def test_gemm_epilogue_codegen(target):
    scheduled = try_schedule_gemm(gemm_program(dtype="bfloat16", out_dtype="bfloat16", bias=True, relu=True))
    source = tilelang.lower(scheduled, target=target).kernel_source
    if target == "pto":
        assert "tl.PTOGemmL1Template(" in source
        assert "pto.mte_l0c_ub(" in source
        assert "with pto.simt(" in source
    else:
        assert "asc_copy_l0c2ub" in source


@pytest.mark.parametrize("m,n", [(32, 64), (64, 32)], ids=["wide", "tall"])
@pytest.mark.pto
def test_pto_rectangular_gemm_epilogue_uses_nz_source_stride(m, n):
    scheduled = try_schedule_gemm(gemm_program(m=m, n=n, k=16, dtype="bfloat16", out_dtype="bfloat16", bias=True, relu=True))
    artifact = tilelang.lower(scheduled, target="pto")
    drains = [call for func in artifact.device_mod.functions.values() for call in calls(func, "tl.ascend_copy_matrix_cc_to_ub")]
    assert len(drains) == 1
    (drain,) = drains
    # NZ source panels span M rows; the ND destination row spans N columns.
    assert [int(value) for value in drain.args[3:7]] == [n, m, n, m]
    mte = next(line for line in artifact.kernel_source.splitlines() if "pto.mte_l0c_ub(" in line)
    assert f', {m}, {n}, {m}, {n}, 0, layout="nz2nd")' in mte


@pytest.mark.parametrize("kwargs", [{"m": 17}, {"k": 15}, {"dtype": "float32"}])
def test_gemm_rejects_unsupported_geometry_and_dtype(kwargs):
    assert try_schedule_gemm(gemm_program(**kwargs)) is None


def test_gemm_rejects_narrowed_products():
    source = te.placeholder((16, 16), dtype="float16")
    weight = te.placeholder((16, 16), dtype="float16")
    axis = te.reduce_axis((0, 16), "k")
    output = te.compute(
        (16, 16),
        lambda i, j: te.sum(
            (source[i, axis].astype("float32") * weight[axis, j].astype("float32")).astype("float16").astype("float32"), axis=axis
        ),
    )
    # Cube fp32 accumulation must not erase an explicitly requested rounding
    # of each product to fp16 before it is accumulated.
    assert try_schedule_gemm(te.create_prim_func([source, weight, output])) is None


def test_gemm_rejects_externally_visible_accumulator():
    source = te.placeholder((16, 16), dtype="float16")
    weight = te.placeholder((16, 16), dtype="float16")
    axis = te.reduce_axis((0, 16), "k")
    product = te.compute((16, 16), lambda i, j: te.sum(source[i, axis].astype("float32") * weight[axis, j].astype("float32"), axis=axis))
    output = te.compute((16, 16), lambda i, j: te.max(product[i, j], tirx.const(0, "float32")))
    # A FuseTIR group with both a producer result and its epilogue returned has
    # two output ABI buffers. Replacing the producer with only an UB handoff
    # would lose the externally visible result.
    assert try_schedule_gemm(te.create_prim_func([source, weight, product, output])) is None


def test_bad_schedule_configuration_is_diagnostic():
    with pytest.raises(ValueError, match="num_cores"):
        try_schedule_reduction(reduction_program(), SimpleNamespace(num_cores=0))


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target,execution_backend", _BACKENDS)
@pytest.mark.parametrize("op,dtype", [("sum", "float16"), ("mean", "bfloat16"), ("max", "float32")])
def test_graph_reduction_numerics(op, dtype, target, execution_backend):
    import torch

    scheduled = try_schedule_reduction(reduction_program(op, dtype=dtype))
    compiled = _compile(scheduled, target, execution_backend)
    values = torch.randn((7, 128), dtype=getattr(torch, dtype), device="npu")
    if op == "max":
        values[0].fill_(-float("inf"))
        values[1, 3] = float("nan")
        values[2, 5] = float("inf")
    expected = {"sum": torch.sum, "mean": torch.mean, "max": torch.amax}[op](values, dim=-1)
    actual = compiled(values)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2, equal_nan=True)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target,execution_backend", _BACKENDS)
@pytest.mark.parametrize("transpose_b,bias", [(False, False), (True, False), (False, True)])
def test_graph_gemm_numerics(transpose_b, bias, target, execution_backend):
    import torch

    program = gemm_program(m=32, n=64, k=16, dtype="bfloat16", out_dtype="bfloat16", transpose_b=transpose_b, bias=bias, relu=bias)
    scheduled = try_schedule_gemm(program)
    compiled = _compile(scheduled, target, execution_backend)
    a = torch.randn((32, 16), dtype=torch.bfloat16, device="npu") * 0.125
    b = torch.randn((64, 16) if transpose_b else (16, 64), dtype=torch.bfloat16, device="npu") * 0.125
    expected = (a.float() @ (b.float().T if transpose_b else b.float())).to(torch.bfloat16)
    arguments = [a, b]
    if bias:
        offset = torch.randn((64,), dtype=torch.bfloat16, device="npu")
        expected = torch.relu(expected + offset)
        arguments.append(offset)
    actual = compiled(*arguments)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target,execution_backend", _BACKENDS)
def test_graph_gemm_pipeline_and_tile_epochs_numerics(target, execution_backend):
    import torch

    program = gemm_program(m=128, n=192, k=256, dtype="bfloat16", out_dtype="bfloat16", bias=True, relu=True)
    scheduled = try_schedule_gemm(program, SimpleNamespace(num_cores=2))
    compiled = _compile(scheduled, target, execution_backend)
    a = torch.randn((128, 256), dtype=torch.bfloat16, device="npu") * 0.125
    b = torch.randn((256, 192), dtype=torch.bfloat16, device="npu") * 0.125
    bias = torch.randn((192,), dtype=torch.bfloat16, device="npu")
    expected = torch.relu((a.float() @ b.float()).to(torch.bfloat16) + bias)
    actual = compiled(a, b, bias)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    # Retaining the previous result also covers output/epoch reuse on a second
    # launch with changed inputs and fresh callee-allocated output storage.
    previous = actual.clone()
    again = compiled(a * 2, b, bias)
    torch.testing.assert_close(actual, previous, rtol=0, atol=0)
    torch.testing.assert_close(again, torch.relu((a.float() * 2 @ b.float()).to(torch.bfloat16) + bias), rtol=2e-2, atol=2e-2)

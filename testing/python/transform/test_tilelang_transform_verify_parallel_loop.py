import tilelang as tl
import pytest
import tvm_tirx as tvm
from tvm_tirx import ir, tirx
from tilelang import ir as tl_ir


def _make_parallel_store(*, injective_indices: bool, varying_value: bool = True):
    output = ir.Var("output", tirx.TensorType((32, 128), "int32", layout="default"))
    ti = ir.Var("ti", "int32")
    tj = ir.Var("tj", "int32")
    i = ir.Var("i", "int32")
    j = ir.Var("j", "int32")

    indices = [i, j] if injective_indices else [0, 0]
    value = ti * 128 + tj if varying_value else 1
    store = ir.TensorStore(output, indices, value)
    body = ir.SeqStmt([ir.Bind(i, ti), ir.Bind(j, tj), store])
    body = ir.For(tj, 0, 128, 1, ir.ForKind.PARALLEL, body)
    body = ir.For(ti, 0, 32, 1, ir.ForKind.PARALLEL, body)
    return tvm.IRModule({"main": tirx.Function([output], body, ret_type=ir.PrimType("void"))})


def _verify(mod, capfd):
    tl.transform.VerifyParallelLoop()(mod)
    return capfd.readouterr().err


def test_flat_bind_indices_are_proven_race_free(capfd):
    mod = _make_parallel_store(injective_indices=True)

    assert "Data race detected" not in _verify(mod, capfd)


def test_irrelevant_flat_binds_do_not_hide_race(capfd):
    mod = _make_parallel_store(injective_indices=False)

    assert "Data race detected" in _verify(mod, capfd)


def test_same_value_parallel_store_is_allowed(capfd):
    mod = _make_parallel_store(injective_indices=False, varying_value=False)

    assert "Data race detected" not in _verify(mod, capfd)


def _make_two_racy_stores():
    out_a = ir.Var("out_a", tirx.TensorType((32,), "int32", layout="default"))
    out_b = ir.Var("out_b", tirx.TensorType((32,), "int32", layout="default"))
    ti = ir.Var("ti", "int32")
    store_a = ir.TensorStore(out_a, [0], ti)
    store_b = ir.TensorStore(out_b, [0], ti)
    body = ir.SeqStmt([store_a, store_b])
    body = ir.For(ti, 0, 32, 1, ir.ForKind.PARALLEL, body)
    mod = tvm.IRModule({"main": tirx.Function([out_a, out_b], body, ret_type=ir.PrimType("void"))})
    return mod, store_a, store_b


def test_data_race_diagnostic_includes_span(capfd):
    mod, store_a, store_b = _make_two_racy_stores()
    tl_ir.set_stmt_span(store_a, tl_ir.make_span("kernel.py", 21))
    tl_ir.set_stmt_span(store_b, tl_ir.make_span("kernel.py", 22))

    err = _verify(mod, capfd)
    assert err.count("Data race detected") == 1
    assert "[1]" in err
    assert "--> kernel.py:21:1" in err
    assert "[2]" in err
    assert "--> kernel.py:22:1" in err


def test_race_check_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TILELANG_ENABLE_DATA_RACE_CHECK", raising=False)
    from tilelang.backend.pass_pipeline import pipeline_utils

    assert pipeline_utils.should_enable_race_check() is False


def test_race_check_enabled_via_env_var(monkeypatch):
    monkeypatch.setenv("TILELANG_ENABLE_DATA_RACE_CHECK", "1")
    from tilelang.backend.pass_pipeline import pipeline_utils

    assert pipeline_utils.should_enable_race_check() is True


def test_race_check_pass_config_overrides_env_var(monkeypatch):
    monkeypatch.setenv("TILELANG_ENABLE_DATA_RACE_CHECK", "1")
    from tilelang.backend.pass_pipeline import pipeline_utils
    from tilelang.transform import PassContext

    with PassContext(config={"tl.disable_data_race_check": True}):
        assert pipeline_utils.should_enable_race_check() is False


def _native_module(output, body):
    return tvm.IRModule({"main": tirx.Function([output], body, ret_type=ir.PrimType("void"))})


@pytest.mark.parametrize("bind_read", [False, True])
def test_distinct_parallel_mutable_reads_do_not_prove_same_value(bind_read, capfd):
    output = ir.Var("output", tirx.TensorType((1,), "int32", layout="default"))
    local = ir.Var("local", tirx.TensorType((1,), "int32", scope="local", layout="default"))
    index = ir.Var("index", "int32")
    body = [
        ir.Bind(local, ir.Call("tirx.alloc_tensor", [], ty_args=[local.ty])),
        ir.TensorStore(local, [0], index),
    ]
    value = ir.TensorLoad(local, [0])
    if bind_read:
        captured = ir.Var("captured", "int32")
        body.append(ir.Bind(captured, value))
        value = captured
    body.append(ir.TensorStore(output, [0], value))
    mod = _native_module(output, [ir.For(index, 0, 32, 1, ir.ForKind.PARALLEL, body)])
    assert "Data race detected" in _verify(mod, capfd)


def test_nested_guard_is_restored_before_following_store(capfd):
    output = ir.Var("output", tirx.TensorType((2,), "int32", layout="default"))
    index = ir.Var("index", "int32")
    safe = ir.If(index == 0, [ir.TensorStore(output, [0], index)], None)
    unsafe = ir.TensorStore(output, [1], index)
    mod = _native_module(output, [ir.For(index, 0, 32, 1, ir.ForKind.PARALLEL, [safe, unsafe])])
    err = _verify(mod, capfd)
    assert "1 potential race(s)" in err


def test_bound_boolean_guard_keeps_source_order_constraints(capfd):
    output = ir.Var("output", tirx.TensorType((1,), "int32", layout="default"))
    index = ir.Var("index", "int32")
    condition = ir.Var("condition", "bool")
    body = [ir.Bind(condition, index == 0), ir.If(condition, [ir.TensorStore(output, [0], index)], None)]
    mod = _native_module(output, [ir.For(index, 0, 32, 1, ir.ForKind.PARALLEL, body)])
    assert "Data race detected" not in _verify(mod, capfd)


def test_read_captured_before_parallel_region_remains_shared(capfd):
    output = ir.Var("output", tirx.TensorType((1,), "int32", layout="default"))
    captured = ir.Var("captured", "int32")
    index = ir.Var("index", "int32")
    body = [
        ir.Bind(captured, ir.TensorLoad(output, [0])),
        ir.For(index, 0, 32, 1, ir.ForKind.PARALLEL, [ir.TensorStore(output, [0], captured)]),
    ]
    assert "Data race detected" not in _verify(_native_module(output, body), capfd)


def test_native_pass_honors_explicit_disable_config(capfd):
    from tilelang.transform import PassContext

    with PassContext(config={"tl.disable_data_race_check": True}):
        assert "Data race detected" not in _verify(_make_parallel_store(injective_indices=False), capfd)


if __name__ == "__main__":
    pytest.main([__file__])

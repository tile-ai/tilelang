"""Two-CTA requirements belong to the native kernel that owns each GEMM."""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx


def integer(value):
    return ir.IntImm("int32", value)


def objects(value, kind):
    result = []
    tvm_ffi.structural_walk(value, (kind, result.append), order="pre")
    return result


def kernel(body, dims=None, **hints):
    metadata = dict(hints)
    if dims is not None:
        metadata["cluster_dims"] = dims
    values = {"node": None, "key": "tl.kernel", "value": metadata}
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")(values)
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def hints(region):
    values = tvm_ffi.get_global_func("tl.DecodeAnnotations")(region.args[0], region.attrs["annotation_schema"])
    return values["value"]


def gemm(enabled=1, blockscaled=False, name="gemm"):
    from tilelang.language._native import call_intrin

    tensors = [
        ir.Var("A", tirx.TensorType((128, 32), "float16", scope="shared")),
        ir.Var("B", tirx.TensorType((128, 32), "float16", scope="shared")),
        ir.Var("C", tirx.TensorType((128, 128), "float32", scope="shared.tmem")),
    ]
    args = [
        *tensors,
        ir.IntImm("bool", 0),
        ir.IntImm("bool", 1),
        integer(128),
        integer(128),
        integer(32),
        integer(0),
        ir.IntImm("bool", 1),
        integer(0),
        integer(0),
        integer(0),
    ]
    if blockscaled:
        for name in ("SFA", "SFB"):
            scale = ir.Var(name, tirx.TensorType((128, 4), "uint32", scope="shared.tmem"))
            tensors.append(scale)
            args.append(scale)
        args.append(integer(0))
    attributes = {} if enabled is None else {"use_2cta": enabled if isinstance(enabled, ir.Expr) else integer(enabled)}
    call = call_intrin("handle", "tl.tileop.gemm_blockscaled" if blockscaled else "tl.tileop." + name, *args, annotations=attributes)
    # Verify these are actual native TileOperator objects, rather than a test
    # registration that happens to have the expected operation name.
    assert tvm_ffi.get_global_func("tl.testing.ParseNativeOperator")(call) is not None
    return [*(ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty])) for tensor in tensors), ir.Evaluate(call)]


def apply(body, tmp_path, arch="sm_100a", params=()):
    function = (
        tirx.Function(list(params), body)
        .with_attr("global_symbol", "main")
        .with_attr("target", tvm_tirx.target.Target({"kind": "cuda", "arch": arch}))
    )
    assert tirx.analysis.check_well_formed(function)
    before = ir.save_json(function)
    operation = tvm_ffi.get_global_func("tl.cuda.transform.LowerBlackwell2SM")()
    result = operation(tvm_tirx.IRModule({"main": function}))
    ir.assert_structural_equal(operation(result), result)
    assert tirx.analysis.check_well_formed(result)
    assert ir.save_json(function) == before
    (tmp_path / "before.json").write_text(before)
    (tmp_path / "after.json").write_text(ir.save_json(result))
    return result["main"]


@pytest.mark.parametrize("dims", [(2, 1, 1), (1, 2, 1)])
@pytest.mark.parametrize("name,blockscaled", [("gemm", False), ("tcgen05_gemm", False), ("wgmma_gemm", False), ("gemm_blockscaled", True)])
def test_dense_and_blockscaled_gemm_annotate_their_kernel(dims, name, blockscaled, tmp_path):
    result = apply([kernel(gemm(blockscaled=blockscaled, name=name), dims, unrelated="preserve")], tmp_path)
    (region,) = [value for value in objects(result, ir.RegionStmt) if value.op.name == "tl.attr"]
    metadata = hints(region)
    assert int(metadata["use_2cta"]) == 1
    assert [int(item) for item in metadata["cluster_dims"]] == list(dims)
    assert metadata["unrelated"] == "preserve"


@pytest.mark.parametrize("dims", [None, (1, 1, 1), (2, 2, 1), (1, 1, 2), (2, 1), (4, 1, 1)])
def test_two_cta_requires_the_owning_kernel_dimensions(dims, tmp_path):
    with pytest.raises(ValueError, match="on its own kernel"):
        apply([kernel(gemm(), dims)], tmp_path)


def test_valid_neighbor_cannot_justify_an_invalid_kernel(tmp_path):
    with pytest.raises(ValueError, match="on its own kernel"):
        apply([kernel(gemm(), (2, 1, 1)), kernel(gemm(), (1, 1, 1))], tmp_path)


def test_two_cta_does_not_leak_to_a_one_cta_sibling(tmp_path):
    result = apply([kernel(gemm(), (2, 1, 1)), kernel(gemm(0), (1, 1, 1))], tmp_path)
    regions = [value for value in objects(result, ir.RegionStmt) if value.op.name == "tl.attr"]
    assert int(hints(regions[0])["use_2cta"]) == 1
    assert "use_2cta" not in hints(regions[1])


@pytest.mark.parametrize("enabled", [0, None])
def test_ordinary_gemm_does_not_acquire_a_two_cta_requirement(enabled, tmp_path):
    result = apply([kernel(gemm(enabled))], tmp_path)
    assert "use_2cta" not in hints(result.body.seq[0])


def test_non_blackwell_target_is_left_to_its_instruction_selector(tmp_path):
    result = apply([kernel(gemm())], tmp_path, arch="sm_90")
    assert "use_2cta" not in hints(result.body.seq[0])


def test_dynamic_use_two_cta_fails_before_allocation(tmp_path):
    flag = ir.Var("flag", "int32")
    with pytest.raises(ValueError, match="use_2cta must be a constant"):
        apply([kernel(gemm(flag), (2, 1, 1))], tmp_path, params=[flag])


def test_native_kernel_before_materialization_is_supported(tmp_path):
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"cluster_dims": [2, 1, 1]})
    region = ir.RegionStmt(
        "tl.launch_kernel",
        [ir.Tuple([integer(0)]), ir.Tuple([integer(8)]), ir.Tuple([integer(1)]), ir.Tuple([integer(128), integer(1), integer(1)]), payload],
        [ir.Var(name, "int32") for name in ("block", "tx", "ty", "tz")],
        {"schema_version": 1, "hint_schema": schema, "threads_was_specified": True},
        gemm(),
    )
    result = apply([region], tmp_path)
    changed = result.body.seq[0]
    metadata = tvm_ffi.get_global_func("tl.DecodeAnnotations")(changed.args[4], changed.attrs["hint_schema"])
    assert int(metadata["use_2cta"]) == 1

"""TileLang owns storage policy while preserving typed native allocations."""

import pytest
import torch
import tvm_ffi
from tvm_tirx import IRModule, ir, tirx
from tvm_tirx.target import Target

import tilelang
import tilelang.cpu.language as T


def allocation_vars(module):
    result = []

    def visit(node):
        if isinstance(node.value, ir.Call) and node.value.op.name == "tirx.alloc_tensor":
            result.append(node.var)

    tvm_ffi.structural_walk(module, (ir.Bind, visit), order="pre")
    return result


def physical_module(function):
    module = IRModule({"main": function})
    module = tirx.transform.BindTarget(Target({"kind": "cuda", "arch": "sm_80"}))(module)
    # Follow the production preparation boundary: source-allocation ownership
    # metadata is consumed before either storage planner receives physical IR.
    module = tilelang.transform.LowerSourceAllocations()(module)
    module = tilelang.transform.LowerSourceAnnotations()(module)
    module = tilelang.transform.PrepareTensorLayouts()(module)
    module = tirx.transform.LowerTensorLayouts()(module)
    return tirx.transform.FlattenTensor()(module)


def kernel(scope="local"):
    @T.prim_func
    def main(output: T.Tensor((128,), "int32")):
        A = T.alloc_local((128,), "int32", scope=scope)
        for i in T.serial(128):
            A[i] = i
        B = T.alloc_local((128,), "int32", scope=scope)
        for i in T.serial(128):
            B[i] = A[i] + 3
        for i in T.serial(128):
            output[i] = B[i]

    return main


@pytest.mark.parametrize("detect_inplace", [None, False, True])
@pytest.mark.parametrize("scope", ["shared", "local"])
def test_config_is_read_at_pass_execution(detect_inplace, scope):
    # A pass constructed outside the context still honors the context at use.
    rewrite = tilelang.transform.StorageRewrite()
    config = {} if detect_inplace is None else {"tl.storage_rewrite_detect_inplace": detect_inplace}
    # Match the real pipeline: storage planning follows target binding and
    # physical flattening. This case inspects IR without running CUDA code.
    module = physical_module(kernel(scope))
    before = ir.save_json(module)
    with ir.transform.PassContext(config=config):
        after = rewrite(module)
    assert len(allocation_vars(after)) == (1 if detect_inplace and scope == "shared" else 2)
    assert tirx.analysis.check_well_formed(after)
    assert ir.save_json(module) == before
    # Applying the same pass again outside the configured context uses defaults.
    assert len(allocation_vars(rewrite(module))) == 2


def test_default_policy_does_not_reuse_disjoint_lifetimes():
    @T.prim_func
    def main(output: T.Tensor((2,), "int32")):
        A = T.alloc_shared((128,), "int32", scope="shared")
        for i in T.serial(128):
            A[i] = i
        output[0] = A[127]
        B = T.alloc_shared((128,), "int32", scope="shared")
        for i in T.serial(128):
            B[i] = i + 3
        output[1] = B[127]

    module = physical_module(main)
    assert len(allocation_vars(tilelang.transform.StorageRewrite()(module))) == 2
    assert len(allocation_vars(tirx.transform.StorageRewrite()(module))) == 1


@pytest.mark.parametrize("detect_inplace", [False, True])
def test_public_cpu_pipeline_storage_policy_preserves_result(detect_inplace):
    target = Target("c")
    compiled = tilelang.compile(
        kernel(),
        out_idx=[0],
        target=target,
        target_host=target,
        execution_backend="tvm_ffi",
        pass_configs={"tl.storage_rewrite_detect_inplace": detect_inplace},
    )
    torch.testing.assert_close(compiled(), torch.arange(128, dtype=torch.int32) + 3, rtol=0, atol=0)


def test_scalar_storage_element_type_is_preserved():
    @T.prim_func
    def main(output: T.Tensor((4,), "float32")):
        A = T.alloc_local((4,), "float32")
        A[T.Ramp(0, 1, 4)] = T.Broadcast(T.float32(5), 4)
        output[T.Ramp(0, 1, 4)] = A[T.Ramp(0, 1, 4)]

    after = tilelang.transform.StorageRewrite()(IRModule({"main": main}))
    assert str(allocation_vars(after)[0].ty.dtype) == "float32"
    assert after["main"].params[0].same_as(main.params[0])
    assert tirx.analysis.check_well_formed(after)


def _pointwise_pair(case="pointwise", scope="shared", size=8):
    typ = tirx.TensorType([size], "int32", scope=scope)
    source, destination = ir.Var("source", typ), ir.Var("destination", typ)
    output = ir.Var("output", tirx.TensorType([size], "int32"))
    i = ir.Var("i", "int32")
    source_index = (i + 1) % size if case == "shifted" else ir.const(0) if case == "repeated" else i
    destination_index = ir.const(0) if case == "repeated" else i
    prefix = [ir.Bind(source, ir.Call("tirx.alloc_tensor", [], ty_args=[typ]))]
    if case == "escaped":
        alias = ir.Var("alias", typ)
        prefix.append(ir.Bind(alias, ir.Call("tirx.decl_tensor", [ir.Call("tirx.tensor_data_ptr", [source])], ty_args=[typ])))
    prefix.append(
        ir.Bind(destination, ir.Call("tirx.alloc_tensor", [], attrs={"tirx.volatile": True} if case == "volatile" else {}, ty_args=[typ]))
    )
    prefix.append(
        ir.For(
            i, 0, size, 1, ir.ForKind.DEFAULT, [ir.TensorStore(destination, [destination_index], ir.TensorLoad(source, [source_index]) + 3)]
        )
    )
    prefix.append(ir.TensorStore(output, [0], ir.TensorLoad(destination, [0])))
    if case in ("live_source", "escaped"):
        prefix.append(ir.TensorStore(output, [1], ir.TensorLoad(source if case == "live_source" else alias, [0])))
    attributes = {"global_symbol": "main", "target": Target("c")}
    if case == "alignment_metadata":
        attributes["tl.smem_alignment_map"] = {"destination": 128}
    return IRModule({"main": tirx.Function([output], prefix, attrs=ir.DictAttrs(attributes))})


@pytest.mark.parametrize("case", ["shifted", "repeated", "live_source", "escaped", "volatile", "alignment_metadata"])
def test_inplace_rejects_nonpointwise_or_observable_allocation_identity(case):
    module = _pointwise_pair(case)
    before = ir.save_json(module)
    with ir.transform.PassContext(config={"tl.storage_rewrite_detect_inplace": True}):
        result = tilelang.transform.StorageRewrite()(module)
    assert len(allocation_vars(result)) == 2
    assert tirx.analysis.check_well_formed(result)
    assert ir.save_json(module) == before


@pytest.mark.parametrize(
    "scope,size,expected", [("shared", 8, 1), ("shared.dyn", 8, 1), ("global", 8, 1), ("local", 8, 2), ("local", 1, 1)]
)
def test_inplace_preserves_scope_policy_and_scalar_register_elimination(scope, size, expected):
    module = _pointwise_pair(scope=scope, size=size)
    with ir.transform.PassContext(config={"tl.storage_rewrite_detect_inplace": True}):
        result = tilelang.transform.StorageRewrite()(module)
    assert len(allocation_vars(result)) == expected
    assert tirx.analysis.check_well_formed(result)
    assert all(str(var.ty.scope) == scope for var in allocation_vars(result))


def _pointwise_pair_with_loop(dtype, begin, end, step):
    module = _pointwise_pair(size=256)
    loop = next(node for node in module["main"].body.seq if isinstance(node, ir.For))
    index = ir.Var("index", dtype)
    store = loop.body.seq[0]
    replacement = ir.For(
        index,
        ir.const(begin, dtype),
        ir.const(end, dtype),
        ir.const(step, dtype),
        ir.ForKind.DEFAULT,
        [ir.TensorStore(store.dest, [index], ir.TensorLoad(store.value.a.source, [index]) + 3)],
    )
    body = [replacement if node.same_as(loop) else node for node in module["main"].body.seq]
    return IRModule({"main": module["main"].with_body(body)})


@pytest.mark.parametrize("dtype,begin,end,step", [("uint8", 250, 251, 10), ("int8", 120, 121, 127), ("int32", 0, 3, -1)])
def test_inplace_does_not_assume_wrapping_or_descending_loop_indices_are_unique(dtype, begin, end, step):
    # These IR-only negative cases are not executed: wrapping can revisit an
    # index, and a negative step does not implement a descending native For.
    module = _pointwise_pair_with_loop(dtype, begin, end, step)
    before = ir.save_json(module)
    with ir.transform.PassContext(config={"tl.storage_rewrite_detect_inplace": True}):
        result = tilelang.transform.StorageRewrite()(module)
    assert len(allocation_vars(result)) == 2
    assert tirx.analysis.check_well_formed(result)
    assert ir.save_json(module) == before


@pytest.mark.parametrize("dtype,begin,end,step", [("uint8", 240, 250, 5), ("int8", 100, 120, 3), ("int32", 1, 8, 2)])
def test_inplace_accepts_provably_nonwrapping_nonunit_loop_steps(dtype, begin, end, step):
    module = _pointwise_pair_with_loop(dtype, begin, end, step)
    with ir.transform.PassContext(config={"tl.storage_rewrite_detect_inplace": True}):
        result = tilelang.transform.StorageRewrite()(module)
    assert len(allocation_vars(result)) == 1
    assert tirx.analysis.check_well_formed(result)

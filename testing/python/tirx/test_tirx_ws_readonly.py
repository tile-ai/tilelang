import pytest
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
from tilelang import language as T
from tilelang._compat import stmt_functor


pytestmark = pytest.mark.usefixtures("native_tirx_passes")


def nodes(root, kind):
    result = []
    stmt_functor.post_order_visit(root, lambda node: result.append(node) if isinstance(node, kind) else None)
    return result


def masked_copy(mutate):
    @T.prim_func
    def kernel(source: T.Tensor((2, 32, 128), T.float16), metadata: T.Tensor((2,), T.int32), output: T.Tensor((2, 32, 128), T.float16)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((32, 128), T.float16)
            fragment = T.alloc_fragment((32, 128), T.float16)
            mask = T.alloc_fragment((2,), T.int32)
            T.copy(metadata, mask)
            for iteration in T.Pipelined(2, num_stages=1):
                if mask[iteration] != 0:
                    T.copy(source[iteration, 0, 0], shared, prefer_instruction="tma")
                    T.copy(shared, fragment)
                    T.copy(fragment, output[iteration, 0, 0])
            if mutate:
                metadata[0] = 1

    return kernel


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("policy", ["default", "noalias", "may_alias", "nonrestrict_pair", "nonrestrict_single", "mutable"])
def test_readonly_metadata_replay_respects_parameter_alias_contract(automatic, policy):
    function = masked_copy(policy == "mutable").with_attr("global_symbol", "main")
    if policy in ("noalias", "may_alias"):
        function = function.with_attr("tirx.noalias", policy == "noalias")
    if policy.startswith("nonrestrict"):
        function = function.with_attr("tl.non_restrict_param_indices", [1, 2] if policy == "nonrestrict_pair" else [1])
    module = ir.IRModule({"main": function})
    module = tirx.transform.BindTarget(Target({"kind": "cuda", "arch": "sm_100a"}))(module)
    module = tilelang.transform.MaterializeKernelLaunch()(module)
    module = tilelang.transform.Simplify()(module)
    rejected = policy in ("may_alias", "nonrestrict_pair", "mutable")

    def schedule():
        if automatic:
            with ir.transform.PassContext(config={"tl.enable_auto_warp_specialization": "role_based"}):
                result = tilelang.cuda.transform.AutoWarpSpecialization()(module)
            return tilelang.cuda.transform.MaterializeWSSchedule()(result)
        return tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(module)

    if rejected:
        with pytest.warns(RuntimeWarning, match="another role's private state"):
            result = schedule()
        assert not any(region.attrs.get("tl.ws_concurrent") for region in nodes(result, ir.RegionStmt))
        return
    result = schedule()
    assert tirx.analysis.check_well_formed(result)
    ir.assert_structural_equal(result, ir.load_json(ir.save_json(result)))
    groups = [region for region in nodes(result, ir.RegionStmt) if region.attrs.get("tl.ws_concurrent")]
    assert len(groups) == 1
    masks = {variable for variable in nodes(result, ir.Var) if variable.name == "mask" and isinstance(variable.ty, tirx.TensorType)}
    assert len(masks) == 2
    used_by_roles = [{variable for variable in nodes(role, ir.Var) if variable in masks} for role in groups[0].body.seq]
    assert len([used for used in used_by_roles if used]) == 2
    assert all(len(used) == 1 for used in used_by_roles if used)

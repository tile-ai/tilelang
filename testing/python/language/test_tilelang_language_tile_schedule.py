import tilelang.language as T
import tilelang.testing
from tilelang.testing.ir import collect_calls, collect_nodes
from tvm_tirx import ir, sym, tirx


def test_persistent_tile_scheduler_scalar_state_access():
    @T.prim_func
    def persistent(A: T.Tensor((8,), T.int32)):
        with T.Kernel(1, threads=1) as (block_id,):
            sched = T.PersistentTileScheduler(4, 2, num_workers=1, swizzle_size=1, name="sched")
            sched.init(block_id)
            while sched.valid():
                A[sched.m_idx] = sched.n_idx + sched.current_iter
                sched.next_tile()

    script = persistent.script()
    assert "sched_m_idx" in script
    assert "sched_n_idx" in script
    assert "sched_current_iter" in script
    assert "while" in script
    assert tirx.analysis.check_well_formed(persistent)
    state = {
        binding.var.name: binding.var
        for binding in collect_nodes(persistent, ir.Bind)
        if isinstance(binding.value, ir.Call) and binding.value.op.same_as(ir.Op.get("tirx.alloc_tensor"))
    }
    assert set(state) == {"sched_m_idx", "sched_n_idx", "sched_linear_idx", "sched_current_iter"}
    assert all(str(variable.ty.scope) == "local.var" for variable in state.values())
    (loop,) = collect_nodes(persistent, ir.While)
    ir.assert_structural_equal(loop.condition, ir.TensorLoad(state["sched_linear_idx"], [0]) < 8)
    for name in ("sched_current_iter", "sched_linear_idx"):
        (update,) = [store for store in collect_nodes(loop.body, ir.TensorStore) if store.dest.same_as(state[name])]
        ir.assert_structural_equal(update.value, ir.TensorLoad(state[name], [0]) + 1)


def test_persistent_tile_scheduler_stateless_coord_access():
    @T.prim_func
    def stateless(A: T.Tensor((8,), T.int32)):
        with T.Kernel(1, threads=1) as (block_id,):
            sched = T.PersistentTileScheduler(4, 2, num_workers=1, swizzle_size=1, stateful=False)
            bx, by = sched.coord(block_id)
            A[bx] = by

    script = stateless.script()
    assert "alloc_var" not in script
    for function in (stateless, ir.load_json(ir.save_json(stateless))):
        assert tirx.analysis.check_well_formed(function)
        assert not collect_calls(function, op="tirx.alloc_tensor")
        assert not collect_nodes(function, ir.While)
        (store,) = collect_nodes(function, ir.TensorStore)
        assert store.dest.same_as(function.params[0])
        (launch,) = [region for region in collect_nodes(function, ir.RegionStmt) if region.op.same_as(ir.Op.get("tl.launch_kernel"))]
        # This fixture has one block. Resolve its two scalar Bind values and
        # require coord(0) == (0, 0), without depending on printer alpha names.
        analyzer = sym.Analyzer()
        constraints = sym.ConstraintContext()
        constraints.append(sym.BindingConstraint(launch.body_params[0], ir.const(0, "int32")))
        for statement in launch.body:
            if isinstance(statement, ir.Bind):
                constraints.append(sym.BindingConstraint(statement.var, statement.value))
        analyzer.sync_constraint_context(constraints)
        assert len(store.indices) == 1
        assert int(analyzer.simplify(store.indices[0])) == 0
        assert int(analyzer.simplify(store.value)) == 0


if __name__ == "__main__":
    tilelang.testing.main()

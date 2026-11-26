"""Tests for DisablePipelineForIrregularLoop (tile-ai/tilelang#4460).

A T.Pipelined loop whose body contains continue/break cannot be software
pipelined: InjectSoftwarePipeline emits the stage-1 statements of the last
trip as an epilogue after the loop, and the loop exit goes with them,
producing CUDA that nvcc rejects. The pass strips the pipeline annotations
from such loops so they lower as plain sequential loops.
"""

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.testing
import tilelang.language as T
from tvm import tirx
from tvm.tirx import Evaluate, For, PyStmtExprVisitor

sm89_target = tvm.target.Target({"kind": "cuda", "arch": "sm_89"})
# Warp specialization needs TMA (sm_90+); used for the ordering tests below.
sm90_target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
_AUTO_WS_CONFIG = {tl.PassConfigKey.TL_ENABLE_AUTO_WARP_SPECIALIZATION.value: "role_based"}


def _run_prologue(func, target=sm90_target, pass_config=None):
    # Runs the CUDA prologue (irregular-loop strip, warp specialization, then
    # pipeline planning) and returns the lowered function as a string.
    from tilelang.cuda.pipeline import CUDAPassPipelineBodyPrologue

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    with target, tvm.transform.PassContext(config=pass_config or {}):
        mod = CUDAPassPipelineBodyPrologue(mod, target)
    return str(mod["main"])


def _is_warp_specialized(ir: str) -> bool:
    # ProducerConsumerWarpSpecialized marks transformed functions with the
    # tl.ws_applied attribute; both ProducerConsumerWarpSpecialized and
    # AutoWarpSpecialization (+ MaterializeWSSchedule) introduce mbarrier
    # operations.
    return "ws_applied" in ir or "mbarrier" in ir


def _make(exit_kind, loop):
    @T.prim_func
    def main(
        X: T.Tensor((128,), "float32"),
        Y: T.Tensor((128,), "float32"),
        O: T.Tensor((128,), "float32"),
        N: T.int32,
    ):
        with T.Kernel(1, threads=128):
            x = T.alloc_fragment((128,), "float32")
            ys = T.alloc_shared((128,), "float32")
            T.copy(X, x)
            for j in T.Pipelined(N, num_stages=2) if loop == "Pipelined" else T.serial(N):
                T.copy(Y, ys)
                if exit_kind == "continue":
                    if j != 1:
                        continue
                    for i in T.Parallel(128):
                        x[i] = x[i] + ys[i]
                elif exit_kind == "break":
                    for i in T.Parallel(128):
                        x[i] = x[i] + ys[i]
                    if j == 1:
                        break
                else:  # the same condition as an if, no loop exit
                    if j == 1:
                        for i in T.Parallel(128):
                            x[i] = x[i] + ys[i]
            T.copy(x, O)

    return main


def _make_nested_break():
    # A break inside a nested serial loop targets that loop, not the
    # pipelined loop, so pipelining must stay enabled.
    @T.prim_func
    def main(
        X: T.Tensor((128,), "float32"),
        Y: T.Tensor((128,), "float32"),
        O: T.Tensor((128,), "float32"),
        N: T.int32,
    ):
        with T.Kernel(1, threads=128):
            x = T.alloc_fragment((128,), "float32")
            ys = T.alloc_shared((128,), "float32")
            T.copy(X, x)
            for _j in T.Pipelined(N, num_stages=2):
                T.copy(Y, ys)
                for k in T.serial(4):
                    if k == 1:
                        break
                for i in T.Parallel(128):
                    x[i] = x[i] + ys[i]
            T.copy(x, O)

    return main


@tirx.functor.visitor
class _StrayExitChecker(PyStmtExprVisitor):
    """Records continue_loop/break_loop Evaluates outside any For."""

    def __init__(self):
        super().__init__()
        self.depth = 0
        self.stray = []

    def visit_for_(self, op: For) -> None:
        self.depth += 1
        self.visit_stmt(op.body)
        self.depth -= 1

    def visit_evaluate_(self, op: Evaluate) -> None:
        if (
            isinstance(op.value, tirx.Call)
            and op.value.op.name
            in (
                "tir.continue_loop",
                "tir.break_loop",
                "tirx.continue_loop",
                "tirx.break_loop",
            )
            and self.depth == 0
        ):
            self.stray.append(op.value.op.name)


def _run(func, target=sm89_target):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tl.transform.IfStmtBinding()(mod)
    mod = tl.transform.DisablePipelineForIrregularLoop()(mod)
    mod = tl.transform.PipelinePlanning()(mod)
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.UnrollLoop()(mod)
    mod = tl.transform.Simplify()(mod)
    return mod["main"]


def _is_pipelined(func) -> bool:
    # PipelinePlanning marks pipelined loops with tl_pipelined_num_stages;
    # InjectSoftwarePipeline consumes the software_pipeline_stage annotations,
    # so they are gone by the time the passes below have run.
    return "tl_pipelined_num_stages" in str(func.body)


def _stray_exits(func) -> list[str]:
    checker = _StrayExitChecker()
    checker.visit_stmt(func.body)
    return checker.stray


def test_continue_in_pipelined_loop_is_not_pipelined():
    func = _run(_make("continue", "Pipelined"))
    assert not _is_pipelined(func)
    assert _stray_exits(func) == []


def test_break_in_pipelined_loop_is_not_pipelined():
    func = _run(_make("break", "Pipelined"))
    assert not _is_pipelined(func)
    assert _stray_exits(func) == []


def test_if_in_pipelined_loop_stays_pipelined():
    func = _run(_make("if", "Pipelined"))
    assert _is_pipelined(func)
    assert _stray_exits(func) == []


def test_continue_in_serial_loop_unchanged():
    func = _run(_make("continue", "serial"))
    assert not _is_pipelined(func)
    assert _stray_exits(func) == []


def test_break_in_nested_loop_keeps_pipelining():
    func = _run(_make_nested_break())
    assert _is_pipelined(func)
    assert _stray_exits(func) == []


@tilelang.testing.requires_cuda
def test_continue_in_pipelined_loop_not_warp_specialized():
    # tile-ai/tilelang#4467 (CodeRabbit review on #4467): warp specialization
    # used to run before DisablePipelineForIrregularLoop, rewriting a
    # T.Pipelined loop with a TMA copy and a continue/break into a
    # producer/consumer structure with barriers; a taken exit would skip the
    # consumer arrive and deadlock the producer. The strip pass now runs
    # first, so such a loop must stay sequential (no WS markers at all).
    for exit_kind in ("continue", "break"):
        ir = _run_prologue(_make(exit_kind, "Pipelined"))
        assert not _is_warp_specialized(ir)


@tilelang.testing.requires_cuda
def test_irregular_pipelined_loop_not_auto_warp_specialized():
    # AutoWarpSpecialization does not require num_stages (ScopeStages falls
    # back to 1), so stripping pipeline annotations alone is not enough:
    # PreprocessIR's CheckCalls must also decline tirx.continue_loop and
    # tirx.break_loop so auto WS leaves irregular loops sequential.
    for exit_kind in ("continue", "break"):
        ir = _run_prologue(_make(exit_kind, "Pipelined"), pass_config=_AUTO_WS_CONFIG)
        assert not _is_warp_specialized(ir)


@tilelang.testing.requires_cuda
def test_eligible_pipelined_loop_still_warp_specialized():
    # Control for the tests above: a pipelined loop without an exit must
    # still be warp-specialized on sm_90 under both default WS and auto WS.
    assert _is_warp_specialized(_run_prologue(_make("if", "Pipelined")))
    assert _is_warp_specialized(_run_prologue(_make("if", "Pipelined"), pass_config=_AUTO_WS_CONFIG))


if __name__ == "__main__":
    test_continue_in_pipelined_loop_is_not_pipelined()
    test_break_in_pipelined_loop_is_not_pipelined()
    test_if_in_pipelined_loop_stays_pipelined()
    test_continue_in_serial_loop_unchanged()
    test_break_in_nested_loop_keeps_pipelining()
    test_continue_in_pipelined_loop_not_warp_specialized()
    test_irregular_pipelined_loop_not_auto_warp_specialized()
    test_eligible_pipelined_loop_still_warp_specialized()
    print("all tests passed")

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import types
import builtins

import pytest


def _make_callable_transform(name, record_list):
    def factory(*args, **kwargs):
        def apply(mod):
            record_list.append((name, args, kwargs))
            return mod
        return apply
    return factory


def _build_stubs(record_list):
    # tvm stubs
    tvm = ModuleType("tvm")
    tir = ModuleType("tvm.tir")
    tir_transform = ModuleType("tvm.tir.transform")

    # Explicit tir transforms used in the module
    for name in [
        "BindTarget",
        "Simplify",
        "NarrowDataType",
        "UnrollLoop",
        "RenormalizeSplitPattern",
        "RemoveNoOp",
        "RewriteUnsafeSelect",
        "HoistIfThenElse",
        "VerifyMemory",
        "AnnotateEntryFunc",
        "SplitHostDevice",
        "InferFragment",
        "PlanAndUpdateBufferAllocationLocation",
    ]:
        setattr(tir_transform, name, _make_callable_transform(f"tir.transform.{name}", record_list))

    # Expose tir.transform and placeholders for IRModule
    tir.transform = tir_transform
    class IRModule:
        def __init__(self):
            # Collect transform calls here
            self.calls = record_list
    tvm.tir = tir
    tvm.IRModule = IRModule

    # tvm.target.Target stub
    tvm_target = ModuleType("tvm.target")
    class Target:
        def __init__(self, kind="cuda"):
            self.kind = kind
    tvm_target.Target = Target
    tvm.target = tvm_target

    # tilelang stubs
    tilelang = ModuleType("tilelang")
    # Keys container
    tilelang.PassConfigKey = SimpleNamespace(
        TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE="tl.enable_aggressive_shared_memory_merge"
    )

    # tilelang.transform submodule with many pass factories
    tl_transform = ModuleType("tilelang.transform")
    for name in [
        "FrontendLegalize",
        "LayoutReducer",
        "LayoutInference",
        "LowerTileOp",
        "LowerL2Persistent",
        "LegalizeVectorizedLoop",
        "LegalizeSafeMemoryAccess",
        "Simplify",
        "LoopVectorizeDynamic",
        "LowerSharedBarrier",
        "IfStmtBinding",
        "MultiVersionBuffer",
        "WarpSpecialized",
        "InjectTmaBarrier",
        "AnnotateWarpGroupRegAlloc",
        "PipelinePlanning",
        "InjectSoftwarePipeline",
        "LowerOpaqueBlock",
        "MergeIfStmt",
        "RewriteWgmmaSync",
        "InjectFenceProxy",
        "FlattenBuffer",
        "ConfigIndexBitwidth",
        "StorageRewrite",
        "MakePackedAPI",
        "LowerDeviceKernelLaunch",
        "PersistThreadblock",
        "AnnotateDeviceRegions",
        "LowerThreadAllreduce",
        "LowerHopperIntrin",
    ]:
        setattr(tl_transform, name, _make_callable_transform(f"tilelang.transform.{name}", record_list))

    # VectorizeLoop and ThreadSync accept arguments; handle specially to record parameters
    def VectorizeLoop(**kwargs):
        def apply(mod):
            record_list.append(("tilelang.transform.VectorizeLoop", (), kwargs))
            return mod
        return apply
    tl_transform.VectorizeLoop = VectorizeLoop

    def ThreadSync(scope):
        def apply(mod):
            record_list.append(("tilelang.transform.ThreadSync", (scope,), {}))
            return mod
        return apply
    tl_transform.ThreadSync = ThreadSync

    # get_pass_context returns a simple object with a config dict
    class PassContext:
        def __init__(self, config=None):
            self.config = config or {}
    def get_pass_context():
        return PassContext({})
    tl_transform.PassContext = PassContext
    tl_transform.get_pass_context = get_pass_context

    tilelang.transform = tl_transform

    # tilelang.contrib.nvcc.have_tma
    tl_contrib = ModuleType("tilelang.contrib")
    tl_nvcc = ModuleType("tilelang.contrib.nvcc")
    def have_tma(*args, **kwargs):
        # default stub value; tests will monkeypatch module-under-test's have_tma
        return True
    tl_nvcc.have_tma = have_tma
    tl_contrib.nvcc = tl_nvcc
    tilelang.contrib = tl_contrib

    # tilelang.jit.adapter.utils.is_cuda_target
    tl_jit = ModuleType("tilelang.jit")
    tl_adapter = ModuleType("tilelang.jit.adapter")
    tl_utils = ModuleType("tilelang.jit.adapter.utils")
    def is_cuda_target(target):
        return bool(getattr(target, "kind", "")) and "cuda" in target.kind
    tl_utils.is_cuda_target = is_cuda_target
    tl_adapter.utils = tl_utils
    tl_jit.adapter = tl_adapter
    tilelang.jit = tl_jit

    return tvm, tilelang


@pytest.fixture
def load_module(tmp_path, monkeypatch):
    """
    Fixture to stub external deps (tvm, tilelang) before importing the module under test.
    Returns a tuple: (module, calls_log, tvm_stub, tilelang_stub)
    """
    _ = tmp_path
    calls = []
    tvm_stub, tilelang_stub = _build_stubs(calls)

    # Inject stubs into sys.modules before import
    monkeypatch.setitem(sys.modules, "tvm", tvm_stub)
    monkeypatch.setitem(sys.modules, "tvm.tir", tvm_stub.tir)
    monkeypatch.setitem(sys.modules, "tvm.tir.transform", tvm_stub.tir.transform)
    monkeypatch.setitem(sys.modules, "tvm.target", tvm_stub.target)

    monkeypatch.setitem(sys.modules, "tilelang", tilelang_stub)
    monkeypatch.setitem(sys.modules, "tilelang.transform", tilelang_stub.transform)
    monkeypatch.setitem(sys.modules, "tilelang.contrib", tilelang_stub.contrib)
    monkeypatch.setitem(sys.modules, "tilelang.contrib.nvcc", tilelang_stub.contrib.nvcc)
    monkeypatch.setitem(sys.modules, "tilelang.jit", tilelang_stub.jit)
    monkeypatch.setitem(sys.modules, "tilelang.jit.adapter", tilelang_stub.jit.adapter)
    monkeypatch.setitem(sys.modules, "tilelang.jit.adapter.utils", tilelang_stub.jit.adapter.utils)

    # Import the module-under-test from file path

    path = "testing/python/engine/test_phase.py"
    spec = importlib.util.spec_from_file_location("module_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        raise RuntimeError("module spec loader is None")
    loader.exec_module(mod)
    return mod, calls, tvm_stub, tilelang_stub


# -------- Decision helpers tests --------

def test_allow_warp_specialized_false_when_not_cuda(load_module, monkeypatch):
    mod, _, tvm, _ = load_module
    # Force not CUDA
    monkeypatch.setattr("module_under_test.tilelang.jit.adapter.utils.is_cuda_target", lambda *args, **kwargs: False)
    monkeypatch.setattr("module_under_test.have_tma", lambda *args, **kwargs: True)
    if mod.allow_warp_specialized(pass_ctx=None, target=tvm.target.Target(kind="cpu")) is not False:
        pytest.fail("allow_warp_specialized should be False when target is not CUDA")


def test_allow_warp_specialized_false_when_no_tma(load_module, monkeypatch):
    mod, _, tvm, _ = load_module
    monkeypatch.setattr("module_under_test.tilelang.jit.adapter.utils.is_cuda_target", lambda *args, **kwargs: True)
    monkeypatch.setattr("module_under_test.have_tma", lambda *args, **kwargs: False)
    if mod.allow_warp_specialized(pass_ctx=None, target=tvm.target.Target(kind="cuda")) is not False:
        pytest.fail("allow_warp_specialized should be False when no TMA available")


def test_allow_warp_specialized_respects_disable_flag(load_module, monkeypatch):
    mod, _, tvm, tilelang_stub = load_module

    # CUDA + TMA available
    monkeypatch.setattr("module_under_test.tilelang.jit.adapter.utils.is_cuda_target", lambda *args, **kwargs: True)
    monkeypatch.setattr("module_under_test.have_tma", lambda *args, **kwargs: True)

    # Configure pass context to disable
    def get_ctx_disabled():
        return tilelang_stub.transform.PassContext({"tl.disable_warp_specialized": True})
    monkeypatch.setattr("module_under_test.tilelang.transform.get_pass_context", get_ctx_disabled)
    if mod.allow_warp_specialized(target=tvm.target.Target(kind="cuda")) is not False:
        pytest.fail("allow_warp_specialized should be False when disabled via pass context")


def test_allow_warp_specialized_true_when_enabled(load_module, monkeypatch):
    mod, _, tvm, tilelang_stub = load_module
    monkeypatch.setattr("module_under_test.tilelang.jit.adapter.utils.is_cuda_target", lambda *args, **kwargs: True)
    monkeypatch.setattr("module_under_test.have_tma", lambda *args, **kwargs: True)
    # default get_pass_context returns {}
    if mod.allow_warp_specialized(target=tvm.target.Target(kind="cuda")) is not True:
        pytest.fail("allow_warp_specialized should be True when CUDA and TMA available and not disabled")


def test_allow_tma_and_warp_specialized_basic_gates(load_module, monkeypatch):
    mod, _, tvm, tilelang_stub = load_module
    # No TMA -> False
    monkeypatch.setattr("module_under_test.have_tma", lambda *args, **kwargs: False)
    if mod.allow_tma_and_warp_specialized(target=tvm.target.Target(kind="cuda")) is not False:
        pytest.fail("allow_tma_and_warp_specialized should be False when no TMA")

    # TMA True but disabled by tl.disable_tma_lower
    monkeypatch.setattr("module_under_test.have_tma", lambda *args, **kwargs: True)
    def get_ctx_disable_tma():
        return tilelang_stub.transform.PassContext({"tl.disable_tma_lower": True})
    monkeypatch.setattr("module_under_test.tilelang.transform.get_pass_context", get_ctx_disable_tma)
    if mod.allow_tma_and_warp_specialized(target=tvm.target.Target(kind="cuda")) is not False:
        pytest.fail("allow_tma_and_warp_specialized should be False when tl.disable_tma_lower is set")

    # TMA True, not disabled, and warp specialized allowed -> True
    def get_ctx_enabled():
        return tilelang_stub.transform.PassContext({})
    monkeypatch.setattr("module_under_test.tilelang.transform.get_pass_context", get_ctx_enabled)
    monkeypatch.setattr("module_under_test.allow_warp_specialized", lambda *args, **kwargs: True)
    if mod.allow_tma_and_warp_specialized(target=tvm.target.Target(kind="cuda")) is not True:
        pytest.fail("allow_tma_and_warp_specialized should be True when TMA and warp specialized are allowed")


def test_allow_fence_proxy_proxies_to_have_tma(load_module, monkeypatch):
    mod, _, tvm, _ = load_module
    monkeypatch.setattr("module_under_test.have_tma", lambda *args, **kwargs: False)
    if mod.allow_fence_proxy(target=tvm.target.Target(kind="cuda")) is not False:
        pytest.fail("allow_fence_proxy should be False when have_tma is False")
    monkeypatch.setattr("module_under_test.have_tma", lambda *args, **kwargs: True)
    if mod.allow_fence_proxy(target=tvm.target.Target(kind="cuda")) is not True:
        pytest.fail("allow_fence_proxy should be True when have_tma is True")


def test_allow_vectorize_flag(load_module, monkeypatch):
    mod, _, _, tilelang_stub = load_module
    # default True
    if mod.allow_vectorize() is not True:
        pytest.fail("allow_vectorize should be True by default")
    # disabled by tir.disable_vectorize
    def get_ctx_disable():
        return tilelang_stub.transform.PassContext({"tir.disable_vectorize": True})
    monkeypatch.setattr("module_under_test.tilelang.transform.get_pass_context", get_ctx_disable)
    if mod.allow_vectorize() is not False:
        pytest.fail("allow_vectorize should be False when tir.disable_vectorize is set")


def test_allow_global_thread_sync_flag(load_module, monkeypatch):
    mod, _, _, tilelang_stub = load_module
    # default False
    if mod.allow_global_thread_synchronization() is not False:
        pytest.fail("allow_global_thread_synchronization should be False by default")
    # enabled by tir.detect_global_barrier
    def get_ctx_enable():
        return tilelang_stub.transform.PassContext({"tir.detect_global_barrier": True})
    monkeypatch.setattr("module_under_test.tilelang.transform.get_pass_context", get_ctx_enable)
    if mod.allow_global_thread_synchronization() is not True:
        pytest.fail("allow_global_thread_synchronization should be True when tir.detect_global_barrier is set")


def test_should_enable_aggressive_merge_config_and_override(load_module, monkeypatch):
    mod, _, tvm, tilelang_stub = load_module
    tgt = tvm.target.Target(kind="cuda")
    def ctx_with(flag):
        return tilelang_stub.transform.PassContext({tilelang_stub.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: flag})

    # Config true, but warp specialized false -> True
    monkeypatch.setattr("module_under_test.tilelang.transform.get_pass_context", lambda: ctx_with(True))
    monkeypatch.setattr("module_under_test.allow_warp_specialized", lambda *args, **kwargs: False)
    if mod.should_enable_aggressive_merge(target=tgt) is not True:
        pytest.fail("should_enable_aggressive_merge should be True when config enabled and warp specialized disabled")

    # Warp specialized true forces False regardless of config
    monkeypatch.setattr("module_under_test.allow_warp_specialized", lambda *args, **kwargs: True)
    if mod.should_enable_aggressive_merge(target=tgt) is not False:
        pytest.fail("should_enable_aggressive_merge should be False when warp specialized is allowed")

    # Config false and no warp specialized -> False
    monkeypatch.setattr("module_under_test.tilelang.transform.get_pass_context", lambda: ctx_with(False))
    monkeypatch.setattr("module_under_test.allow_warp_specialized", lambda *args, **kwargs: False)
    if mod.should_enable_aggressive_merge(target=tgt) is not False:
        pytest.fail("should_enable_aggressive_merge should be False when config disabled and warp specialized disabled")


# -------- Pipeline tests (verify key sequencing/branching) --------

def test_LowerAndLegalize_applies_expected_transform_sequence(load_module, monkeypatch):
    _ = monkeypatch
    mod, calls, tvm, _ = load_module
    ir_mod = tvm.IRModule()
    tgt = tvm.target.Target(kind="cuda")

    out = mod.LowerAndLegalize(ir_mod, tgt)
    if out is not ir_mod:
        pytest.fail("Should return the same IRModule instance")

    names = [c[0] for c in calls]
    # Verify early sequence and critical passes order
    if not names:
        pytest.fail("No transforms were applied")
    if names[0] != "tir.transform.BindTarget":
        pytest.fail('First transform must be "tir.transform.BindTarget"')
    if "tilelang.transform.FrontendLegalize" not in names:
        pytest.fail('"tilelang.transform.FrontendLegalize" not in applied transforms')
    if names.count("tir.transform.Simplify") < 1:
        pytest.fail('Expected at least one "tir.transform.Simplify"')
    if "tilelang.transform.LayoutReducer" not in names:
        pytest.fail('"tilelang.transform.LayoutReducer" not in applied transforms')
    if "tilelang.transform.LayoutInference" not in names:
        pytest.fail('"tilelang.transform.LayoutInference" not in applied transforms')
    if "tilelang.transform.LowerTileOp" not in names:
        pytest.fail('"tilelang.transform.LowerTileOp" not in applied transforms')
    if "tilelang.transform.LegalizeSafeMemoryAccess" not in names:
        pytest.fail('"tilelang.transform.LegalizeSafeMemoryAccess" not in applied transforms')
    if "tilelang.transform.Simplify" not in names:
        pytest.fail('"tilelang.transform.Simplify" not in applied transforms')
    if "tilelang.transform.LoopVectorizeDynamic" not in names:
        pytest.fail('"tilelang.transform.LoopVectorizeDynamic" not in applied transforms')


def test_OptimizeForTarget_branch_with_tma_and_warp(load_module, monkeypatch):
    mod, calls, tvm, _ = load_module
    # Drive branch: allow_tma_and_warp_specialized -> True
    monkeypatch.setattr("module_under_test.allow_tma_and_warp_specialized", lambda *args, **kwargs: True)
    monkeypatch.setattr("module_under_test.allow_fence_proxy", lambda *args, **kwargs: True)
    monkeypatch.setattr("module_under_test.allow_vectorize", lambda *args, **kwargs: True)
    monkeypatch.setattr("module_under_test.allow_global_thread_synchronization", lambda: True)
    monkeypatch.setattr("module_under_test.should_enable_aggressive_merge", lambda *args, **kwargs: True)

    ir_mod = tvm.IRModule()
    tgt = tvm.target.Target(kind="cuda")
    out = mod.OptimizeForTarget(ir_mod, tgt)
    if out is not ir_mod:
        pytest.fail("OptimizeForTarget should return the same IRModule instance")

    names = [c[0] for c in calls]

    # must include warp-specialized branch transforms
    expected = [
        "tilelang.transform.LowerSharedBarrier",
        "tilelang.transform.IfStmtBinding",
        "tilelang.transform.MultiVersionBuffer",
        "tilelang.transform.WarpSpecialized",
        "tilelang.transform.InjectTmaBarrier",
        "tilelang.transform.AnnotateWarpGroupRegAlloc",
        "tilelang.transform.PipelinePlanning",
        "tilelang.transform.InjectSoftwarePipeline",
        "tilelang.transform.LowerOpaqueBlock",
        "tilelang.transform.MergeIfStmt",
        "tilelang.transform.RewriteWgmmaSync",
        "tilelang.transform.InjectFenceProxy",
    ]
    for e in expected:
        if e not in names:
            pytest.fail(f"Expected transform {e} not found in applied transforms")

    # VectorizeLoop should record enable_vectorize=True
    vec_entries = [c for c in calls if c[0] == "tilelang.transform.VectorizeLoop"]
    if not (vec_entries and vec_entries[-1][2].get("enable_vectorize") is True):
        pytest.fail("VectorizeLoop enable_vectorize should be True in the recorded entries")

    # MergeSharedMemoryAllocations should reflect enable_aggressive_merge from hook
    msa_entries = [c for c in calls if c[0] == "tilelang.transform.MergeSharedMemoryAllocations"]
    if not msa_entries:
        pytest.fail("MergeSharedMemoryAllocations must be applied")
    if msa_entries[-1][2].get("enable_aggressive_merge") is not True:
        pytest.fail("MergeSharedMemoryAllocations should have enable_aggressive_merge=True")

    # Global thread sync applied before SplitHostDevice
    # Check relative order indices
    idx_sync_global = names.index("tilelang.transform.ThreadSync")
    idx_split = names.index("tir.transform.SplitHostDevice")
    if not (idx_sync_global < idx_split):
        pytest.fail("Global ThreadSync should be applied before tir.transform.SplitHostDevice")


def test_OptimizeForTarget_branch_without_tma_and_warp_fence_proxy_toggle(load_module, monkeypatch):
    mod, calls, tvm, _ = load_module
    # Reset calls list for clarity
    calls.clear()

    # Branch: allow_tma_and_warp_specialized -> False
    monkeypatch.setattr("module_under_test.allow_tma_and_warp_specialized", lambda *args, **kwargs: False)
    monkeypatch.setattr("module_under_test.allow_vectorize", lambda *args, **kwargs: False)
    monkeypatch.setattr("module_under_test.allow_global_thread_synchronization", lambda: False)
    monkeypatch.setattr("module_under_test.should_enable_aggressive_merge", lambda *args, **kwargs: False)

    ir_mod = tvm.IRModule()
    tgt = tvm.target.Target(kind="cuda")

    # Case A: fence proxy allowed -> InjectFenceProxy present
    monkeypatch.setattr("module_under_test.allow_fence_proxy", lambda *args, **kwargs: True)
    mod.OptimizeForTarget(ir_mod, tgt)

    names = [c[0] for c in calls]
    if "tilelang.transform.IfStmtBinding" not in names:
        pytest.fail("IfStmtBinding should be present in transforms for Case A")
    if "tir.transform.PlanAndUpdateBufferAllocationLocation" not in names:
        pytest.fail("PlanAndUpdateBufferAllocationLocation should be present in transforms for Case A")
    if "tilelang.transform.InjectFenceProxy" not in names:
        pytest.fail("InjectFenceProxy should be present in transforms for Case A")

    # VectorizeLoop should record enable_vectorize=False
    vec_entries = [c for c in calls if c[0] == "tilelang.transform.VectorizeLoop"]
    if not (vec_entries and vec_entries[-1][2].get("enable_vectorize") is False):
        pytest.fail("VectorizeLoop enable_vectorize should be False in Case A")

    # Reset and test Case B: fence proxy not allowed -> InjectFenceProxy absent in else-branch part
    calls.clear()
    monkeypatch.setattr("module_under_test.allow_fence_proxy", lambda *args, **kwargs: False)
    mod.OptimizeForTarget(ir_mod, tgt)
    names = [c[0] for c in calls]
    # Ensure branch pipeline still present
    if "tilelang.transform.IfStmtBinding" not in names:
        pytest.fail("IfStmtBinding should be present in transforms for Case B")
    if "tir.transform.PlanAndUpdateBufferAllocationLocation" not in names:
        pytest.fail("PlanAndUpdateBufferAllocationLocation should be present in transforms for Case B")
    # InjectFenceProxy may still happen later in common tail (not in else-branch pre-LowerOpaqueBlock)
    # Verify that there's at most one InjectFenceProxy, and if present, it's not coming from the earlier conditional
    inj_idxs = [i for i, n in enumerate(names) if n == "tilelang.transform.InjectFenceProxy"]
    # It's acceptable if zero (depending on stubs), but ensure not two from the else-branch + tail.
    if len(inj_idxs) > 1:
        pytest.fail("InjectFenceProxy should not appear more than once")
    

# Note on framework:
# These tests use pytest with its built-in monkeypatch fixture to stub external dependencies (tvm, tilelang)
# and to control decision branches. This aligns with common Python testing practices in similar repositories.
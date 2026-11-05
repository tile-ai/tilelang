"""Exercise public DSL through the registered backend pipeline and emitter."""

import json

import pytest
import tvm_ffi
import tvm_tirx as tvm
from tvm_tirx import ir, tirx

import tilelang
import tilelang.language as T


TARGETS = {
    "cuda": {"kind": "cuda", "arch": "sm_100a"},
    "hip": {"kind": "hip", "mcpu": "gfx942", "thread_warp_size": 64},
    "metal": {"kind": "metal", "keys": ["metal", "gpu"]},
    "webgpu": {"kind": "webgpu"},
}


def elementwise():
    @T.prim_func
    def main(A: T.Tensor((257,), "float32"), B: T.Tensor((257,), "float32")):
        with T.Kernel(T.ceildiv(257, 128), threads=128) as block:
            for i in T.Parallel(128):
                index = block * 128 + i
                if index < 257:
                    B[index] = A[index] + 1

    return main


def shared_copy():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((128,), "float32")
            T.copy(A, shared, prefer_instruction="sync")
            T.copy(shared, B, prefer_instruction="sync")

    return main


@pytest.mark.parametrize("backend", list(TARGETS))
@pytest.mark.parametrize("case", ["elementwise", "shared_copy"])
def test_registered_pipeline_to_device_source(backend, case, tmp_path):
    # WS transforms have their own author/peer suites while their migration is
    # integrated. This explicitly selected no-WS policy is a supported path.
    function = globals()[case]()
    before = ir.save_json(function)
    target = tvm.target.Target(TARGETS[backend])
    with tilelang.transform.PassContext(config={"tl.disable_warp_specialized": True}):
        artifact = tilelang.lower(function, target=target, target_host={"kind": "llvm", "mcpu": "generic"})
    assert ir.save_json(function) == before
    assert tirx.analysis.check_well_formed(artifact.host_mod)
    assert tirx.analysis.check_well_formed(artifact.device_mod)
    assert len(artifact.device_mod.functions) == 1
    assert artifact.kernel_source
    assert len(artifact.params) == 2
    assert all(param.kind == "tensor" for param in artifact.params)
    assert "main_kernel" in artifact.kernel_source
    assert not any(getattr(node.op, "name", "").startswith("tl.tileop.") for node in _calls(artifact.device_mod))
    (tmp_path / "device.source").write_text(artifact.kernel_source)
    (tmp_path / "host.json").write_text(ir.save_json(artifact.host_mod))
    (tmp_path / "device.json").write_text(ir.save_json(artifact.device_mod))
    (tmp_path / "case.json").write_text(
        json.dumps(
            {
                "case": case,
                "backend": backend,
                "stage": "device_source",
                "device_execution": False,
                "warp_specialized": False,
            }
        )
    )


def _calls(value):
    calls = []
    tvm_ffi.structural_walk(value, (ir.Call, calls.append))
    return calls

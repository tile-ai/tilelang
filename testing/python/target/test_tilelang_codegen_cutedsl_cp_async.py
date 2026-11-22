import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang.engine.lower import lower
from tilelang.backend.target import determine_target
from tilelang.cuda.target import normalize_cutedsl_target


@pytest.mark.cutedsl
@tilelang.testing.requires_cuda(support_required="compile-only")
def test_cutedsl_dict_target_normalizes_to_cuda_marker(monkeypatch):
    from tilelang.jit.adapter.cutedsl import checks

    monkeypatch.setattr(checks, "check_cutedsl_available", lambda: None)
    target = determine_target({"kind": "cutedsl", "arch": "sm_80"}, return_object=True)

    assert target.kind.name == "cuda"
    assert target.attrs["arch"] == "sm_80"
    assert {"cuda", "gpu", "cutedsl"}.issubset(set(target.keys))


@pytest.mark.cutedsl
@tilelang.testing.requires_cuda(support_required="compile-only")
def test_cutedsl_string_target_uses_detected_cuda_arch(monkeypatch):
    """Verify bare CuTeDSL targets use TileLang's CUDA arch normalization."""

    from tilelang.jit.adapter.cutedsl import checks

    monkeypatch.setattr(checks, "check_cutedsl_available", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))

    target = determine_target("cutedsl", return_object=True)

    assert target.kind.name == "cuda"
    assert target.attrs["arch"] == "sm_90a"
    assert {"cuda", "gpu", "cutedsl"}.issubset(set(target.keys))


@tilelang.testing.requires_cutedsl(support_required="compile-only")
def test_cutedsl_codegen_supports_tl_ptx_cp_async():
    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": "sm_80"})
    assert target is not None

    @T.prim_func
    def prog(A: T.Tensor((16,), "uint8"), B: T.Tensor((16,), "uint8")):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((16,), "uint8", scope="shared")
            T.ptx_cp_async(T.access_ptr(A_shared[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
            B[0] = A_shared[0]

    artifact = lower(prog.with_attr("global_symbol", "main"), target=target)
    assert "tl.cp_async_gs(" in artifact.kernel_source


if __name__ == "__main__":
    tilelang.testing.main()

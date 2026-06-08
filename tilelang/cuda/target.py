from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import register_target_detector


def _detect_cuda_target() -> Target | str | None:
    import torch

    if torch.version.hip is not None:
        return None
    try:
        from tilelang.contrib import nvcc

        nvcc.find_cuda_path()
    except Exception:
        return None

    if not torch.cuda.is_available():
        return "cuda"
    cap = torch.cuda.get_device_capability(0)
    if not cap:
        return "cuda"
    return Target({"kind": "cuda", "arch": f"sm_{nvcc.get_target_arch(cap)}"})


register_target_detector("cuda", _detect_cuda_target, priority=100, override=True)

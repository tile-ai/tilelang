from tilelang.backend.target import Target, register_target_kind, register_target_preset

from . import intrinsics  # noqa: F401
from . import op  # noqa: F401
from . import pipeline  # noqa: F401
from . import transform  # noqa: F401

_CUDA_ARCH_PRESETS: dict[str, str] = {
    "a100": "sm_80",
    "h100": "sm_90a",
    "h200": "sm_90a",
    "b100": "sm_100a",
}


def _cuda_arch_target_preset(arch: str):
    def resolve(spec):
        attrs = dict(spec.attrs)
        preset_arch = attrs.pop("arch", arch)
        attrs.pop("kind", None)
        return {"kind": "cuda", "arch": preset_arch, **attrs}

    return resolve


def _cutedsl_target_preset(spec):
    attrs = dict(spec.attrs)
    raw_keys = attrs.pop("keys", ())
    attrs.pop("kind", None)
    if isinstance(raw_keys, str):
        keys = [raw_keys]
    else:
        keys = list(raw_keys)
    return {"kind": "cuda", **attrs, "keys": list(dict.fromkeys([*keys, "cutedsl"]))}


def _detect_cuda_target():
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
    return Target("cuda", arch=f"sm_{nvcc.get_target_arch(cap)}")


register_target_kind("cuda", tvm_kind="cuda", detect=_detect_cuda_target, priority=100, override=True)
register_target_kind("cutedsl", normalize=_cutedsl_target_preset, priority=90, override=True)
for _preset_name, _preset_arch in _CUDA_ARCH_PRESETS.items():
    register_target_preset(_preset_name, _cuda_arch_target_preset(_preset_arch), override=True)

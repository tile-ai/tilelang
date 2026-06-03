from tilelang.backend.target import Target, register_target_kind, register_target_preset

from . import intrinsics  # noqa: F401
from . import op  # noqa: F401
from . import pipeline  # noqa: F401

_ROCM_ARCH_PRESETS: dict[str, str] = {
    "mi300": "gfx942",
    "mi300x": "gfx942",
}


def _rocm_arch_target_preset(mcpu: str):
    def resolve(spec):
        attrs = dict(spec.attrs)
        preset_mcpu = attrs.pop("mcpu", mcpu)
        attrs.pop("kind", None)
        return {"kind": "hip", "mcpu": preset_mcpu, **attrs}

    return resolve


def _normalize_rocm_arch(arch: str | None) -> str | None:
    if arch is None:
        return None
    normalized = str(arch).strip().split(":", maxsplit=1)[0]
    return normalized if normalized.startswith("gfx") else None


def _detect_rocm_target():
    try:
        from tvm.contrib import rocm

        rocm.find_rocm_path()
    except Exception:
        return None

    try:
        import torch

        if torch.cuda.is_available():
            arch = _normalize_rocm_arch(getattr(torch.cuda.get_device_properties(0), "gcnArchName", None))
            if arch is not None:
                return Target("hip", mcpu=arch)
    except Exception:
        pass
    return "hip"


register_target_kind("hip", tvm_kind="hip", detect=_detect_rocm_target, priority=80, override=True)
register_target_kind("rocm", tvm_kind="hip", override=True)
for _preset_name, _preset_mcpu in _ROCM_ARCH_PRESETS.items():
    register_target_preset(_preset_name, _rocm_arch_target_preset(_preset_mcpu), override=True)

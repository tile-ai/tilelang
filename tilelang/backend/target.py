from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from platform import mac_ver
from typing import TYPE_CHECKING, Literal

from tvm.target import Target

if TYPE_CHECKING:
    import torch

TargetInput = str | Mapping[str, object] | Target
TargetDetector = Callable[[], TargetInput | None]
TargetConfig = dict[str, object]
TargetLike = str | TargetConfig | Target

SUPPORTED_TARGETS: dict[str, str] = {
    "auto": "Auto-detect CUDA/HIP/Metal based on availability.",
    "cuda": "CUDA GPU target. Use dict options such as {'kind': 'cuda', 'arch': 'sm_90'}.",
    "hip": "ROCm HIP target. Use dict options such as {'kind': 'hip', 'mcpu': 'gfx942'}.",
    "metal": "Apple Metal target for arm64 Macs.",
    "llvm": "LLVM CPU target. Use dict options such as {'kind': 'llvm', 'mcpu': 'native'}.",
    "webgpu": "WebGPU target for browser/WebGPU runtimes.",
    "c": "C source backend.",
    "cutedsl": "CuTe DSL GPU target. Use dict options such as {'kind': 'cutedsl', 'arch': 'sm_90'}.",
}

ROCM_MTRIPLE = "amdgcn-amd-amdhsa-hcc"


@dataclass(frozen=True, slots=True)
class TargetDetectorSpec:
    name: str
    detect: TargetDetector


_TARGET_DETECTORS: dict[str, TargetDetectorSpec] = {}
_LAZY_TARGET_DETECTORS: dict[str, str] = {}
_LOADED_TARGET_DETECTORS: set[str] = set()


def register_target_detector(
    name: str,
    detect: TargetDetector,
    *,
    override: bool = False,
) -> TargetDetectorSpec:
    if name in _TARGET_DETECTORS and not override:
        raise ValueError(f"Target detector {name!r} is already registered")
    spec = TargetDetectorSpec(name=name, detect=detect)
    _TARGET_DETECTORS[name] = spec
    return spec


def register_lazy_target_detector(name: str, import_path: str) -> None:
    _LAZY_TARGET_DETECTORS[name] = import_path


def _ensure_target_detectors_loaded() -> list[str]:
    errors: list[str] = []
    for name, import_path in tuple(_LAZY_TARGET_DETECTORS.items()):
        if name in _LOADED_TARGET_DETECTORS:
            continue
        try:
            import_module(import_path)
        except Exception as err:
            errors.append(f"{name}: {err}")
        finally:
            _LOADED_TARGET_DETECTORS.add(name)
    return errors


def auto_detect_target() -> TargetInput:
    errors = _ensure_target_detectors_loaded()
    for spec in _TARGET_DETECTORS.values():
        try:
            detected = spec.detect()
        except Exception as err:
            errors.append(f"{spec.name}: {err}")
            continue
        if detected is not None:
            return detected

    details = f" Tried: {', '.join(errors)}." if errors else ""
    raise ValueError(f"No CUDA or HIP or MPS available on this system.{details}")


def list_target_detectors() -> tuple[str, ...]:
    _ensure_target_detectors_loaded()
    return tuple(_TARGET_DETECTORS)


def normalize_rocm_arch(arch: str | None) -> str | None:
    if arch is None:
        return None
    normalized = str(arch).strip().split(":", maxsplit=1)[0]
    return normalized if normalized.startswith("gfx") else None


def target_get_mcpu(target: str | Target | None) -> str | None:
    if target is None:
        return None
    if isinstance(target, str):
        target = Target(target)
    return normalize_rocm_arch(target.attrs.get("mcpu"))


def rocm_warp_size_for_arch(arch: str | None) -> int | None:
    if arch is None:
        return None
    if arch.startswith("gfx9"):
        return 64
    if arch.startswith(("gfx10", "gfx11", "gfx12")):
        return 32
    return None


def with_rocm_target_attrs(target: Target) -> Target:
    if target.kind.name != "hip":
        return target
    arch = target_get_mcpu(target)
    if arch is None:
        return target

    target_dict = dict(target.export())
    target_dict.setdefault("mtriple", ROCM_MTRIPLE)
    warp_size = rocm_warp_size_for_arch(arch)
    if warp_size is not None:
        target_dict["thread_warp_size"] = warp_size
    else:
        target_dict.pop("thread_warp_size", None)
    return Target(target_dict)


def _detect_torch_rocm_arch() -> str | None:
    import torch

    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return normalize_rocm_arch(getattr(props, "gcnArchName", None))


def _rocm_target_from_arch(arch: str | None) -> Target | str:
    if arch is None:
        return "hip"
    target_dict: dict[str, object] = {
        "kind": "hip",
        "mcpu": arch,
        "mtriple": ROCM_MTRIPLE,
    }
    warp_size = rocm_warp_size_for_arch(arch)
    if warp_size is not None:
        target_dict["thread_warp_size"] = warp_size
    return Target(target_dict)


def _detect_torch_cuda_arch() -> str | None:
    """Return the CUDA SM architecture detected from PyTorch, if available."""
    import torch

    from tilelang.contrib import nvcc

    if not torch.cuda.is_available():
        return None
    cap = torch.cuda.get_device_capability(0)
    return f"sm_{nvcc.get_target_arch(cap)}" if cap else None


def _cuda_target_from_arch(arch: str | None) -> Target | str:
    """Build a CUDA target while preserving the legacy bare string fallback."""
    if arch is None:
        return "cuda"
    return Target({"kind": "cuda", "arch": arch})


def describe_supported_targets() -> dict[str, str]:
    """
    Return a mapping of supported target names to usage descriptions.
    """
    return dict(SUPPORTED_TARGETS)


def check_cuda_availability() -> bool:
    """
    Check if CUDA is available on the system by locating the CUDA path.
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    try:
        from tilelang.contrib import nvcc

        nvcc.find_cuda_path()
        return True
    except Exception:
        return False


def check_hip_availability() -> bool:
    """
    Check if HIP (ROCm) is available on the system by locating the ROCm path.
    Returns:
        bool: True if HIP is available, False otherwise.
    """
    try:
        from tvm.contrib import rocm

        rocm.find_rocm_path()
        return True
    except Exception:
        return False


def check_metal_availability() -> bool:
    mac_release, _, arch = mac_ver()
    if not mac_release:
        return False
    # todo: check torch version?
    return arch == "arm64"


def determine_fp8_type(fp8_format: Literal["e4m3", "e5m2"] = "e4m3") -> str:
    """
    Select the correct FP8 dtype string for the current platform.
    - CUDA defaults to FP8 E4M3FN / E5M2.
    - ROCm uses FNUZ except gfx950 (OCP), which prefers non-FNUZ when available.
    """
    import torch

    from tilelang import language as T

    if fp8_format not in {"e4m3", "e5m2"}:
        raise ValueError(f"Unsupported FP8 format: {fp8_format}")
    if torch.version.hip is None:
        return T.float8_e4m3fn if fp8_format == "e4m3" else T.float8_e5m2
    if not torch.cuda.is_available():
        return T.float8_e4m3fnuz if fp8_format == "e4m3" else T.float8_e5m2fnuz
    props = torch.cuda.get_device_properties(0)
    gcn_arch = getattr(props, "gcnArchName", "")
    if fp8_format == "e4m3":
        if gcn_arch.startswith("gfx950"):
            return T.float8_e4m3fn
        return T.float8_e4m3fnuz
    if gcn_arch.startswith("gfx950") and hasattr(T, "float8_e5m2"):
        return T.float8_e5m2
    return T.float8_e5m2fnuz


def determine_torch_fp8_type(fp8_format: Literal["e4m3", "e5m2"] = "e4m3") -> torch.dtype:
    import torch

    dtype_name = determine_fp8_type(fp8_format)
    torch_dtype = getattr(torch, dtype_name, None)
    if torch_dtype is None:
        raise RuntimeError(f"PyTorch does not expose dtype {dtype_name}")
    return torch_dtype


def _with_cutedsl_key(target: Target | str) -> Target:
    if not isinstance(target, Target):
        target = Target(target)
    target_dict = dict(target.export())
    target_dict["keys"] = list(dict.fromkeys([*target_dict.get("keys", ()), "cutedsl"]))
    return Target(target_dict)


def normalize_cutedsl_target(target: TargetLike) -> Target | None:
    if isinstance(target, Target):
        if target.kind.name == "cuda" and "cutedsl" in target.keys:
            return target
        return None

    if isinstance(target, dict):
        if target.get("kind") == "cutedsl":
            cuda_target = dict(target)
            cuda_target["kind"] = "cuda"
            try:
                return _with_cutedsl_key(Target(cuda_target))
            except Exception:
                return None
        try:
            temp_target = Target(target)
        except Exception:
            return None
        if temp_target.kind.name == "cuda" and "cutedsl" in temp_target.keys:
            return temp_target
        return None

    if target.strip() == "cutedsl":
        try:
            return _with_cutedsl_key(_cuda_target_from_arch(_detect_torch_cuda_arch()))
        except Exception:
            return None

    return None


def determine_target(target: TargetLike | Literal["auto"] = "auto", return_object: bool = False) -> str | TargetConfig | Target:
    """
    Determine the appropriate target for compilation (CUDA, HIP, or manual selection).

    Args:
        target (Union[str, dict, Target, Literal["auto"]]): User-specified target.
            - If "auto", the system will automatically detect whether CUDA or HIP is available.
            - If a string, dict, or Target, it is directly validated.

    Returns:
        Union[str, dict, Target]: The selected target ("cuda", "hip", a config dict, or a Target object).

    Raises:
        ValueError: If no CUDA or HIP is available and the target is "auto".
        AssertionError: If the target is invalid.
    """

    return_var: str | TargetConfig | Target = target

    if target == "auto":
        target = Target.current(allow_none=True)
        if target is not None:
            return with_rocm_target_attrs(target)

        return_var = auto_detect_target()

    else:
        possible_cutedsl_target = normalize_cutedsl_target(target)
        if possible_cutedsl_target is not None:
            try:
                from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available  # lazy

                check_cutedsl_available()
            except ImportError as e:
                raise AssertionError(f"CuTeDSL backend is not available. Please install tilelang-cutedsl package. {str(e)}") from e

            return_var = possible_cutedsl_target
        else:
            # Validate the target if it's not "auto"
            if isinstance(target, Target):
                return_var = with_rocm_target_attrs(target)
            elif isinstance(target, dict):
                try:
                    parsed_target = Target(target)
                except Exception as err:
                    raise AssertionError(
                        f"Target {target} is not supported. Pass a valid target config dict, e.g. `{{'kind': 'cuda', 'arch': 'sm_80'}}`."
                    ) from err
                if parsed_target.kind.name == "hip" and target_get_mcpu(parsed_target) is not None:
                    return_var = with_rocm_target_attrs(parsed_target)
                else:
                    return_var = target
            elif isinstance(target, str):
                normalized_target = target.strip()
                if not normalized_target:
                    raise AssertionError(f"Target {target} is not supported")
                try:
                    parsed_target = Target(normalized_target)
                except Exception as err:
                    examples = ", ".join(f"`{name}`" for name in SUPPORTED_TARGETS)
                    raise AssertionError(
                        f"Target {target} is not supported. Supported targets include: {examples}. "
                        "Pass target options as a dict, e.g. `{'kind': 'cuda', 'arch': 'sm_80'}`."
                    ) from err
                if parsed_target.kind.name == "hip" and target_get_mcpu(parsed_target) is not None:
                    return_var = with_rocm_target_attrs(parsed_target)
                else:
                    return_var = normalized_target
            else:
                raise AssertionError(f"Target {target} is not supported")

    if isinstance(return_var, Target):
        return with_rocm_target_attrs(return_var)
    if return_object:
        if isinstance(return_var, Target):
            return return_var
        return Target(return_var)
    return return_var


def _target_ffi_api():
    from tilelang import _ffi_api

    return _ffi_api


def target_is_cuda(target: Target) -> bool:
    return _target_ffi_api().TargetIsCuda(target)


def target_is_hip(target: Target) -> bool:
    return _target_ffi_api().TargetIsRocm(target)


def target_is_metal(target: Target) -> bool:
    return _target_ffi_api().TargetIsMetal(target)


def target_is_volta(target: Target) -> bool:
    return _target_ffi_api().TargetIsVolta(target)


def target_is_turing(target: Target) -> bool:
    return _target_ffi_api().TargetIsTuring(target)


def target_is_ampere(target: Target) -> bool:
    return _target_ffi_api().TargetIsAmpere(target)


def target_is_hopper(target: Target) -> bool:
    return _target_ffi_api().TargetIsHopper(target)


def target_is_sm120(target: Target) -> bool:
    return _target_ffi_api().TargetIsSM120(target)


def target_is_cdna(target: Target) -> bool:
    return _target_ffi_api().TargetIsCDNA(target)


def target_is_rdna(target: Target) -> bool:
    return _target_ffi_api().TargetIsRDNA(target)


def target_is_gfx950(target: Target) -> bool:
    return _target_ffi_api().TargetIsGfx950(target)


def target_has_async_copy(target: Target) -> bool:
    return _target_ffi_api().TargetHasAsyncCopy(target)


def target_has_ldmatrix(target: Target) -> bool:
    return _target_ffi_api().TargetHasLdmatrix(target)


def target_has_stmatrix(target: Target) -> bool:
    return _target_ffi_api().TargetHasStmatrix(target)


def target_has_bulk_copy(target: Target) -> bool:
    return _target_ffi_api().TargetHasBulkCopy(target)


def target_get_warp_size(target: Target) -> int:
    return _target_ffi_api().TargetGetWarpSize(target)


def target_get_rdna_generation(target: Target) -> int:
    return _target_ffi_api().TargetGetRDNAGeneration(target)

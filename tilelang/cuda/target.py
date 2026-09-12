from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_detector, register_target_normalizer


def _target_ffi_api():
    from tilelang import _ffi_api

    return _ffi_api


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


def _detect_torch_cuda_arch() -> str | None:
    """Return the CUDA SM architecture detected from PyTorch, if available."""
    import torch

    from tilelang.contrib import nvcc

    if not torch.cuda.is_available():
        return None
    cap = torch.cuda.get_device_capability(torch.cuda.current_device())
    return f"sm_{nvcc.get_target_arch(cap)}" if cap else None


def _cuda_target_from_arch(arch: str | None) -> Target:
    """Build a concrete-arch CUDA target, refusing a bare "cuda" (TVM would
    silently fill arch=sm_50 and mis-compile). Guards auto-detect and explicit
    target="cuda" alike; an explicit arch compiles without a GPU."""
    if arch is None:
        raise ValueError(
            "CUDA target has no arch and no GPU is available to detect one. "
            "Pin the arch for your target GPU, e.g. "
            "tilelang.compile(func, target={'kind': 'cuda', 'arch': 'sm_90a'}) or "
            'TILELANG_DEFAULT_TARGET=\'{"kind": "cuda", "arch": "sm_90a"}\'.'
        )
    return Target({"kind": "cuda", "arch": arch})


def _detect_cuda_target() -> Target | None:
    import torch

    if torch.version.hip is not None:
        return None
    if not check_cuda_availability():
        return None

    return _cuda_target_from_arch(_detect_torch_cuda_arch())


def normalize_cuda_target(target: TargetLike) -> Target | None:
    if not isinstance(target, str) or target.strip() != "cuda":
        return None
    return _cuda_target_from_arch(_detect_torch_cuda_arch())


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


def _normalize_cutedsl_target_for_resolve(target: TargetLike) -> Target | None:
    normalized = normalize_cutedsl_target(target)
    if normalized is None:
        return None
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except ImportError as err:
        raise AssertionError(f"CuTeDSL backend is not available. Please install tilelang-cutedsl package. {str(err)}") from err
    return normalized


def target_is_cuda(target: Target) -> bool:
    return _target_ffi_api().TargetIsCuda(target)


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


def target_has_async_copy(target: Target) -> bool:
    return _target_ffi_api().TargetHasAsyncCopy(target)


def target_has_ldmatrix(target: Target) -> bool:
    return _target_ffi_api().TargetHasLdmatrix(target)


def target_has_stmatrix(target: Target, is_m16n8: bool = False) -> bool:
    return _target_ffi_api().TargetHasStmatrix(target, is_m16n8)


def target_has_bulk_copy(target: Target) -> bool:
    return _target_ffi_api().TargetHasBulkCopy(target)


register_target_detector("cuda", _detect_cuda_target, override=True)
register_target_normalizer("cuda", normalize_cuda_target, override=True)
register_target_normalizer("cutedsl", _normalize_cutedsl_target_for_resolve, override=True)

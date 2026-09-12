from __future__ import annotations

import os

from tvm import IRModule, tirx
from tvm.target import Target

import tilelang
from tilelang.transform import PassContext


def retarget_private_host_launchers(mod: IRModule) -> IRModule:
    """Retarget split private schedule wrappers to their host target.

    Programmatic schedule functions initially need the full target so their
    T.Kernel regions can be lowered and split.  After splitting, the remaining
    wrapper is an internal host function.  Retargeting it here keeps ordinary
    host-to-host calls internal while LowerDeviceKernelLaunch rewrites only its
    calls to the extracted device kernels.
    """
    updates = {}
    for global_var, base_func in mod.functions.items():
        if not isinstance(base_func, tirx.PrimFunc) or base_func.attrs is None:
            continue
        if not base_func.attrs.get("tl.is_host_launcher"):
            continue
        target = base_func.attrs.get("target")
        if target is None or target.host is None:
            raise ValueError("a private TileLang host launcher requires a target with a host")
        updates[global_var] = base_func.with_attr("target", target.host)
    if updates:
        mod.update(IRModule(updates))
    return mod


def internalize_private_host_launcher_abis(mod: IRModule) -> IRModule:
    """Replace private launchers' external buffer handles with data pointers.

    This runs after buffer flattening and immediately before host/device
    splitting.  The private entry is called from already-unpacked host code,
    so its ABI is the buffer data pointers used by the extracted device
    kernels, not public DLTensor handles requiring MakePackedAPI.
    """
    updates = {}
    for global_var, base_func in mod.functions.items():
        if not isinstance(base_func, tirx.PrimFunc) or base_func.attrs is None:
            continue
        if not base_func.attrs.get("tl.is_host_launcher"):
            continue
        params = [base_func.buffer_map[param].data if param in base_func.buffer_map else param for param in base_func.params]
        updates[global_var] = tirx.PrimFunc(
            params,
            base_func.body,
            base_func.ret_type,
            {},
            base_func.attrs,
            base_func.span,
        )
    if updates:
        mod.update(IRModule(updates))
    return mod


def inline_private_host_launchers(mod: IRModule) -> IRModule:
    """Inline only host launch wrappers after their device calls are lowered.

    LowerDeviceKernelLaunch gives every remaining PrimFunc a symbol for codegen.
    Restore marked schedule wrappers to private status, then use TVM's existing
    private-function inliner.  Extracted device kernels retain their public
    symbols and remain reusable single definitions.
    """
    updates = {}
    for global_var, base_func in mod.functions.items():
        if not isinstance(base_func, tirx.PrimFunc) or base_func.attrs is None:
            continue
        if not base_func.attrs.get("tl.is_host_launcher"):
            continue
        updates[global_var] = base_func.without_attr("global_symbol").without_attr("tl.is_host_launcher")
    if updates:
        mod.update(IRModule(updates))
        mod = tirx.transform.InlinePrivateFunctions()(mod)
    return mod


def _env_data_race_check_enabled() -> bool:
    """Whether the data race check is enabled via the environment.

    The check is disabled by default; users can opt in by setting the
    ``TILELANG_ENABLE_DATA_RACE_CHECK`` environment variable to a truthy value
    (e.g. ``1``, ``true``, ``yes``, ``on``).
    """
    value = os.environ.get("TILELANG_ENABLE_DATA_RACE_CHECK")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def allow_vectorize(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    disable_vectorize = pass_ctx.config.get("tirx.disable_vectorize", False)
    return not disable_vectorize


def allow_global_thread_synchronization(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    enable_global_thread_sync = pass_ctx.config.get("tir.detect_global_barrier", False)
    return enable_global_thread_sync


def should_enable_aggressive_merge(pass_ctx: PassContext | None = None, target: Target | None = None) -> bool:
    del target
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx.config.get(tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE, False))


def should_force_let_inline(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx and pass_ctx.config.get(tilelang.PassConfigKey.TL_FORCE_LET_INLINE, False))


def should_enable_layout_visual(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE, False)


def should_enable_race_check(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    # The check is disabled by default because it can report false positives
    # (e.g. shared buffer stores whose per-thread addresses cannot be proven
    # distinct). Users can opt in via the TILELANG_ENABLE_DATA_RACE_CHECK
    # environment variable, or override it per-compile through the
    # `tl.disable_data_race_check` pass config.
    default_disable = not _env_data_race_check_enabled()
    disable = pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK, default_disable)
    return not disable


def should_disable_shared_memory_reuse(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    return bool(pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE, False))


def get_layout_visual_formats(pass_ctx: PassContext | None = None) -> list[str]:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    formats_value = pass_ctx.config.get(tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS, "")
    if not formats_value:
        return ["txt"]

    formats_str = formats_value.strip().lower()
    valid_formats = ["txt", "png", "pdf", "svg", "all"]

    if formats_str == "all":
        return ["txt", "png", "pdf", "svg"]

    if "," in formats_str:
        formats_list = [f.strip() for f in formats_str.split(",")]
    else:
        formats_list = [formats_str]

    invalid_formats = [f for f in formats_list if f not in valid_formats]
    if invalid_formats:
        raise ValueError(
            f"Invalid formats for TL_LAYOUT_VISUALIZATION_FORMATS: {invalid_formats}. "
            f"Valid formats are: {valid_formats}. "
            f"You can choose one of the valid formats or a comma-separated list of formats.(e.g., 'txt,png,pdf')"
        )
    return formats_list


def LayoutVisual(mod: IRModule) -> None:
    """Apply layout visualization pass if enabled."""
    if should_enable_layout_visual():
        formats = get_layout_visual_formats()
        tilelang.analysis.LayoutVisual(formats=formats)(mod)

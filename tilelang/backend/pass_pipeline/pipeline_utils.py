from __future__ import annotations

import os

from tvm import IRModule
from tvm.target import Target

import tilelang
from tilelang.transform import PassContext


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


def should_enable_gemm_accum_init_check(pass_ctx: PassContext | None = None) -> bool:
    if pass_ctx is None:
        pass_ctx = tilelang.transform.get_pass_context()
    # Enabled by default, unlike the data race check: this check only fires
    # when `clear_accum` is literally false and no write to the accumulator was
    # seen anywhere earlier, so correct kernels do not trip it.
    disable = pass_ctx.config.get(tilelang.PassConfigKey.TL_DISABLE_GEMM_ACCUM_INIT_CHECK, False)
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

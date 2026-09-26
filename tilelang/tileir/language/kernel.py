"""TileIR launch annotations on the common kernel launch."""

from __future__ import annotations

from tvm import tirx

from tilelang.language.kernel import KernelLaunchFrame, kernel_launch_factory, launch_kernel

__all__ = ["Kernel"]


@kernel_launch_factory
def Kernel(
    *blocks: int | tirx.PrimExpr,
    threads: int | list[int] | tuple[int, ...] | None = None,
    num_ctas: int | None = None,
    occupancy: int | None = None,
    num_worker_warps: int | None = None,
    tileir_hints: dict[str, dict[str, int]] | None = None,
) -> KernelLaunchFrame:
    """Construct a launch with CUDA Tile IR optimization hints.

    ``num_ctas`` is a power of two in [1, 16], ``occupancy`` is in [1, 32],
    and ``num_worker_warps`` is 4 or 8. Unset hints let the Tile IR compiler
    choose. ``threads`` describes the frontend thread extent; the compiler
    chooses the physical launch configuration.

    ``tileir_hints`` maps architecture names to entry-hint dictionaries, for
    example ``{"sm_100": {"num_cta_in_cga": 2}}``. This dictionary cannot be
    combined with scalar hint keywords. Load/store hints belong on ``T.copy``.
    TileIR lowering validates the hint names and ranges.
    """
    attrs: dict = {}
    for key, value in (
        ("num_ctas", num_ctas),
        ("occupancy", occupancy),
        ("num_worker_warps", num_worker_warps),
    ):
        if value is not None:
            attrs[f"tileir.{key}"] = int(value)
    if tileir_hints:
        if not isinstance(tileir_hints, dict):
            raise TypeError(f"tileir_hints must be a dict of per-arch hint dicts; got {type(tileir_hints).__name__}.")
        try:
            attrs["tileir.hints"] = {
                str(arch): {str(key): int(value) for key, value in (hints or {}).items()} for arch, hints in tileir_hints.items()
            }
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(
                "tileir_hints: each per-arch value must be a dict of numeric hint values "
                f"(e.g., {{'sm_100': {{'num_cta_in_cga': 2}}}}). Error encoding hints: {exc}"
            ) from exc
    return launch_kernel(blocks, threads=threads, **attrs)

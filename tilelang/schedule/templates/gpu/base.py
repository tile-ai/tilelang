"""Base schedule rule for GPU operators."""

from tilelang import tvm

from tvm.target import Target
from tvm.s_tir.dlight import ScheduleRule


def _as_const_int(expr) -> int | None:
    if isinstance(expr, int):
        return expr
    if isinstance(expr, tvm.tirx.IntImm):
        return int(expr.value)
    return None


def spatial_tile_from_config(config) -> list[int] | None:
    """Extract a positive spatial tile list from a fixed schedule config."""
    if config is None:
        return None
    if isinstance(config, dict):
        config = config.get("spatial_tile") or config.get("tile") or config.get("block")
    elif hasattr(config, "spatial_tile"):
        config = config.spatial_tile
    elif hasattr(config, "tile"):
        config = config.tile
    elif hasattr(config, "block"):
        config = config.block
    if config is None:
        return None
    try:
        tile = [int(value) for value in config]
    except TypeError:
        return None
    if not tile or any(value <= 0 for value in tile):
        return None
    return tile


def align_spatial_tile_to_extents(tile, extents) -> list[int] | None:
    """Align a full output tile to the surviving spatial schedule loops.

    TVM block normalization drops unit spatial axes in several rules.  Fusion
    tile search still works with the full buffer rank, so this helper removes
    those implicit unit dimensions by matching the tile to the visible loop
    extents in order.
    """
    tile = spatial_tile_from_config(tile)
    if tile is None:
        return None
    if len(tile) == len(extents):
        return list(tile)
    if len(tile) < len(extents):
        return None

    result = []
    start = 0
    remaining = len(extents)
    for extent in extents:
        remaining -= 1
        extent_int = _as_const_int(extent)
        stop = len(tile) - remaining
        chosen = None
        for idx in range(start, stop):
            candidate = tile[idx]
            if extent_int is not None and candidate > extent_int:
                continue
            if chosen is None:
                chosen = idx
            if (extent_int is None or extent_int > 1) and candidate > 1:
                chosen = idx
                break
        if chosen is None:
            return None
        result.append(tile[chosen])
        start = chosen + 1
    return result


def spatial_tile_product_for_extents(tile, extents) -> int | None:
    """Return the product of a fixed tile aligned to the given loop extents."""
    aligned = align_spatial_tile_to_extents(tile, extents)
    if aligned is None:
        return None
    product = 1
    for value in aligned:
        product *= int(value)
    return max(product, 1)


class GPUScheduleRule(ScheduleRule):  # pylint: disable=too-few-public-methods
    """The Schedule Rule specific to GPU targets, will return None if the target is not GPU."""

    def is_target_available(self, target: Target) -> bool:
        """Check whether the target is available for gpu rule.

        Parameters
        ----------
        target : Target
            The compilation target to check.

        Returns
        -------
        available : bool
            Whether the target is available for this rule.
        """
        return super().is_target_available(target) and "gpu" in target.keys

    def apply_config(self, func, target: Target, config, _: bool = False):
        """Apply this rule with a fixed spatial tile configuration.

        Rules that have a tile-oriented schedule override this method.  The
        default keeps existing behavior so every GPU rule can participate in a
        fixed-config scheduling pass without special casing.
        """
        return self.apply(func, target, _)

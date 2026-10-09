from ..scan_tiling_data import ScanStrategy

STATUS = "DESIGN_ONLY"
STRATEGY = ScanStrategy(
    "oneway_sklansky",
    "one core owns one resident row",
    True,
    "ceil(log2(R))",
    "static R; each level broadcasts the group anchor to the upper half",
)


def level_pairs(length: int, level: int):
    """Return (anchor, target) pairs for one forward Sklansky level."""
    half = 1 << level
    group = half << 1
    return [(start + half - 1, target) for start in range(0, length, group) for target in range(start + half, min(start + group, length))]

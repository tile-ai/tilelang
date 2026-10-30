from dataclasses import dataclass


@dataclass(frozen=True)
class Transpose021Plan:
    batches: int
    height: int
    width: int
    split_axis: str


def select_plan(batches: int, height: int, width: int, num_cores: int = 64) -> Transpose021Plan:
    split_axis = "H" if batches < num_cores and height >= num_cores else "N"
    return Transpose021Plan(batches, height, width, split_axis)


def reference(x):
    """Exact [N,H,W] -> [N,W,H] semantics."""
    return x.transpose(1, 2).contiguous()

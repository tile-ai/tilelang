from dataclasses import dataclass


@dataclass(frozen=True)
class TransposeTiling:
    block_x: int
    block_y: int
    num_cores: int = 64
    strategy: str = "base"

    def validate(self, shape_x: int, shape_y: int) -> None:
        if shape_x % self.block_x or shape_y % self.block_y:
            raise ValueError("shape must be exactly tiled; dispatch tails separately")


def select_tiling(shape_x: int, shape_y: int, num_cores: int = 64) -> TransposeTiling:
    block_x = 128 if shape_x % 128 == 0 else 64
    block_y = 128 if shape_y % 128 == 0 else 64
    if shape_x % block_x or shape_y % block_y:
        raise ValueError("PTO transpose supports dimensions divisible by 64")
    if min(shape_x, shape_y) <= 64:
        strategy = "small_shape"
    elif max(shape_x, shape_y) >= 4096:
        strategy = "big_dim"
    else:
        strategy = "base"
    return TransposeTiling(block_x, block_y, num_cores, strategy)

from dataclasses import dataclass


def ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class BroadcastTiling:
    rows: int
    cols: int
    tile_cols: int
    num_cores: int

    @property
    def active_cores(self) -> int:
        return min(self.rows, self.num_cores)

    def validate(self) -> None:
        if min(self.rows, self.cols, self.tile_cols, self.num_cores) <= 0:
            raise ValueError("broadcast dimensions and resources must be positive")

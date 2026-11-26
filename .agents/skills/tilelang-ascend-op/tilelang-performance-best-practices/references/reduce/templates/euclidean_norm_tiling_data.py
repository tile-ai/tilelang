from dataclasses import dataclass


@dataclass(frozen=True)
class EuclideanNormTiling:
    rows: int
    cols: int
    tile_cols: int
    num_cores: int
    strategy: str

    @property
    def active_cores(self) -> int:
        return min(self.rows, self.num_cores)

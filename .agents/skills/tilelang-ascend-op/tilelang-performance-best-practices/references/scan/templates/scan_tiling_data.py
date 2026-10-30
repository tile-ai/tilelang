from dataclasses import dataclass


@dataclass(frozen=True)
class ScanTiling:
    rows: int
    cols: int
    tile_cols: int
    num_cores: int
    strategy: str

    @property
    def active_cores(self) -> int:
        return min(self.rows, self.num_cores)


@dataclass(frozen=True)
class ScanStrategy:
    name: str
    ownership: str
    ub_resident: bool
    parallel_depth: str
    requirements: str

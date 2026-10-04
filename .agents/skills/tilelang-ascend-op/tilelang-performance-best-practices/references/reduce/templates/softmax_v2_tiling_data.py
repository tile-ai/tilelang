from dataclasses import dataclass


@dataclass(frozen=True)
class SoftmaxTiling:
    rows: int
    cols: int
    tile_cols: int
    num_cores: int
    strategy: str

    @property
    def active_cores(self) -> int:
        return min(self.rows, self.num_cores)


@dataclass(frozen=True)
class SoftmaxStrategy:
    name: str
    layout: str
    input_passes: int
    fp32_state: tuple[str, ...]
    selection: str

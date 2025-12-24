"""Rasteration Plan For L2 Cache Locality"""


class Rasterization:
    panel_width_ = None

    def __init__(self) -> None:
        pass

    def get_code(self) -> list[str]:
        raise NotImplementedError()

    @property
    def panel_width(self):
        assert self.panel_width_ is not None
        return self.panel_width_


class NoRasterization(Rasterization):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return "<NoRasterization>"

    def get_code(self) -> list[str]:
        return []


class Rasterization2DRow(Rasterization):
    """
    Rasterization by Row, each Row line width is panel_width
         _________
         _________|
        |_________
        __________|
    """

    def __init__(self, panel_width=4) -> None:
        super().__init__()
        self.panel_width_ = panel_width

    def __repr__(self) -> str:
        return f"<Rasterization2DRow({self.panel_width_})>"

    def get_code(self) -> list[str]:
        raise NotImplementedError()


class Rasterization2DColumn(Rasterization):
    """
    Rasterization by Column, each column line width is panel_width
            _
         | | | |
         | | | |
         |_| |_|
    """

    def __init__(self, panel_width=4) -> None:
        super().__init__()
        self.panel_width_ = panel_width

    def __repr__(self) -> str:
        return f"<Rasterization2DColumn({self.panel_width_})>"

    def get_device_function(self) -> str:
        return """
__device__ __inline__ dim3 rasterization2DColumn(const int panel_width) {
    const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
    const auto totalPanel = (gridDim.x * gridDim.y +panel_width * gridDim.x - 1) / (panel_width * gridDim.x);
    const auto totalBlock = gridDim.x * gridDim.y;
    const auto panelIdx = baseBlockIdx / (panel_width *gridDim.x);
    const auto strideLd = panelIdx + 1 < totalPanel ?panel_width : (totalBlock - panelIdx * (panel_width *gridDim.x)) / gridDim.x;
    const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * panel_width * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * panel_width *gridDim.x) / strideLd;
    const auto by = (baseBlockIdx - panelIdx * panel_width *gridDim.x) % strideLd + panelIdx * panel_width;
    const auto bz = blockIdx.z;

    dim3 blockIdx(bx, by, bz);
    return blockIdx;
}
    """

    def get_code(self, panel_width: int = None) -> list[str]:
        if panel_width is None:
            panel_width = self.panel_width_
        return [
            self.get_device_function(),
            f"const dim3 blockIdx = rasterization2DColumn({panel_width});\n",
        ]

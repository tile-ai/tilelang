from dataclasses import dataclass


@dataclass(frozen=True)
class Layout5HDPlan:
    n: int
    c: int
    h: int
    w: int
    c0: int = 16

    @property
    def c1(self) -> int:
        return (self.c + self.c0 - 1) // self.c0

    @property
    def output_shape(self) -> tuple[int, int, int, int, int]:
        return self.n, self.c1, self.h, self.w, self.c0


def nd_to_5hd_reference(x, c0: int = 16):
    """Reference mapping [N,C,H,W] -> [N,C1,H,W,C0] with zero channel padding."""
    import torch

    n, c, h, w = x.shape
    c1 = (c + c0 - 1) // c0
    padded = torch.zeros((n, c1 * c0, h, w), dtype=x.dtype, device=x.device)
    padded[:, :c] = x
    return padded.view(n, c1, c0, h, w).permute(0, 1, 3, 4, 2).contiguous()


def fivehd_to_nd_reference(x, channels: int):
    """Reference inverse [N,C1,H,W,C0] -> [N,C,H,W], cropping padded channels."""
    n, c1, h, w, c0 = x.shape
    return x.permute(0, 1, 4, 2, 3).reshape(n, c1 * c0, h, w)[:, :channels].contiguous()

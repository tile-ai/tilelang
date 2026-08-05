import torch
import torch.nn.functional as F
from torch import fx
from torch import nn

from tilelang.graph.converter import fx_to_relax


def test_conv_transpose2d_reads_kwargs():
    class Model(nn.Module):
        def forward(self, x, weight):
            return F.conv_transpose2d(x, weight, stride=2, padding=0)

    graph_module = fx.symbolic_trace(Model())
    mod, _ = fx_to_relax(
        graph_module,
        [
            torch.empty(1, 16, 8, 8, dtype=torch.float16),
            torch.empty(16, 8, 2, 2, dtype=torch.float16),
        ],
    )

    script = mod.script()
    assert "conv2d_transpose" in script
    assert "strides=[2, 2]" in script

import pytest

import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from examples.dequantize_gemm.quantize import _tir_u8_to_f4_to_bf16


@pytest.mark.parametrize(
    "use_buffer_scale",
    [False, True],
    ids=["python-int-zero", "uint16-buffer-load"],
)
def test_u8_to_f4_to_bf16_traces_symbolic_exponent_clamp(use_buffer_scale):
    @T.prim_func
    def main(
        packed: T.Tensor((1,), "uint8"),
        scale: T.Tensor((1,), "uint16"),
        output: T.Tensor((1,), "bfloat16"),
    ):
        with T.Kernel(1, threads=1):
            for i in T.Parallel(1):
                exponent_scale = scale[0] if use_buffer_scale else 0
                output[i] = _tir_u8_to_f4_to_bf16(
                    4,
                    packed[i],
                    i,
                    exponent_scale,
                    dtype="bfloat16",
                )

    min_nodes = []
    tvm.tirx.stmt_functor.post_order_visit(
        main.body,
        lambda node: min_nodes.append(node) if isinstance(node, tvm.tirx.Min) else None,
    )

    assert len(min_nodes) == 1
    assert min_nodes[0].dtype == T.uint16


if __name__ == "__main__":
    tilelang.testing.main()

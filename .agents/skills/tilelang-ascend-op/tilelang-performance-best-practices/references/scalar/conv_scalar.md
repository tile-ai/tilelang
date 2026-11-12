# PTO Conv/GEMM Scalar Streamlining

When a convolution is lowered to im2col/grouped GEMM, specialize shape, stride, padding, dilation, groups, and BM/BN/BK in a Python factory. Keep only output-tile and K-tile indices in the hot path, and reuse T.AscendTileScheduler from `examples/ascend/example_gemm_various_shapes.py`.

For depthwise convolution or small K, scalar overhead is proportionally high: generate a dedicated group/layout kernel, combine address calculations, and avoid extensive unrolling. For standard large GEMM, prioritize Cube/MTE pipeline optimization; scalar changes require timeline evidence.

Cover every layout, padding mode, group tail, and bias ordering. Compare generated-code size, scalar gaps, and end-to-end latency.

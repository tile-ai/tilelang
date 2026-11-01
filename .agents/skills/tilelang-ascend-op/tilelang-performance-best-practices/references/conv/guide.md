# PTO Convolution Optimization Index

## Supported Paths

First check whether PTO lowering in the current repository provides a convolution primitive that satisfies the required semantics. If not, transform the convolution explicitly into im2col/grouped GEMM and reuse `examples/ascend/example_gemm.py`. Do not invoke nonexistent Load3D or layout APIs.

The Python factory specializes N/C/H/W, kernel, stride, padding, dilation, groups, and layout. Generate im2col tiles only in UB/L1; do not write the complete intermediate matrix to GM. For standard convolution, output spatial×Cout forms GEMM M/N, while Cin×kH×kW forms K. Batch depthwise convolution by group and generate a specialized vector or grouped-GEMM path for small K.

## Gates

Accuracy coverage must include NCHW/supported layouts, padding/stride/dilation, groups, bias, every boundary window, and tail blocks. Use an fp32 accumulator for fp16/bf16. Performance comparisons include end-to-end latency, im2col address calculation, GM bytes, Cube/MTE/Vector time, and workspace. A primitive that has not passed a lowering microtest must not enter the code.

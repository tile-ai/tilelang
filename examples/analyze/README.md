# Performance Analyzer Examples

These examples use `tilelang.tools.Analyzer` to estimate FLOPs, global-memory
traffic, and roofline execution time from TileLang TIR. See the
[Performance Analyzer guide](../../docs/tools/analyzer.md) for the supported
operations, result fields, and current modeling limitations.

| File | Coverage |
| --- | --- |
| `example_gemm_analyze.py` | Tiled GEMM analysis and FLOP validation |
| `example_conv_analyze.py` | Convolution lowered through `T.im2col` and `T.gemm` |
| `test_example_analyze.py` | Test entry points for both examples |

Run the examples from the repository root:

```bash
python examples/analyze/example_gemm_analyze.py
python examples/analyze/example_conv_analyze.py
```

The examples construct a CUDA or CDNA device model from the active PyTorch
runtime. They therefore require a visible GPU even though the analyzer does not
execute the kernel.

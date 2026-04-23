**Notes**: This example targets the simple SM120 `mma.sync` FP4 (`float4_e2m1fn`) path exposed through `T.gemm`.

It now:
- compiles the kernel and prints the generated CUDA source
- runs a zero-input sanity check
- runs a numerical check against a float32 LUT-based FP4 reference
- prints a small benchmark

Current limitation:
- the frontend/runtime flow is exercised end-to-end, but the FP4 shared-to-register load path still uses a temporary fallback instead of the dedicated `b4x16_p64` load. Zero-check passes, while the numerical check currently reports a mismatch so the remaining backend gap is visible.

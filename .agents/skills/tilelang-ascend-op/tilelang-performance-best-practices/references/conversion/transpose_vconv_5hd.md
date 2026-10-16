# 5HD Layout Conversion

## Semantics

Convert logical input `[N, C, H, W]` to physical layout `[N, C1, H, W, C0]`:

```text
C1 = ceil_div(C, C0)
out[n, c // C0, h, w, c % C0] = x[n, c, h, w]
```

When `C` is not divisible by `C0`, invalid channels in the final C1 block must be zero-padded, and the reverse conversion must crop to the original C. See `templates/dav3510/transpose_transdata_5hd.py` for the exact PyTorch round-trip reference.

## PTO Data Flow

Python tiling first selects the core-partitioning axis among N/C/H/W, then selects the R/C split in UB. Prefer `T.copy` for contiguous C0 blocks. For rearrangement across channel blocks, use SIMD gather/scatter validated in the target repository; use the SIMT fallback for dtypes that cannot be expressed. Pad the two-dimensional footprints of both UB input and output to the full SIMD register width.

This semantic transformation cannot be replaced by a simple two-dimensional `[H*W, C1*C0]` transpose. The implementation must explicitly preserve N, C padding, the inner C0 dimension, and the physical output strides.

## Correctness and Performance Gates

Cover `C<C0`, `C=C0`, `C=C0±1`, multiple C1 blocks, H/W tile boundaries, every supported dtype, and ND→5HD→ND round trips. Performance reports must include valid bytes, padding bytes, GM bursts, SIMD gather cost, UB usage, and generated-code length.

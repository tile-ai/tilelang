# Type System

This page lists the data types supported by TileLang and how to specify them in
kernels. For full details and the authoritative list, see the API Reference
(`autoapi/tilelang/index`) and `tilelang.language.v2.dtypes`.

How to specify dtypes
- Use any of the following forms; TileLang normalizes them internally:
  - String: `'float32'`, `'int8'`, `'bfloat16'`, ...
  - TileLang dtype object: `T.float32`, `T.int8`, `T.bfloat16`, ...
  - Framework dtype: `torch.float32`, `torch.int8`, `torch.bfloat16`, ...

Common scalar types
- Boolean: `bool`
- Signed integers: `int8`, `int16`, `int32`, `int64`
- Unsigned integers: `uint8`, `uint16`, `uint32`, `uint64`
- Floating‑point: `float16` (half), `bfloat16`, `float32`, `float64`

Float8 and low‑precision families
- Float8: `float8_e3m4`, `float8_e4m3`, `float8_e4m3b11fnuz`, `float8_e4m3fn`,
  `float8_e4m3fnuz`, `float8_e5m2`, `float8_e5m2fnuz`, `float8_e8m0fnu`
- Float6: `float6_e2m3fn`, `float6_e3m2fn`
- Float4: `float4_e2m1fn`

Vectorized element types (SIMD packs)
- For many base types, vector‑packed variants are available by lane count:
  `x2`, `x4`, `x8`, `x16`, `x32`, `x64`.
- Examples:
  - Integers: `int8x2`, `int8x4`, ..., `int32x2`, `int32x4`, ...
  - Unsigned: `uint8x2`, `uint8x4`, ...
  - Floats: `float16x2`, `float16x4`, `float32x2`, `float32x4`, ...
  - Float8/6/4 families also provide `x2/x4/x8/x16/x32/x64` where applicable,
    e.g., `float8_e4m3x2`, `float8_e4m3x4`, `float6_e2m3fnx8`, `float4_e2m1fnx16`.

Notes
- Availability of certain low‑precision formats (float8/6/4) depends on target
  architecture and backend support.
- Choose accumulation dtypes explicitly for mixed‑precision compute (e.g.,
  GEMM with `float16` inputs and `float32` accumulators).
- The complete, up‑to‑date list is exposed in
  `tilelang.language.v2.dtypes` and rendered in the API Reference.

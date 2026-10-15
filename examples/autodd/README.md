# AutoDD Example

This directory contains a deliberately noisy TileLang program with a GEMM shape
mismatch and an expected reduced reproducer. See the
[AutoDD guide](../../docs/tools/autodd.md) for the complete CLI, backend
tradeoffs, timeout and concurrency guidance, and freeze annotations.

| File | Purpose |
| --- | --- |
| `tilelang_buggy.py` | Original program with irrelevant setup and a GEMM shape mismatch |
| `tilelang_minimized_expected.py` | Representative reduced reproducer |

Run the reduction from this directory:

```bash
python -m tilelang.autodd tilelang_buggy.py \
  --err-msg "T.gemm K shape check failed" \
  -o minimized.py
```

Then run `python minimized.py` to confirm that the selected failure remains.
The exact reduced source can change as AutoDD's rewrite rules evolve; compare
the result with `tilelang_minimized_expected.py` as a reference, not a byte-for-
byte golden file.

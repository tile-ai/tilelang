# FlashMLA on Ascend

Sparse attention kernels for Ascend 950, adapted from
[FlashMLA PR #229](https://github.com/deepseek-ai/FlashMLA/pull/229)
at `a123d0b0191e0da7aa1e044f8644e0989cc63220`.

| Files | Purpose |
| --- | --- |
| `kernel.py` | Persistent prefill and decode with `T.Stage` and compiler-managed buffer versions |
| `decode.py`, `quantization.py` | Packed FP8 main KV and optional FP8/FP4 extra KV |
| `run.py`, `run_decode.py` | Correctness checks and benchmarks against FlashMLA |
| `test_flashmla.py` | Kernel regression tests |

The kernels use H=64, Dqk=Dv=512, BF16 Q/output, and topk extents divisible
by 64. Prefill and decode support attention sinks, invalid indices, and per-query lengths.
Prefill accepts contiguous BF16 KV; decode requires contiguous packed KV pages,
with each cache smaller than 4 GiB.

## Run

Build TileLang with Ascend support; see the [backend guide](../../../tilelang/ascend/README.md).
The kernels require [#3623](https://github.com/tile-ai/tilelang/pull/3623)
for buffer view layouts and [#3624](https://github.com/tile-ai/tilelang/pull/3624)
for configuration-register dependencies.

The runners use the pinned FlashMLA checkout's input generators, numerical
references, quantization, and `kernelkit.bench` profiler. Build its Ascend
extension first:

```bash
cd /path/to/FlashMLA
MAX_JOBS=16 FLASH_MLA_BUILD_TARGET_PLATFORM=ASCEND python setup.py build_ext --inplace
```

From the TileLang repository root, select an idle NPU with
`ASCEND_RT_VISIBLE_DEVICES`, then run:

```bash
export PYTHONPATH=.
export FLASHMLA_REFERENCE=/path/to/FlashMLA
python examples/ascend/flashmla/run.py --reference "$FLASHMLA_REFERENCE" \
  --mode correctness --output /tmp/flashmla-prefill.json
python examples/ascend/flashmla/run_decode.py --reference "$FLASHMLA_REFERENCE" \
  --mode correctness --output /tmp/flashmla-decode.json
python -m pytest examples/ascend/flashmla/test_flashmla.py -q
```

Use `--mode bench --rounds 3 --repeat 10` for benchmarks and `--case N` for
one case. `--implementation` accepts `reference`, `tilelang`, or
`both` (the default).
Benchmark runners check outputs before timing and write per-launch durations
to `--output`.

Pair gathers use a zero-or-two burst count. The local `T.assume(n_burst > 0)`
suppresses the compiler's copy guard while retaining the runtime DMA count;
it is a code-generation workaround, not a valid input invariant. A zero
burst must preserve the UB contents and issue no GM read.

## License

The adapted FlashMLA code and scheduling algorithms retain the following notice:

```text
MIT License

Copyright (c) 2025 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

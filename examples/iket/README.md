# TileLang IKET Examples

These examples exercise the experimental IKET integration for TileLang's CUDA
backend. See the [IKET profiling guide](../../docs/tools/iket.md) for setup,
API details, cache and callback behavior, payload semantics, trace collection,
and troubleshooting.

## Examples

| File | Coverage |
| --- | --- |
| `minimal.py` | Instant markers and a warp-local range |
| `payload_minimal.py` | One `int32` runtime payload marker |
| `all_features.py` | Ranges, no-payload markers, and runtime payload markers |

The examples require TileLang, CUDA, PyTorch, and the external IKET package.

## Run Directly

Compile and run the minimal marker example:

```bash
python examples/iket/minimal.py \
  --iket-output-dir /tmp/tilelang_iket_minimal
```

Run the minimal payload example with runtime capture enabled:

```bash
python examples/iket/payload_minimal.py \
  --iket-output-dir /tmp/tilelang_iket_payload_minimal \
  --iket-runtime-payloads
```

Run the comprehensive example:

```bash
python examples/iket/all_features.py \
  --iket-output-dir /tmp/tilelang_iket_all_features \
  --iket-runtime-payloads
```

These commands validate the kernels and write generated CUDA source. Use the
external profiler to collect a trace.

## Collect a Trace

```bash
rm -rf /tmp/tilelang_iket_all_features_profile

python -m iket.cli.main \
  --output-dir /tmp/tilelang_iket_all_features_profile \
  --clobber \
  profile \
  --postprocess all \
  -- \
  python examples/iket/all_features.py \
    --iket-output-dir /tmp/tilelang_iket_all_features_profile \
    --iket-runtime-payloads
```

Serve the output directory and open the generated
`iket_pid_0x....html` file:

```bash
cd /tmp/tilelang_iket_all_features_profile
python3 -m http.server 8080
```

The expected Perfetto timeline is shown in
[`assets/iket_perfetto_all_features.png`](assets/iket_perfetto_all_features.png).

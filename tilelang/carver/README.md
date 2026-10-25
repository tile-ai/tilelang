# Carver hardware helpers on standalone TIRx

The Schedule/TE-based Carver templates, graph nodes, shape inference, and policy
search are retired in the TIRx rebase. TIRx does not provide the legacy TVM
Schedule or TE APIs, and TileLang does not carry a second compiler to supply them.

The following independent helpers remain available:

- `tilelang.carver.arch`: CUDA, CPU, CDNA, RDNA, and Metal hardware models.
- `tilelang.carver.arch.driver`: device-property queries.
- `tilelang.carver.roller.hint`: configuration data and serialization.
- `tilelang.carver.roller.rasterization`: rasterization descriptors.

Use `tilelang.autotuner.AutoTuner` or `tilelang.autotuner.autotune` with explicit
candidate configurations for native TileLang kernels. The GEMM autotuning
examples use this path. The former `with_roller=True` option reports an explicit
migration error; it does not silently substitute a different search algorithm.

See `MIGRATION_TIRX.md` for dependency installation and API migration details.

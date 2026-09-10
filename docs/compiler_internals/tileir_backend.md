# TileIR Backend

The TileIR backend is a CUDA backend that lowers the high-level TileLang TIR AST
to CUDA Tile IR before CUDA-specific TileLang passes rewrite tile operations into
PTX intrinsics. It assembles the resulting TileIR bytecode with `tileiras`.

This backend does not lower through the cuTile Python DSL. The dependency
boundary is:

- CUDA Tile IR Python bindings from the public `NVIDIA/cuda-tile` `v13.4.0`
  release built with
  `CUDA_TILE_ENABLE_BINDINGS_PYTHON=ON`. TileLang requires CUDA Tile IR 13.4
  bindings so the backend can use the 13.4 dialect surface.
- `tileiras` from CUDA Toolkit 13.4 or the matching NVIDIA Python toolchain
  wheels.
- cuTile 1.5's native dispatcher runtime for loading and launching the assembled
  cubin. TileLang pins this minor version because the precompiled-cubin bridge
  follows cuTile's internal dispatcher ABI; it does not use the cuTile Python
  DSL.

Install the assembler wheels:

```bash
pip install 'tilelang[tileir]'
```

The extra installs the supported cuTile 1.5 runtime and CUDA 13.4 assembler
stack. cuTile 1.5 supports Python 3.10 through 3.14, matching TileLang's declared
Python-version range. Build the remaining CUDA Tile IR Python bindings from the
matching public release:

```bash
pip install 'nanobind>=2.9,<3.0'
git clone --branch v13.4.0 --depth 1 https://github.com/NVIDIA/cuda-tile.git
cmake -G Ninja -S cuda-tile -B cuda-tile/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DCUDA_TILE_ENABLE_BINDINGS_PYTHON=ON
cmake --build cuda-tile/build
```

Expose that build's Python bindings:

```bash
export PYTHONPATH=/path/to/cuda-tile/build/python_packages:${PYTHONPATH}
```

For a manually managed `tileiras` binary whose `--version` output omits a
numeric version, declare the verified ABI explicitly:

```bash
export TILELANG_TILEIRAS_VERSION=13.4
```

Compile with the backend:

```python
kernel = tilelang.compile(
    prim_func,
    target="tileir -arch=sm_120",
    execution_backend="tileir",
)
```

The lowering implementation must use structured CUDA Tile IR builder APIs. Do
not lower TileLang TIR by formatting textual MLIR, and do not route through
TileLang's CUDA/PTX lowering passes before entering TileIR.

## TileLang Semantic IR

The TileIR frontend boundary starts from TileLang's high-level TIR AST. The
Python semantic contract lives in `tilelang/tileir/semantic.py` and is exposed as
`extract_semantic_program`.

The backend is split into two packages: `tilelang/tileir/` owns the compiler
(Semantic IR extraction, structured CUDA Tile IR lowering, assembly, and
toolchain detection), and `tilelang/jit/adapter/tileir/` owns the runtime adapter
that turns a lowered kernel into a torch-callable function.

Shared buffers with data-dependent scatter or atomic indices require addressable
memory. The runtime supplies a hidden byte workspace with a disjoint, aligned
slice for each tile block. This avoids CUDA 13.4's observed duplication of
`alloca` across producer and consumer warp groups. Workspace allocation follows
the launch stream and CUDA Graph lifetime; ordinary SSA tiles need no workspace.
The version 3 cache artifact records the bytes required per block and validates
them against the active TIR before loading a cached cubin.

The semantic model records:

- program parameters and global temporary allocations;
- per-kernel launch grid/thread extents;
- block-local allocation shape, dtype, and scope, including `shared.dyn` and
  `local.fragment`;
- structured control flow, including `let`, `if`, serial/parallel/vectorized/
  unrolled/thread-binding loops, and `T.Pipelined` loops via `num_stages` and
  `tl_pipeline_*` annotations;
- TileLang tile operations with explicit buffer regions for `fill`, `copy`,
  `gemm`, and `reduce`.

Unsupported statements, attrs, or tile operations raise
`TileLangSemanticError`. They must be represented explicitly before they can be
lowered; they must not be hidden behind a generic fallback node.

`lower_primfunc_to_tileir` extracts this semantic program before any CUDA Tile IR
builder call. Multi-kernel functions are first represented as semantic kernels,
then split into per-kernel PrimFuncs for argument pruning and assembly. The
device lowerer receives a `SemanticKernel`. Semantic statements carry explicit
binding, region, index, and control-flow payloads. Whole `PrimFunc` and `Stmt`
back-references do not cross the semantic boundary; expression payloads remain
typed TIR `PrimExpr` leaves, including `Call`, until semantic-to-TileIR lowering
converts them into TileIR values.

Before assembly, TileLang invokes the CUDA Tile IR optimizer at the selected
optimization level, then writes TileIR bytecode for `tileiras`. TileLang does
not tune CUDA Tile IR pass
ordering or synthesize performance hints. It only validates and forwards
explicit user hints from `T.Kernel(...)` and `T.copy(...)`.

There is no alternate bytecode-JIT or CUDA-driver launcher path. If structured
lowering, `tileiras`, or the native dispatcher is unavailable, TileIR compilation
or launch fails at that boundary.

## Validation

TileIR support is gated by CUDA Tile IR 13.4 bindings, `tileiras` 13.4, and the
cuTile 1.5 native dispatcher. In developer environments where those
dependencies are present, run the focused JIT suite:

```bash
python -m pytest testing/python/jit/test_tilelang_jit_tileir.py -q --tb=short
```

If the CUDA Tile IR Python bindings or assembler are missing, TileIR-specific
tests skip or fail at the dependency boundary instead of falling back to another
execution backend. This keeps coverage tied to the structured TileIR path that
the backend actually exposes.

The dedicated `tileir` CI job builds CUDA Tile 13.4 bindings from pinned CUDA
Tile and LLVM commits with its active Python 3.12 interpreter, then installs the
`tileir` extra and checks toolchain availability before running the TileIR tests
and examples. The ordinary CUDA, ROCm, and Metal jobs do not require these
optional dependencies or a preconfigured bindings path.

To reproduce the bindings build in an active virtual environment with `uv`, Git,
and a C++ compiler available:

```bash
bash .github/scripts/build_cuda_tile.sh /tmp/cuda-tile-ci-build
export PYTHONPATH=/tmp/cuda-tile-ci-build/build/python_packages
python -m pip install '.[tileir]'
python -c 'from tilelang.tileir.checks import check_tileir_available; print(check_tileir_available())'
```

Use a fresh build directory. The first build also compiles the matching LLVM/MLIR
libraries and can take substantially longer than a TileLang-only build.

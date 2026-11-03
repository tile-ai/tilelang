# Ascend 950 SuperKernel adaptation

The Ascend C backend can emit SuperKernel (SK) subfunctions alongside its
ordinary kernels. The ordinary launch ABI and TVM-FFI/ACL launch path stay the
same. The caller registers the operator with Torch and enables graph fusion;
TileLang does not directly launch the SK entry or enable framework fusion.

## Enable generation

For an existing PrimFunc `func`:

```python
import tilelang

kernel = tilelang.compile(
    func,
    target="ascend",
    execution_backend="tvm_ffi",
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_ENABLE_SUPERKERNEL: True,
    },
)
```

Pass the same `pass_configs` dictionary to `@tilelang.jit` when using JIT
compilation. Source-only lowering also honors the options through
`tvm.transform.PassContext(config=...)`.

Generation is disabled by default. When enabled, each device kernel retains its
ordinary entry and gets an argument structure, a templated `__sk__` entry with
`sk::SkSystemArgs*`, and `SK_BIND`. The template argument creates equivalent
symbols, not different computations. Parameter fields follow the final device
signature, including dynamic parameters; scalar fields smaller than four bytes
use `alignas(4)`. Grid-constant by-value parameters are currently unsupported.

## Task context

| Kernel type | Block index | Runtime block count | Sub-block index |
| --- | --- | --- | --- |
| Vector | `SkGetBlockIdx()` | `SkGetNumBlocks()` | Original implementation |
| Cube | Original implementation | `SkGetNumBlocks()` | Original implementation |
| MIX 1:2 | Original implementation | `SkGetNumBlocks()` | Original implementation |

The SK entry resolves task context before calling VF helpers. Pure Vector VF
helpers receive the task index explicitly. Runtime queries written with
`T.call_extern` to `AscendC::GetBlockNum` or `asc_get_block_num` use the SK count;
Vector index queries to `AscendC::GetBlockIdx` or `asc_get_block_idx` use the SK
index. Arbitrary imported C/C++ functions are not rewritten. Their authors must
pass the required task context explicitly.

`T.get_block_extent()` is a compile-time launch extent and retains its existing
meaning. Hard-coded loop strides are not automatically turned into runtime
counts. This matters when opting into grid expansion.

The SK entry calls `asc_init()` and `SkSystemArgs::SkInit()`. The latter
initializes the SDK early-start synchronization configuration. Existing
computation, synchronization and hardware-state resets are retained. Imported
helpers remain responsible for their own asynchronous resource and Cache
contracts, including TPipe cleanup when applicable.

## Binding options

All keys below have the prefix `tl.ascend_` and corresponding
`PassConfigKey.TL_ASCEND_*` enum members. Options apply to the kernels generated
by one compilation. Supplying an SK-specific option without enabling SK is an
error. The PTO backend rejects these options.

| Key suffix | Default | Meaning |
| --- | --- | --- |
| `enable_superkernel` | `False` | Emit SK entries and bindings |
| `superkernel_num_variants` | `4` | Bind 1 to 4 SK entry symbols |
| `superkernel_disable_dcci` | `False` | Cap 4: suppress DCCI ALL around calls |

Only these three options are exposed. `superkernel_disable_dcci=False` emits
`cap=0`; setting it to `True` emits `cap=4`. Only disable DCCI when the kernel
already guarantees GM cache consistency.

## Code generation organization

SuperKernel generation is a separate step after all ordinary Ascend entries
have been emitted:

```cpp
for (...) {
  cg.AddFunction(f);
}
EmitSuperKernel(mod, &cg);
return cg.Finish();
```

`src/ascend/codegen/codegen_superkernel.cc` owns configuration validation, symbol
checks, SK argument structures, entry signatures, task-context IR rewriting,
parameter unpacking and `SK_BIND`. The ordinary generator only extracts its
existing per-function setup into `PrepareFunction`, shared with the SK emitter.
It has no SK mode flag or SK-specific statement/expression visitor branches.

For pure Vector kernels the SK copy replaces block-index bindings with explicit
IR variables. Ordinary VF capture analysis then passes task context to helpers.
The source IR module and its ordinary launch metadata are not modified.

## SDK and verification

Use a CANN SDK that supports `__sk__`, `SK_BIND`, and the task-context methods on
Ascend 950 (`dav-3510`). Public documentation and released SDK support tables can
differ; source generation alone does not establish toolchain support.

The existing device compile/link flow is used. `SK_BIND` emits `.ascend.meta`
binding records; these and the SK entry symbols must survive the final device
link. No host argument-packing or ordinary runtime launch change is required.

Run the source and SDK artifact tests after building with `USE_ASCEND=ON`:

```bash
source /home/cann/9.3.0_1014/ascend-toolkit/set_env.sh
python -m pytest testing/ascend/target/test_ascend_superkernel.py -q
```

`TILELANG_TEST_ASCEND_COMPILE_FLAGS` can supply toolchain-specific flags for SDK
artifact tests. For example, when this Bisheng version is used on a host whose
GCC 13 C++ headers fail to compile the SDK, select installed GCC 11 headers with
`-nostdinc++` and the appropriate `-isystem` paths. This is a host toolchain
setting, not a default TileLang compile option.

On a 950 system, compare ordinary execution with SK generation off/on, then
run two dependent Torch-registered operators through `npugraph_ex` with
SuperKernel enabled. Check numerical results and inspect fusion logs/artifacts
and device tasks to establish actual SK execution. Test different block counts,
tails, dynamic parameters, MIX synchronization, and DCCI behavior separately.
Speed changes alone do not prove fusion.

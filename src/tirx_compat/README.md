# Private TIRx extension declarations

These headers describe the pinned TIRx helpers and runtime module factories used
by TileLang. TileLang uses the same declarations for source and installed
dependency builds; an installed TIRx package need not install its private headers.

C-family code generators are no longer private declarations of dependency
classes. TileLang owns those implementations in `src/*/codegen/tirx` and
`src/backend/common/codegen/tirx`, under `tvm::tl::codegen_impl`. They register
`tl.codegen.*` builders without replacing native `target.build.*` registrations.
The LLVM boundary lowers TileLang constructs before calling the native LLVM
builder; it does not copy or replace the LLVM implementation.

`transform.h` declares the existing upstream `ConvertSSA(Stmt)` and
`PointerValueTypeRewrite(Function, ...)` helpers used by TileLang-owned passes.
It does not introduce a dependency-side API or copy their implementations.

`src/transform/storage_rewrite.cc` wraps the latter helper. When TileLang
metadata refers to storage, it uses the native options to preserve storage and
parameter element types rather than leaving metadata attached to obsolete
scalar/vector definitions. Vectorized accesses are retained independently.

| TileLang header | Corresponding internal TIRx header |
| --- | --- |
| `source_module.h` | Factory declarations from `src/codegen/c/codegen_source_base.h` |
| `build_common.h` | `src/codegen/build_common.h` |
| `launch_param.h` | `src/runtime/launch_param.h` |
| `metadata.h` | `src/runtime/metadata.h` |
| `transform.h` | `ConvertSSA` from `src/tirx/transform/ir_utils.h`; `PointerValueTypeRewrite` implemented in `src/tirx/transform/storage_rewrite.cc` |
| `cuda_fallback_module.h` | `src/backend/cuda/codegen/cuda_fallback_module.h` |
| `rocm_fallback_module.h` | Factory declaration from `src/backend/rocm/codegen/rocm_fallback_module.h` |
| `metal_fallback_module.h` | `src/backend/metal/codegen/metal_fallback_module.h` |
| `webgpu_fallback_module.h` | `src/backend/webgpu/codegen/webgpu_fallback_module.h` |

The declarations retain the dependency's types and class layouts. Update them
alongside any change to that private ABI and rebuild both libraries. The local
include paths and import annotations are intentionally different from the
internal headers. This is a pinned compatibility surface, not a stable upstream
C++ SDK promise.

`cmake/tirx_extension_exports.cmake` owns the matching symbol visibility policy.
Source builds apply it after adding TIRx's targets. The dependency provisioning
helper injects the same policy through `CMAKE_PROJECT_tvm_tirx_INCLUDE` for an
independent TIRx wheel build. No source patch or reverse TileLang include is
applied to TIRx. An arbitrary binary built without these exports is not assumed
to support TileLang's private C++ extension interface.

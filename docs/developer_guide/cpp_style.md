# C++ Style Guide

TileLang C++ code should be easy to read for contributors who already know TVM.
Use TVM's C++ conventions as the baseline, with small TileLang-specific rules
where the compiler, runtime, or generated kernel templates need them.

This guide applies to C++ source under `src/` and TileLang-owned headers. Do not
apply broad style-only churn to vendored code under `3rdparty/` or generated
runtime templates unless a local change already needs to touch that code.

## Formatting

Run the repository formatter before sending a pull request:

```bash
bash format.sh --files <changed-file>...
```

For all changed files relative to the merge base:

```bash
bash format.sh
```

Formatting is mechanical. Keep style changes separate from behavior changes when
the diff would otherwise become noisy.

TileLang aims to stay close to TVM's Google-based C++ formatting. If the
repository `.clang-format` is changed, do it in a focused pull request and avoid
mixing that change with semantic edits.

## Naming

Use the following names in new C++ code.

| Entity | Style | Example |
| --- | --- | --- |
| File names | `lower_snake_case` | `lower_tile_op.cc`, `layout_reducer.h` |
| Namespaces | `lower_snake_case` | `tvm`, `tl`, `tirx` |
| Types, classes, structs | `PascalCase` | `LayoutNode`, `TileOperator`, `LowerArgs` |
| Object nodes | `PascalCaseNode` | `GemmNode`, `FragmentNode` |
| Object refs | `PascalCase` | `Gemm`, `Fragment`, `Layout` |
| Functions and methods | `PascalCase` | `MakeLinearLayout`, `InferLayout`, `GetAccessRegions` |
| Boolean functions | `Is`/`Has`/`Can` + `PascalCase` | `IsFragmentBuffer`, `HasTmaBarrier` |
| Parameters and locals | `lower_snake_case` | `layout_map`, `thread_bounds`, `buffer_remap` |
| Private/protected members | `lower_snake_case_` | `layout_remap_`, `analyzer_` |
| Constants and enum values | `kPascalCase` | `kWarpSize`, `kAccessRead` |
| Macros | `UPPER_SNAKE_CASE` | `TVM_REGISTER_GLOBAL`, `TIR_REGISTER_TL_TILE_OP` |

Prefer:

```cpp
Layout MakeLinearLayout(Array<PrimExpr> shape);
bool IsSharedBuffer(const Buffer& buffer);

Stmt Lower(const LowerArgs& args, arith::Analyzer* analyzer) const;
LayoutMap InferLayout(const LayoutInferArgs& args, InferLevel level) const;
```

Avoid adding new names like:

```cpp
Layout makeLinearLayout(Array<PrimExpr> shape);
bool isSharedBuffer(const Buffer& buffer);
Stmt Lower(const LowerArgs& T, arith::Analyzer* analyzer) const;
```

Existing lowerCamelCase names can be migrated when related code is touched. Do
not rename large areas solely to satisfy this guide unless the change is scoped
and easy to review.

## TVM Object And FFI Types

TileLang C++ code uses TVM object handles heavily. Treat `Stmt`, `PrimExpr`,
`Buffer`, `Var`, `SBlock`, `Layout`, and similar types as `ObjectRef` handles.
Treat `StmtNode`, `BufferNode`, `VarNode`, and other `*Node` types as raw node
views used mainly for visitor callbacks and local inspection.

Use handle types across function boundaries:

```cpp
Optional<For> FindPipelineLoop(const Stmt& stmt);
Map<Buffer, Layout> InferBufferLayouts(const SBlock& block);
```

Keep raw node pointers local to visitor callbacks, pattern checks, and
`CopyOnWrite()` mutation sites:

```cpp
Stmt VisitStmt_(const ForNode* op) final {
  For loop = GetRef<For>(op);
  Optional<For> pipeline_loop = FindPipelineLoop(loop->body);
  if (pipeline_loop.defined() && pipeline_loop.value().same_as(loop)) {
    ...
  }
  return StmtMutator::VisitStmt_(op);
}
```

For identity comparisons between handles, use `.same_as(other)`. For structural
comparisons, use TVM structural equality utilities such as `StructuralEqual`.
Do not use `handle.get() == other.get()` outside narrow adapter code.

For unordered identity maps or sets keyed by TVM handles, use TVM pointer hash
and equality helpers:

```cpp
using BufferSet = std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>;
using VarMap = std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>;
```

Use `Optional<T>` for nullable TVM handle values. Avoid storing `const *Node`
pointers as nullable state.

## ObjectNode Fields

For TVM-style `ObjectNode` classes, public reflected fields should use
`lower_snake_case` without a trailing underscore:

```cpp
class CopyNode : public TileOperatorNode {
 public:
  Buffer src;
  Buffer dst;
  Array<Range> ranges;

  static void RegisterReflection() {
    namespace refl = reflection;
    refl::ObjectDef<CopyNode>()
        .def_ro("src", &CopyNode::src)
        .def_ro("dst", &CopyNode::dst)
        .def_ro("ranges", &CopyNode::ranges);
  }
};
```

Use trailing underscores for private or protected implementation state:

```cpp
class LayoutRemapRewriter : public arith::IRMutatorWithAnalyzer {
 private:
  Map<Buffer, Layout> layout_remap_;
};
```

Be careful when renaming reflected fields. FFI-visible field names are part of
the Python/debugging surface, so compatibility should be considered before
changing them.

## Function Parameters

Use descriptive parameter names:

```cpp
Stmt Lower(const LowerArgs& args, arith::Analyzer* analyzer) const;
LayoutMap InferLayout(const LayoutInferArgs& args, InferLevel level) const;
```

Avoid `T` as a parameter name for lowering context. In TileLang code, `T` is
already associated with the Python DSL namespace, and in C++ it is commonly used
as a template type parameter.

Pass non-trivial inputs by `const&` unless the function intentionally consumes
the value. Take a value when the function will store it and move from it:

```cpp
explicit DeviceRegionAnnotator(Target device_target)
    : device_target_(std::move(device_target)) {}
```

## Headers

Keep headers narrow and predictable.

- Avoid `using namespace` in installed public headers or headers that are widely
  included across TileLang subsystems.
- Prefer explicit qualifications or narrow aliases in the smallest namespace
  where they are needed.
- Include the headers that define the types used by the file.
- Keep implementation-only helpers in `.cc` files or anonymous namespaces.
- Keep public macros rare and name them with a TileLang or TVM prefix.

Prefer:

```cpp
namespace tvm {
namespace tl {

using LayoutMap = ffi::Map<tirx::Buffer, Layout>;

}  // namespace tl
}  // namespace tvm
```

Avoid:

```cpp
using namespace tirx;
using namespace ffi;
```

## Namespace Usage

Namespace rules should make public APIs predictable without making implementation
files noisy. TVM itself is not mechanically uniform across the whole repository,
so TileLang uses the following policy as a review guideline rather than as a
blanket rewrite rule.

TileLang currently does not maintain a separate installed C++ header tree like
TVM's `include/tvm/`. The strictest rule applies to any future installed public
headers and to TileLang-owned headers that behave like shared interfaces inside
the repository: headers that define cross-module APIs, ObjectRef/ObjectNode
types, FFI-visible fields, or interfaces included across multiple subsystems. Do
not add namespace-wide imports there. Use qualified names for ordinary types:

```cpp
class LayoutNode : public ffi::Object {
 public:
  ffi::Array<tirx::PrimExpr> input_shape;
  ffi::Optional<tirx::Buffer> buffer;
};
```

Use a narrow alias in a header only when the alias is part of the API or removes
real repetition:

```cpp
using LayoutMap = ffi::Map<tirx::Buffer, Layout>;
```

Implementation files may use file-local namespace imports when the file is
dense with IR or DSL expressions:

```cpp
namespace tvm {
namespace tl {

using namespace tirx;

}  // namespace tl
}  // namespace tvm
```

Narrowly included implementation headers under `src/` may follow the same rule
as `.cc` files. This mirrors TVM's internal schedule and transform headers,
where `tirx` imports are sometimes used to keep IR visitor code readable. If an
internal helper header grows into a cross-module interface, revisit its
namespace imports.

For `ffi`, prefer explicit names such as `ffi::Any`, `ffi::Array`,
`ffi::Map`, and `ffi::make_object` in core-style code. A `.cc` file may use
`using namespace ffi;` when FFI helpers are pervasive and the shorter names make
the implementation materially easier to read. Keep the choice consistent within
the file.

Do not run mechanical namespace rewrites across unrelated code. When touching a
file for functional work, avoid making its namespace style less clear, but keep
large style-only cleanups isolated in focused pull requests.

## Comments And Documentation

Use Doxygen comments for public APIs, public classes, and non-obvious fields.
Use short line comments for local implementation notes only when they explain
why the code exists or why a constraint matters.

Good comments describe invariants:

```cpp
// The barrier buffer is allocated lazily because only TMA paths require it.
Optional<Buffer> mbarrier_buffer;
```

Avoid comments that restate the code:

```cpp
// Set the target.
args.target = target;
```

## Error Handling

Use TVM-style checks and fatal errors for compiler invariants:

```cpp
ICHECK(layout.defined()) << "Expected layout for buffer " << buffer;
LOG(FATAL) << "Unsupported copy instruction: " << instruction;
```

Make error messages actionable. Include the operation, relevant buffer or
layout, and the unsupported condition where possible.

Use user-facing validation errors where the problem can be caused by a TileLang
program rather than an internal invariant. Keep internal invariant failures
distinguishable from frontend validation failures.

## Includes

Group includes consistently:

1. The paired header for a `.cc` file, when one exists.
2. C and C++ standard library headers.
3. TVM and other third-party headers.
4. TileLang local headers.

Keep include order stable within each group and let the formatter handle
spacing. Avoid adding transitive includes just because another header currently
pulls in the type.

## Exceptions

Generated kernel templates, CUDA/HIP runtime shims, and external API adapters
may need to match the naming of CUDA, HIP, Metal, C APIs, or generated source.
In those files, prefer consistency with the external interface over forced
renaming.

Do not reformat or rename vendored files under `3rdparty/` as part of TileLang
style cleanup.

## Migration Policy

Style convergence should be incremental.

- New C++ code should follow this guide.
- Code touched for functional work should avoid adding new inconsistencies.
- Mechanical renames should be small, focused, and easy to review.
- Avoid combining broad formatting changes with behavior changes.
- Preserve FFI-visible names unless a compatibility plan is documented.

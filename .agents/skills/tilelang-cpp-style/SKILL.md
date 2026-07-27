---
name: tilelang-cpp-style
description: Use when editing, reviewing, or documenting TileLang C++ code style, including naming conventions, clang-format choices, headers, TVM ObjectRef/ObjectNode style, FFI-visible fields, and incremental style cleanup plans.
---

# TileLang C++ Style

Use this skill when C++ style decisions matter in TileLang. For broad policy
changes, review the full guide in `docs/developer_guide/cpp_style.md`.

## Baseline

TileLang C++ should be readable to TVM contributors. Follow TVM/Google C++ style
unless a TileLang runtime template, generated kernel, backend shim, or external
API adapter has a concrete reason to match another convention.

Do not apply broad style churn to `3rdparty/` or generated/template code unless
the task explicitly targets those files.

Before writing new C++:

- Identify whether the file is TileLang-owned compiler/runtime code, a
  `src/tl_templates` runtime template, a CUDA/HIP/Metal shim, or an external API
  adapter. Owned compiler/runtime code follows this skill. Templates and shims
  may follow the generated or external interface they model.
- Prefer the nearest local pattern in the same subsystem (`op`, `layout`,
  `transform`, `backend`, `cuda/op`, `rocm/op`, etc.) over inventing a new
  abstraction.
- Keep style-only edits separate from behavior changes unless the style cleanup
  is needed to make the behavioral change reviewable.

## Naming Checklist

- File names: `lower_snake_case`.
- Namespaces: `lower_snake_case`.
- Types, classes, structs, ObjectRefs, ObjectNodes: `PascalCase`.
- Object nodes: `PascalCaseNode`; ObjectRefs: `PascalCase`.
- Public functions and methods: `PascalCase`.
- Private/protected methods: `PascalCase_`.
- Inherited TVM visitor, mutator, and codegen hooks keep upstream names such as
  `VisitExpr_`, `VisitStmt_`, `VisitStmtDefault_`, and `InitFuncState_`.
- Boolean helpers: `Is`/`Has`/`Can` + `PascalCase`.
- Parameters and local variables: `lower_snake_case`.
- Private/protected data members: `lower_snake_case_`.
- Constants and enum values: `kPascalCase`.
- Macros: `UPPER_SNAKE_CASE`.

Prefer new C++ APIs like:

```cpp
Layout MakeLinearLayout(Array<PrimExpr> shape);
bool IsSharedBuffer(const Buffer& buffer);
Stmt Lower(const LowerArgs& args, arith::Analyzer* analyzer) const;
```

Avoid adding new lowerCamelCase helpers or ambiguous context names like
`const LowerArgs& T`.

## API Boundaries And Ownership

Prefer concrete type declarations over `auto` when the type is short or the
value crosses an API boundary. `auto` is fine for obvious pattern checks,
iterators, lambdas, `Downcast<T>()`, and `node.as<T>()` results where spelling
the type would obscure the logic.

Pass TVM object handles and non-trivial inputs by `const&` unless the callee
will store the value. If a constructor or helper stores the value, take it by
value and `std::move` into the field.

Mark query methods and helpers `const` whenever possible. For visitor classes,
keep mutable state explicit in private fields and prefer short static entry
points such as `Collect`, `Rewrite`, or `Analyze` that hide the visitor object.

Use TVM `Array`, `Map`, `Optional`, and ObjectRefs at FFI/API boundaries.
Internal analysis code may use `std::vector`, `std::unordered_map`, and
`std::unordered_set` when that is simpler, but convert back at the boundary.

Use `TVM_DLL` only for functions and constructors that need cross-translation
unit or extension visibility, following nearby exported APIs. Keep file-local
helpers in anonymous namespaces or private methods.

## TVM Object Conventions

Treat `Stmt`, `PrimExpr`, `Buffer`, `Var`, `SBlock`, `Layout`, and related types
as TVM `ObjectRef` handles. Use handles across function boundaries and for
stored state.

Keep raw `*Node` pointers local to visitor callbacks, pattern checks, and
`CopyOnWrite()` mutation sites. Convert callback nodes with `GetRef<T>(op)` if
the value must be passed elsewhere or retained.

Use `.same_as(other)` for identity comparisons between handles. Use TVM
structural equality utilities for structural comparisons. Avoid `.get() ==
other.get()` outside narrow adapter code.

For identity maps/sets keyed by handles, use TVM pointer hash/equality helpers:

```cpp
using BufferSet = std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>;
using VarMap = std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>;
```

Use `Optional<T>` for nullable ObjectRef results. Return an undefined
`Optional<T>` or ObjectRef for "not found" cases instead of retaining raw node
pointers or relying on unrelated sentinel state.

Construct ObjectRefs with `make_object<Node>()`, populate reflected fields, then
assign `data_ = std::move(node)`. For `Clone()`, a shallow
`make_object<Node>(*this)` is fine only when embedded mutable ObjectRefs do not
need independent copies; clone nested mutable operators explicitly when needed.

When mutating existing IR/ObjectRefs, call `CopyOnWrite()` on a handle. Do not
mutate raw callback nodes directly.

## Symbolic Arithmetic And Analyzer Use

Do not force a `PrimExpr` to a constant integer if symbolic arithmetic can carry
the logic. Keeping expressions symbolic preserves dynamic shape support.

When a constant is truly required, extract it as `int64_t` (`as_const_int`,
`IntImmNode::value`, or an equivalent helper), check the failure path, and
rebuild constants with the intended dtype (`Integer`, `IntImm(expr->dtype, v)`,
or `make_const(dtype, v)`). Avoid narrowing to `int` unless the target API
requires it and the range is checked.

Use a populated `arith::Analyzer` for `Simplify`, `CanProve`,
`CanProveEqual`, bounds, and modular reasoning. Bind loop/shape variable ranges
before asking the analyzer to prove facts. If a function accepts an optional
analyzer pointer, create a local fallback only for simple self-contained
reasoning.

Be careful mixing `int32` and `int64` in analyzer-sensitive expressions. In
layout/swizzle code, preserve the native dtype of the expression spine when
casts would prevent simplification.

When constructing constants, prefer dtype-aware helpers:

```cpp
PrimExpr zero = make_const(buffer->dtype, 0);
PrimExpr one_i32 = IntImm(DataType::Int(32), 1);
PrimExpr same_dtype = IntImm(expr->dtype, value);
```

## ObjectNode Fields

For TVM-style `ObjectNode` classes, public reflected fields should use
`lower_snake_case` without a trailing underscore. Use trailing underscores for
private/protected implementation state.

Before renaming reflected fields, check whether the field name is FFI-visible or
used by Python/debugging surfaces. Do not break compatibility without an
explicit migration plan.

For a new FFI-visible Object:

- Node classes use `TVM_FFI_DECLARE_OBJECT_INFO` or
  `TVM_FFI_DECLARE_OBJECT_INFO_FINAL`.
- ObjectRef classes use `TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE` unless the
  handle is intentionally non-nullable.
- `RegisterReflection()` defines public fields with stable string names.
- A `TVM_FFI_STATIC_INIT_BLOCK()` registers reflection and global functions.

Treat registered field names and global names as API. Python-facing global
names usually follow existing strings such as `tl.transform.Simplify`,
`tl.make_linear_layout`, or target-specific `tl.cuda.transform.*`.

## Tile Operators And Backend Implementations

New tile operators should follow the `TileOperatorNode` pattern:

- Parse `ffi::Array<PrimExpr>` arguments and `Map<String, ObjectRef>`
  annotations in the ObjectRef constructor.
- Fill access regions with `SetAccessRegions` so analysis sees reads/writes.
- Implement `Lower`, `InferLayout`, and `Clone`.
- Register the TIR op with `TIR_REGISTER_TL_TILE_OP` or a matching
  `TVM_REGISTER_OP` variant. Set call effects and printer names consistently.

Target-specific lowering belongs in target directories such as `src/cuda/op`,
`src/rocm/op`, `src/cpu/op`, or `src/metal/op`. Use small target predicates and
registration structs (`RegisterCopyImpl`, `RegisterAtomic...Impl`, etc.) instead
of hardcoding target branches in common code. Put shared fallback logic in
`src/backend/common` or the common op file.

Use target helper APIs (`TargetIsCuda`, `TargetHasAsyncCopy`,
`TargetCudaGetWarpSize`, etc.) instead of string-matching targets at call sites.

For annotations, prefer existing `attr::k...` constants when available. When a
string key is required for compatibility, parse it through a small helper and
validate type/range close to the parse site.

## Passes And Visitors

Transform passes usually use local visitor/mutator classes plus a small public
factory:

```cpp
tvm::transform::Pass MyPass() {
  auto pass_func = [](PrimFunc func, const IRModule& mod,
                      PassContext ctx) -> PrimFunc {
    arith::Analyzer analyzer;
    return MyRewriter::Rewrite(std::move(func), &analyzer);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MyPass", {});
}
```

Register pass factories in `TVM_FFI_STATIC_INIT_BLOCK()` under
`tl.transform.*` or target-specific namespaces like `tl.cuda.transform.*`.

Visitor rules:

- Use `StmtExprVisitor`, `StmtExprMutator`, `StmtMutator`, or
  `arith::IRMutatorWithAnalyzer` according to whether the pass needs expression
  traversal, mutation, or analyzer context.
- Provide static entry points such as `Collect`, `Rewrite`, `Substitute`, or
  `Apply`; keep mutable traversal state private.
- In callback overrides, call the base visitor/mutator unless intentionally
  stopping traversal. Convert callback nodes with `GetRef<T>(op)` when passing
  or storing the handle.
- Preserve annotations, spans, and read/write metadata when rebuilding IR
  unless the pass intentionally changes them.

## Headers And Includes

- TileLang currently has no separate installed public C++ header tree. Avoid
  `using namespace` in any future installed public headers and in headers that
  behave like shared cross-module interfaces inside the repository.
- Prefer explicit qualifications in shared headers, or narrow aliases when they
  are part of the API or remove real repetition.
- In `.cc` files and narrowly included internal `src/` helper headers,
  file-local namespace imports are acceptable when they improve readability.
  `tirx` imports are common for IR-heavy code; prefer explicit `ffi::` names in
  core-style code unless FFI helpers are pervasive in the file.
- Include the headers that define the types used by the file.
- Keep implementation-only helpers in `.cc` files or anonymous namespaces.
- Group includes as paired header, standard library, TVM/third-party, then local
  TileLang headers.
- Use header guards shaped like `TVM_TL_<PATH>_<FILE>_H_` where possible, and
  close namespaces and guards with comments.

## Error Handling And Diagnostics

Use `ICHECK`/`ICHECK_EQ` for internal compiler invariants. Use `LOG(FATAL)` for
unsupported or unreachable branches where a streamed message is clearer than a
boolean check. Avoid new bare `ICHECK(0)` without context.

Use `CHECK(..., ErrorKind)` only for user-facing FFI validation paths where a
typed TVM FFI error is intended. Runtime ABI validation should follow existing
generated `AssertStmt` or `runtime/error_helpers` patterns.

Make messages actionable: include the op, target, buffer name, scope, dtype,
shape/range, or annotation key that caused the failure. Prefer messages that
explain the violated constraint over "not supported".

## Comments And Documentation

Use Doxygen comments for public APIs, ObjectRefs/ObjectNodes, pass factories,
and non-obvious fields. For local code, comment invariants and target-specific
constraints, not line-by-line restatements.

Good comments in TileLang often explain why a fallback, annotation, dtype cast,
or analyzer binding is required. Keep comments short unless they describe a
subtle lowering invariant.

## Clang-Format And Macro Boundaries

Use `clang-format off/on` narrowly around dense generated tables, inline asm, or
registration blocks only when the formatter makes the code materially worse.
Keep the disabled region as small as possible.

Write macros so clang-format can still parse surrounding C++: macro functions
should look like calls at use sites, and declaration macros should leave the
declaration syntactically complete, including a semicolon when appropriate.

## Migration Policy

Style convergence should be incremental:

- New C++ code should follow the style guide.
- Touched code should avoid adding new inconsistencies.
- Mechanical renames should be small and isolated.
- Do not mix broad formatting changes with behavior changes.
- Keep `.clang-format` changes in a focused PR.

Validate style/documentation edits with:

```bash
python3 -m pre_commit run --files <changed-file>...
```

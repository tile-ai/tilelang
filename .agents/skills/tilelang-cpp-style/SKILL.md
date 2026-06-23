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

## Naming Checklist

- File names: `lower_snake_case`.
- Types, classes, structs, ObjectRefs, ObjectNodes: `PascalCase`.
- Object nodes: `PascalCaseNode`; ObjectRefs: `PascalCase`.
- Functions and methods: `PascalCase`.
- Boolean helpers: `Is`/`Has`/`Can` + `PascalCase`.
- Parameters and local variables: `lower_snake_case`.
- Private/protected members: `lower_snake_case_`.
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

## ObjectNode Fields

For TVM-style `ObjectNode` classes, public reflected fields should use
`lower_snake_case` without a trailing underscore. Use trailing underscores for
private/protected implementation state.

Before renaming reflected fields, check whether the field name is FFI-visible or
used by Python/debugging surfaces. Do not break compatibility without an
explicit migration plan.

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

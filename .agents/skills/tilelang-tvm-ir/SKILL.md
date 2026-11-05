---
name: tilelang-tvm-ir
description: Use when editing TileLang C++ passes or TVM TIRX code that handles ObjectRef/NodeRef types such as For, tirx::TensorVar, Var, RegionStmt, Stmt, PrimExpr, or their *Node raw node counterparts; especially when choosing function parameters, optional values, identity maps/sets, or equality checks.
---

# TileLang TVM IR Handle Conventions

## Core Rule

In native TIRx C++, `For`, `tirx::TensorVar`, `Var`, `RegionStmt`, `Stmt`, `PrimExpr`, `SeqStmt`, etc. are `ObjectRef` smart handles. `ForNode`, `VarNode`, `RegionStmtNode`, etc. are raw node structs reached through visitor callbacks, `as<TNode>()`, `operator->`, or `.get()`.

When a value needs to cross a function boundary, be stored, be optional, be used as an identity key, or survive beyond a local inspection branch, prefer the handle type over `const *Node`.

`TensorVar` wraps an ordinary `Var` whose type is `TensorType`; its raw node
is `VarNode`. Access tensor metadata through `shape()`, `dtype()`, `scope()`
and `data()`. `Function` replaces `PrimFunc` and has no `buffer_map`; its
defined body is a `SeqStmt`. The public DSL spelling `@T.prim_func` is retained.
Generic expressions use `Expr`, while `PrimExpr` requires a primitive type.

## Preferred Patterns

- Function parameters and return values: use handles such as `For`, `tirx::TensorVar`, `Var`, `RegionStmt`, `Stmt`, or `SeqStmt`.
- Nullable AST values: use `Optional<For>`, `Optional<SeqStmt>`, etc., not `const ForNode* = nullptr`.
- Identity maps and sets: use handle keys with TVM identity hashing:

```cpp
using BufferSet = std::unordered_set<tirx::TensorVar, ObjectPtrHash, ObjectPtrEqual>;
using BufferMap = std::unordered_map<tirx::TensorVar, tirx::TensorVar, ObjectPtrHash, ObjectPtrEqual>;
using VarMap = std::unordered_map<Var, Expr, ObjectPtrHash, ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
```

- Identity comparisons: use `.same_as(other)` when comparing two handles.
- Visitor callback node pointers: convert to a handle with `GetRef<T>(op)` when the value must be retained or passed elsewhere.

```cpp
ffi::UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode mode) final {
  For loop = ffi::GetRef<For>(op);
  Optional<For> candidate = FindPipelineLoop(loop->body);
  if (candidate.has_value() && candidate.value().same_as(loop)) {
    // Inspect or retain the handle here.
  }
  return StmtExprMutator::Mutate_(op, mode);
}
```

- Pattern matching and local mutation may still use node pointers:
  - `if (const auto* seq = stmt.as<SeqStmtNode>()) { ... }`
  - `TensorStoreNode* n = store.CopyOnWrite();`
  - native overrides such as `Mutate_(const SeqStmtNode* op, InplaceMode mode)`

Keep these raw pointers local to the immediate inspection or mutation site.

## TileLang Intrinsic Calls

When adding a new internal TileLang intrinsic, represent it as a registered
TileLang `Op` plus `call_intrin`, not as `call_extern`.

- Register the intrinsic in C++ first, usually with a `TVM_TIRX_DLL const Op&`
  declaration in `src/op/builtin.h` and a matching `TIR_DEFINE_TL_BUILTIN(...)`
  registration in `src/op/builtin.cc`.
- In C++ transforms and lowering passes, build the call with the registered op,
  for example `Call(result_type, tl::my_intrin(), args)`.
- In Python language helpers or tests, use `T.call_intrin(...)` or the
  internal `tilelang.language._native.call_intrin(...)` facade. Native op
  lookup is `tvm_tirx.ir.Op.get("tl.my_intrin")`. The facade constructs a
  native `Call`; it does not preserve old compiler object types.
- Native C++ calls take a result `Type` and `Array<Expr>` operands. Calls
  carrying TileLang annotations use `MakeAnnotatedCall` so dynamic metadata
  remains in the explicit final `Tuple`, not hidden in `Call.attrs`.
- Add backend codegen handling by matching the op identity, for example
  `op->op.same_as(tl::my_intrin())`, and set any required template/header flags
  from that intrinsic branch.

Do not use `tirx::call_extern_op()`, the internal extern facade, or `T.call_extern` to
smuggle internal TileLang runtime-template helpers through IR. Reserve
`call_extern` for real external ABI calls, user-provided/custom externs, or
lowered packed/runtime calls whose contract is genuinely external. Avoid
special-casing `PrintCallExtern` by global symbol name for new TileLang
intrinsics; the include and emission logic should live on the intrinsic's
`call_intrin` codegen path.

## Avoid

- Passing `const ForNode*`, `const VarNode*`, or `const RegionStmtNode*` between helper functions when a handle exists.
- Storing raw node pointers in `std::unordered_map` or `std::unordered_set` for identity tracking.
- Using `.get()` as a key unless a callee requires a raw TVM node API and the pointer is not retained.
- Comparing handles through `.get() == other.get()`; prefer `.same_as()`.
- Reconstructing handles from raw pointers repeatedly when a handle is already available.

## Common Refactors

```cpp
// Before
const SeqStmtNode* pipeline_body_seq = nullptr;
pipeline_body_seq = seq_stmt;
ICHECK(pipeline_body_seq != nullptr);

// After
Optional<SeqStmt> pipeline_body_seq;
pipeline_body_seq = GetRef<SeqStmt>(seq_stmt);
ICHECK(pipeline_body_seq.has_value());
SeqStmt pipeline_body = pipeline_body_seq.value();
```

```cpp
// Before
std::unordered_set<const VarNode*> seen;
seen.insert(buffer.get());
if (const auto* source = read->source.as<VarNode>()) {
  if (seen.count(source)) { ... }
}

// After
BufferSet seen;
seen.insert(buffer);
if (auto source = read->source.as<Var>();
    source && (*source)->ty.as<tirx::TensorType>()) {
  if (seen.count(tirx::TensorVar(*source))) { ... }
}
```

```cpp
// `var` is a Var handle supplied by the traversal.
// Before: retaining raw callback pointers.
std::unordered_set<const VarNode*> vars;
vars.insert(loop->var.get());
bool uses = vars.count(var.get()) > 0;

// After: the same identity query with retained handles.
VarSet vars;
vars.insert(loop->var);
bool uses = vars.count(var) > 0;
```

A `TensorLoad` source is a general `Expr`; check that it is a tensor-typed
`Var` before putting it in a `TensorVar` identity set. When collecting uses,
follow the native structural visitor's definition/use context: a definition
is not a use, while tensor types and explicit layout/annotation operands can
contain uses that must remain in scope.

## Review Checklist

When reviewing TileLang TIR passes, search for:

```bash
rg -n "std::unordered_(set|map)<const .*Node \\*|const (For|SeqStmt|RegionStmt).*Node \\*|\\.get\\(\\) ==|\\.find\\([^\\n]*\\.get\\(\\)|\\.count\\([^\\n]*\\.get\\(\\)|\\.insert\\([^\\n]*\\.get\\(\\)" src/transform
```

Do not mechanically remove every raw node pointer. Keep visitor signatures, `as<TNode>()` pattern checks, and `CopyOnWrite()` mutation pointers. Refactor only the places that store, pass, compare, or key identities through raw pointers.

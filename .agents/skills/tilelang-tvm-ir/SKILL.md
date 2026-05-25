---
name: tilelang-tvm-ir
description: Use when editing TileLang C++ passes or TVM TIRX code that handles ObjectRef/NodeRef types such as For, Buffer, Var, SBlock, Stmt, PrimExpr, or their *Node raw node counterparts; especially when choosing function parameters, optional values, identity maps/sets, or equality checks.
---

# TileLang TVM IR Handle Conventions

## Core Rule

In TVM C++, `For`, `Buffer`, `Var`, `SBlock`, `Stmt`, `PrimExpr`, `SeqStmt`, etc. are `ObjectRef` smart handles. `ForNode`, `BufferNode`, `VarNode`, `SBlockNode`, etc. are raw node structs reached through visitor callbacks, `as<TNode>()`, `operator->`, or `.get()`.

When a value needs to cross a function boundary, be stored, be optional, be used as an identity key, or survive beyond a local inspection branch, prefer the handle type over `const *Node`.

## Preferred Patterns

- Function parameters and return values: use handles such as `For`, `Buffer`, `Var`, `SBlock`, `Stmt`, or `SeqStmt`.
- Nullable AST values: use `Optional<For>`, `Optional<SeqStmt>`, etc., not `const ForNode* = nullptr`.
- Identity maps and sets: use handle keys with TVM identity hashing:

```cpp
using BufferSet = std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>;
using BufferMap = std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>;
using VarMap = std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>;
```

- Identity comparisons: use `.same_as(other)` when comparing two handles.
- Visitor callback node pointers: convert to a handle with `GetRef<T>(op)` when the value must be retained or passed elsewhere.

```cpp
Stmt VisitStmt_(const ForNode* op) final {
  For loop = GetRef<For>(op);
  Optional<For> candidate = FindPipelineLoop(loop->body);
  if (candidate.defined() && candidate.value().same_as(loop)) {
    ...
  }
}
```

- Pattern matching and local mutation may still use node pointers:
  - `if (const auto* seq = stmt.as<SeqStmtNode>()) { ... }`
  - `BufferStoreNode* n = store.CopyOnWrite();`
  - visitor overrides such as `VisitStmt_(const SeqStmtNode* op)`

Keep these raw pointers local to the immediate inspection or mutation site.

## Avoid

- Passing `const ForNode*`, `const BufferNode*`, `const VarNode*`, or `const SBlockNode*` between helper functions when a handle exists.
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
ICHECK(pipeline_body_seq.defined());
SeqStmt pipeline_body = pipeline_body_seq.value();
```

```cpp
// Before
std::unordered_set<const BufferNode*> seen;
seen.insert(buffer.get());
if (seen.count(read->buffer.get())) { ... }

// After
BufferSet seen;
seen.insert(buffer);
if (seen.count(read->buffer)) { ... }
```

```cpp
// Before
std::unordered_set<const VarNode*> vars;
vars.insert(loop->loop_var.get());
bool uses = UsesVar(expr, [&](const VarNode* vn) {
  return vars.count(vn) > 0;
});

// After
VarSet vars;
vars.insert(loop->loop_var);
bool uses = UsesVar(expr, [&](const VarNode* vn) {
  return vars.count(GetRef<Var>(vn)) > 0;
});
```

## Review Checklist

When reviewing TileLang TIR passes, search for:

```bash
rg -n "std::unordered_(set|map)<const .*Node \\*|const (For|SeqStmt|SBlock).*Node \\*|\\.get\\(\\) ==|\\.find\\([^\\n]*\\.get\\(\\)|\\.count\\([^\\n]*\\.get\\(\\)|\\.insert\\([^\\n]*\\.get\\(\\)" src/transform
```

Do not mechanically remove every raw node pointer. Keep visitor signatures, `as<TNode>()` pattern checks, and `CopyOnWrite()` mutation pointers. Refactor only the places that store, pass, compare, or key identities through raw pointers.

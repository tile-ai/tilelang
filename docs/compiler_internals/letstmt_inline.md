# Binding Inlining in TileLang

The public `LetInline` pass name is retained, while its statement input is now
native TIRx `Bind`. A `Bind` defines a variable for subsequent statements in a
`SeqStmt`; it does not own a nested `body` like the former `LetStmt`.
The expression form `prim::Let` still has its own expression body.

## Ordinary Simplification

`src/transform/simplify.cc` implements TileLang's statement policy around the
native symbolic Analyzer. `NativeSimplifier::Mutate_(const BindNode*, ...)`
first simplifies the value. A primitive expression with at most `kPure`
effects is recorded for condition reasoning and, when
`enable_simplify_let_inline` is true, bound in the owning Analyzer. Later
numeric uses may simplify through that binding. String literals use the
native variable-remapping mechanism.

The binding is not dropped merely because one numeric use was simplified.
`CollectBindingUsage` follows the structural definition/use protocol, including
variable types and explicit annotation operands. `RemoveUnusedBindings` then
removes unused primitive bindings to a fixed point. A remaining use in a tensor
type or layout therefore keeps its definition alive. `TensorType` metadata is
not numerically rewritten by `NativeSimplifier`; changing that contract needs
a typed replacement of the variable and its uses.

This replaces the former Buffer-field collection rule. Native tensor
definitions and views use ordinary variables with `TensorType`, and their
shape/layout expressions must remain in scope after every transformation.

## Memory Reads and Effects

Consider a temporary that captures a load:

```python
tmp = A[i]
A[i] = 0
B[i] = tmp
```

Duplicating `A[i]` at the last statement would change the result. A tensor
load has a read-state effect and is not treated as a pure value for eager
substitution. Allocations, tensor declarations, view identities and effectful
calls also retain their defining statement. Ordinary simplification can remove
an unused nonvolatile read, but keeps potentially volatile reads, including
through aliases. Its optional known-store forwarding is a separate analysis
with scope and effect checks; it does not make arbitrary loads pure.

## Explicit `LetInline`

`src/transform/frontend_legalize.cc` implements `LetInliner`. For a native
`Bind`, it substitutes only a primitive, pure value using `VarRemapSet` and
replaces the defining statement with an empty `SeqStmt`. For `prim::Let`, the
same purity condition permits substitution into the expression body. The
native mutator handles remapping and lexical scope. This pass does not force
inlining of reads, allocations or effectful values.

The `tl.force_let_inline` option asks the selected backend pipeline to run this
eager pass. The helper `should_force_let_inline` lives in
`tilelang/backend/pass_pipeline/pipeline_utils.py`; each backend owns its
actual pass order in `tilelang/<backend>/pipeline.py`.

```python
import tilelang

kernel = tilelang.compile(
    program,
    target="cuda",
    pass_configs={tilelang.PassConfigKey.TL_FORCE_LET_INLINE: True},
)
```

The option defaults to false. Ordinary Simplify can still propagate pure
bindings under its own `tl.Simplify` settings. Enabling the eager pass changes
when substitution occurs, while preserving evaluation order and typed uses.

## Relevant Sources and Regression Coverage

- `src/transform/simplify.cc`: pure binding propagation, structural-use
  collection, unused binding removal and volatile-memory handling.
- `src/transform/frontend_legalize.cc`: native `Bind` and `prim::Let` inlining.
- `tilelang/backend/pass_pipeline/pipeline_utils.py`: explicit eager-pass option.
- `testing/python/transform/test_tilelang_transform_let_inline.py`: public pass
  behavior and definition/use invariants.

When changing these passes, verify a variable used in a tensor type, nested
scope remapping, a load followed by an aliasing write, volatile reads, and
side-effecting calls. Comparing printed IR alone does not establish equivalent
evaluation order.

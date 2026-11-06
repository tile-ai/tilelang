# Layout declarations in native TIRx

`T.annotate_layout({tensor: layout})` preserves the TileLang language interface.
It emits an opaque, void `tl.layout_map` call with no operands or type arguments:

```python
ir.Call("tl.layout_map", [], attrs={"layout_map": {tensor: layout}})
```

The map contains actual TensorVar keys and TileLang Layout objects, including
Fragment, PartialFragment and Ascend layout subtypes. Source declarations also
carry `tl.source_block_annotation=True`. They describe their entire nearest
source block; their physical position remains after any definitions they use.
Mixed source metadata emits adjacent declarations, with the source marker on
both. LayoutInference and LowerTileOp consume these declarations; an unconsumed
declaration is an error at VerifyLoweredAnnotations.

## Transforming declarations

Native TIRx Call traversal intentionally leaves attrs to the owning dialect.
FFI Map traversal also visits values without rewriting keys. A pass that changes
variables while declarations are present must therefore use TileLang's
`LayoutMapMutator` (or its constraint-aware analyzer base) from
`src/ir/stmt_functor.h`. The Call hook visits each tensor key and Layout value
using the same native remap state that rewrites the surrounding function. It
rebuilds the Call only when an entry changes. Two keys mapping to the same tensor
are rejected by this low-level helper; an alias-owning pass must explicitly
check compatibility and combine those entries itself.

`tl::RenewDef` adds the same Call hook to structural definition renewal. The
upstream `tirx::RenewDef` does not know this TileLang declaration contract.
Whole-statement scalar substitution uses TileLang's
`SubstituteWithDataTypeLegalization`; expression-only substitutions can use the
native implementation. The registry entries `tl.RenewDef`,
`tl.SubstituteLayoutMap` and `tl.CheckLayoutMapScopes` expose these internal
operations for IR tooling and tests.

`CheckLayoutMapScopes` validates tensor keys and free layout captures at the
original lexical declaration position. It creates a temporary view containing
ordinary expression uses and calls the native scope verifier. The view never
enters a compiler pipeline or cache, and the source function is unchanged.
Layout coordinates are local definitions and are excluded from its free uses.

## Layout persistence

Each Layout owns explicit coordinate definitions, including replication where
applicable. Its stored index/thread expressions refer to those definitions.
Public forward-index accessors project to the canonical coordinates used by
existing layout algorithms. JSON therefore preserves the actual binding graph;
loading a layout does not rely on globally interned placeholder identities or
variable names. Import the owning Python layout module before loading JSON
when its Python convenience methods are needed. Existing compiled caches are
invalidated through compiler identity; this does not promise compatibility with
standalone Layout JSON written by a different compiler revision. Printing
reconstructs the subtype through `from_mapping` with an IR lambda that binds its
coordinates and preserves external captures.

Other metadata, including metadata that contains a Layout for another purpose,
can still use the existing annotation payload codec. The layout_map declaration
path does not encode or decode that payload.

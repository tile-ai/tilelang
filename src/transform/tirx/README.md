# TileLang-owned TIRx passes

These implementations operate directly on standalone TIRx IR. They are ported
from the selected upstream baseline
`9af5544df8553c35b6a85927cd2589f0a596f2fc` and the previously paired integration
changes, with TileLang-specific behavior maintained here rather than in TIRx.
They are not the old TVM/TIR implementation or a second compiler/runtime.

The policy is to reuse upstream passes when their semantics fit, compose them
when preparation or finalization suffices, and implement missing transformations
inside TileLang. In particular:

- `pipeline_adapters.cc` composes upstream target binding and entry annotation.
- Disabled vectorization uses the upstream pass. Enabled vectorization owns its
  dialect callback, under `tl.FVectorizeCall`, including effectful-call handling.
- Storage planning owns reuse/in-place/shared-allocation policies. Its final
  pointer-type rewrite reuses the existing upstream implementation.
- Device outlining and launch lowering are separate TileLang transformations;
  the original upstream combined pass and its registration are untouched.
- Tensor/layout lowering keeps TileLang's retained region metadata consistent.
- Simplification, no-op removal, and conditional hoisting preserve non-unit and
  descending-loop behavior. Hoisting uses the matching downstream cleanup passes.
- `LoopAwareMutator` supplies downstream loop constraints and annotation
  traversal. TIRx's original `For` mutator remains compatible with its own passes.
- Intrinsic lowering uses this loop-aware traversal. Host-builtin lowering keeps
  small CPU-local allocations on the stack without changing their IR scope.
- Packed-wrapper construction lives here, but packed-argument decoding still
  calls the paired dependency's decoder. Moving that separate ABI implementation
  is not part of the pass-ownership change.

Pass-internal implementation classes have local linkage; `LoopAwareMutator` is
shared by TileLang transforms. Pass entry points register as
`tl._tirx_passes.*`; they never override `tirx.transform.*`. Native pass signatures
and vectorization attributes are restored to the upstream baseline. The two
existing private helpers used here are declared in `src/tirx_compat/transform.h`;
no private TIRx implementation file is included or compiled into TileLang.

`testing/python/tools/test_tirx_pass_boundary.py` audits upstream source hashes
and registration ownership. The relocated regression cases live under
`testing/python/tirx/passes/`, alongside the existing TileLang pass suites.

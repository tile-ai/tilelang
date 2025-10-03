# Ticket: Default TT Annotation Helper

## Goal
Provide a Python helper that stamps default Tenstorrent schedule/sharding metadata (contiguous schedule + interleaved TensorAccessor layout) when user code omits TT annotations.

## Context
- Workstream 1 specifies introducing `python/tilelang_tt/target.py` to centralize default policy synthesis.
- Helper should be reusable by future workstreams when additional defaults are needed.

## Key Tasks
- Create `python/tilelang_tt/target.py` exporting a function (e.g., `apply_tt_defaults(mod)`) that injects attrs on each PrimFunc.
- Hook into the lowering pipeline before TT-specific transforms run.
- Document the default choices (contiguous schedule, interleaved DRAM tensors) within the helper for clarity.

## Dependencies
- Requires the engine adapter ticket so the TT path has a hook to call the helper.
- Metadata produced here must align with Workstream 2 inference passes to avoid duplication.

## Validation
- Verified indirectly by `tests/python/tt/test_target_registration.py`, asserting the lowered IR contains TT default attrs.

Status: TODO

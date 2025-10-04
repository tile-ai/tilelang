# Ticket: Add Tenstorrent Engine Adapter

## Goal
Teach the TileLang lowering entry point to delegate device codegen to a dedicated Tenstorrent helper when the `tenstorrent` target is active.

## Context
- Workstream 1 requires a TT-specific `lower` helper in `tilelang.engine` to split host/device responsibilities.
- Must preserve existing CUDA/HIP/CPU behavior.

## Key Tasks
- Introduce `tilelang/engine/tt/lower.py` (or similar) to encapsulate TT-specific lowering orchestration.
- Update `tilelang/engine/__init__.py` and `tilelang/engine/lower.py` to branch on `target.kind.name == "tenstorrent"` and invoke the TT helper.
- Ensure TT path integrates with `CompileArtifact` data structures without affecting current consumers.

## Dependencies
- Depends on `ws1_target_registration.md` so the target can be selected.
- Precedes annotation helper wiring that will rely on the new entry point.

## Validation
- Smoke test through `tests/python/tt/test_target_registration.py` to confirm the TT branch executes without raising.

Status: In Review (changes pending on branch `ws1-engine-adapter`)

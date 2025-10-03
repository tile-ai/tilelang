# Ticket: Wire TT Defaults into Lowering Entry Point

## Goal
Invoke the Tenstorrent default annotation helper during lowering so TT metadata is always present before TT-specific passes execute.

## Context
- Complements the default helper by ensuring it runs automatically when the TT target is selected.
- Should be inserted immediately after target resolution but before pass pipelines.

## Key Tasks
- Modify `tilelang/engine/lower.lower` (or the TT helper introduced in `ws1_engine_adapter.md`) to call the default annotation routine on the IRModule.
- Guarantee idempotency so repeated lowering passes do not duplicate attrs.
- Add logging or debug hooks (if appropriate) to confirm defaults were applied.

## Dependencies
- Depends on `ws1_engine_adapter.md` and `ws1_default_annotation_helper.md`.

## Validation
- Covered by the Workstream 1 test (`tests/python/tt/test_target_registration.py`) inspecting the transformed IR.

Status: TODO

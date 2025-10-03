# Ticket: Register `tenstorrent` Target

## Goal
Enable explicit opt-in for the Tenstorrent backend by adding `"tenstorrent"` to TileLang's target resolution so downstream lowering can select the TT pass pipeline.

## Context
- Driven by Workstream 1 of `project_1.md`.
- Must avoid impacting existing auto-detection logic for CUDA/HIP backends.

## Key Tasks
- Extend `tilelang/utils/target.py` with a constant entry and validation branch for `"tenstorrent"`.
- Ensure `determine_target(..., return_object=True)` returns a `Target` whose `kind.name` is `"tenstorrent"` when explicitly requested.
- Skip auto-selection when the user passes `tenstorrent`; require explicit opt-in when unavailable.

## Dependencies
- None; pure frontend change but should land before other TT-specific wiring.

## Validation
- Covered by `tests/python/tt/test_target_registration.py` (see Workstream 1 testing ticket).

Status: TODO

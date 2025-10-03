# Ticket: Register `tenstorrent` Target

## Goal
Enable explicit opt-in for the Tenstorrent backend by adding `"tenstorrent"` to TileLang's target resolution so downstream lowering can select the TT pass pipeline.

## Context
- Driven by Workstream 1 of `project_1.md`.
- Must avoid impacting existing auto-detection logic for CUDA/HIP backends.

## Key Tasks
- Update `tilelang/utils/target.py`:
  - Append `"tenstorrent"` to `AVALIABLE_TARGETS` and document that auto detection remains CUDA/HIP only.
  - In `determine_target`, add an explicit branch handling the string/Target case where `target == "tenstorrent"` and return `Target("tenstorrent")` when `return_object=True`.
  - Guard the `auto` path from ever choosing TT by ensuring the CUDA/HIP checks remain first and that TT raises if requested but not compiled with TT support.
- Define an informative error/warning path (e.g., `raise ValueError("Tenstorrent backend requires TL_TT_BACKEND build flag")`) for configurations built without TT support; place the check adjacent to the new branch so failure is immediate.
- Ensure the returned `Target` exposes `kind.name == "tenstorrent"` so later code can branch on it.
- Add inline comments noting that TT auto-detection is intentionally disabled until the backend can probe hardware.

## Dependencies
- None; pure frontend change but should land before other TT-specific wiring.

## Validation
- Covered by `tests/python/tt/test_target_registration.py` (see Workstream 1 testing ticket).

Status: TODO

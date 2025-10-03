# Ticket: Target Registration Test Coverage

## Goal
Add Python regression coverage ensuring the Tenstorrent target can be selected and that default metadata injection occurs during lowering.

## Context
- Testing requirement from Workstream 1 to guard new frontend wiring.
- Relies on pytest infrastructure in `tests/python`.

## Key Tasks
- Create `tests/python/tt/test_target_registration.py` that:
  - Requests `target="tenstorrent"` through TileLang lowering APIs.
  - Verifies the resulting IRModule carries `tt.schedule` and `tt.shard` attrs seeded by the default helper.
  - Ensures no CUDA/HIP-specific passes execute when the TT path is chosen.
- Add fixtures/utilities if needed to inspect PrimFunc attrs.

## Dependencies
- Depends on prior tickets that add the target registration and default helper wiring.

## Validation
- Test passes under `pytest tests/python/tt/test_target_registration.py` and is added to CI once Workstream 1 lands.

Status: TODO

# Host-Side Check Repro Scripts

This folder contains standalone scripts that deliberately trigger host-side (and adapter-side) validation errors described in `docs/compiler_internals/tensor_checks.md`. Each script can be run directly and will reproduce the corresponding error with a minimal example.

Prerequisites
- CUDA-capable environment (most scripts compile a CUDA-targeted kernel)
- Python packages: torch, tilelang

Usage
- Run any script, e.g.:
  - `python 01_num_args_mismatch.py`
  - `python 02_pointer_type_error.py`
  - ... up to `10_scalar_type_mismatch.py`

- Or run all at once with a summary:
  - `python run_all.py`
  - Logs per test are saved under `logs/` as `<script>.out` / `<script>.err`.

Notes
- Scripts assume at least one CUDA device. For the device-id mismatch case (08), two GPUs are required; the script will skip with a note if only one is available.
- The adapter raises some errors before the host stub (e.g., wrong input count). The messages are aligned with the host checks as far as possible.

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

The runner uses the current Python executable and a fresh subprocess for every script. A `PASS` requires the actual uncaught exception to match that script's declared argument error: `ValueError` for the adapter input count in 01, and `RuntimeError` for the other argument checks. Import errors, loader failures, compilation failures, normal returns, and missing scripts fail the run. Any `FAIL` or `MISSING` makes the runner exit with status 1.

Only 08 may report `SKIP`: it must return normally with exit status 0 and print its exact two-device requirement once as the final stdout line. A skip marker from another script, a partial message, or a failed child does not qualify. In addition to the unchanged `.out` and `.err` logs, `<script>.result.json` records the child's terminal exception and the classification. Printed host source or traceback text does not supply exception evidence.

Script 09 passes `None` for a used tensor; it does not construct a DLTensor whose data pointer is NULL. In 10, the first integer mismatch raises before the later boolean call, so this script alone does not cover the boolean mismatch.

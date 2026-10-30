# PipeGuard quick start

These four TileLang kernels illustrate shared-memory ordering and TMA barrier
accounting. They are lowered and checked statically; no GPU kernel is launched.

From the TileLang checkout, with TileLang installed and Python 3.10+:

```bash
git submodule update --init 3rdparty/pipeguard
python -m pip install ./3rdparty/pipeguard
python examples/pipeguard/check_kernels.py
```

| Case | Kernel behavior | Expected verdict |
|---|---|---|
| `shared_safe` | Thread 0 writes shared memory; a CTA barrier precedes thread 64's read | `SAFE` |
| `shared_race` | The same cross-warp handoff without the CTA barrier | `DATA_HAZARD` |
| `tma_safe` | Load a 32×32 tile using TMA; all 128 threads arrive and wait before consumption | `SAFE` |
| `tma_missing_arrival` | Issue the same TMA and wait, but omit the required thread arrivals | `SYNC_HAZARD` |

Run one case and inspect the complete JSON report:

```bash
python examples/pipeguard/check_kernels.py --case shared_race --json
```

The script exits with zero when all selected cases match their expected verdicts,
including the intentional hazard cases. A mismatch, unsupported input, or timeout
returns a nonzero status. Capture and checking have a 120-second limit per case,
after lowering and serialization.

All kernels use one CTA of 128 threads and target SM90a. The script disables
TileLang's automatic thread-storage synchronization so the compiler does not
repair the intentionally missing CTA barrier. Explicit synchronization remains
in the safe cases. The TMA examples recover the 2D descriptor from lowered host
IR; there is no hand-written PGIR or guessed descriptor.

The TMA negative case demonstrates that completed data transfers alone cannot
satisfy a barrier's pending thread arrivals. Its wait cannot complete under the
specified protocol. Do not launch the intentionally faulty kernels on a GPU.

See the [integration guide](../../docs/tutorials/pipeguard.md) for checking your
own kernels, required launch evidence, and the scope of a SAFE result.

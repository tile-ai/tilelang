# PipeGuard quick start

These six TileLang kernels illustrate shared-memory ordering, TMA barrier
accounting, and mbarrier phase reuse. They are lowered and checked statically; no GPU kernel is launched.

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
| `mbarrier_reuse_safe` | Observe phase 0, then use a CTA barrier before phase-1 arrivals | `SAFE` |
| `mbarrier_reuse_race` | One warp starts phase 1 without ordering its arrival after another warp's phase-0 observation | `SYNC_HAZARD` |

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

## A scheduling-dependent mbarrier error

```bash
python examples/pipeguard/check_kernels.py --case mbarrier_reuse_race --json
python examples/pipeguard/check_kernels.py --case mbarrier_reuse_safe
```

The barrier expects two arrivals per phase. Threads 0 and 64 first arrive in
phase 0; a CTA barrier ensures both have arrived. Phase 0 is complete, but nobody
has yet observed that completion with an mbarrier wait.

In the racy version, thread 0 next arrives in phase 1 while thread 64 can observe
phase 0. No synchronization orders those two operations:

- If thread 64 successfully observes phase 0 first, the phase-1 arrival is valid.
- If thread 0 arrives first, it reuses an unobserved phase and violates the protocol.

The subsequent CTA barrier keeps phase 1 incomplete until both phase-0 waits
return; it cannot retroactively order the earlier arrival. The safe version puts
the phase-0 observations and CTA handoff before the next arrival.

PTX requires at least one successful observation of each phase before an arrival
in the next phase; it does not require every thread to observe it. See
[mbarrier phases](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#mbarrier).

The checker reports the possible phase-reuse violation even though some
execution orders satisfy the protocol. Adding the explicit observation handoff
makes the safe variant pass.

"""Runnable PipeGuard examples: lower and check kernels, without launching them."""

import argparse
import json

import tilelang.language as T
from tilelang.contrib.pipeguard import check


def shared_handoff(synchronized):
    """Thread 0 publishes one shared value to thread 64 in another warp."""

    @T.prim_func
    def kernel(Out: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((1,), "float32")
            if tx == 0:
                shared[0] = 1.0
            if synchronized:
                T.sync_threads()
            if tx == 64:
                Out[0] = shared[0]

    return kernel


def tma_handoff(arrive):
    """A 2D TMA tile needs both transaction completion and 128 arrivals."""

    @T.prim_func
    def kernel(A: T.Tensor((64, 64), "float16"), Out: T.Tensor((1,), "float16")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((32, 32), "float16")
            barrier = T.alloc_barrier([128])
            T.tma_copy(A[0:32, 0:32], shared, barrier=barrier)
            if arrive:
                T.mbarrier_arrive(barrier)
            T.mbarrier_wait_parity(barrier, 0)
            if T.get_thread_binding() == 0:
                Out[0] = shared[0, 0]

    return kernel


def mbarrier_handoff(ordered, producer=0):
    """Phase reuse must follow a successful observation by at least one thread."""
    observer = 64 if producer == 0 else 0

    @T.prim_func
    def kernel(Out: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            barrier = T.alloc_barrier([2])
            if tx == 0 or tx == 64:
                T.mbarrier_arrive(barrier)
            # Phase 0 is complete, but nobody has observed it yet.
            T.sync_threads()
            if ordered:
                if tx == 0 or tx == 64:
                    T.mbarrier_wait_parity(barrier, 0)
                T.sync_threads()
                if tx == producer:
                    T.mbarrier_arrive(barrier)
            else:
                if tx == producer:
                    # Valid only if the other warp has already observed phase 0.
                    T.mbarrier_arrive(barrier)
                if tx == 0 or tx == 64:
                    T.mbarrier_wait_parity(barrier, 0)
                # Keep phase 1 incomplete until both phase-0 waits have returned.
                # This later join cannot order the producer's earlier arrival.
                T.sync_threads()
            if tx == observer:
                T.mbarrier_arrive(barrier)
            if tx == 0 or tx == 64:
                T.mbarrier_wait_parity(barrier, 1)
            if tx == producer:
                Out[0] = 1

    return kernel


CASES = {
    "shared_safe": (shared_handoff, True, "SAFE"),
    "shared_race": (shared_handoff, False, "DATA_HAZARD"),
    "tma_safe": (tma_handoff, True, "SAFE"),
    "tma_missing_arrival": (tma_handoff, False, "SYNC_HAZARD"),
    "mbarrier_reuse_safe": (mbarrier_handoff, True, "SAFE"),
    "mbarrier_reuse_race": (mbarrier_handoff, False, "SYNC_HAZARD"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *CASES], default="all")
    parser.add_argument("--json", action="store_true", help="print complete checker reports")
    args = parser.parse_args()
    failed = False
    names = CASES if args.case == "all" else [args.case]
    for name in names:
        factory, enabled, expected = CASES[name]
        report = check(
            factory(enabled),
            target={"kind": "cuda", "arch": "sm_90a"},
            launch_grid=(1,),
            # Keep the intentional missing-barrier bug visible in lowered IR.
            pass_configs={"tl.disable_thread_storage_sync": True},
            timeout=120,
        )
        verdict = report["verdict"]
        print(f"{name}: {verdict} (expected {expected})")
        if args.json:
            print(json.dumps(report, indent=2))
        elif verdict != expected:
            print(json.dumps(report.get("diagnostics", []), indent=2))
        failed |= verdict != expected
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

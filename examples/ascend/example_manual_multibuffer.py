"""Manual + auto multi-buffer coexisting on Ascend NPU.

A small kernel that shows why the *manual* multi-buffer mechanism exists and how
it coexists with the *auto* one:

  * **Auto** multi-buffer (outer ``T.Pipelined`` row loop): AutoSchedule versions
    the per-row input buffer ``in_ub`` so the MTE2 load of row ``r+1`` overlaps
    the compute of row ``r``. Nothing special is written by the user — the pass
    owns the version count and the sync.

  * **Manual** multi-buffer (inner ``T.serial`` step loop): a ping-pong state
    buffer the user hand-versions as ``state_ub[s % 2]`` (read) /
    ``state_ub[1 - s % 2]`` (write). This is NOT expressible by the auto path
    for two reasons that only appear together here:
      1. Within a single iteration the code reads one slot and writes the OTHER
         slot (auto versioning uses one slot per iteration).
      2. Step ``s + 1`` reads what step ``s`` wrote (loop-carried RAW), while at
         the same time MTE3 drains the freshly-written slot to GM. That
         concurrent drain is the cross-iteration WAR that forces a real 2-slot
         ring: step ``s + 2`` (which reuses the slot) must wait for the drain.
    ``T.annotate_manual_multi_buffer`` tells AutoSchedule to analyze these
    hand-written indices and emit the ``set_flag`` / ``wait_flag`` ring itself.

Semantics: ``state`` starts at ``inp`` and adds one ``delta`` vector per step, so
``out[r, s] = inp[r] + delta[r, 0 .. s].sum(0)`` is stored after each step.
"""

import tilelang
import tilelang.ascend.language as T
import torch
from tilelang.profiler import do_bench

NUM_CORES = 64


def manual_multibuffer(width, num_steps, dtype="float32"):
    num_rows = T.dynamic("m")

    NUM_STAGES = 4

    @T.prim_func
    def main(
        inp: T.Tensor((num_rows, width), dtype),
        delta: T.Tensor((num_rows, num_steps, width), dtype),
        out: T.Tensor((num_rows, num_steps, width), dtype),
    ):
        with T.Kernel(NUM_CORES) as core_id:
            delta_ub = T.alloc_shared((width,), dtype)  # per-step delta (inner)
            state_ub = T.alloc_shared((NUM_STAGES, width), dtype)  # manual ping-pong
            T.annotate_manual_multi_buffer(state_ub)

            T.assume(num_rows % NUM_CORES == 0)
            for r in T.serial(num_rows // NUM_CORES):
                row = r * NUM_CORES + core_id
                T.copy(inp[row, :], state_ub[0, :])  # MTE2 -> state_ub

                for s in T.Pipelined(num_steps, num_stages=NUM_STAGES):
                    read_slot = s % NUM_STAGES
                    write_slot = (s + 1) % NUM_STAGES
                    T.copy(delta[row, s, :], delta_ub)  # MTE2 -> delta_ub
                    with T.SimdVF():
                        for k in T.Parallel(width):
                            # read read_slot, write write_slot (different slots)
                            state_ub[write_slot, k] = state_ub[read_slot, k] + delta_ub[k]
                    T.copy(state_ub[write_slot, :], out[row, s, :])  # MTE3

    return main


def ref_program(inp, delta):
    # out[r, s] = inp[r] + delta[r, 0..s].sum(0)
    cum = torch.cumsum(delta.float(), dim=1)  # (num_rows, num_steps, width)
    return (inp.float().unsqueeze(1) + cum).to(inp.dtype)


def run_manual_multibuffer(width=4096, num_steps=8, num_rows=8192, *, verify=True, bench=False, print_source=False):
    device = torch.device("npu")
    program = manual_multibuffer(width, num_steps)
    kernel = tilelang.compile(program, target="ascend")

    if print_source:
        print(kernel.get_kernel_source())

    inp = torch.randn(num_rows, width, dtype=torch.float32, device=device)
    delta = torch.randn(num_rows, num_steps, width, dtype=torch.float32, device=device)
    out = torch.empty(num_rows, num_steps, width, dtype=torch.float32, device=device)
    kernel(inp, delta, out)
    torch.npu.synchronize()

    if verify:
        expected = ref_program(inp, delta)
        torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)

    if bench:
        latency_ms = do_bench(lambda: kernel(inp, delta, out), backend="msprof", _n_warmup=30, _n_repeat=50)
        bytes_moved = (num_rows * width + num_rows * num_steps * width + num_rows * num_steps * width) * 4
        bw_gbs = bytes_moved / (latency_ms * 1e-3) / 1e9
        print(f"    [width={width}, steps={num_steps}, rows={num_rows}] {latency_ms * 1e3:.1f} us  {bw_gbs:.0f} GB/s")


if __name__ == "__main__":
    run_manual_multibuffer(verify=True, bench=True, print_source=True)
    print("PASS")

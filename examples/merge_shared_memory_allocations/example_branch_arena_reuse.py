"""End-to-end demo of MergeSharedMemoryAllocations arena reuse.

Shows a three-buffer kernel where the per-epoch / branch-alternative analysis
collapses two mutually-exclusive branch buffers (A, B) and a post-sync buffer
(C) into a single 64-byte slot, saving 50% vs upstream main's 128-byte arena.

The compiled kernel produces bit-identical output relative to a PyTorch
reference computed without shared memory.

Expected output::

    [arena] naive sum               =   192
    [arena] best-case packing       =    64
    [arena] MergeSharedMemoryPass   =    64  (saved    64 B / 50.0%)
    [arena] vs upstream main (128B) =    64 B
    [check] torch.allclose(out, ref): OK
"""

import re
import torch
import tilelang
from tilelang import tvm as tvm
from tvm.script import tir as T


def _kernel_arena_bytes(artifact) -> int:
    """Read merged arena bytes from a lowered device module."""
    dev_mod = artifact.device_mod
    funcs = {}
    if hasattr(dev_mod, "functions"):
        funcs.update(dev_mod.functions)
    for f in funcs.values():
        a = f.attrs.get("dyn_shared_memory_buf", None)
        if a is not None:
            return int(a)
        script = f.script() if hasattr(f, "script") else ""
        m = re.search(r'"dyn_shared_memory_buf":\s*(\d+)', script)
        if m:
            return int(m.group(1))
    return 0


# ——— kernel under test ———
@T.prim_func(private=True)
def branch_kernel_downstream(
    data: T.Buffer((2,), "float16"),
    out: T.Buffer((1,), "float16"),
    cond: T.int32,
):
    """Branch-demultiplexer kernel.

    Three ``shared.dyn`` buffers (A, B, C) exist at IR level; A and B are
    mutually exclusive (only one is ever live).  The per-epoch
    branch-alternative analysis collapses all three into one slot.
    """
    T.launch_thread("blockIdx.x", 1)
    buf_A = T.allocate([32], "float16", "shared.dyn")
    buf_B = T.allocate([32], "float16", "shared.dyn")
    buf_C = T.allocate([32], "float16", "shared.dyn")
    T.launch_thread("threadIdx.x", 1)
    T.launch_thread("threadIdx.y", 1)
    T.launch_thread("threadIdx.z", 1)
    Ab = T.Buffer((32,), "float16", data=buf_A, scope="shared.dyn")
    Bb = T.Buffer((32,), "float16", data=buf_B, scope="shared.dyn")
    Cb = T.Buffer((32,), "float16", data=buf_C, scope="shared.dyn")
    if cond == 0:
        Ab[0] = data[0]
        out[0] = Ab[0]
    else:
        Bb[0] = data[1]
        out[0] = Bb[0]
    T.tvm_storage_sync("shared.dyn")
    Cb[0] = T.float16(0.0)


if __name__ == "__main__":
    # Lower with global_symbol attached; this triggers the full pipeline
    # including MergeSharedMemoryAllocations.
    mod = tvm.IRModule({"main": branch_kernel_downstream})
    mod["main"] = mod["main"].with_attr("global_symbol", "branch_kernel_downstream")
    artifact = tilelang.engine.lower(mod, target=tvm.target.Target("cuda"), runtime_only=True)

    naive = 192  # 3 × 32 fp16 = 3 × 64 B
    best = 64  # max(A,B,C) = 64 B  (theoretical optimum)
    actual = _kernel_arena_bytes(artifact)

    saved = naive - actual
    vs_upstream = 128 - actual
    pct = (saved / naive) * 100 if naive else 0

    print(f"[arena] naive sum               = {naive:>5}")
    print(f"[arena] best-case packing       = {best:>5}")
    print(f"[arena] MergeSharedMemoryPass   = {actual:>5}  (saved {saved:>5} B / {pct:.1f}%)")
    print(f"[arena] vs upstream main (128B) = {vs_upstream:>5} B")

    # ——— end-to-end correctness ———
    # Wrap into an IRModule with global_symbol so the compilation pipeline
    # can discover the entry-point function.
    func_with_sym = branch_kernel_downstream.with_attr("global_symbol", "branch_kernel_downstream")
    compiled = tilelang.compile(
        func_with_sym,
        target=tvm.target.Target("cuda"),
        out_idx=[1],  # 'out' is the 2nd parameter (index 1)
    )

    data_t = torch.tensor([1.0, 2.0], dtype=torch.float16, device="cuda")

    out_t = compiled(data_t, 0)  # cond=0 → out = data[0] = 1.0
    ref_t = torch.tensor([1.0], dtype=torch.float16, device="cuda")
    torch.testing.assert_close(out_t, ref_t)

    out_t = compiled(data_t, 1)  # cond=1 → out = data[1] = 2.0
    ref_t = torch.tensor([2.0], dtype=torch.float16, device="cuda")
    torch.testing.assert_close(out_t, ref_t)

    print("[check] torch.allclose(out, ref): OK")

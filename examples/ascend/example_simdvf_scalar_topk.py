"""SimdVF + scalar top-k on UB: MTE2 copy → SimdVF vector add → scalar selection-sort.

This test exercises the auto-schedule pass's ability to classify pure-scalar
tasks (which run on PIPE_S and carry no specialized pipe bit). Before the fix,
a scalar consumer of UB data produced by a vector/DMA task would cause
``GetPipeName()`` to return "UNKNOWN", crashing the barrier insertion pass.

Pattern
-------
- MTE2 copy: GM → UB (``T.copy``)
- SimdVF: vector add on UB (``vld`` / ``vadd`` / ``vsts``)
- Scalar: selection-sort top-k over UB, results buffered in UB
- MTE3 copy: UB → GM for the final output
"""

import tilelang
import tilelang.language as T
from tilelang.language import simd as S

VL = 64  # float32 lanes per 2048-bit vector register
NUM_EXPERTS = 128
NUM_TOPK = 8
NUM_CORES = 32


def make_kernel(backend="asc"):
    num_vregs = NUM_EXPERTS // VL
    num_tokens = T.dynamic("num_tokens")

    @tilelang.jit(out_idx=-1, target=backend)
    def _build():
        @T.prim_func
        def main(
            logits: T.Tensor[(num_tokens, NUM_EXPERTS), T.float32],
            out_idx: T.Tensor[(num_tokens, NUM_TOPK), T.int32],
        ):
            with T.Kernel(NUM_CORES) as core_id:
                scores_ub = T.alloc_shared((NUM_EXPERTS,), T.float32)
                result_ub = T.alloc_shared((NUM_TOPK,), T.int32)
                # ASC uses alloc_var for scalar carry. PTO local.var does not
                # carry loop-updated scalars on this path, so PTO uses 1-element UB.
                if backend == "pto":
                    best_val_ub = T.alloc_shared((1,), T.float32)
                    best_idx_ub = T.alloc_shared((1,), T.int32)
                else:
                    best_val = T.alloc_var(dtype=T.float32)
                    best_idx = T.alloc_var(dtype=T.int32)

                for w in T.Pipelined(T.ceildiv(num_tokens, NUM_CORES), num_stages=1):
                    token = w * NUM_CORES + core_id
                    if token < num_tokens:
                        T.copy(logits[token, :], scores_ub)

                        # VECTOR: add 1.0 (monotonic, preserves ordering)
                        with T.SimdVF():
                            if backend == "pto":
                                full = T.vmi.create_mask(VL, size=VL)
                                one = T.vmi.vbrc(T.float32(1), size=VL)
                                for r in T.serial(num_vregs):
                                    x = T.vmi.vload(scores_ub[r * VL], size=VL)
                                    T.vmi.vstore(T.vmi.vadd(x, one, full), scores_ub[r * VL], full)
                            else:
                                full = S.pset(32, "PAT_ALL")
                                one = S.vdup(T.float32(1), "float32", full)
                                for r in T.serial(num_vregs):
                                    x = S.vld(scores_ub[r * VL])
                                    S.vsts(
                                        scores_ub[r * VL],
                                        S.vadd(x, one, full),
                                        full,
                                    )

                        # SCALAR: selection-sort top-k, only touches UB
                        if backend == "pto":
                            for k in T.serial(NUM_TOPK):
                                best_val_ub[0] = -T.infinity(T.float32)
                                best_idx_ub[0] = -1
                                for e in T.serial(NUM_EXPERTS):
                                    if scores_ub[e] > best_val_ub[0]:
                                        best_val_ub[0] = scores_ub[e]
                                        best_idx_ub[0] = e
                                result_ub[k] = best_idx_ub[0]
                                scores_ub[best_idx_ub[0]] = -T.infinity(T.float32)
                        else:
                            for k in T.serial(NUM_TOPK):
                                best_val = -T.infinity(T.float32)
                                best_idx = -1
                                for e in T.serial(NUM_EXPERTS):
                                    if scores_ub[e] > best_val:
                                        best_val = scores_ub[e]
                                        best_idx = e
                                result_ub[k] = best_idx
                                scores_ub[best_idx] = -T.infinity(T.float32)

                        # MTE3: copy results from UB to GM
                        T.copy(result_ub, out_idx[token, :])

        return main

    return _build()


def ref_program(logits, num_topk):
    """Reference: torch.topk (largest=True, sorted=True, stable tie-break)."""
    import torch

    return torch.topk(logits + 1, num_topk, dim=-1, largest=True, sorted=True).indices.to(torch.int32)


def simulator_safe_randn(shape, *, dtype, device):
    import torch

    return torch.randn(shape, dtype=dtype, device="cpu").to(device)


if __name__ == "__main__":
    import torch

    kernel = make_kernel()
    print("compiled OK")

    device = torch.device("npu")
    n = 4096
    torch.manual_seed(42)
    logits = torch.randn(n, NUM_EXPERTS, dtype=torch.float32, device=device)

    out = kernel(logits)
    torch.npu.synchronize()

    expected = ref_program(logits, NUM_TOPK)
    bad = (out != expected).any(dim=1).sum().item()
    assert bad == 0, f"top-{NUM_TOPK} mismatch on {bad}/{n} rows"
    print(f"PASS: {n} tokens, {NUM_EXPERTS} experts, top-{NUM_TOPK}")

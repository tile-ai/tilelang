import torch
import tilelang
import tilelang.language as T


THREADS = 128


@tilelang.jit(
    out_idx=[2],
)
def fp8_vecadd_kernel(n: int):
    @T.prim_func
    def main(
        A: T.Tensor((n,), T.float8_e4m3fn),
        B: T.Tensor((n,), T.float8_e4m3fn),
        C: T.Tensor((n,), T.float8_e4m3fn),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((n,), T.float8_e4m3fn)
            b_ub = T.alloc_shared((n,), T.float8_e4m3fn)
            c_ub = T.alloc_shared((n,), T.float8_e4m3fn)

            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimtVF(threads=THREADS):
                a_local = T.alloc_fragment((n,), T.float8_e4m3fn)
                b_local = T.alloc_fragment((n,), T.float8_e4m3fn)
                c_local = T.alloc_fragment((n,), T.float8_e4m3fn)

                T.copy(a_ub, a_local)
                T.copy(b_ub, b_local)
                for i in T.Parallel(n):
                    c_local[i] = a_local[i] + b_local[i]
                T.copy(c_local, c_ub)
            T.copy(c_ub, C)

    return main


def _raw_prefix(tensor: torch.Tensor, count: int = 16) -> str:
    try:
        return str(tensor.view(torch.uint8)[:count].cpu())
    except Exception:
        return "<raw view unavailable>"


def run_case(num_fp8_per_thread: int) -> bool:
    n = THREADS * num_fp8_per_thread
    print(f"\n=== num_fp8_per_thread={num_fp8_per_thread}, n={n} ===")

    kernel = fp8_vecadd_kernel(n)
    source = kernel.get_kernel_source()

    print(source)

    a_f32 = torch.linspace(-128.0, 128.0, n, device="npu", dtype=torch.float32)
    b_f32 = torch.linspace(64.0, -64.0, n, device="npu", dtype=torch.float32)
    a = a_f32.to(torch.float8_e4m3fn)
    b = b_f32.to(torch.float8_e4m3fn)

    c = kernel(a, b)
    torch.npu.synchronize()

    ref = (a.to(torch.float32) + b.to(torch.float32)).to(torch.float8_e4m3fn)
    c_f32 = c.to(torch.float32)
    ref_f32 = ref.to(torch.float32)
    ok = torch.equal(c_f32, ref_f32)

    if not ok:
        mismatch = torch.nonzero(c_f32 != ref_f32).flatten()
        first = int(mismatch[0].item()) if mismatch.numel() else -1
        print(f"first mismatch index: {first}")
        print(f"a f32: {a.to(torch.float32)[first : first + 8].cpu()}")
        print(f"b f32: {b.to(torch.float32)[first : first + 8].cpu()}")
        print(f"tilelang f32: {c_f32[first : first + 8].cpu()}")
        print(f"torch f32: {ref_f32[first : first + 8].cpu()}")
        print(f"tilelang raw: {_raw_prefix(c)}")
        print(f"torch raw: {_raw_prefix(ref)}")
    return ok


def main():
    results = {num_fp8_per_thread: run_case(num_fp8_per_thread) for num_fp8_per_thread in (16, 8, 4, 2, 1)}
    print("\n=== summary ===")
    for num_fp8_per_thread, ok in results.items():
        print(f"num_fp8_per_thread={num_fp8_per_thread}: {'PASS' if ok else 'FAIL'}")
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

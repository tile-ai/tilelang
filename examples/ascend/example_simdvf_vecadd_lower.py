import tilelang
import tilelang.ascend.language as T
from tilelang.profiler import do_bench

NUM_BLOCKS = 64  # AI cores
NUM_THREADS = 2048  # SIMT threads per core
TILE = NUM_THREADS * 4  # 8192 floats = 32 KB per tile
NUM_STAGES = 2
# Same problem size for AscendC and PTO so latency / bandwidth are comparable.
DEFAULT_N = 2**30


def vector_add(N, backend="asc"):
    if N % (TILE * NUM_BLOCKS) != 0:
        raise ValueError(f"N must be a multiple of {TILE * NUM_BLOCKS}, got {N}")

    TOTAL_TILES = N // (TILE * NUM_BLOCKS)
    lanes = 64

    @T.prim_func
    def main(
        A: T.Buffer((N,), "float32"),
        B: T.Buffer((N,), "float32"),
        C: T.Buffer((N,), "float32"),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            temp1 = T.alloc_shared((TILE,), "float32")
            temp2 = T.alloc_shared((TILE,), "float32")
            temp3 = T.alloc_shared((TILE,), "float32")
            T.annotate_buffer_versions({temp1: NUM_STAGES, temp2: NUM_STAGES, temp3: NUM_STAGES})

            for iter in T.Pipelined(TOTAL_TILES, num_stages=NUM_STAGES):
                begin = (iter * NUM_BLOCKS + bx) * TILE
                end = (iter * NUM_BLOCKS + bx + 1) * TILE

                T.copy(A[begin:end], temp1)
                T.copy(B[begin:end], temp2)
                with T.SimdVF():
                    if backend == "pto":
                        mask = T.vmi.create_mask(lanes, size=lanes)
                        for i in range(TILE // lanes):
                            r0 = T.vmi.vload(temp1[i * lanes], size=lanes)
                            r1 = T.vmi.vload(temp2[i * lanes], size=lanes)
                            T.vmi.vstore(T.vmi.vadd(r0, r1, mask), temp3[i * lanes], mask)
                    else:
                        mask = T.simd.pset(32)
                        for i in range(TILE // 64):
                            r0 = T.simd.vld(temp1[i * 64])
                            r1 = T.simd.vld(temp2[i * 64])
                            r2 = T.simd.vadd(r0, r1, mask)
                            T.simd.vsts(temp3[i * 64], r2, mask)
                T.copy(temp3, C[begin:end])

    return main


def ref_program(a, b):
    """Reference implementation: C = A + B"""
    return a + b


NUM_REPEATS = 10


def run_regression_perf(N=None, backend="asc"):
    import torch

    if N is None:
        N = DEFAULT_N
    device = torch.device("npu")
    target = backend
    program = vector_add(N, backend)
    kernel = tilelang.compile(program, target=target, out_idx=-1)
    a = torch.randn(N, dtype=torch.float32, device="cpu").to(device)
    b = torch.randn(N, dtype=torch.float32, device="cpu").to(device)
    kernel(a, b)
    torch.npu.synchronize()
    latency_ms = do_bench(lambda: kernel(a, b), backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
    gbps = (N * a.element_size() * 3) / (latency_ms / 1e3) / 1e9
    print(f"    [{backend} N={N}] {latency_ms * 1e3:.2f} us  |  {gbps:.1f} GB/s")
    return latency_ms


if __name__ == "__main__":
    import torch

    N = DEFAULT_N
    dtype = torch.float32
    device = torch.device("npu")

    # Create and compile kernel
    print(f"Compiling vector_add kernel (N={N})...")
    program = vector_add(N)
    kernel = tilelang.compile(program, out_idx=-1)
    print("Compilation succeeded!")

    # Print generated source code
    print("\n--- Generated Ascend Source ---")
    print(kernel.get_kernel_source())

    # Create test data
    a = torch.randn(N, dtype=dtype, device=device)
    b = torch.randn(N, dtype=dtype, device=device)

    # Run kernel
    print("Running kernel on NPU...")
    c = kernel(a, b)
    torch.npu.synchronize()

    # Verify
    expected = ref_program(a, b)
    if not torch.equal(c, expected):
        max_diff = torch.max(torch.abs(c - expected)).item()
        raise AssertionError(f"Results mismatch! Max diff: {max_diff}")
    print("\nVerification passed!")

    # Benchmark
    latency_ms = do_bench(lambda: kernel(a, b), backend="msprof", _n_warmup=30, _n_repeat=NUM_REPEATS)
    print(f"{latency_ms:.1f} ms/iter  |  {(N * a.element_size() * 3) / (latency_ms / 1e3) / 1e9:.1f} GB/s")

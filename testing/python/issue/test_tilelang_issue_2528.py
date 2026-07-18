# Pipelined T.copy into a 1-D fp16 shared buffer: block_K=32 with num_stages>=2
# faults with "CUDA error: misaligned address". The pipeline double-buffers
# dt_shared with a 64-byte per-stage stride, so the odd version's shared
# destination is at 64 mod 128 -- and cp.async.bulk.tensor.1d (sm_90) requires
# a 128-byte-aligned shared destination. block_K=64 (128-byte stride) and
# num_stages=1 (no double-buffer) both run.
import tilelang
import torch
import tilelang.language as T

tilelang.disable_cache()


def make_kernel(chunk_size, block_K, num_stages, threads=128):
    dtype = T.float16
    accum = T.float32

    @T.prim_func
    def f(
        dt: T.Tensor((chunk_size,), dtype),
        Out: T.Tensor((chunk_size,), accum),
    ):
        with T.Kernel(1, threads=threads) as _:
            dt_shared = T.alloc_shared((block_K,), dtype)  # 1-D pipelined TMA dest
            dt_local = T.alloc_fragment((block_K,), accum)
            for k in T.Pipelined(T.ceildiv(chunk_size, block_K), num_stages=num_stages):
                T.copy(dt[k * block_K : (k + 1) * block_K], dt_shared)  # -> cp.async.bulk.tensor.1d on sm_90
                T.copy(dt_shared, dt_local)
                for i in T.Parallel(block_K):
                    Out[k * block_K + i] = dt_local[i] * 2.0

    return tilelang.compile(f)


def trial(block_K, num_stages):
    chunk_size = 256
    torch.manual_seed(0)
    dt = torch.randint(0, 3, (chunk_size,), device="cuda").half()
    out = torch.empty((chunk_size,), device="cuda", dtype=torch.float32)
    make_kernel(chunk_size, block_K, num_stages)(dt, out)
    torch.cuda.synchronize()
    print(f"  block_K={block_K} num_stages={num_stages}: OK  match={torch.allclose(out, dt.float() * 2.0)}")


if __name__ == "__main__":
    print("block_K=64 num_stages=2 (expected OK):")
    trial(64, 2)  # -> OK  match=True
    print("block_K=32 num_stages=1 (expected OK):")
    trial(32, 1)  # -> OK  match=True
    print("block_K=32 num_stages=2 (expected CRASH):")
    trial(32, 2)  # -> CUDA error: misaligned address

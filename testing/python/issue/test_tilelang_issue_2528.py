import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def make_kernel(chunk_size, block_k, num_stages, threads=128):
    @T.prim_func
    def main(
        source: T.Tensor((chunk_size,), T.float16),
        output: T.Tensor((chunk_size,), T.float32),
    ):
        with T.Kernel(1, threads=threads):
            source_shared = T.alloc_shared((block_k,), T.float16)
            source_local = T.alloc_fragment((block_k,), T.float32)
            for k in T.Pipelined(T.ceildiv(chunk_size, block_k), num_stages=num_stages):
                T.copy(source[k * block_k : (k + 1) * block_k], source_shared)
                T.copy(source_shared, source_local)
                for i in T.Parallel(block_k):
                    output[k * block_k + i] = source_local[i] * 2.0

    return tilelang.compile(main)


@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_pipeline_versioned_1d_copy_uses_linear_bulk():
    chunk_size = 256
    source = torch.randint(0, 3, (chunk_size,), device="cuda").half()

    for num_stages in (2, 3):
        kernel = make_kernel(chunk_size, block_k=32, num_stages=num_stages)
        code = kernel.get_kernel_source()
        assert "tl::tma_load" in code
        assert "CUtensorMap" not in code
        assert ".arrive_and_expect_tx(64)" in code

        output = torch.empty(chunk_size, device="cuda", dtype=torch.float32)
        kernel(source, output)
        torch.testing.assert_close(output, source.float() * 2.0)


if __name__ == "__main__":
    tilelang.testing.main()

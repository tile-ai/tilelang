import tilelang.tools.bench
import fp8_lighting_indexer
import sparse_mla_bwd
import sparse_mla_fwd
import sparse_mla_fwd_pipelined
import topk_selector


def bench_topk_selector():
    tilelang.tools.bench.process_func(topk_selector.test_topk_selector)



def bench_fp8_lighting_indexer():
    tilelang.tools.bench.process_func(fp8_lighting_indexer.test_fp8_lighting_indexer)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def bench_sparse_mla_fwd():
    tilelang.tools.bench.process_func(sparse_mla_fwd.test_sparse_mla_fwd)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def bench_sparse_mla_fwd_pipelined():
    tilelang.tools.bench.process_func(sparse_mla_fwd_pipelined.test_sparse_mla_fwd_pipelined)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def bench_sparse_mla_bwd():
    tilelang.tools.bench.process_func(sparse_mla_bwd.test_sparse_mla_bwd)

def main():
    tilelang.tools.bench.main()

if __name__ == "__main__":
    main()

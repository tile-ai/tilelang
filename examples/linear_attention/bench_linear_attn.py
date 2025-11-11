import tilelang.tools.bench
import example_linear_attn_bwd
import example_linear_attn_fwd

@tilelang.testing.requires_cuda
def bench_example_linear_attn_fwd():
    tilelang.tools.bench.process_func(example_linear_attn_fwd.main)


@tilelang.testing.requires_cuda
def bench_example_linear_attn_bwd():
    tilelang.tools.bench.process_func(example_linear_attn_bwd.main)

def main():
    tilelang.tools.bench.main()

if __name__ == "__main__":
    main()

import tilelang.tools.bench
import example_fusedmoe_tilelang


def bench_example_fusedmoe_tilelang():
    tilelang.tools.bench.process_func(
        example_fusedmoe_tilelang.benchmark,
        d_hidden=1024,
        d_expert=256,
        n_routed_experts=8,
        n_shared_experts=1,
        n_experts_per_token=4,
        batch_size=1,
        seq_len=1024)


if globals().get("__name__") == "__main__":
    tilelang.tools.bench.main()

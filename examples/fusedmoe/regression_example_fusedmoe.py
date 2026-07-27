import tilelang.testing
import example_fusedmoe_tilelang


def regression_example_fusedmoe_tilelang():
    tilelang.testing.process_func(
        example_fusedmoe_tilelang.run_regression_perf,
        d_hidden=1024,
        d_expert=256,
        n_routed_experts=8,
        n_shared_experts=1,
        n_experts_per_token=4,
        batch_size=1,
        seq_len=1024,
    )


if __name__ == "__main__":
    tilelang.testing.regression()

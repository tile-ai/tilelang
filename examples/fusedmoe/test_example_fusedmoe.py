import tilelang.testing
import example_fusedmoe_tilelang
import bench_example_fusedmoe


def test_example_fusedmoe_tilelang():
    example_fusedmoe_tilelang.main(
        d_hidden=1024,
        d_expert=256,
        n_routed_experts=8,
        n_shared_experts=1,
        n_experts_per_token=4,
        batch_size=1,
        seq_len=1024)


def test_bench_example_fusedmoe():
    bench_example_fusedmoe.main()


if __name__ == "__main__":
    tilelang.testing.main()

import tilelang.testing
import tilelang
import tilelang_example_sparse_tensorcore


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_tilelang_example_sparse_tensorcore():
    tilelang_example_sparse_tensorcore.main()


if __name__ == "__main__":
    tilelang.testing.main()

import tilelang.testing

import example_hadamard


# Cover the distinct kernel code paths:
#   dim=256   -> thread-local + warp shuffle only (no shared-memory exchange)
#   dim=1024  -> single shared-memory exchange across warps
#   dim=8192  -> single exchange with full 32KB shared memory
#   dim=32768 -> multiple shared-memory exchange rounds
@tilelang.testing.requires_cuda
def test_example_hadamard():
    for batch, dim in [(8, 256), (8, 1024), (8, 8192), (8, 32768)]:
        example_hadamard.main(["--batch", str(batch), "--dim", str(dim)])


if __name__ == "__main__":
    tilelang.testing.main()

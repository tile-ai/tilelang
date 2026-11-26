"""RNG initialization follows runtime control flow within one helper."""

import pytest
import tilelang
import tilelang.ascend.language as T


@pytest.mark.parametrize("conditional", [False, True])
def test_rng_state_outlives_initialization_control_flow(conditional):
    @T.prim_func
    def main(A: T.Tensor((64,), T.float32), seed: T.int32):
        with T.Kernel(1), T.SimtVF(threads=64):
            tx = T.get_thread_binding()
            if conditional:
                if tx < 32:
                    T.rng_init(seed, seq=tx)
                else:
                    T.rng_init(seed + 1, seq=tx)
            else:
                for i in T.serial(2):
                    T.rng_init(seed + i, seq=tx)
            A[tx] = T.rng_rand_float()

    source = tilelang.lower(main, target="ascend").kernel_source
    assert source.count("tl::AscendPhiloxState ") == 1
    control = "if (" if conditional else "for ("
    assert source.index("tl::AscendPhiloxState ") < source.index(control)

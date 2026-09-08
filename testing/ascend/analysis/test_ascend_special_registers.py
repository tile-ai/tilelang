import re

import tilelang.language as T
import tilelang.testing

from tilelang.engine.lower import lower


TILE = 256
FLAG = 3
LOOP_TRIPS = 2


def _core_branch(source: str, token: str) -> str:
    position = source.index(token)
    aic = source.rfind("if ASCEND_IS_AIC", 0, position)
    aiv = source.rfind("if ASCEND_IS_AIV", 0, position)
    assert max(aic, aiv) >= 0
    return "AIC" if aic > aiv else "AIV"


def _make_loop_nested_wait_program():
    @T.prim_func
    def main(
        values: T.Tensor((64, 64), "float32"),
        exchange: T.Tensor((64, 64), "float32"),
        independent: T.Tensor((64, 64), "float32"),
        output: T.Tensor((64, 64), "float32"),
    ):
        with T.Kernel(64) as bx:
            write_ub = T.alloc_shared((64,), "float32")
            loop_ub = T.alloc_shared((64,), "float32")
            read_ub = T.alloc_shared((64,), "float32")
            T.copy(values[bx, :], write_ub)
            T.copy(write_ub, exchange[bx, :])
            iteration = T.alloc_var("int32")
            iteration = 0
            while iteration < LOOP_TRIPS:
                T.copy(exchange[bx, :], loop_ub)
                T.ascend_sync_inter_wait("PIPE_MTE2", FLAG)
                iteration = iteration + 1
            T.copy(independent[bx, :], read_ub)
            T.copy(read_ub, output[bx, :])

    return main


def test_loop_nested_wait_orders_following_pipe_user():
    source = lower(_make_loop_nested_wait_program(), target="ascend").kernel_source
    assert "while (1)" in source
    wait = source.index(f"AscendC::CrossCoreWaitFlag<0, PIPE_MTE2>({FLAG});")
    mte2_loads = [match.start() for match in re.finditer("copy_gm_to_ubuf", source)]
    assert len(mte2_loads) == 3
    assert wait < mte2_loads[-1]


def _make_loop_nested_pad_writer_program():
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
        X: T.Tensor((1, 30), "float32"),
        C: T.Tensor((TILE, TILE), "float32"),
        Y: T.Tensor((1, 32), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            x_ub = T.alloc_shared((1, 32), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
            T.copy(accum, C)
            iteration = T.alloc_var("int32")
            iteration = 0
            while iteration < LOOP_TRIPS:
                with T.Task():
                    T.ascend_set_copy_pad_value(-1.0, dtype="float32")
                iteration = iteration + 1
            T.copy(X, x_ub[:, :30], data_select=True)
            T.copy(x_ub, Y)

    return main


def test_loop_nested_pad_writer_follows_external_reader_core():
    source = lower(_make_loop_nested_pad_writer_program(), target="ascend").kernel_source
    writer = source.index("set_mov_pad_val")
    reader = source.index("copy_gm_to_ubuf_align_v2")
    assert source.rfind("while (1)", 0, writer) >= 0
    assert _core_branch(source, "set_mov_pad_val") == "AIV"
    assert writer < reader


def _make_nested_loop_break_program():
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
        X: T.Tensor((64,), "float32"),
        C: T.Tensor((TILE, TILE), "float32"),
        Y: T.Tensor((64,), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            x_ub = T.alloc_shared((64,), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            i = T.alloc_var("int32")
            i = 0
            while i < 2:
                T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
                T.copy(accum, C)
                for _ in T.Serial(2):
                    T.copy(X, x_ub)
                    T.copy(x_ub, Y)
                i = i + 1

    return main


def test_nested_loop_break_is_emitted_on_every_core():
    source = lower(_make_nested_loop_break_program(), target="ascend").kernel_source
    aic_start = source.index("if ASCEND_IS_AIC")
    aiv_start = source.index("if ASCEND_IS_AIV")
    assert "break;" in source[aic_start:aiv_start]
    assert "break;" in source[aiv_start:]


def _make_split_atomic_writer_program():
    @T.prim_func
    def main(
        X: T.Tensor((64,), "float32"),
        Y: T.Tensor((64,), "float32"),
        C: T.Tensor((16, 16), "float32"),
    ):
        with T.MixedKernel(1):
            ub = T.alloc_shared((64,), "float32")
            l0c = T.alloc_l0c((16, 16), "float32")
            T.copy(X, ub)
            T.set_atomic("add", "float32")
            T.copy(ub, Y)
            T.set_atomic_none()
            T.set_atomic("add", "float32")
            T.copy(l0c, C)
            T.set_atomic_none()

    return main


def test_atomic_writers_are_broadcast_across_split_core_readers():
    source = lower(_make_split_atomic_writer_program(), target="ascend").kernel_source
    assert source.count("AscendC::SetAtomicAdd<float>();") == 4
    assert source.count("AscendC::SetAtomicNone();") == 4
    assert _core_branch(source, "copy_ubuf_to_gm") == "AIV"
    assert _core_branch(source, "copy_matrix_cc_to_gm") == "AIC"


if __name__ == "__main__":
    tilelang.testing.main()

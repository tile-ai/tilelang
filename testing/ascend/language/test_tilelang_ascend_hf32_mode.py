import pytest

import tilelang.ascend.language as T
from tilelang.engine.lower import lower


def _pto_hf32_gemm(hf32):
    @T.prim_func
    def main(
        A: T.Buffer((256, 128), "float32"),
        B: T.Buffer((256, 128), "float32"),
        C: T.Buffer((256, 256), "float32"),
    ):
        with T.Kernel(1):
            T.set_hf32_mode(hf32)
            a_l1 = T.alloc_l1((256, 128), "float32")
            b_l1 = T.alloc_l1((256, 128), "float32")
            acc = T.alloc_l0c((256, 256), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, C)

    return main


@pytest.mark.pto
@pytest.mark.parametrize(
    ("hf32", "pto_mode"),
    [
        ("nearest_zero", "pto.Tf32Mode.ROUND_AWAY"),
        ("nearest_even", "pto.Tf32Mode.ROUND_EVEN"),
    ],
)
def test_pto_fp32_gemm_hf32_codegen(hf32, pto_mode):
    source = lower(
        _pto_hf32_gemm(hf32),
        target="pto",
    ).kernel_source

    assert 'kernel_kind="cube"' in source
    assert "pto.section" not in source
    assert f"tf32_mode={pto_mode}" in source


def _pto_two_gemms(first_mode, second_mode):
    @T.prim_func
    def main(
        A: T.Buffer((256, 128), "float32"),
        B: T.Buffer((256, 128), "float32"),
        C0: T.Buffer((256, 256), "float32"),
        C1: T.Buffer((256, 256), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((256, 128), "float32")
            b_l1 = T.alloc_l1((256, 128), "float32")
            acc = T.alloc_l0c((256, 256), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.set_hf32_mode(first_mode)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, C0)
            T.set_hf32_mode(second_mode)
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, C1)

    return main


def _gemm_run_lines(source):
    return [line for line in source.splitlines() if "_tl_gemm_l1.run_l1_tile(" in line]


@pytest.mark.pto
def test_pto_hf32_enable_then_restore_codegen():
    source = lower(
        _pto_two_gemms("nearest_even", None),
        target="pto",
    ).kernel_source
    runs = _gemm_run_lines(source)

    assert len(runs) == 2
    assert "tf32_mode=pto.Tf32Mode.ROUND_EVEN" in runs[0]
    assert "tf32_mode=" not in runs[1]


@pytest.mark.pto
def test_pto_hf32_modes_are_bound_per_gemm():
    source = lower(
        _pto_two_gemms("nearest_zero", "nearest_even"),
        target="pto",
    ).kernel_source
    runs = _gemm_run_lines(source)

    assert len(runs) == 2
    assert "tf32_mode=pto.Tf32Mode.ROUND_AWAY" in runs[0]
    assert "tf32_mode=pto.Tf32Mode.ROUND_EVEN" in runs[1]


@pytest.mark.pto
def test_pto_rejects_control_flow_dependent_hf32_mode():
    @T.prim_func
    def main(
        A: T.Buffer((256, 128), "float32"),
        B: T.Buffer((256, 128), "float32"),
        C: T.Buffer((256, 256), "float32"),
        flag: T.int32,
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((256, 128), "float32")
            b_l1 = T.alloc_l1((256, 128), "float32")
            acc = T.alloc_l0c((256, 256), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            if flag == 0:
                T.set_hf32_mode(None)
            else:
                T.set_hf32_mode("nearest_even")
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, C)

    with pytest.raises(Exception, match="HF32 mode is control-flow dependent"):
        lower(main, target="pto")


def _pto_hf32_set_in_loop_then_gemm(extent):
    @T.prim_func
    def main(
        A: T.Buffer((256, 128), "float32"),
        B: T.Buffer((256, 128), "float32"),
        C: T.Buffer((256, 256), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((256, 128), "float32")
            b_l1 = T.alloc_l1((256, 128), "float32")
            acc = T.alloc_l0c((256, 256), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            for _ in T.serial(extent):
                T.set_hf32_mode("nearest_even")
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, C)

    return main


@pytest.mark.pto
def test_pto_hf32_mode_set_in_positive_loop_reaches_following_gemm():
    source = lower(
        _pto_hf32_set_in_loop_then_gemm(2),
        target="pto",
    ).kernel_source

    assert "tf32_mode=pto.Tf32Mode.ROUND_EVEN" in source


@pytest.mark.pto
def test_pto_rejects_possibly_zero_loop_hf32_mode():
    @T.prim_func
    def main(
        A: T.Buffer((256, 128), "float32"),
        B: T.Buffer((256, 128), "float32"),
        C: T.Buffer((256, 256), "float32"),
        extent: T.int32,
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((256, 128), "float32")
            b_l1 = T.alloc_l1((256, 128), "float32")
            acc = T.alloc_l0c((256, 256), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            for _ in T.serial(extent):
                T.set_hf32_mode("nearest_even")
            T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=True)
            T.copy(acc, C)

    with pytest.raises(Exception, match="HF32 mode is control-flow dependent"):
        lower(main, target="pto")


if __name__ == "__main__":
    pytest.main([__file__])

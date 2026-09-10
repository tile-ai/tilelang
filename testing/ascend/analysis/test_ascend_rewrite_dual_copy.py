import pytest

import tilelang
import tilelang.ascend.language as T
from tilelang import tvm
from tvm.tirx.stmt_functor import post_order_visit


def _unsupported_gm_to_l1_dual_copy():
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "bfloat16"),
        B: T.Tensor((128, 128), "bfloat16"),
        C: T.Tensor((64, 128), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((64, 128), "bfloat16")
            b_l1 = T.alloc_l1((128, 128), "bfloat16")
            accum = T.alloc_l0c((64, 128), "float32")
            T.dual_copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
            T.copy(accum, C)

    return main


def _odd_extent_gm_to_ub_dual_copy():
    @T.prim_func
    def main(A: T.Tensor((127, 128), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((63, 128), "float32")
            T.dual_copy(A, temp)

    return main


def _odd_region_l0c_to_ub_dual_copy():
    @T.prim_func
    def main():
        with T.Kernel(1):
            accum = T.alloc_l0c((128, 128), "float32")
            temp = T.alloc_shared((64, 128), "float32")
            T.dual_copy(accum[:127, :], temp[:63, :])

    return main


def _unaligned_n_split_l0c_to_ub_dual_copy():
    @T.prim_func
    def main():
        with T.Kernel(1):
            accum = T.alloc_l0c((64, 64), "float32")
            temp = T.alloc_shared((64, 32), "float32")
            T.dual_copy(accum[:, :48], temp[:, :24])

    return main


def _one_dimensional_software_dual_copies():
    @T.prim_func
    def main(
        A: T.Tensor((128,), "float32"),
        C: T.Tensor((128,), "float32"),
        D: T.Tensor((16, 16), "float32"),
    ):
        with T.Kernel(1):
            temp = T.alloc_shared((64,), "float32")
            output_l1 = T.alloc_l1((128,), "float32")
            a_l1 = T.alloc_l1((16, 16), "bfloat16")
            b_l1 = T.alloc_l1((16, 16), "bfloat16")
            accum = T.alloc_l0c((16, 16), "float32")
            T.dual_copy(A, temp)
            T.dual_copy(temp, C)
            T.dual_copy(temp, output_l1)
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
            T.copy(accum, D)

    return main


def _manual_one_dimensional_software_dual_copies():
    @T.prim_func
    def main(
        A: T.Tensor((128,), "float32"),
        C: T.Tensor((128,), "float32"),
        D: T.Tensor((16, 16), "float32"),
    ):
        with T.Kernel(1):
            temp = T.alloc_shared((64,), "float32")
            output_l1 = T.alloc_l1((128,), "float32")
            a_l1 = T.alloc_l1((16, 16), "bfloat16")
            b_l1 = T.alloc_l1((16, 16), "bfloat16")
            accum = T.alloc_l0c((16, 16), "float32")
            with T.Cube():
                T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
                T.copy(accum, D)
            with T.Vector():
                T.dual_copy(A, temp)
                T.dual_copy(temp, C)
                T.dual_copy(temp, output_l1)

    return main


def _manual_mixed_kernel_software_dual_copy():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
        with T.MixedKernel(1):
            temp = T.alloc_shared((64,), "float32")
            T.dual_copy(A, temp)
            T.dual_copy(temp, C)

    return main


def _manual_unit_flag_dual_copies():
    @T.prim_func
    def main(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((64,), "float32")
            accum = T.alloc_l0c((128, 128), "float32")
            output = T.alloc_shared((64, 128), "float32")
            with T.Cube():
                T.dual_copy(accum, output, unit_flag_ctrl=3)
            with T.Vector():
                T.dual_copy(A, temp, unit_flag_ctrl=3)
                T.dual_copy(temp, C, unit_flag_ctrl=3)

    return main


def _unaligned_one_dimensional_ub_to_l1_dual_copy():
    @T.prim_func
    def main():
        with T.Kernel(1):
            temp = T.alloc_shared((3,), "float32")
            output_l1 = T.alloc_l1((6,), "float32")
            with T.Cube():
                pass
            with T.Vector():
                T.dual_copy(temp, output_l1)

    return main


def _one_dimensional_hardware_dual_copy():
    @T.prim_func
    def main():
        with T.Kernel(1):
            accum = T.alloc_l0c((128,), "float32")
            temp = T.alloc_shared((64,), "float32")
            T.dual_copy(accum, temp)

    return main


def _odd_one_dimensional_software_dual_copy():
    @T.prim_func
    def main(A: T.Tensor((127,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((63,), "float32")
            T.dual_copy(A, temp)

    return main


def _odd_one_dimensional_software_region(half_extent):
    @T.prim_func
    def main(A: T.Tensor((128,), "float32")):
        with T.Kernel(1):
            temp = T.alloc_shared((64,), "float32")
            T.dual_copy(A[:127], temp[:half_extent])

    return main


@pytest.mark.parametrize("target", ["ascend"])
def test_rewrite_dual_copy_rejects_unsupported_memory_path(target):
    with pytest.raises(ValueError, match="RewriteDualCopy supports only L0C->UB"):
        tilelang.lower(_unsupported_gm_to_l1_dual_copy(), target=target)


def test_dual_copy_rejects_odd_static_extent():
    with pytest.raises(ValueError, match="Exactly one trailing dimension must be halved"):
        _odd_extent_gm_to_ub_dual_copy()


def test_rewrite_dual_copy_rejects_odd_static_hardware_region():
    with pytest.raises(ValueError, match="exact 2:1 extent ratio"):
        tilelang.lower(_odd_region_l0c_to_ub_dual_copy(), target="ascend")


def test_rewrite_dual_copy_rejects_unaligned_hardware_n_split():
    with pytest.raises(ValueError, match="source N extent to be a multiple of 32"):
        tilelang.lower(_unaligned_n_split_l0c_to_ub_dual_copy(), target="ascend")


def test_rewrite_dual_copy_supports_one_dimensional_software_paths():
    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE.value: True}):
        source = tilelang.lower(_one_dimensional_software_dual_copies(), target="ascend").kernel_source
    assert "asc_copy_gm2ub_align" in source
    assert "asc_copy_ub2gm_align" in source
    assert "asc_copy_ub2l1" in source
    assert "A[(sid * 64)]" in source
    assert "C[(sid * 64)]" in source
    assert source.count("sid * 64") >= 3


def test_rewrite_dual_copy_supports_manual_one_dimensional_software_paths():
    with tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE.value: False}):
        source = tilelang.lower(_manual_one_dimensional_software_dual_copies(), target="ascend").kernel_source
    assert "__global__ __mix__(1, 2)" in source
    assert source.count("asc_get_sub_block_id()") == 1
    assert "A[(sid * 64)]" in source
    assert "C[(sid * 64)]" in source


def test_rewrite_dual_copy_rejects_unscoped_manual_software_path():
    with (
        tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE.value: False}),
        pytest.raises(ValueError, match=r"explicit T\.Vector\(\) block"),
    ):
        tilelang.lower(_one_dimensional_software_dual_copies(), target="ascend")


def test_rewrite_dual_copy_rejects_manual_mixed_kernel_software_path():
    with (
        tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE.value: False}),
        pytest.raises(ValueError, match="T.MixedKernel is not supported in manual mode"),
    ):
        tilelang.lower(_manual_mixed_kernel_software_dual_copy(), target="ascend")


def test_rewrite_dual_copy_strips_software_unit_flag_ctrl():
    rewritten_annotations = []

    @tvm.ir.instrument.pass_instrument
    class CaptureRewriteDualCopy:
        def run_after_pass(self, mod, info):
            if info.name != "tl.RewriteDualCopy":
                return

            def visit(node):
                if isinstance(node, tvm.tirx.Call) and node.op.name == "tl.tileop.copy":
                    rewritten_annotations.append(node.annotations)

            post_order_visit(mod["main"].body, visit)

    with tvm.transform.PassContext(
        config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE.value: False},
        instruments=[CaptureRewriteDualCopy()],
    ):
        source = tilelang.lower(_manual_unit_flag_dual_copies(), target="ascend").kernel_source

    hardware = [annotations for annotations in rewritten_annotations if "dual_dst_ctl" in annotations]
    software = [annotations for annotations in rewritten_annotations if "dual_dst_ctl" not in annotations]
    assert len(hardware) == 1 and hardware[0]["unit_flag_ctrl"] == 3
    assert len(software) == 2
    assert all("unit_flag_ctrl" not in annotations for annotations in software)
    assert all("dual_dst_ctl" not in annotations for annotations in software)
    assert (
        "static_cast<asc_dual_dst_mode>(1), static_cast<asc_unit_flag_mode>(3), "
        "static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), 0, 1, 0, 0);" in source
    )


def test_rewrite_dual_copy_rejects_unaligned_one_dimensional_ub_to_l1():
    with (
        tvm.transform.PassContext(config={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE.value: False}),
        pytest.raises(ValueError, match="payload that is a multiple of 32 bytes"),
    ):
        tilelang.lower(_unaligned_one_dimensional_ub_to_l1_dual_copy(), target="ascend")


def test_rewrite_dual_copy_rejects_one_dimensional_hardware_path():
    with pytest.raises(ValueError, match="hardware dual_copy requires at least two-dimensional"):
        tilelang.lower(_one_dimensional_hardware_dual_copy(), target="ascend")


def test_dual_copy_rejects_odd_one_dimensional_extent():
    with pytest.raises(ValueError, match="sole dimension must have an exact 2:1 extent ratio"):
        _odd_one_dimensional_software_dual_copy()


@pytest.mark.parametrize("half_extent", [63, 64])
def test_rewrite_dual_copy_rejects_odd_one_dimensional_region(half_extent):
    with pytest.raises(ValueError, match="exact 2:1 extent ratio"):
        tilelang.lower(_odd_one_dimensional_software_region(half_extent), target="ascend")


if __name__ == "__main__":
    pytest.main([__file__])

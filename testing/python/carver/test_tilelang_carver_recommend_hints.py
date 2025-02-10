# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tilelang.testing
from tilelang import carver
from tilelang.carver.arch import auto_infer_current_arch
from typing import List


def run_general_reduction_recommend_hints(structure: str = "SSR",
                                          shape: List[int] = None,
                                          dtype: str = "float16",
                                          topk: int = 20):
    arch = auto_infer_current_arch()
    carve_template = carver.GeneralReductionTemplate(
        structure=structure,
        shape=shape,
        dtype=dtype,
    ).with_arch(arch)

    func = carve_template.equivalent_function()
    assert func is not None, "Function is None"

    hints = carve_template.recommend_hints(topk=topk)
    assert len(hints) > 0, "Hints length is zero"


def test_general_reduction_recommend_hints():
    run_general_reduction_recommend_hints("SSR", [1024, 1024, 1024], "float16")
    run_general_reduction_recommend_hints("SS", [1024, 1024], "float16")
    run_general_reduction_recommend_hints("SRS", [1024, 1024, 1024], "float16")


def run_elementwise_recommend_hints(shape: List[int] = None,
                                    dtype: str = "float16",
                                    topk: int = 20):
    arch = auto_infer_current_arch()
    carve_template = carver.ElementwiseTemplate(
        shape=shape,
        dtype=dtype,
    ).with_arch(arch)

    func = carve_template.equivalent_function()
    assert func is not None, "Function is None"

    hints = carve_template.recommend_hints(topk=topk)
    assert len(hints) > 0, "Hints length is not topk"


def test_elementwise_recommend_hints():
    run_elementwise_recommend_hints([1024, 1024], "float16")
    run_elementwise_recommend_hints([1024], "float16")
    run_elementwise_recommend_hints([1024, 1024, 1024], "float16")


def run_matmul_recommend_hints(
    M: int = 1024,
    N: int = 1024,
    K: int = 1024,
    in_dtype: str = "float16",
    out_dtype: str = "float16",
    accum_dtype: str = "float16",
):
    arch = auto_infer_current_arch()
    carve_template = carver.MatmulTemplate(
        M=M,
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    ).with_arch(arch)

    func = carve_template.equivalent_function()
    assert func is not None, "Function is None"

    hints = carve_template.recommend_hints(topk=20)
    assert len(hints) > 0, "Hints length is not 20"


def test_matmul_recommend_hints():
    run_matmul_recommend_hints(1024, 1024, 1024, "float16", "float16", "float16")
    run_matmul_recommend_hints(1024, 1024, 1024, "int8", "int32", "int32")
    run_matmul_recommend_hints(1024, 1024, 1024, "float16", "float32", "float16")


def run_gemv_recommend_hints(N: int = 1024,
                             K: int = 1024,
                             in_dtype: str = "float16",
                             out_dtype: str = "float16",
                             accum_dtype: str = "float16"):
    arch = auto_infer_current_arch()
    carve_template = carver.GEMVTemplate(
        N=N,
        K=K,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
    ).with_arch(arch)

    func = carve_template.equivalent_function()
    assert func is not None, "Function is None"

    hints = carve_template.recommend_hints(topk=20)
    assert len(hints) > 0, "Hints length is not 20"


def test_gemv_recommend_hints():
    run_gemv_recommend_hints(1024, 1024, "float16", "float16", "float16")
    run_gemv_recommend_hints(1024, 1024, "int8", "int32", "int32")
    run_gemv_recommend_hints(1024, 1024, "float16", "float32", "float16")


if __name__ == "__main__":
    tilelang.testing.main()

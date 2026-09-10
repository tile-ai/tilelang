"""Source-level coverage for PTO modules containing multiple kernels."""

import pytest

from tilelang import tvm
import tilelang.ascend.language as T
from tilelang.backend.target import determine_target
from tilelang.engine.lower import lower
from tilelang.jit.adapter.wrapper import TLPTOSourceWrapper


def _two_kernel_program():
    @T.prim_func
    def main(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), C: T.Buffer((128,), "float32")):
        with T.Kernel(1), T.SimtVF(threads=64):
            for i in T.Parallel(128):
                C[i] = A[i] + B[i]

        with T.Kernel(1), T.SimtVF(threads=64):
            for i in T.Parallel(128):
                C[i] = C[i] * T.float32(2.0)

    return main


@pytest.mark.pto
def test_pto_wrapper_supports_multiple_device_kernels_in_host_call_order():
    program = _two_kernel_program()
    module = tvm.IRModule({"main": program})
    artifact = lower(program, target="pto")
    reversed_device_mod = tvm.IRModule(dict(reversed(list(artifact.device_mod.functions.items()))))
    wrapper = TLPTOSourceWrapper(
        module,
        str(artifact.kernel_source),
        determine_target("pto", return_object=True),
        reversed_device_mod,
        artifact.host_mod,
    )

    assert wrapper.pto_kernel_names == ["main_kernel", "main_kernel_1"]
    assert wrapper.pto_kernel_source.count("@pto.jit") == 2
    first_launch = wrapper.lib_code.index("main_kernel<<<")
    second_launch = wrapper.lib_code.index("main_kernel_1<<<")
    assert first_launch < second_launch

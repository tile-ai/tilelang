import pytest

from tilelang import tvm
from tvm import tirx


def _build_vector_select_source():
    lanes = 4
    input_buffer = tirx.decl_buffer((lanes,), "float32", name="input")
    output_buffer = tirx.decl_buffer((lanes,), "float32", name="output")
    ramp = tirx.Ramp(0, 1, lanes)
    input_vector = tirx.BufferLoad(input_buffer, [ramp])
    zero = tirx.Broadcast(tirx.const(0, "float32"), lanes)
    selected = tirx.Select(input_vector > zero, input_vector, zero)
    body = tirx.BufferStore(output_buffer, selected, [ramp])
    func = tirx.PrimFunc(
        [input_buffer.data, output_buffer.data],
        body,
        buffer_map={
            input_buffer.data: input_buffer,
            output_buffer.data: output_buffer,
        },
    )
    func = func.with_attr("global_symbol", "vector_select")
    func = func.with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    mod = tvm.IRModule({"vector_select": func})
    build = tvm.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True)
    if build is None:
        pytest.skip("TileLang was built without the ROCm code generator")
    return build(mod, tvm.target.Target("rocm")).inspect_source()


def test_vector_condition_select_is_scalarized_in_hip_source():
    source = _build_vector_select_source()

    assignments = [line for line in source.splitlines() if "?" in line and "=" in line]
    assert len(assignments) == 4
    assert all("bool(" in line for line in assignments)


if __name__ == "__main__":
    test_vector_condition_select_is_scalarized_in_hip_source()

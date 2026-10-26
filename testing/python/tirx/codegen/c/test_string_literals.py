import numpy as np
import pytest

from tvm_tirx import ir, runtime, tirx


@pytest.mark.parametrize("message", ['line one\n"line two"\t\\end', "\0" + "7\x01af", "中文\ré"])
def test_string_literals_preserve_bytes(message):
    expected = np.frombuffer(message.encode("utf-8"), dtype="uint8").copy()
    source = ir.Var("source", tirx.TensorType([len(expected)], "uint8"))
    output = ir.Var("output", tirx.TensorType([1], "int32"))
    comparison = ir.Call(
        "tirx.call_pure_extern",
        [
            ir.StringImm("compare_literal_bytes"),
            ir.StringImm(message),
            source.data,
            ir.const(len(expected), "uint64"),
        ],
        ty="int32",
    )
    function = tirx.Function([source, output], ir.TensorStore(output, [0], comparison))
    function = function.with_attr("global_symbol", "compare_string")
    module = ir.IRModule({"compare_string": function}).with_attr(
        "codegen.preamble",
        """#include <stdint.h>
#ifdef __cplusplus
extern "C"
#endif
int32_t compare_literal_bytes(const char* actual, uint8_t* expected, uint64_t size) {
  for (uint64_t index = 0; index < size; ++index) {
    if ((uint8_t)actual[index] != expected[index]) return 1;
  }
  return 0;
}
""",
    )
    executable = tirx.compile(module, target="c")
    actual = runtime.empty((1,), "int32")
    executable(runtime.tensor(expected), actual)
    np.testing.assert_array_equal(actual.numpy(), [0])

"""Exercise TileLang's own C emitter against native TIRx IR and TVM FFI."""

from pathlib import Path

import numpy as np
import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.tirx.compilation_pipeline import tirx_pipeline


def build_own_c(func, tmp_path, *, retain_allocations=False):
    target = tvm_tirx.target.Target("c", host="c")
    func = func.with_attr("global_symbol", "main")
    lowering, host, _ = tirx_pipeline(target=target)
    lowered = lowering(tvm_tirx.IRModule({"main": func}))
    if retain_allocations:
        # Exercise the emitter's native allocation boundary independently of
        # LowerTVMBuiltin, which promotes non-global CPU allocations to runtime
        # workspace calls. Packed ABI lowering has already run above.
        lowered = tvm_tirx.tirx.transform.LowerIntrin()(lowered)
    else:
        lowered = host()(lowered)
    module = tvm_ffi.get_global_func("target.build.tilelang_c")(lowered, target)
    code = module.inspect_source()
    (tmp_path / "kernel.cc").write_text(code)
    (tmp_path / "lowered.py").write_text(lowered.script())
    root = Path(__file__).resolve().parents[3]
    library = tmp_path / "kernel.so"
    tvm_tirx.codegen.Executable(module).export_library(str(library), cc="c++", options=["-std=c++17", "-fopenmp", f"-I{root / 'src'}"])
    return tvm_tirx.runtime.load_module(str(library)), code


@pytest.mark.parametrize("n", [1, 17, 257])
def test_dynamic_float32_add(n, tmp_path):
    size = T.dynamic("size")

    @T.function
    def add(source: T.Tensor((size,), "float32"), output: T.Tensor((size,), "float32")):
        for i in range(size):
            output[i] = source[i] + T.float32(1.25)

    module, code = build_own_c(add, tmp_path)
    rng = np.random.default_rng(23)
    source = rng.uniform(-4, 4, n).astype("float32")
    output = tvm_tirx.runtime.tensor(np.empty_like(source))
    module(tvm_tirx.runtime.tensor(source), output)
    np.testing.assert_array_equal(output.numpy(), source + np.float32(1.25))
    assert "tl_templates/cpp/common.h" in code
    assert "TVMFFIAny" in code


def test_bool_complement(tmp_path):
    @T.function
    def complement(source: T.Tensor((5,), "int32"), output: T.Tensor((5,), "bool")):
        for i in range(5):
            output[i] = T.bitwise_not(source[i] != 0)

    module, _ = build_own_c(complement, tmp_path)
    source = np.array([-3, -1, 0, 1, 2], dtype="int32")
    output = tvm_tirx.runtime.tensor(np.empty(5, dtype="bool"))
    module(tvm_tirx.runtime.tensor(source), output)
    np.testing.assert_array_equal(output.numpy(), np.logical_not(source))


def test_parallel_collapse_and_thread_count(tmp_path):
    @T.function
    def row_add(source: T.Tensor((3, 5), "float32"), output: T.Tensor((3, 5), "float32")):
        for i in T.parallel(3):
            for j in T.parallel(5):
                output[i, j] = source[i, j] + T.float32(2)

    outer = row_add.body.seq[0]
    parallel = tvm_tirx.ir.For(outer.var, outer.begin, outer.end, outer.step, outer.kind, outer.body, annotations={"tl.cpu_num_threads": 2})
    row_add = row_add.with_body([parallel, *row_add.body.seq[1:]])
    module, code = build_own_c(row_add, tmp_path)
    assert code.count("#pragma omp parallel for") == 1
    assert "#pragma omp parallel for collapse(2) num_threads(2)" in code
    source = np.arange(15, dtype="float32").reshape(3, 5)
    output = tvm_tirx.runtime.tensor(np.empty_like(source))
    module(tvm_tirx.runtime.tensor(source), output)
    np.testing.assert_array_equal(output.numpy(), source + np.float32(2))


def test_nonfinite_float_literals(tmp_path):
    @T.function
    def fill(output: T.Tensor((3,), "float32")):
        output[0] = T.float32("inf")
        output[1] = T.float32("-inf")
        output[2] = T.float32("nan")

    module, code = build_own_c(fill, tmp_path)
    assert "INFINITY" in code and "NAN" in code
    output = tvm_tirx.runtime.tensor(np.zeros(3, dtype="float32"))
    module(output)
    np.testing.assert_array_equal(output.numpy(), np.array([np.inf, -np.inf, np.nan], dtype="float32"))


@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
@pytest.mark.parametrize("retain_allocations", [False, True])
def test_cpu_shared_allocation(scope, retain_allocations, tmp_path):
    # TileLang alloc_shared defaults to shared.dyn.  The CPU emitter treats
    # these constant-size allocations as local C arrays, as the old emitter did.
    @T.function
    def scratch_add(source: T.Tensor((17,), "float32"), output: T.Tensor((17,), "float32")):
        scratch = T.alloc_tensor(ty_args=[T.Tensor((17,), "float32", scope=scope)])
        for i in range(17):
            scratch[i] = source[i] * T.float32(2)
        for i in range(17):
            output[i] = scratch[i] + T.float32(1)

    module, code = build_own_c(scratch_add, tmp_path, retain_allocations=retain_allocations)
    if retain_allocations:
        assert "scratch_ptr[17]" in code
        assert "TVMBackendAllocWorkspace" not in code
    else:
        assert "TVMBackendAllocWorkspace" in code
        assert "TVMBackendFreeWorkspace" in code
    assert "__shared__" not in code
    source = np.arange(17, dtype="float32") - np.float32(8)
    output = tvm_tirx.runtime.tensor(np.empty_like(source))
    module(tvm_tirx.runtime.tensor(source), output)
    np.testing.assert_array_equal(output.numpy(), source * np.float32(2) + np.float32(1))

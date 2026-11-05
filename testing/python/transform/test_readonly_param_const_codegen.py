import tilelang.language as T
import re
from tilelang.engine.lower import lower
from tilelang.jit.adapter.utils import match_declare_kernel
import tilelang.testing
from tvm_tirx.tirx import Function


def _simple_add_kernel():
    @T.prim_func
    def main(
        x: T.Tensor((128,), T.float32),
        y: T.Tensor((128,), T.float32),
    ):
        # One-dimensional kernel; writes y from x without modifying x
        with T.Kernel(128, threads=32) as pid:
            y[pid] = x[pid] + 1.0

    return main


@tilelang.testing.requires_cuda
def test_codegen_emits_const_for_readonly_params():
    # Lower without device compilation to retrieve CUDA source reliably
    func = _simple_add_kernel()
    artifact = lower(func, target="cuda", enable_device_compile=False)

    src = artifact.kernel_source
    print(src)
    assert 'extern "C" __global__' in src

    # Extract kernel signature and check qualifiers
    declaration = re.match(
        r"__global__\s+void\s+(?:__launch_bounds__\([^)]*\)\s+)?(?P<name>\w+)\s*\((?P<parameters>[^)]*)\)",
        src[match_declare_kernel(src) :],
    )
    assert declaration is not None, "Expected the CUDA device entry declaration"
    signature = declaration.group("parameters")

    # Check the two device arguments in their ABI order. Native lowering can
    # name the data-pointer parameters x_ptr/y_ptr; their C++ declaration must
    # agree with the corresponding native Function parameters below.
    parameters = [parameter.strip() for parameter in signature.split(",")]
    assert len(parameters) == 2
    readonly = re.fullmatch(r"const float\s*\*\s*__restrict__\s+(\w+)", parameters[0])
    writable = re.fullmatch(r"float\s*\*\s*__restrict__\s+(\w+)", parameters[1])
    assert readonly is not None, "Expected a const, restrict-qualified input pointer"
    assert writable is not None, "Expected a writable, restrict-qualified output pointer"
    parameter_names = [readonly.group(1), writable.group(1)]

    # Also validate the function attribute carries read-only param indices
    # Expect only the first handle parameter (x) to be marked read-only
    device_mod = artifact.device_mod
    device_funcs = [f for f in device_mod.functions.values() if isinstance(f, Function)]
    entries = [
        f
        for f in device_funcs
        if str(f.attrs.get("global_symbol", "")) == declaration.group("name")
        and len(f.params) == 2
        and [parameter.name for parameter in f.params] == parameter_names
    ]
    assert len(entries) == 1, "Expected one native device entry with the x/y parameter contract"
    (pf,) = entries
    ro = pf.attrs.get("tl.readonly_param_indices")
    assert ro is not None, "Expected tl.readonly_param_indices to be present"
    ro_list = [int(i) for i in ro]
    assert ro_list == [0]


if __name__ == "__main__":
    test_codegen_emits_const_for_readonly_params()

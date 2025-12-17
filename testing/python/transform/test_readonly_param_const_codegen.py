import tilelang.language as T
from tilelang.engine.lower import lower
from tilelang.jit.adapter.utils import match_declare_kernel


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


def test_codegen_emits_const_for_readonly_params():
    # Lower without device compilation to retrieve CUDA source reliably
    func = _simple_add_kernel()
    artifact = lower(func, target="cuda", enable_device_compile=False)

    src = artifact.kernel_source
    print(src)
    assert 'extern "C" __global__' in src

    # Extract kernel signature and check qualifiers
    lparen = match_declare_kernel(src)
    rparen = src.find(")", lparen)
    assert rparen != -1
    signature = src[lparen:rparen]

    # x is read-only: should be `const` and `__restrict__`
    assert "const float* __restrict__" in signature
    # y is written: must not be const, but still `__restrict__` due to noalias
    # We ensure there is a non-const float* parameter with __restrict__ as well
    assert "const float* __restrict__ x" in src or "const float *__restrict__ x" in src
    assert " float* __restrict__ y" in src or " float *__restrict__ y" in src

    # Also validate the function attribute carries read-only param indices
    # Expect only the first handle parameter (x) to be marked read-only
    device_mod = artifact.device_mod
    prim_funcs = [f for f in device_mod.functions.values() if hasattr(f, "attrs")]
    assert prim_funcs, "No PrimFunc found in device module"
    pf = prim_funcs[0]
    ro = pf.attrs.get("tl.readonly_param_indices")
    assert ro is not None, "Expected tl.readonly_param_indices to be present"
    ro_list = [int(i) for i in ro]
    assert 0 in ro_list and 1 not in ro_list


if __name__ == "__main__":
    test_codegen_emits_const_for_readonly_params()

import pytest
from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
from tilelang.intrinsics.hexagon import hmx


def has_hexagon_codegen():
    try:
        if not tvm.runtime.enabled("llvm"):
            return False
        tvm.target.Target("llvm -mtriple=hexagon -mcpu=hexagonv73")
        return True
    except Exception:
        return False


def build_hmx_matmul(M, N, K):
    @T.prim_func
    def main(
        A_host: T.Tensor((M, K), "int8"),
        B_host: T.Tensor((K, N), "int8"),
        C_host: T.Tensor((M, N), "int32"),
    ):
        A_vtcm = T.alloc_fragment((M, K), "int8", scope="global.vtcm")
        B_vtcm = T.alloc_fragment((K, N), "int8", scope="global.vtcm")
        C_acc  = T.alloc_fragment((M, N), "int32", scope="global.hmx.acc")

        for i, k in T.grid(M, K):
            A_vtcm[i, k] = A_host[i, k]
        for k, j in T.grid(K, N):
            B_vtcm[k, j] = B_host[k, j]
        for i, j in T.grid(M, N):
            C_acc[i, j] = T.cast(0, "int32")

        hmx.mma(A_vtcm, B_vtcm, C_acc)

        for i, j in T.grid(M, N):
            C_host[i, j] = C_acc[i, j]

    return main


# Diagnostics (always run, no skipif)
def test_000_environment():
    """Report the full environment so we know exactly what we're working with."""
    print("\n")
    print("=" * 60)
    print("ENVIRONMENT REPORT")
    print("=" * 60)
    print(f"  tvm.__file__         : {tvm.__file__}")
    print(f"  tvm.__version__      : {tvm.__version__}")
    print(f"  llvm enabled         : {tvm.runtime.enabled('llvm')}")
    print(f"  has_hexagon_codegen(): {has_hexagon_codegen()}")

    try:
        t = tvm.target.Target("llvm -mtriple=hexagon -mcpu=hexagonv73")
        print(f"  hexagon target       : OK → {t}")
    except Exception as e:
        print(f"  hexagon target       : FAILED → {e}")
    print("=" * 60)


@pytest.mark.skipif(not has_hexagon_codegen(), reason="Hexagon LLVM not available")
def test_001_ir_dump():
    """Dump the full kernel_source so we can see what was actually generated."""
    M, N, K = 32, 32, 32
    func   = build_hmx_matmul(M, N, K)
    target = tvm.target.Target("llvm -mtriple=hexagon -mcpu=hexagonv73")
    kernel = tl.compile(func, target=target)
    ir     = kernel.kernel_source

    print("\n")
    print("=" * 60)
    print("FULL KERNEL SOURCE")
    print("=" * 60)
    print(ir)
    print("=" * 60)

    # Report which assertions would pass/fail without actually asserting
    checks = {
        'target triple = "hexagon"'  : 'target triple',
        "A_vtcm"                     : 'VTCM alloc A',
        "B_vtcm"                     : 'VTCM alloc B',
        "C_acc"                      : 'HMX accumulator',
        "hmx_mma_placeholder"        : 'placeholder NOT lowered (bad)',
        "HexKL_mma_i8acc32"          : 'HexKL intrinsic (good)',
        "HexKL_mma_i8i32"            : 'HexKL alt spelling',
        "call_extern"                : 'any call_extern',
        "llvm.hexagon"               : 'LLVM hexagon intrinsic',
    }

    print("\nASSERTION PROBE RESULTS:")
    all_good = True
    for needle, label in checks.items():
        found = needle in ir
        status = "✓ FOUND  " if found else "✗ MISSING"
        print(f"  {status}  [{label}]  '{needle}'")
        if label == 'placeholder NOT lowered (bad)' and found:
            all_good = False
        if label in ('HexKL intrinsic (good)', 'HexKL alt spelling') and found:
            pass  # at least one should be present

    print()
    # Only hard-assert on things we're sure about
    assert 'target triple = "hexagon"' in ir, \
        "Not even targeting Hexagon — target string is wrong or codegen didn't run"


@pytest.mark.skipif(not has_hexagon_codegen(), reason="Hexagon LLVM not available")  
def test_002_hmx_lowering_status():
    """Specifically check whether HMX intrinsics were lowered or are still placeholders."""
    M, N, K = 32, 32, 32
    func   = build_hmx_matmul(M, N, K)
    target = tvm.target.Target("llvm -mtriple=hexagon -mcpu=hexagonv73")
    kernel = tl.compile(func, target=target)
    ir     = kernel.kernel_source

    placeholder_present = "hmx_mma_placeholder" in ir
    hexkl_present       = any(s in ir for s in [
        "HexKL_mma_i8acc32",
        "HexKL_mma_i8i32",
        "HexKL_mma",
    ])
    llvm_intrin_present = "llvm.hexagon" in ir

    print(f"\n  placeholder still in IR : {placeholder_present}")
    print(f"  HexKL intrinsic in IR   : {hexkl_present}")
    print(f"  llvm.hexagon in IR      : {llvm_intrin_present}")

    if placeholder_present:
        pytest.fail(
            "hmx_mma_placeholder was NOT lowered.\n"
            "_lower_hexagon_intrinsics is not wired into the compile pipeline.\n"
            "Check lower() in tilelang/engine/lower.py"
        )
    elif not hexkl_present and not llvm_intrin_present:
        pytest.fail(
            "HMX placeholder is gone but no HexKL/llvm.hexagon intrinsic was emitted.\n"
            "The lowering pass may be silently dropping the MMA op.\n"
            "Check LowerHMXIntrinsics implementation."
        )
    else:
        print("  ✓ HMX intrinsics correctly lowered")
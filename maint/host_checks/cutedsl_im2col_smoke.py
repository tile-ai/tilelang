"""Small CuTeDSL im2col/TMA smoke test using the convolution example."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import tilelang


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "examples" / "convolution"))

    import example_convolution

    n, c, h, w, f, k, s, d, p = 1, 128, 8, 8, 128, 3, 1, 1, 1
    block_m, block_n, block_k, num_stages, threads = 64, 128, 32, 3, 256

    prim_func = example_convolution.convolution.get_tir(
        N=n,
        C=c,
        H=h,
        W=w,
        F=f,
        K=k,
        S=s,
        D=d,
        P=p,
        block_M=block_m,
        block_N=block_n,
        block_K=block_k,
        num_stages=num_stages,
        threads=threads,
    )
    kernel = tilelang.compile(
        prim_func,
        target="cutedsl",
        execution_backend="cutedsl",
    )
    print(f"adapter={type(kernel.adapter).__name__}")
    pymodule = getattr(kernel.adapter, "pymodule", None)
    print(f"has_tma_descs={getattr(pymodule, '_has_tma_descs', None)}")
    print(f"cutlass_host_launcher_supported={getattr(pymodule, '_cutlass_host_launcher_supported', None)}")
    print(f"cutlass_host_launcher_disabled_reason={getattr(pymodule, '_cutlass_host_launcher_disabled_reason', None)}")
    source = kernel.get_kernel_source(kernel_only=False) or ""
    print(f"source_has_im2col_offsets={'im2col_offsets' in source}")

    a = torch.randn(n, h, w, c, device="cuda", dtype=torch.float16)
    b = torch.randn(k, k, c, f, device="cuda", dtype=torch.float16)
    out = kernel(a, b)
    ref = example_convolution.ref_program(s, p, d)(a, b)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print("All checks passed.")


if __name__ == "__main__":
    main()

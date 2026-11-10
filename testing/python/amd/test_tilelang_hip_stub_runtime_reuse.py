"""Regression test for tile-ai/tilelang#3711.

PyTorch's ROCm wheels load their bundled ``libamdhip64`` with ``RTLD_LOCAL``,
so the HIP stub cannot see it through ``RTLD_DEFAULT``. It used to fall back to
``dlopen("libamdhip64.so")`` and load a second runtime from the system ROCm,
after which ``hipModuleGetFunction`` failed with "named symbol not found".
The stub must reuse a runtime that is already mapped into the process.
"""

import subprocess
import sys
import textwrap

import pytest

import tilelang.testing

_SCRIPT = textwrap.dedent(
    """
    import os
    import torch
    import tilelang
    import tilelang.language as T

    @tilelang.jit
    def add(N=1024, block=128):
        @T.prim_func
        def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
            with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
                for i in T.Parallel(block):
                    C[bx * block + i] = A[bx * block + i] + B[bx * block + i]

        return main

    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    c = torch.empty_like(a)
    add()(a, b, c)
    torch.cuda.synchronize()
    assert torch.allclose(c, a + b)

    with open("/proc/self/maps") as f:
        libs = {os.path.realpath(line.split()[-1]) for line in f if "libamdhip64" in line}
    print("HIP_RUNTIMES", sorted(libs))
    assert len(libs) == 1, libs
    """
)


@tilelang.testing.requires_rocm
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc/self/maps")
def test_hip_stub_reuses_loaded_runtime(tmp_path):
    # tilelang.jit reads the kernel source, so the script has to be a real file.
    script = tmp_path / "hip_stub_runtime_reuse_kernel.py"
    script.write_text(_SCRIPT)
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    tilelang.testing.main()

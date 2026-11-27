"""GEMM operand shape checks must raise even under python -O."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

import tilelang.language as T
import tilelang.testing


def _mismatched_k_gemm():
    """Build a prim_func whose A/B K dimensions disagree under transpose_B."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 32), "float16"),
        B: T.Tensor((64, 64), "float16"),
        C: T.Tensor((64, 64), "float32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((64, 32), "float16")
            b = T.alloc_shared((64, 64), "float16")
            c = T.alloc_fragment((64, 64), "float32")
            T.copy(A, a)
            T.copy(B, b)
            # transpose_B=True -> K = A.shape[-1] = 32, K_B = B.shape[-1] = 64
            T.gemm(a, b, c, transpose_B=True)
            T.copy(c, C)

    return main


def test_gemm_mismatched_k_raises_value_error():
    with pytest.raises(ValueError, match=r"T\.gemm K shape check failed: K_A = 32, K_B = 64"):
        _mismatched_k_gemm()


def test_gemm_mismatched_k_raises_under_python_optimize(tmp_path):
    # Bare assert disappears under -O; ValueError must still fire.
    # Write a real file so T.prim_func can recover source lines.
    script = textwrap.dedent(
        """\
        import tilelang.language as T

        @T.prim_func
        def main(
            A: T.Tensor((64, 32), "float16"),
            B: T.Tensor((64, 64), "float16"),
            C: T.Tensor((64, 64), "float32"),
        ):
            with T.Kernel(1):
                a = T.alloc_shared((64, 32), "float16")
                b = T.alloc_shared((64, 64), "float16")
                c = T.alloc_fragment((64, 64), "float32")
                T.copy(A, a)
                T.copy(B, b)
                T.gemm(a, b, c, transpose_B=True)
                T.copy(c, C)
        """
    )
    script_path = tmp_path / "mismatched_k_gemm.py"
    script_path.write_text(script)
    env = os.environ.copy()
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [repo_root, env.get("PYTHONPATH", "")]))
    result = subprocess.run(
        [sys.executable, "-O", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "ValueError" in result.stderr
    assert "T.gemm K shape check failed: K_A = 32, K_B = 64" in result.stderr
    assert "AssertionError" not in result.stderr


if __name__ == "__main__":
    tilelang.testing.main()

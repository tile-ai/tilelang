"""Metal unrolls short constant loops by default.

Apple's shader compiler does not unroll the loops TileLang emits on its own, so
the Metal pipeline marks short constant loops as unrolled and the codegen emits
``TILELANG_PRAGMA_UNROLL`` for them, unless the caller configures
``tl.UnrollLoop``.
"""

import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

ROWS = 96
WIDTH = 4
ROLLED = {"tl.UnrollLoop": {"auto_max_step": 0, "auto_max_extent": 0, "explicit_unroll": False}}


def row_sums():
    @T.prim_func
    def main(A: T.Tensor((ROWS, WIDTH), "float32"), B: T.Tensor((ROWS,), "float32")):
        with T.Kernel(T.ceildiv(ROWS, 32), threads=32) as block:
            i = block * 32 + T.get_thread_binding(0)
            if i < ROWS:
                B[i] = 0.0
                # A carried dependence keeps the loop out of the vectorizer.
                for j in T.serial(WIDTH):
                    B[i] = B[i] + A[i, j] * (j + 1)

    return main


def kernel_body(pass_configs=None) -> str:
    with tvm.target.Target("metal"), tvm.transform.PassContext(opt_level=3, config=pass_configs):
        source = tilelang.lower(row_sums(), target="metal").kernel_source
    return source[source.index("main_kernel(") :]


def check(kernel) -> None:
    a = torch.arange(ROWS * WIDTH, dtype=torch.float32, device="mps").reshape(ROWS, WIDTH)
    b = torch.zeros(ROWS, device="mps")
    kernel(a, b)
    torch.mps.synchronize()
    weights = torch.arange(1, WIDTH + 1, dtype=torch.float32)
    torch.testing.assert_close(b.cpu(), (a.cpu() * weights).sum(1))


UNROLLED_LOOP = re.compile(r"TILELANG_PRAGMA_UNROLL\s*\n\s*for \(")


def test_short_constant_loops_are_unrolled_by_default():
    body = kernel_body()
    assert len(re.findall(r"\bfor \(", body)) == 1
    assert UNROLLED_LOOP.search(body) is not None


def test_caller_pass_configuration_takes_precedence():
    body = kernel_body(ROLLED)
    assert len(re.findall(r"\bfor \(", body)) == 1
    assert UNROLLED_LOOP.search(body) is None


@tilelang.testing.requires_metal
@pytest.mark.parametrize("pass_configs", [None, ROLLED], ids=["unrolled", "rolled"])
def test_row_sums_execute(pass_configs):
    kernel = tilelang.compile(row_sums(), target="metal", execution_backend="torch", pass_configs=pass_configs)
    check(kernel)


if __name__ == "__main__":
    tilelang.testing.main()

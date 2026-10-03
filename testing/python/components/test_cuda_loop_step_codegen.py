"""The CUDA/HIP loop emitters have to honour `ForNode::step`.

Written through the raw TVMScript frontend, because that is the one that hands a
`For` node with a non-unit step straight to codegen -- the tilelang frontend folds
the step into the index expression instead:

    for _tmp in range(4):
        i = _tmp * 2

The base `CodeGenC` already emits `vid += step`; the TileLang GPU overrides built
the header from `min` and `extent` alone and always printed `++vid`.
"""

import re

import torch

import tilelang
import tilelang.testing
from tvm.script import tirx as T

N = 8
STEP = 2


@T.prim_func
def stepped_serial(B: T.Buffer((N,), "int32")):
    for _bx in T.thread_binding(1, thread="blockIdx.x"):
        for _tx in T.thread_binding(1, thread="threadIdx.x"):
            for i in T.serial(0, N, step=STEP):
                B[i] = 1


def _loop_headers(source):
    return [line.strip() for line in source.splitlines() if re.search(r"for \(.*;.*;.*\)", line)]


@tilelang.testing.requires_cuda
def test_stepped_serial_loop_emits_the_step():
    source = tilelang.lower(stepped_serial, target="cuda").kernel_source
    headers = _loop_headers(source)
    assert headers, f"no loop emitted:\n{source}"
    assert any(f"+= {STEP}" in header for header in headers), headers


@tilelang.testing.requires_cuda
def test_stepped_serial_loop_visits_only_every_step_th_index():
    B = torch.zeros(N, dtype=torch.int32, device="cuda")
    tilelang.compile(stepped_serial, target="cuda")(B)
    torch.cuda.synchronize()
    written = [i for i, value in enumerate(B.cpu().tolist()) if value == 1]
    assert written == list(range(0, N, STEP))


@tilelang.testing.requires_cuda
def test_unit_step_loops_still_emit_a_plain_increment():
    """Control: an absent step keeps the `++vid` spelling."""

    @T.prim_func
    def plain(B: T.Buffer((N,), "int32")):
        for _bx in T.thread_binding(1, thread="blockIdx.x"):
            for _tx in T.thread_binding(1, thread="threadIdx.x"):
                for i in T.serial(0, N):
                    B[i] = 1

    headers = _loop_headers(tilelang.lower(plain, target="cuda").kernel_source)
    assert headers, "no loop emitted"
    assert any("++" in header for header in headers), headers


if __name__ == "__main__":
    tilelang.testing.main()

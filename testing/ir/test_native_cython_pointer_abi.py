"""Raw pointer calls retain their ABI after the frontend constructs tensor types."""

import pytest
import torch

import tilelang
import tilelang.language as T


@pytest.mark.parametrize("raw_pointers", [False, True])
@pytest.mark.parametrize("offset", [0, 3])
def test_cython_tensor_address_and_scalar_arguments(raw_pointers, offset):
    @T.prim_func
    def main(a: T.ptr, b: T.ptr, n: T.int32, bias: T.float32):
        A = T.make_tensor(a, (n,), T.float32)
        B = T.make_tensor(b, (n,), T.float32)
        with T.Kernel(1, threads=1):
            for i in T.serial(n):
                B[i] = A[i] + bias

    kernel = tilelang.compile(
        main,
        target="c",
        target_host={"kind": "llvm", "mcpu": "generic"},
        execution_backend="cython",
    )
    source = torch.arange(offset + 7, dtype=torch.float32)[offset:]
    storage = torch.full((offset + 7 + 2,), -101.0, dtype=torch.float32)
    output = storage[offset : offset + 7]
    arguments = [source.data_ptr(), output.data_ptr()] if raw_pointers else [source, output]
    kernel(*arguments, 7, 1.5)
    torch.testing.assert_close(output, source + 1.5, rtol=0, atol=0)
    assert torch.all(storage[:offset] == -101)
    assert torch.all(storage[offset + 7 :] == -101)

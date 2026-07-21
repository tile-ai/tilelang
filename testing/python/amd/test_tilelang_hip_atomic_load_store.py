import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


_MEMORY_ORDER_IDS = {
    "relaxed": 0,
    "consume": 1,
    "acquire": 2,
    "release": 3,
    "seq_cst": 5,
}


def _atomic_load_store_kernel(load_order, store_order):
    @T.prim_func
    def kernel(source: T.Tensor((32,), T.int32), destination: T.Tensor((32,), T.int32)):
        with T.Kernel(1, threads=32):
            index = T.get_thread_binding()
            value = T.atomic_load(source[index], memory_order=load_order)
            T.atomic_store(destination[index], value, memory_order=store_order)

    return kernel


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(
    "load_order,store_order",
    [
        ("relaxed", "relaxed"),
        ("consume", "release"),
        ("acquire", "seq_cst"),
        ("seq_cst", "relaxed"),
    ],
)
def test_atomic_load_store_hip(load_order, store_order):
    kernel = tilelang.compile(
        _atomic_load_store_kernel(load_order, store_order),
        out_idx=[1],
        target="hip",
    )
    source = kernel.get_kernel_source()

    load_pattern = rf"AtomicLoad\([^;\n]*, {_MEMORY_ORDER_IDS[load_order]}\)"
    store_pattern = rf"AtomicStore\([^;\n]*, {_MEMORY_ORDER_IDS[store_order]}\);"
    assert re.search(load_pattern, source)
    assert re.search(store_pattern, source)

    input_tensor = torch.arange(32, dtype=torch.int32, device="cuda")
    output_tensor = kernel(input_tensor)

    torch.testing.assert_close(output_tensor, input_tensor)


if __name__ == "__main__":
    tilelang.testing.main()

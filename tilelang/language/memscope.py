# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from tvm._ffi.registry import register_func
from tvm.ir import make_node


@register_func("tvm.info.mem.local.var")
def mem_info_local_var():
    return make_node(
        "MemoryInfo",
        unit_bits=8,
        max_num_bits=64,
        max_simd_bits=128,
        head_address=None,
    )

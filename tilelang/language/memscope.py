from tvm.ffi.registry import register_func
from tvm.ir import make_node


@register_func("tvm.info.mem.local.var")
def mem_info_local_var():
    """Get memory information for local variable memory.

    Returns:
        tvm.ir.make_node: A node containing memory information
    """
    return make_node(
        "target.MemoryInfo",
        unit_bits=8,
        max_num_bits=64,
        max_simd_bits=128,
        head_address=None,
    )

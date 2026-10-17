"""Shared buffer-reuse fixtures and alias-map inspection."""

import tilelang.ascend.language as T
from tilelang.ascend import transform
from tilelang import tvm
from testing.ascend._ir import schedule, nodes


def _make_cyclic_scalar_program(interleave: bool = False, versions: int = 1):
    @T.prim_func
    def main(
        x: T.Tensor((8,), "int32"),
        y: T.Tensor((8,), "int32"),
        out_x: T.Tensor((8,), "int32"),
        out_y: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1):
            a = T.alloc_shared((8,), "int32")
            b = T.alloc_shared((8,), "int32")
            if versions > 1:
                T.annotate_buffer_versions({a: versions, b: versions})
            for k in T.Pipelined(8, num_stages=1):
                a[0] = x[k]
                if interleave:
                    b[0] = y[k]
                out_x[k] = a[0]
                if not interleave:
                    b[0] = y[k]
                out_y[k] = b[0]

    return main


def _insert_sync_alias_pairs(program, *, disable_reuse=False) -> set[tuple[str, str]]:
    with tvm.transform.PassContext(config={"tl.disable_shared_memory_reuse": disable_reuse}):
        before = schedule(program, through="ResolveCore", auto_schedule=False)
        mod = transform.InsertSync()(before)
    pairs = set()
    for block in nodes(mod, tvm.tirx.SBlock):
        for storage, compatible in block.annotations.get("tl.buffer_alias_map", {}).items():
            for other in compatible:
                pairs.add(tuple(sorted((storage.name, other.name))))
    return pairs

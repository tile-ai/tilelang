import tilelang
import tilelang.language as T
from tvm import tirx


@tilelang.jit
def _constant_rebind_after_loop(n: int):

    @T.prim_func
    def main(A: T.Tensor([n], T.float32)):
        with T.Kernel(1, threads=1) as block_id:
            for w in T.serial(1):
                tile_id = block_id + w
                bid = tile_id // 1
                if bid < n:
                    A[0] = 1.0

            T.sync_grid()

            for w in T.serial(1):
                tile_id = block_id + w
                hid = tile_id // n
                bid = tile_id % 1
                if bid < n and hid < n:
                    A[bid] = 2.0

    return main


def test_constant_rebind_shadows_stale_tir_scope():
    prim_func = _constant_rebind_after_loop.get_tir(4)
    ops = set()

    def visit(node):
        if isinstance(node, tirx.Call):
            ops.add(getattr(node.op, "name", str(node.op)))

    tirx.stmt_functor.post_order_visit(prim_func.body, visit)

    assert "tl.sync_grid" in ops

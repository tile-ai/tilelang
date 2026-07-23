"""Tests for TL_PIPELINE_COPY_STRATEGY pass config."""

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
from tvm.tirx.stmt_functor import post_order_visit

M, N, K = 512, 512, 512
block_M, block_K, block_N = 64, 64, 64


@T.prim_func
def kernel(
    A: T.Tensor((M, K), T.float16),
    B: T.Tensor((K, N), T.float16),
    Out_A: T.Tensor((M, K), T.float16),
    Out_B: T.Tensor((K, N), T.float16),
    Out_A2: T.Tensor((M, K), T.float16),
):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_s = T.alloc_shared((block_M, block_K), T.float16)
        B_s = T.alloc_shared((block_K, block_N), T.float16)

        for k in T.Pipelined(K // block_K, num_stages=2):
            T.copy(A[by * block_M, k * block_K], A_s)  # stmt 0: copy A
            T.copy(B[k * block_K, bx * block_N], B_s)  # stmt 1: copy B
            T.copy(A_s, Out_A[by * block_M, k * block_K])  # stmt 2: first_use of A
            T.copy(B_s, Out_B[k * block_K, bx * block_N])  # stmt 3: first&last use of B
            T.copy(A_s, Out_A2[by * block_M, k * block_K])  # stmt 4: last_use of A


def run_pipeline_planning(strategy: str):
    """Run PipelinePlanning with given strategy and return (stages, orders)."""
    target = tvm.target.Target("cuda")
    with tvm.transform.PassContext(
        config={
            "tl.pipeline_copy_strategy": strategy,
        }
    ):
        mod = tvm.IRModule.from_expr(kernel.with_attr("global_symbol", "main"))
        mod = tvm.tirx.transform.BindTarget(target)(mod)
        mod = tl.transform.MaterializeKernelLaunch()(mod)
        mod = tl.transform.IfStmtBinding()(mod)
        mod = tl.transform.PipelinePlanning()(mod)

    annotations = []

    def visit(node):
        if isinstance(node, tvm.tirx.For) and "software_pipeline_stage" in node.annotations:
            annotations.append(node.annotations)

    post_order_visit(mod["main"].body, visit)

    assert annotations, f"No pipeline annotations found for strategy '{strategy}'"
    anno = annotations[0]
    stages = [int(s) for s in anno["software_pipeline_stage"]]
    orders = [int(o) for o in anno["software_pipeline_order"]]
    return stages, orders


def test_tilelang_transform_pipeline_copy_strategy():
    """Test pipeline copy strategies via IR annotations."""
    stages_occ, orders_occ = run_pipeline_planning("occupancy")
    stages_lat, orders_lat = run_pipeline_planning("latency")
    stages_bal, orders_bal = run_pipeline_planning("balance")

    # occupancy: copies placed by last_use (B.last_use=3 < A.last_use=4, so B first)
    assert stages_occ == [0, 0, 2, 2, 2]
    assert orders_occ == [4, 2, 0, 1, 3]

    # latency: copies placed by first_use (A.first_use=2 < B.first_use=3, so A first)
    assert stages_lat == [0, 0, 2, 2, 2]
    assert orders_lat == [1, 3, 0, 2, 4]

    # balance: first_use ordering + last_use chain constraint
    assert stages_bal == [0, 0, 1, 1, 1]
    assert orders_bal == [0, 1, 2, 3, 4]

    print("All assertions passed!")


if __name__ == "__main__":
    test_tilelang_transform_pipeline_copy_strategy()

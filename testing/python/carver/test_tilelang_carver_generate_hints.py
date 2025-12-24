import tilelang.testing
from tilelang import carver
from tilelang.carver.roller import PrimFuncNode, OutputNode, Edge
from tilelang.carver.arch import auto_infer_current_arch
from tvm import te
from tilelang.language import dtypes as T


def run_general_matmul_emit_configs(M, N, K, topk: int = 20):
    arch = auto_infer_current_arch()

    def gemm(M, N, K):
        A = te.placeholder((M, K), name="A", dtype=T.float16)
        B = te.placeholder((N, K), name="B", dtype=T.float16)

        # Describe the matrix multiplication in TE
        k = te.reduce_axis((0, K), name="k")

        C = te.compute((M, N), lambda i, j: te.sum(A[i, k].astype(T.float16) * B[j, k].astype(T.float16), axis=[k]), name="C")

        return A, B, C

    arg1 = gemm(M, N, K)
    args = arg1

    func = te.create_prim_func(args)

    tensorized_func, tags = carver.utils.get_tensorized_func_and_tags(func, arch.target)
    print(tags)
    policy = carver.TensorCorePolicy.from_prim_func(func=tensorized_func, arch=arch, tags=tags, name="matmul_0")

    hints = policy.emit_config(topk=topk)

    for hint in hints:
        print(hint)

    assert len(hints) > 0, "Hints length is zero"

    prim_func_node = PrimFuncNode(tensorized_func, name="matmul_1")
    output_nodes = [OutputNode(prim_func_node)]
    policy = carver.TensorCorePolicy.from_output_nodes(output_nodes, arch=arch, tags=tags)

    hints = policy.emit_config(topk=10)

    for config in hints:
        print(config)

    assert len(hints) > 0, "Hints length is zero"


def test_general_matmul_emit_configs():
    run_general_matmul_emit_configs(128, 128, 128)


def run_general_matmul_matmul_emit_configs(M, N, K, topk: int = 20):
    arch = auto_infer_current_arch()

    def gemm(M, N, K):
        A = te.placeholder((M, K), name="A", dtype=T.float16)
        B = te.placeholder((N, K), name="B", dtype=T.float16)

        # Describe the matrix multiplication in TE
        k = te.reduce_axis((0, K), name="k")

        C = te.compute((M, N), lambda i, j: te.sum(A[i, k].astype(T.float16) * B[j, k].astype(T.float16), axis=[k]), name="C")

        return A, B, C

    arg1 = gemm(M, N, K)
    args = arg1

    func = te.create_prim_func(args)

    tensorized_func, tags = carver.utils.get_tensorized_func_and_tags(func, arch.target)
    print(tags)

    node_0 = PrimFuncNode(tensorized_func, name="matmul_0")
    node_1 = PrimFuncNode(tensorized_func, name="matmul_1")

    edge = Edge(node_0, node_1, 0, 0)
    node_0._out_edges.append(edge)
    node_1.set_inputs(0, edge)

    output_nodes = [OutputNode(node_1)]
    policy = carver.TensorCorePolicy.from_output_nodes(output_nodes, arch=arch, tags=tags)

    hints = policy.emit_config(topk=topk)

    for config in hints:
        print(config)

    assert len(hints) > 0, "Hints length is zero"


def test_general_matmul_matmul_emit_configs():
    run_general_matmul_matmul_emit_configs(128, 128, 128)


if __name__ == "__main__":
    tilelang.testing.main()

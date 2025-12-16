from dataclasses import dataclass
from .base import BaseTemplate
from tvm import te
from ..arch import TileDevice
from ..roller import Hint
from ..roller import PrimFuncNode, OutputNode, Edge
from ..utils import get_roller_hints_from_output_nodes, get_tensorized_func_and_tags


@dataclass
class FlashAttentionTemplate(BaseTemplate):
    _output_nodes: list[OutputNode] = None

    # Operation-related configuration parameters
    batch_size: int = 1
    num_heads: int = 1
    head_dim: int = 1
    seq_length: int = 1
    seq_kv_length: int = 1

    is_causal: bool = False

    in_dtype: str = "float16"
    out_dtype: str = "float16"
    accum_dtype: str = "float16"

    def get_hardware_aware_configs(self, arch: TileDevice = None, topk: int = 10) -> list[Hint]:
        """
        Retrieves optimized hardware-aware configurations.

        Args:
            arch (TileDevice, optional): The target hardware architecture.
            topk (int, optional): Number of top configurations to consider.

        Returns:
            List[Hint]: A list of optimization hints for hardware acceleration.
        """
        roller_hints = get_roller_hints_from_output_nodes(self.output_nodes, arch=arch, topk=topk)
        return roller_hints

    def initialize_function(self) -> None:
        """
        Defines and initializes the matrix multiplication computation.

        This method sets up placeholders for input matrices, computes
        the matrix multiplication using TVM's compute API,
        and optionally applies bias and type casting.

        Raises:
            AssertionError: If M, N, or K are not positive integers.
        """
        batch_size = self.batch_size
        num_heads = self.num_heads
        head_dim = self.head_dim
        seq_length = self.seq_length
        seq_kv_length = self.seq_kv_length

        in_dtype = self.in_dtype
        out_dtype = self.out_dtype
        accum_dtype = self.accum_dtype

        # Equalize the input shaps into a matmul shape
        QK_B, QK_M, QK_N, QK_K = batch_size * num_heads, seq_length, seq_kv_length, head_dim
        SV_B, SV_M, SV_N, SV_K = batch_size * num_heads, seq_length, head_dim, seq_kv_length

        # Define tensor shapes based on transpose flags
        def create_matmul(B, M, N, K):
            # Define tensor shapes based on transpose flags
            input_shape = (B, M, K)
            weight_shape = (B, N, K)
            output_shape = (B, M, N)  # Shape of output matrix C

            # Create TVM placeholders for input tensors
            A = te.placeholder(input_shape, name="A", dtype=in_dtype)  # Input matrix A
            B = te.placeholder(weight_shape, name="B", dtype=in_dtype)  # Weight matrix B

            # Define a reduction axis for matrix multiplication
            k = te.reduce_axis((0, K), name="k")

            def _compute_matmul(b, i, j):
                """
                Compute function for matrix multiplication.

                Args:
                    i (int): Row index.
                    j (int): Column index.

                Returns:
                    Computed value for C[i, j] as a sum over the reduction axis.
                """
                A_indices = [b, i, k]
                B_indices = [b, j, k]
                return te.sum(A[tuple(A_indices)].astype(accum_dtype) * B[tuple(B_indices)].astype(accum_dtype), axis=k)

            # Compute matrix multiplication result
            C = te.compute(
                output_shape,
                fcompute=_compute_matmul,
                name="C",
            )

            # Optionally cast the output to a different type
            if out_dtype != accum_dtype:
                C = te.compute(
                    output_shape,
                    lambda b, i, j: C[b, i, j].astype(out_dtype),
                    name="D",
                )

            args = [A, B, C]
            return te.create_prim_func(args)

        MMA0_prim_func = create_matmul(QK_B, QK_M, QK_N, QK_K)
        MMA1_prim_func = create_matmul(SV_B, SV_M, SV_N, SV_K)

        self.set_function([MMA0_prim_func, MMA1_prim_func])

        def create_node_from_function(func, name):
            tensorized_func, tags = get_tensorized_func_and_tags(func, self.arch.target)
            assert tags is not None
            return PrimFuncNode(tensorized_func, name=name, tags=tags)

        node_0 = create_node_from_function(MMA0_prim_func, name="MMA0")
        node_1 = create_node_from_function(MMA1_prim_func, name="MMA1")

        # connect the two nodes
        edge = Edge(node_0, node_1, 0, 0)
        node_0._out_edges.append(edge)
        node_1.set_inputs(0, edge)

        output_nodes = [OutputNode(node_1)]
        self.set_output_nodes(output_nodes)

    def params_as_dict(self):
        """
        Returns the template parameters as a dictionary.

        Returns:
            dict: Dictionary containing template parameter values.
        """
        return {
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "trans_A": self.trans_A,
            "trans_B": self.trans_B,
            "in_dtype": self.in_dtype,
            "out_dtype": self.out_dtype,
            "accum_dtype": self.accum_dtype,
            "with_bias": self.with_bias,
        }

    @property
    def class_attributes(self):
        """
        Returns the class attributes in dictionary form.

        Returns:
            dict: Dictionary of class attributes.
        """
        return self.params_as_dict()

    def __repr__(self) -> str:
        """
        Returns a string representation of the class instance.

        Returns:
            str: A formatted string representation of the class.
        """
        cls_name = self.__class__.__name__
        fields = self.class_attributes
        field_str = ", ".join(f"{key}={value!r}" for key, value in fields.items())
        return f"{cls_name}({field_str})"

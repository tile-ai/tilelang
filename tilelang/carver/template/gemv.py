from dataclasses import dataclass
from .base import BaseTemplate
from tvm import te
from ..arch import TileDevice
from ..roller import Hint
from ..utils import get_roller_hints_from_func


@dataclass
class GEMVTemplate(BaseTemplate):
    """
    A template for Generalized Matrix-Vector Multiplication (GEMV).

    This template defines the computation for a matrix-vector multiplication
    with configurable parameters such as transposition, data types, and bias addition.
    """

    # Operation-related configuration parameters
    N: int = None  # Number of columns in matrix B (output width)
    K: int = None  # Number of rows in matrix B (input width)
    trans_B: bool = True  # Whether to transpose matrix B
    in_dtype: str = "float16"  # Input data type
    out_dtype: str = "float16"  # Output data type
    accum_dtype: str = "float16"  # Accumulation data type
    with_bias: bool = False  # Whether to add a bias term

    def get_hardware_aware_configs(self, arch: TileDevice = None, topk: int = 10) -> list[Hint]:
        """
        Retrieves optimized hardware-aware configurations.

        Args:
            arch (TileDevice, optional): The target hardware architecture.
            topk (int, optional): Number of top configurations to consider.

        Returns:
            List[Hint]: A list of optimization hints for hardware acceleration.
        """
        roller_hints = get_roller_hints_from_func(self._func, arch=arch, topk=topk)
        return roller_hints

    def initialize_function(self) -> None:
        """
        Defines and initializes the GEMV computation function.

        This method sets up placeholders for input matrices, computes
        the matrix-vector multiplication using TVM's compute API,
        and optionally applies bias and type casting.
        """
        M: int = 1  # Fixed M value, representing a single batch dimension
        N, K = self.N, self.K

        # Ensure M, N, K are valid positive integers
        assert isinstance(M, int) and isinstance(N, int) and isinstance(K, int), "Only Support Integer M, N, K"
        assert M > 0 and N > 0 and K > 0, "M, N, K should be positive"

        # Load configuration parameters
        trans_B = self.trans_B
        in_dtype, out_dtype, accum_dtype = self.in_dtype, self.out_dtype, self.accum_dtype
        with_bias = self.with_bias

        # Define tensor shapes
        input_shape = (M, K)  # Shape of input matrix A
        weight_shape = (K, N) if not trans_B else (N, K)  # Shape of weight matrix B
        output_shape = (M, N)  # Shape of output matrix C
        Bias_shape = (N,)  # Shape of bias vector

        # Create TVM placeholders for input tensors
        A = te.placeholder(input_shape, name="A", dtype=in_dtype)  # Input matrix
        B = te.placeholder(weight_shape, name="B", dtype=in_dtype)  # Weight matrix
        Bias = te.placeholder(Bias_shape, name="Bias", dtype=accum_dtype)  # Bias vector

        # Define a reduction axis for matrix multiplication
        k = te.reduce_axis((0, K), name="k")

        def _compute_matmul(i, j):
            """
            Compute function for matrix-vector multiplication.

            Args:
                i (int): Row index.
                j (int): Column index.

            Returns:
                Computed value for C[i, j] as a sum over the reduction axis.
            """
            A_indices = [i, k]
            B_indices = [k, j] if not trans_B else [j, k]
            return te.sum(A[tuple(A_indices)].astype(accum_dtype) * B[tuple(B_indices)].astype(accum_dtype), axis=k)

        # Compute matrix multiplication result
        C = te.compute(
            output_shape,
            fcompute=_compute_matmul,
            name="C",
        )

        # Optionally apply bias addition
        if with_bias:
            C = te.compute(
                output_shape,
                lambda i, j: C[i, j] + Bias[j],
                name="Bias",
            )

        # Optionally cast the output to a different type
        if out_dtype != accum_dtype:
            C = te.compute(
                output_shape,
                lambda i, j: C[i, j].astype(out_dtype),
                name="D",
            )

        # Set function arguments (including bias if used)
        args = [A, B, Bias, C] if self.with_bias else [A, B, C]
        self.set_function(te.create_prim_func(args))

    def params_as_dict(self):
        """
        Returns the template parameters as a dictionary.

        Returns:
            dict: Dictionary containing template parameter values.
        """
        return {
            "N": self.N,
            "K": self.K,
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

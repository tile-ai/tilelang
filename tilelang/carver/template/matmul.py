from dataclasses import dataclass
from .base import BaseTemplate
from tvm import te
from ..arch import TileDevice
from ..roller import Hint
from ..utils import get_roller_hints_from_func


@dataclass
class MatmulTemplate(BaseTemplate):
    """
    A template for matrix multiplication (MatMul).

    This class defines the computation for a matrix-matrix multiplication
    with configurable parameters such as transposition, data types, and bias addition.

    Attributes:
        M (int): Number of rows in matrix A and matrix C.
        N (int): Number of columns in matrix B and matrix C.
        K (int): Number of columns in matrix A and rows in matrix B.
        trans_A (bool): Whether to transpose matrix A before multiplication.
        trans_B (bool): Whether to transpose matrix B before multiplication.
        in_dtype (str): Data type of input matrices.
        out_dtype (str): Data type of output matrix.
        accum_dtype (str): Data type used for accumulation.
        with_bias (bool): Whether to add a bias term.
    """

    # Operation-related configuration parameters
    M: int = None  # Number of rows in matrix A and matrix C
    N: int = None  # Number of columns in matrix B and matrix C
    K: int = None  # Number of columns in matrix A and rows in matrix B
    trans_A: bool = False  # Whether to transpose matrix A
    trans_B: bool = True  # Whether to transpose matrix B
    in_dtype: str = "float16"  # Data type of input matrices
    out_dtype: str = "float16"  # Data type of output matrix
    accum_dtype: str = "float16"  # Data type for accumulation
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
        roller_hints = get_roller_hints_from_func(self._func, arch=arch, topk=topk, allow_gemv=True)
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
        M, N, K = self.M, self.N, self.K

        # Ensure M, N, K are valid positive integers
        assert isinstance(M, int) and isinstance(N, int) and isinstance(K, int), "Only Support Integer M, N, K"
        assert M > 0 and N > 0 and K > 0, "M, N, K should be positive"

        # Load configuration parameters
        trans_A, trans_B = self.trans_A, self.trans_B
        in_dtype, out_dtype, accum_dtype = self.in_dtype, self.out_dtype, self.accum_dtype
        with_bias = self.with_bias

        # Define tensor shapes based on transpose flags
        input_shape = (M, K) if not trans_A else (K, M)  # Shape of input matrix A
        weight_shape = (K, N) if not trans_B else (N, K)  # Shape of weight matrix B
        output_shape = (M, N)  # Shape of output matrix C
        Bias_shape = (N,)  # Shape of bias vector

        # Create TVM placeholders for input tensors
        A = te.placeholder(input_shape, name="A", dtype=in_dtype)  # Input matrix A
        B = te.placeholder(weight_shape, name="B", dtype=in_dtype)  # Weight matrix B
        Bias = te.placeholder(Bias_shape, name="Bias", dtype=accum_dtype)  # Bias vector

        # Define a reduction axis for matrix multiplication
        k = te.reduce_axis((0, K), name="k")

        def _compute_matmul(i, j):
            """
            Compute function for matrix multiplication.

            Args:
                i (int): Row index.
                j (int): Column index.

            Returns:
                Computed value for C[i, j] as a sum over the reduction axis.
            """
            A_indices = [i, k] if not trans_A else [k, i]  # Adjust indexing if A is transposed
            B_indices = [k, j] if not trans_B else [j, k]  # Adjust indexing if B is transposed
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

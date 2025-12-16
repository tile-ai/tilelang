from dataclasses import dataclass
from .base import BaseTemplate
from tvm import te, tir
from ..roller import Hint
from ..utils import get_roller_hints_from_func


@dataclass
class ConvTemplate(BaseTemplate):
    """
    A template for convolution (Conv).

    This class defines the computation for a matrix-matrix convolution
    with configurable parameters such as transposition, data types, and bias addition.

    Attributes:
        N (int): The number of input samples processed simultaneously in a batch.
        C (int): The number of input feature maps.
        H (int): The height of the input feature maps.
        W (int): The width of the input feature maps.
        F (int): The number of filters (kernels) applied, determining output depth.
        K (int): The spatial dimensions of each convolutional filter.
        S (int): The step size by which the kernel slides across the input.
        D (int): The spacing between kernel elements, controlling receptive field expansion.
        P (int): The number of pixels added to input borders to control output spatial dimensions.
        in_dtype (str): Data type of input matrices.
        out_dtype (str): Data type of output matrix.
        accum_dtype (str): Data type used for accumulation.
        with_bias (bool): Whether to add a bias term.
    """

    # Operation-related configuration parameters
    N: int  # The number of input samples processed simultaneously in a batch.
    C: int  # The number of input feature maps.
    H: int  # The height of the input feature maps.
    W: int  # The width of the input feature maps.
    F: int  # The number of filters (kernels) applied, determining output depth.
    K: int  # The spatial dimensions of each convolutional filter.
    S: int  # The step size by which the kernel slides across the input.
    D: int  # The spacing between kernel elements, controlling receptive field expansion.
    P: int  # The number of pixels added to input borders to control output spatial dimensions.
    in_dtype: str = "float16"  # Data type of input matrices
    out_dtype: str = "float16"  # Data type of output matrix
    accum_dtype: str = "float16"  # Data type for accumulation
    with_bias: bool = False  # Whether to add a bias term

    def get_hardware_aware_configs(self, arch=None, topk=10) -> list[Hint]:
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
        Defines and initializes the convolution computation.

        This method sets up placeholders for input matrices, computes
        the convolution using TVM's compute API,
        and optionally applies bias and type casting.

        Raises:
            AssertionError: If N, C, H, W, F, K, S, D, P are not positive integers.
        """
        N, C, H, W, F, K, S, D, P = self.N, self.C, self.H, self.W, self.F, self.K, self.S, self.D, self.P
        assert (
            isinstance(N, int)
            and isinstance(C, int)
            and isinstance(H, int)
            and isinstance(W, int)
            and isinstance(F, int)
            and isinstance(K, int)
            and isinstance(S, int)
            and isinstance(D, int)
            and isinstance(P, int)
        ), "Only Support Integer Params"
        assert N > 0 and C > 0 and H > 0 and W > 0 and F > 0 and K > 0 and S > 0 and D > 0 and P > 0, "Params should be positive"

        # Load configuration parameters
        in_dtype, out_dtype, accum_dtype = self.in_dtype, self.out_dtype, self.accum_dtype
        with_bias = self.with_bias

        # Calculate kernel dimensions and output dimensions
        KH, KW = K, K
        OH = (H + 2 * P - D * (KH - 1) - 1) // S + 1
        OW = (W + 2 * P - D * (KW - 1) - 1) // S + 1

        # Define tensor shapes
        input_shape = (N, H, W, C)  # NHWC format input tensor
        weight_shape = (KH, KW, C, F)  # HWCF format weight tensor
        output_shape = (N, OH, OW, F)  # NHWC format output tensor
        bias_shape = (F,)  # Bias vector shape

        # Create TVM placeholders for input tensors
        A = te.placeholder(input_shape, name="A", dtype=in_dtype)  # Input tensor
        B = te.placeholder(weight_shape, name="B", dtype=in_dtype)  # Weight tensor
        Bias = te.placeholder(bias_shape, name="Bias", dtype=accum_dtype)  # Bias vector

        # Define reduction axes for convolution
        kh = te.reduce_axis((0, KH), name="kh")
        kw = te.reduce_axis((0, KW), name="kw")
        c = te.reduce_axis((0, C), name="c")

        def _compute_conv(n, h, w, f):
            """
            Compute function for convolution.

            Args:
                n (int): Batch index.
                h (int): Output height index.
                w (int): Output width index.
                f (int): Output channel index.

            Returns:
                Computed value for output[n, h, w, f] as a sum over reduction axes.
            """
            # Calculate input positions considering stride and dilation
            h_in = h * S - P + kh * D
            w_in = w * S - P + kw * D

            # Check if the input position is within bounds (implicit padding with 0)
            return te.sum(
                te.if_then_else(
                    te.all(h_in >= 0, h_in < H, w_in >= 0, w_in < W),
                    A[n, h_in, w_in, c].astype(accum_dtype) * B[kh, kw, c, f].astype(accum_dtype),
                    tir.const(0, accum_dtype),
                ),
                axis=[kh, kw, c],
            )

        # Compute convolution result
        C = te.compute(
            output_shape,
            fcompute=_compute_conv,
            name="C",
        )

        # Optionally apply bias addition
        if with_bias:
            C = te.compute(
                output_shape,
                lambda n, h, w, f: C[n, h, w, f] + Bias[f],
                name="Bias",
            )

        # Optionally cast the output to a different type
        if out_dtype != accum_dtype:
            C = te.compute(
                output_shape,
                lambda n, h, w, f: C[n, h, w, f].astype(out_dtype),
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
            "C": self.C,
            "H": self.H,
            "W": self.W,
            "F": self.F,
            "K": self.K,
            "S": self.S,
            "D": self.D,
            "P": self.P,
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

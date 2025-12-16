# Import necessary modules
from dataclasses import dataclass  # Used for defining data classes
from .base import BaseTemplate  # Importing the base class for templates
from tvm import te  # Importing TVM's tensor expression module
from ..arch import TileDevice  # Importing TileDevice for hardware-specific configurations
from ..roller import Hint  # Importing Hint for optimization hints
from ..utils import get_roller_hints_from_func  # Function to obtain optimization hints


@dataclass
class ElementwiseTemplate(BaseTemplate):
    """
    A template for element-wise operations using TVM.

    Attributes:
        shape (List[int]): The shape of the tensor.
        dtype (str): The data type of the tensor (default: "float16").
    """

    # OP Related Config
    shape: list[int] = None  # Shape of the tensor
    dtype: str = "float16"  # Data type of the tensor

    def get_hardware_aware_configs(self, arch: TileDevice = None, topk: int = 10) -> list[Hint]:
        """
        Retrieves hardware-aware optimization configurations.

        Args:
            arch (TileDevice, optional): The target hardware architecture.
            topk (int, optional): Number of top configurations to consider.

        Returns:
            List[Hint]: A list of optimization hints for the given architecture.
        """
        roller_hints = get_roller_hints_from_func(self._func, arch=arch, topk=topk, allow_gemv=True)
        return roller_hints

    def initialize_function(self) -> None:
        """
        Initializes the element-wise computation function.

        Defines a simple element-wise computation: B = A + 1, where A is an input tensor.
        The computation graph is built using TVM's tensor expressions.
        """
        shape, dtype = self.shape, self.dtype  # Extract shape and dtype

        # Define a placeholder tensor A
        A = te.placeholder(shape, name="A", dtype=dtype)

        # Define the element-wise computation (adding 1 to each element)
        def _compute_elementwise(*indices):
            return A[indices] + 1

        # Define the computation for B based on A
        B = te.compute(
            shape,
            fcompute=_compute_elementwise,  # Function that defines element-wise computation
            name="B",
        )

        # Store input and output tensors as function arguments
        args = [A, B]

        # Create and set the computation function
        self.set_function(te.create_prim_func(args))

    def params_as_dict(self):
        """
        Returns the parameters of the template as a dictionary.

        Returns:
            dict: A dictionary containing shape and dtype.
        """
        return {"shape": self.shape, "dtype": self.dtype}

    @property
    def class_attributes(self):
        """
        Returns class attributes as a dictionary.

        Returns:
            dict: A dictionary representation of the class attributes.
        """
        return self.params_as_dict()

    def __repr__(self) -> str:
        """
        Returns a string representation of the object.

        Returns:
            str: A string describing the instance with its parameters.
        """
        cls_name = self.__class__.__name__
        fields = self.class_attributes
        field_str = ", ".join(f"{key}={value!r}" for key, value in fields.items())
        return f"{cls_name}({field_str})"

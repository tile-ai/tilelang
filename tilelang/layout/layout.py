"""Wrapping Layouts."""

# pylint: disable=invalid-name, unsupported-binary-operation
import tvm_ffi
from tvm.ir import Node, Range
from tvm.tir import IterVar, Var, PrimExpr, IndexMap
from tilelang import _ffi_api


# Register the Layout class as a TVM object under the name "tl.Layout"
@tvm_ffi.register_object("tl.Layout")
class Layout(Node):
    def __init__(self, shape, forward_fn):
        """
        Initialize a Layout object.

        Parameters
        ----------
        shape : list of int
            The shape of the layout, defining the number of elements along each dimension.
        forward_fn : function
            A function that maps index variables to their computed forward index.
        """
        forward_vars = []  # List to store IterVars corresponding to each shape dimension

        # Create an IterVar for each dimension in the shape
        for idx, size in enumerate(shape):
            # Define an IterVar over the range [0, size) with an associated variable name
            iv = IterVar(Range(0, size), Var(f"i{idx}", "int32"), 0)
            forward_vars.append(iv)

        # Extract the variable references from the IterVars
        vars = [iv.var for iv in forward_vars]

        # Compute the forward index using the provided forward function
        forward_index = forward_fn(*vars)

        # Ensure forward_index is a list (to handle cases where a single expression is returned)
        if isinstance(forward_index, PrimExpr):
            forward_index = [forward_index]

        # Call the FFI constructor to create the Layout object in C++ backend
        self.__init_handle_by_constructor__(_ffi_api.Layout, forward_vars, forward_index)

    @property
    def index(self):
        """
        Property to retrieve the forward index of the layout.

        Returns
        -------
        PrimExpr or List[PrimExpr]
            The computed forward index expression(s).
        """
        return _ffi_api.Layout_index(self)

    def get_input_shape(self):
        """
        Get the input shape of the layout.

        Returns
        -------
        List[int]
            The shape of the input layout.
        """
        return _ffi_api.Layout_input_shape(self)

    def get_output_shape(self):
        """
        Get the output shape of the layout.

        Returns
        -------
        List[int]
            The shape of the output layout.
        """
        return _ffi_api.Layout_output_shape(self)

    def get_forward_vars(self):
        """
        Retrieve the iteration variables associated with the layout.

        Returns
        -------
        List[IterVar]
            A list of iteration variables that define the layout transformation.
        """
        return _ffi_api.Layout_forward_vars(self)

    def get_forward_index(self):
        return self.index

    def map_forward_index(self, indices: list[PrimExpr]) -> PrimExpr:
        """
        Compute the forward index mapping for a given set of input indices.

        Parameters
        ----------
        indices : list of PrimExpr
            The input indices to be mapped to their corresponding output indices.

        Returns
        -------
        PrimExpr
            The mapped index expression for the provided input indices.
        """
        # Retrieve the iteration variables used in the layout transformation
        forward_vars = self.get_forward_vars()

        # Retrieve the computed forward index expressions
        forward_indexes = self.index

        # Construct an IndexMap to map the input indices to the computed output indices
        index_map = IndexMap(
            initial_indices=forward_vars,  # The original iteration variables
            final_indices=forward_indexes,  # The computed forward indices
            inverse_index_map=None,  # No inverse mapping provided at this stage
        )

        # Map the provided indices using the constructed index mapping
        return index_map.map_indices(indices)

    def inverse(self) -> "Layout":
        """
        Compute the inverse of the current layout transformation.

        Returns
        -------
        Layout
            A new Layout object representing the inverse transformation.
        """
        return _ffi_api.Layout_inverse(self)

    def is_equal(self, other: "Layout") -> bool:
        """
        Check if the current layout is equal to another layout.

        Parameters
        ----------
        other : Layout
            The layout to compare with.
        """
        return _ffi_api.Layout_is_equal(self, other)

    def __repr__(self):
        return self._DebugOutput()
        # return f"Layout<{self.get_input_shape()}->{self.get_output_shape()}, {self.get_forward_vars()} -> {self.get_forward_index()}>"

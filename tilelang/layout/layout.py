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

    def repeat(self, dim: int, factor: int) -> "Layout":
        """
        Repeat a layout along a single input dimension.

        This is useful for building a larger layout by tiling an "atom" layout.
        Conceptually, repeating on dimension ``dim`` with ``factor`` constructs a
        new layout ``L'`` such that::

            L'(*idx) = [idx[dim] // extent_dim] + L(idx with idx[dim] % extent_dim)

        where ``extent_dim`` is the original extent of the repeated dimension.

        Parameters
        ----------
        dim : int
            The input dimension to repeat (0-based, supports negative indexing).
        factor : int
            The repeat factor. Must be a positive integer.

        Returns
        -------
        Layout
            A new Layout with the repeated input shape and an extra leading
            output dimension representing the repeat-group index.
        """
        if not isinstance(dim, int):
            raise TypeError(f"dim must be an int, got {type(dim)!r}")
        if not isinstance(factor, int):
            raise TypeError(f"factor must be an int, got {type(factor)!r}")
        if factor < 1:
            raise ValueError(f"factor must be >= 1, got {factor}")
        if factor == 1:
            return self

        input_shape = list(self.get_input_shape())
        ndim = len(input_shape)
        if ndim == 0:
            raise ValueError("Cannot repeat a 0-dim layout")

        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise ValueError(f"dim out of range: dim={dim}, ndim={ndim}")
        return _ffi_api.Layout_repeat(self, dim, factor)

    def expand(self, leading_shape) -> "Layout":
        """
        Expand (lift) this layout by prepending new leading input dimensions.

        The new leading dimensions are forwarded unchanged to the output, and
        the original layout is applied to the remaining trailing dimensions.

        Example
        -------
        Given a 2D layout ``L`` over ``[J, K]``, you can lift it to a 3D layout
        over ``[I, J, K]`` by::

            L3 = L.expand([I])
            # [i, j, k] -> [i, *L(j, k)]

        Parameters
        ----------
        leading_shape : int or Sequence[int or PrimExpr]
            The shape of the new leading dimensions to prepend. Use an empty
            list/tuple for a no-op.

        Returns
        -------
        Layout
            A new Layout with input shape ``leading_shape + input_shape`` and
            output indices ``[leading_dims] + old_forward_index``.
        """
        if isinstance(leading_shape, int):
            leading_shape = [leading_shape]
        if not isinstance(leading_shape, (list, tuple)):
            raise TypeError(f"leading_shape must be an int or a sequence, got {type(leading_shape)!r}")

        leading_shape = list(leading_shape)
        if len(leading_shape) == 0:
            return self

        for idx, extent in enumerate(leading_shape):
            if isinstance(extent, int):
                if extent <= 0:
                    raise ValueError(f"leading_shape[{idx}] must be > 0, got {extent}")
            elif not isinstance(extent, PrimExpr):
                raise TypeError(f"leading_shape elements must be int or PrimExpr, got {type(extent)!r} at index {idx}")

        return _ffi_api.Layout_expand(self, leading_shape)

    def inverse(self) -> "Layout":
        """
        Compute the inverse of the current layout transformation.

        Returns
        -------
        Layout
            A new Layout object representing the inverse transformation.
        """
        return _ffi_api.Layout_inverse(self)

    def reshape(self, shape, rescale_num=1, rescale_den=1) -> "Layout":
        """
        Reshape the input shape of the layout.

        Parameters
        ----------
        shape : list[PrimExpr] or list[int]
            The new input shape.
        rescale_num : int
            Rescale numerator for element size changes.
        rescale_den : int
            Rescale denominator for element size changes.
        """
        return _ffi_api.Layout_reshape(self, shape, rescale_num, rescale_den)

    def is_equal(self, other: "Layout") -> bool:
        """
        Check if the current layout is equal to another layout.

        Parameters
        ----------
        other : Layout
            The layout to compare with.
        """
        return _ffi_api.Layout_is_equal(self, other)

    def __call__(self, *args: list[PrimExpr]) -> PrimExpr:
        return self.map_forward_index(args)

    def __repr__(self):
        return self._DebugOutput()

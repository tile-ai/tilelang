from __future__ import annotations
from dataclasses import dataclass
from .base import BaseTemplate
from tvm import te
from ..arch import TileDevice
from ..roller import Hint
from ..utils import get_roller_hints_from_func


@dataclass
class GeneralReductionTemplate(BaseTemplate):
    # OP Related Config
    structure: str | list[str] = None
    shape: list[int] = None
    dtype: str = "float16"

    def get_hardware_aware_configs(self, arch: TileDevice = None, topk: int = 10) -> list[Hint]:
        roller_hints = get_roller_hints_from_func(self._func, arch=arch, topk=topk, allow_gemv=False)
        return roller_hints

    def initialize_function(self) -> None:
        """
        Parse the structure (e.g., 'SSR'), build the TVM compute definition
        with the appropriate spatial and reduce axes, and store it in self._func.
        """
        assert isinstance(self.structure, str), "Structure must be a string Currently."

        if self.structure is None or self.shape is None:
            raise ValueError("Must provide both `structure` and `shape`.")
        if len(self.structure) != len(self.shape):
            raise ValueError("`structure` length must match `shape` length.")
        if not all(isinstance(s, int) and s > 0 for s in self.shape):
            raise ValueError("All dimensions in `shape` must be positive integers.")

        # Separate axes into spatial vs reduce
        spatial_axes = []
        reduce_axes = []
        for i, axis_type in enumerate(self.structure):
            if axis_type.upper() == "S":
                spatial_axes.append((i, self.shape[i]))
            elif axis_type.upper() == "R":
                reduce_axes.append((i, self.shape[i]))
            else:
                raise ValueError(f"Unrecognized axis type '{axis_type}', only 'S'/'R' allowed.")

        # Create input placeholder
        A = te.placeholder(shape=self.shape, dtype=self.dtype, name="A")

        # Build a list of te.reduce_axis (for R) and the final output shape (for S).
        # We'll index them in order so that the compute lambda is consistent.
        # Example for SSR => 2 spatial dims (i, j), 1 reduce dim (k).

        # (1) Prepare the spatial dimensions:
        # The output shape is the product of all spatial axes in the same order they appear.
        # We'll construct a tuple for the final te.compute's shape. Example: (i, j).
        spatial_extents = [ext for (_, ext) in spatial_axes]

        # (2) Prepare reduce axes
        # e.g. (k0, (0, extent)), (k1, (0, extent)), ...
        reduce_axis_objs = []
        for _, ext in reduce_axes:
            reduce_axis_objs.append(te.reduce_axis((0, ext)))

        # We need to build a function that uses the correct index mapping.
        # Let's define a small helper that maps from the "spatial" indices to the
        # correct A[] indexing, and includes the reduce axes as well.

        # The final compute's shape is precisely the number of spatial axes in the same order.
        out_shape = tuple(spatial_extents)

        # We'll create a lambda of the form:
        #   (i, j, ...) -> te.sum(A[i, j, k, ...], axis=[k, ...])
        # We can do this dynamically by constructing indexing for each dimension in `A`.

        def compute_func(*spatial_indices):
            # spatial_indices is a tuple of the same length as spatial_axes
            # We must place each spatial index into the correct dimension of `A`
            # or reduce_axis. Then for the reduce axes, we use the reduce_axis_objs in order.

            # We want to build a full indexing that has length = len(self.shape).
            # E.g. structure='SSR', shape=[S0, S1, R2]
            #   i, j -> A[i, j, k]
            #   where i = spatial_indices[0], j = spatial_indices[1]

            full_index = []
            spatial_iter = 0
            reduce_iter = 0

            # Walk through the structure in order
            for axis_type in self.structure:
                if axis_type.upper() == "S":
                    # use the next spatial_indices item
                    full_index.append(spatial_indices[spatial_iter])
                    spatial_iter += 1
                else:
                    # axis_type is 'R', use the next reduce_axis_obj
                    full_index.append(reduce_axis_objs[reduce_iter])
                    reduce_iter += 1

            # Now we do the sum:
            return te.sum(A[tuple(full_index)], axis=tuple(reduce_axis_objs))

        # Construct the output tensor with te.compute
        C = te.compute(out_shape, compute_func, name="C")

        # Create a PrimFunc from placeholders + output
        args = [A, C]
        prim_func = te.create_prim_func(args)
        self.set_function(prim_func)

    def params_as_dict(self):
        return {"shape": self.shape, "dtype": self.dtype}

    @property
    def class_attributes(self):
        return self.params_as_dict()

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        fields = self.class_attributes
        field_str = ", ".join(f"{key}={value!r}" for key, value in fields.items())
        return f"{cls_name}({field_str})"

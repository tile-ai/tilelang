"""Wrapping Layouts."""

# pylint: disable=invalid-name, unsupported-binary-operation
import tvm
import tvm_ffi
from tvm.ir import Range
from tvm.tir import IterVar, Var, PrimExpr, IndexMap
from tilelang import _ffi_api
from tilelang.layout import Layout


@tvm_ffi.register_object("tl.Fragment")
class Fragment(Layout):
    """
    A Fragment layout object that encapsulates iteration variables (forward_vars),
    thread iteration variables (forward_thread), and index transformations
    (forward_index). This class supports replication (thread_replicate) and
    index mapping for fine-grained control over multi-dimensional data layouts.
    """

    # Disable the linter warning about not calling super().__init__()
    # because this object is created via TVM's FFI constructor mechanism.
    # pylint: disable=super-init-not-called
    def __init__(self, shape, forward_fn=None, forward_thread_fn=None, replicate=1, forward_index_fn=None):
        """
        Initialize the Fragment with iteration variables and optional thread replication.

        Parameters
        ----------
        shape : list[int]
            A list of integer sizes for each dimension of this fragment.
        forward_fn : callable, optional
            A function that takes the iteration variables, plus optionally a replicate
            IterVar, and returns a tuple: (forward_thread, forward_index).
            It is used when you want to compute both thread mapping and index mapping
            from the shape variables.
        forward_thread_fn : callable, optional
            A function that takes iteration variables (plus optionally a replicate Var)
            and returns an IterVar representing the thread index. This is used if
            `forward_fn` is not provided, and only the thread mapping is derived
            here while the index mapping is derived separately via `forward_index_fn`.
        replicate : int, optional
            How many times to replicate the iteration over the threads, typically
            used for multi-threading or replication in the hardware threads. Defaults to 1.
        forward_index_fn : callable, optional
            A function that takes iteration variables and returns an index or list
            of indices for this fragment. Used when `forward_fn` is None and
            the index transformation is derived separately.
        """

        # Create a list of IterVar objects based on shape dimensions
        # Each dimension is assigned a range from 0..size and a Var like i0, i1, etc.
        forward_vars = []
        for idx, size in enumerate(shape):
            iv = IterVar(Range(0, size), Var(f"i{idx}", "int32"), 0)
            forward_vars.append(iv)

        # Collect the underlying variables (i.e., Var objects) from the IterVars
        vars = [iv.var for iv in forward_vars]

        # Initialize placeholders for optional outputs
        forward_thread: IterVar = None
        forward_index: tvm.ir.container.Array = None
        thread_replicate: IterVar = None

        # If a forward_fn is provided, use it to derive both thread mapping and indices
        if forward_fn is not None:
            # If replication is greater than 1, create a replicate IterVar
            # and pass it to forward_fn
            if replicate > 1:
                thread_replicate = IterVar(Range(0, replicate), Var("rep", "int32"), 0)
                forward_thread, forward_index = forward_fn(*vars, thread_replicate)
            else:
                thread_replicate = None
                forward_thread, forward_index = forward_fn(*vars)
        else:
            # If no forward_fn is provided, compute forward_index (if any) via forward_index_fn
            forward_index = forward_index_fn(*vars) if forward_index_fn else None
            # Then compute forward_thread via forward_thread_fn
            if replicate > 1:
                thread_replicate = IterVar(Range(0, replicate), Var("rep", "int32"), 0)
                forward_thread = forward_thread_fn(*vars, thread_replicate.var)
            else:
                thread_replicate = None
                forward_thread = forward_thread_fn(*vars)

        # Ensure forward_index is an array if it isn't None
        if forward_index is None:
            forward_index = []
        elif not isinstance(forward_index, tvm.ir.container.Array):
            forward_index = [forward_index]

        # Call TVM FFI constructor to set up internal data structures
        self.__init_handle_by_constructor__(
            _ffi_api.Fragment,
            forward_vars,
            forward_index,
            forward_thread,
            thread_replicate,
        )

    @property
    def thread(self):
        """
        Returns the forward_thread (IterVar) of the Fragment, representing
        the thread dimension or mapping.
        """
        return _ffi_api.Fragment_thread(self)

    def get_thread_size(self):
        """
        Returns the extent (range size) of the thread dimension.
        If the Fragment was replicated over threads, this will reflect
        the number of threads.
        """
        return _ffi_api.Fragment_thread_size(self)

    def repeat(self, repeats, repeat_on_thread: bool = False, lower_dim_first: bool = True) -> "Fragment":
        """
        Returns a new Fragment that repeats the iteration space a given number of times.

        Parameters
        ----------
        repeats : int
            Number of times to repeat.
        repeat_on_thread : bool, optional
            If set, the repeat will happen on the thread dimension.
        lower_dim_first : bool, optional
            If set to True, repeat on lower dimensions first.

        Returns
        -------
        Fragment
            A new Fragment with the repeated iteration space.
        """
        return _ffi_api.Fragment_repeat(self, repeats, repeat_on_thread, lower_dim_first)

    def replicate(self, replicate: int) -> "Fragment":
        """
        Replicate the Fragment across a new thread dimension.

        Parameters
        ----------
        replicate : int
            The replication factor or number of threads.

        Returns
        -------
        Fragment
            A new Fragment with an additional replicate dimension.
        """
        return _ffi_api.Fragment_replicate(self, replicate)

    def condense_rep_var(self) -> "Fragment":
        """
        Condense or fold the replicate variable into the existing iteration space.
        This operation may be used to reduce dimensionality if the replicate variable
        is no longer needed as a separate dimension.

        Returns
        -------
        Fragment
            A new Fragment where the replicate variable is condensed.
        """
        return _ffi_api.Fragment_condense_rep_var(self)

    def map_forward_thread(self, indices: list[PrimExpr]) -> PrimExpr:
        """
        Get the thread mapping expression for a given set of argument indices.

        Parameters
        ----------
        indices : list of PrimExpr
            Indices for which to compute the thread mapping.

        Returns
        -------
        PrimExpr
            The computed thread expression for the provided indices.
        """
        # Retrieve the forward iteration variables
        forward_vars = self.get_forward_vars()
        # The thread dimension (IterVar) is accessed via the `thread` property
        forward_thread = self.thread
        # Construct an IndexMap to map the provided args into the final thread index
        index_map = IndexMap(initial_indices=forward_vars, final_indices=[forward_thread], inverse_index_map=None)
        return index_map.map_indices(indices)

    def __repr__(self):
        """
        String representation of the Fragment for debugging and logging.

        Returns
        -------
        str
            A string showing the thread dimension and the index dimension.
        """
        return self._DebugOutput()
        # return f"Fragment<{self.get_input_shape()}->{self.get_output_shape()}, thread={self.thread}, index={self.index}>"

    def is_equal(self, other: "Fragment") -> bool:
        """
        Check if the current fragment is equal to another fragment.
        """
        return _ffi_api.Fragment_is_equal(self, other)

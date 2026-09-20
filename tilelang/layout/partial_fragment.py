"""Wrapping the reducer partial-fragment layout."""

# pylint: disable=invalid-name, unsupported-binary-operation
import tvm_ffi
from tvm.ir import Range
from tvm.tirx import IterVar, Var
from tilelang import _ffi_api
from tilelang.layout.fragment import Fragment


@tvm_ffi.register_object("tl.PartialFragment")
class PartialFragment(Fragment):
    """Layout of a reducer's per-thread partials.

    Same algebra as Fragment, but the replication coordinate enumerates
    addends that still await the finalize collective (combine lanes), not
    equal copies of a finished value. Inferred by layout inference for
    ``local.reducer`` buffers, or supplied by the user via
    ``T.annotate_layout({acc: PartialFragment(...)})`` to pin the reducer's
    physical plan.
    """

    # pylint: disable=super-init-not-called
    def __init__(self, shape, forward_fn=None, forward_thread_fn=None, replicate=1, forward_index_fn=None, combine=None):
        """
        Mirror of :class:`Fragment`'s constructor, with partial semantics.

        `replicate` is the total number of replicas per logical element,
        decomposed under the low-bits convention by `combine`:
        `_rep % combine` enumerates addend lanes (threads holding partials
        the finalize collective sums), `_rep // combine` enumerates
        equal-value copy groups (each group independently computes the same
        partials; groups come from the update loop's own replication).

        `combine` defaults to `replicate`: every declared replica is an
        addend lane. It must evenly divide `replicate`.
        """
        if combine is None:
            combine = replicate
        if combine < 1 or replicate % combine != 0:
            raise ValueError(
                f"PartialFragment: combine ({combine}) must be >= 1 and evenly "
                f"divide replicate ({replicate}); `_rep % combine` are addend "
                f"lanes, `_rep // combine` are equal-value copy groups."
            )
        forward_vars = []
        for idx, size in enumerate(shape):
            iv = IterVar(Range(0, size), Var(f"i{idx}", "int32"), 0)
            forward_vars.append(iv)
        vars = [iv.var for iv in forward_vars]

        forward_thread = None
        forward_index = None
        thread_replicate = None
        if forward_fn is not None:
            if replicate > 1:
                thread_replicate = IterVar(Range(0, replicate), Var("rep", "int32"), 0)
                forward_thread, forward_index = forward_fn(*vars, thread_replicate)
            else:
                forward_thread, forward_index = forward_fn(*vars)
        else:
            forward_index = forward_index_fn(*vars) if forward_index_fn else None
            if replicate > 1:
                thread_replicate = IterVar(Range(0, replicate), Var("rep", "int32"), 0)
                forward_thread = forward_thread_fn(*vars, thread_replicate.var)
            else:
                forward_thread = forward_thread_fn(*vars)

        if forward_index is None:
            forward_index = []
        elif not isinstance(forward_index, tvm_ffi.Array):
            forward_index = [forward_index]

        self.__init_handle_by_constructor__(
            _ffi_api.PartialFragment,
            forward_vars,
            forward_index,
            forward_thread,
            thread_replicate,
            combine,
        )

    @staticmethod
    def from_fragment(fragment: Fragment) -> "PartialFragment":
        """Reinterpret a solved Fragment as per-thread partials."""
        return _ffi_api.PartialFragment_from_fragment(fragment)

    def as_post_collective(self) -> Fragment:
        """The same algebraic map read as a plain Fragment (post-finalize)."""
        return _ffi_api.PartialFragment_as_post_collective(self)


def make_fully_replicated_partial_fragment(shape, thread_extent) -> PartialFragment:
    """The wide plan: every participant holds one full-shape partial."""
    return _ffi_api.make_fully_replicated_partial_fragment(shape, thread_extent)

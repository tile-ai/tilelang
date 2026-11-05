"""Local indexing diagnostics without mutating the provider's tensor class."""

from tvm_tirx.ir import Expr


def tensor_getitem(tensor, indices):
    if not isinstance(indices, (tuple, list)):
        indices = [indices]
    ndim = len(tensor.ty.shape)
    if len(indices) != ndim:
        raise IndexError(
            f"Buffer {tensor.name} is {ndim}-dimensional (shape={tensor.ty.shape}), "
            f"but {len(indices)} index(es) were provided: {indices}. "
            f"Please provide exactly {ndim} index/indices or slice(s)."
        )
    return Expr.__getitem__(tensor, indices)

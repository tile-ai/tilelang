"""TileLang-specific override for TIR Buffer __getitem__ to provide
user-friendly error messages when the number of indices does not match
the buffer's dimensionality."""

from __future__ import annotations

from tvm import tir


_original_buffer_getitem = tir.Buffer.__getitem__


def _patched_buffer_getitem(self, indices):
    if not isinstance(indices, (tuple, list)):
        indices = [indices]
    ndim = len(self.shape)
    if len(indices) != ndim:
        raise IndexError(
            f"Buffer {self.name} is {ndim}-dimensional (shape={self.shape}), "
            f"but {len(indices)} index(es) were provided: {indices}. "
            f"Please provide exactly {ndim} index/indices or slice(s)."
        )
    return _original_buffer_getitem(self, indices)


tir.Buffer.__getitem__ = _patched_buffer_getitem

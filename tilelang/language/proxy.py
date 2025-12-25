"""Buffer/Tensor proxy in TileLang."""

from __future__ import annotations

from typing import Any, SupportsIndex, TYPE_CHECKING, Generic, TypeVar
from collections.abc import Sequence
from typing_extensions import Self

from tvm import tir
from tvm.tir import Var, PrimExpr
from tvm.script.ir_builder.tir import buffer, handle, match_buffer
from tilelang.utils import deprecated


class BufferProxy:
    """Buffer proxy class for constructing tir buffer."""

    # Index via T.Buffer(...)
    @deprecated("T.Buffer(...)", "T.Tensor(...)")
    def __call__(
        self,
        shape,
        dtype="float32",
        data=None,
        strides=None,
        elem_offset=None,
        scope="global",
        align=0,
        offset_factor=0,
        buffer_type="",
        axis_separators=None,
    ) -> tir.Buffer:
        return buffer(
            shape,
            dtype=dtype,
            data=data,
            strides=strides,
            elem_offset=elem_offset,
            scope=scope,
            align=align,
            offset_factor=offset_factor,
            buffer_type=buffer_type,
            axis_separators=axis_separators,
        )

    # Index via T.Buffer[...]
    @deprecated("T.Buffer[...]", "T.Tensor(...)")
    def __getitem__(self, keys) -> tir.Buffer:
        if not isinstance(keys, tuple):
            return self(keys)
        if len(keys) >= 2 and not isinstance(keys[1], str):
            return self(keys)
        return self(*keys)  # type: ignore[attr-defined] # pylint: disable=no-member

    def from_ptr(
        self, pointer_var: Var, shape: tuple[PrimExpr, ...], dtype: str = "float32", strides: tuple[PrimExpr, ...] = None
    ) -> Buffer:
        """Create a buffer from a pointer, shape, and data type.

        Args:
            pointer_var: The pointer variable
            shape: The shape of the buffer
            dtype: The data type of the buffer (default: float32)

        Returns:
            A buffer created from the given parameters
        """
        return match_buffer(pointer_var, shape, dtype=dtype, strides=strides)


class BaseTensorProxy:
    """Base proxy class for tensor types with configurable defaults.

    This class serves as a foundation for different tensor proxy types, providing
    customizable default values for scope, alignment, and offset factors. It implements
    the core functionality for creating TIR buffers with specific memory configurations.
    """

    default_scope = "global"
    default_align = 0
    default_offset_factor = 0

    def __call__(
        self,
        shape,
        dtype="float32",
        data=None,
        strides=None,
        elem_offset=None,
        scope=None,  # Changed to None to use class default
        align=None,
        offset_factor=None,
        buffer_type="",
        axis_separators=None,
    ) -> tir.Buffer:
        # Use class defaults if not specified
        scope = scope or self.default_scope
        align = align or self.default_align
        offset_factor = offset_factor or self.default_offset_factor

        return buffer(
            shape,
            dtype=dtype,
            data=data,
            strides=strides,
            elem_offset=elem_offset,
            scope=scope,
            align=align,
            offset_factor=offset_factor,
            buffer_type=buffer_type,
            axis_separators=axis_separators,
        )

    def __getitem__(self, keys) -> tir.Buffer:
        assert isinstance(keys, tuple)
        # Single argument (the shape)
        if all([type(s) not in (tuple, str, list) for s in keys]):
            keys = (keys,)
        return self(*keys)

    def from_ptr(
        self, pointer_var: Var, shape: tuple[PrimExpr, ...], dtype: str = "float32", strides: tuple[PrimExpr, ...] = None
    ) -> tir.Buffer:
        """Create a buffer from a pointer, shape, and data type.

        Args:
            pointer_var: The pointer variable
            shape: The shape of the buffer
            dtype: The data type of the buffer (default: float32)

        Returns:
            A buffer created from the given parameters
        """
        return match_buffer(pointer_var, shape, dtype=dtype, strides=strides)


class TensorProxy(BaseTensorProxy):
    """Main tensor proxy class for global scope buffers.

    This class implements the default tensor proxy with global memory scope,
    the tensor should be by default contiguous.
    """

    @staticmethod
    def _construct_strides(shape: tuple[Any]):
        s, strides = 1, [1]
        for dim in shape[:0:-1]:
            s *= dim
            strides.append(s)
        return tuple(reversed(strides))

    def __call__(self, shape: tuple[Any] | PrimExpr | int, dtype: str = "float32", data=None, scope=None) -> tir.Buffer:
        if isinstance(shape, (int, PrimExpr)):
            shape = (shape,)
        return super().__call__(shape, dtype=dtype, strides=TensorProxy._construct_strides(shape), data=data, scope=scope)


class StridedTensorProxy(BaseTensorProxy):
    """Main tensor proxy class for global scope buffers, with strides supported.

    This class implements the default tensor proxy with global memory scope, with the stride information required.
    """

    def __call__(self, shape: tuple[Any], strides: tuple[Any], dtype: str = "float32", scope=None) -> tir.Buffer:
        if len(shape) != len(strides):
            raise ValueError("Invalid shape/strides' dimensions")
        return super().__call__(shape, dtype=dtype, strides=strides, scope=scope)


class FragmentBufferProxy(BaseTensorProxy):
    """Proxy class for fragment memory buffers.

    This class represents tensor proxies specifically for local fragment memory,
    typically used in GPU tensor core operations.
    """

    default_scope = "local.fragment"


class SharedBufferProxy(BaseTensorProxy):
    """Proxy class for shared memory buffers.

    This class represents tensor proxies for dynamic shared memory,
    commonly used in GPU shared memory operations.
    """

    default_scope = "shared.dyn"


class LocalBufferProxy(BaseTensorProxy):
    """Proxy class for local memory buffers.

    This class represents tensor proxies for local memory scope,
    typically used for temporary computations in GPU kernels.
    """

    default_scope = "local"


Buffer = BufferProxy()  # pylint: disable=invalid-name
# Tensor is an alias for Buffer
# Because when user do jit compile, the input and output will
# be mapped with torch.Tensor.
if TYPE_CHECKING:

    class BaseTensor:
        def __class_getitem__(cls, key):
            return cls

        def __getitem__(self, key) -> Any: ...

        def __setitem__(self, key, value) -> None: ...

        def __init__(
            self,
            shape: Sequence[SupportsIndex],
            dtype="float32",
            data=None,
            strides=None,
            elem_offset=None,
            scope=None,  # Changed to None to use class default
            align=None,
            offset_factor=None,
            buffer_type="",
            axis_separators=None,
        ): ...

        @classmethod
        def from_ptr(
            cls, pointer_var: Var, shape: Sequence[PrimExpr, ...], dtype: str = "float32", strides: tuple[PrimExpr, ...] = None
        ) -> Self: ...

    class Tensor(BaseTensor): ...

    class StridedTensor(BaseTensor): ...

    class FragmentBuffer(BaseTensor): ...

    class SharedBuffer(BaseTensor): ...

    class LocalBuffer(BaseTensor): ...

    _T = TypeVar("_T")

    class Ref(Generic[_T], tir.Var): ...
else:
    Tensor = TensorProxy()  # pylint: disable=invalid-name
    StridedTensor = StridedTensorProxy()  # pylint: disable=invalid-name
    FragmentBuffer = FragmentBufferProxy()  # pylint: disable=invalid-name
    SharedBuffer = SharedBufferProxy()  # pylint: disable=invalid-name
    LocalBuffer = LocalBufferProxy()  # pylint: disable=invalid-name

    class Ref: ...


def ptr(dtype: str | None = None, storage_scope: str = "global", *, is_size_var: bool = False) -> Var:
    """Create a TIR var that represents a pointer.

    Parameters
    ----------
    dtype: str
        The data type of the pointer.

    storage_scope: str
        The storage scope of the pointer.

    is_size_var: bool
        Whether or not to return a SizeVar instead of Var.

    Returns
    -------
    res : PrimExpr
        The new tir.Var with type handle or casted expression with type handle.
    """
    return handle(dtype=dtype, storage_scope=storage_scope, is_size_var=is_size_var)


def make_tensor(ptr: Var, shape: tuple[PrimExpr, ...], dtype: str = "float32", strides: tuple[PrimExpr, ...] = None) -> tir.Buffer:
    return Tensor.from_ptr(ptr, shape, dtype, strides)

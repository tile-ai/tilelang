"""The language interface for tl programs."""

from __future__ import annotations
from typing import Any, Optional, Sequence, SupportsIndex, TYPE_CHECKING, Tuple
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

    def from_ptr(self,
                 pointer_var: Var,
                 shape: tuple[PrimExpr, ...],
                 dtype: str = "float32",
                 strides: tuple[PrimExpr, ...] = None) -> Buffer:
        """
                 Create a tir.Buffer view that matches a pointer variable.
                 
                 This calls `match_buffer` with the provided pointer, shape, dtype, and optional strides to produce a Buffer view over the pointer memory.
                 
                 Parameters:
                     pointer_var: A TIR pointer/handle variable referencing the underlying memory.
                     shape: The buffer shape (sequence of PrimExpr).
                     dtype: Element data type (default "float32").
                     strides: Optional explicit strides for the buffer; when provided they are forwarded to `match_buffer`.
                 
                 Returns:
                     A tir.Buffer corresponding to the matched view.
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
        """
        Create a TIR Buffer with the given shape and attributes, using class defaults for scope, alignment, and offset factor when those arguments are None.
        
        Parameters:
            shape: Sequence of integer/PrimExpr dimensions for the buffer.
            dtype (str): Element data type (default "float32").
            data: Optional backing pointer or handle for the buffer.
            strides: Optional sequence of strides for each dimension; if None, caller-level proxies may compute contiguous strides.
            elem_offset: Optional element offset within the backing storage.
            scope: Storage scope; if None the proxy's class-level default_scope is used.
            align: Optional alignment in bytes; if None the proxy's class-level default_align is used.
            offset_factor: Optional offset factor; if None the proxy's class-level default_offset_factor is used.
            buffer_type (str): Optional string tag describing the buffer type.
            axis_separators: Optional per-axis separator metadata.
        
        Returns:
            tir.Buffer: A TIR buffer object describing the requested tensor view.
        """
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
        """
        Create and return a tir.Buffer for the provided indexing or shape specifier.
        
        Parameters:
            keys (tuple): A tuple describing either (1) a simple shape where each element is a scalar (e.g., (n, m))
                          or (2) one or more detailed buffer arguments (which may be nested tuples/strides/etc.).
                          The method asserts `keys` is a tuple. If `keys` is a simple shape (no element is a tuple,
                          list, or string), it is treated as a single shape and wrapped as `(keys,)` before delegating
                          to the callable interface.
        
        Returns:
            tir.Buffer: The constructed buffer returned by delegating to self(*keys).
        
        Raises:
            AssertionError: If `keys` is not a tuple.
        """
        assert isinstance(keys, tuple)
        # Single argument (the shape)
        if all([type(s) not in (tuple, str, list) for s in keys]):
            keys = (keys,)
        return self(*keys)

    def from_ptr(self,
                 pointer_var: Var,
                 shape: tuple[PrimExpr, ...],
                 dtype: str = "float32",
                 strides: tuple[PrimExpr, ...] = None) -> tir.Buffer:
        """
                 Create a TIR Buffer view that matches an existing pointer.
                 
                 Parameters:
                     pointer_var: The pointer/handle variable that refers to the underlying memory.
                     shape: Tuple of PrimExpr describing the buffer dimensions.
                     dtype: Element data type for the buffer (default "float32").
                     strides: Optional tuple of PrimExpr specifying per-dimension strides; if omitted a contiguous layout is assumed.
                 
                 Returns:
                     tir.Buffer: A buffer view bound to the given pointer with the specified shape, dtype, and optional strides.
                 """
        return match_buffer(pointer_var, shape, dtype=dtype, strides=strides)


class TensorProxy(BaseTensorProxy):
    """Main tensor proxy class for global scope buffers.
    
    This class implements the default tensor proxy with global memory scope,
    the tensor should be by default contiguous.
    """

    @staticmethod
    def _construct_strides(shape: Tuple[Any]):
        """
        Compute contiguous row-major strides for a given tensor shape.
        
        Given a sequence of dimension sizes (ints or PrimExpr), return a tuple of strides where each stride is the product of all subsequent dimensions (i.e., stride[i] = prod(shape[i+1:])), and the last stride is 1. The returned tuple has the same length as `shape`.
        """
        s, strides = 1, [1]
        for dim in shape[:0:-1]:
            s *= dim
            strides.append(s)
        return tuple(reversed(strides))

    def __call__(self, shape: Tuple[Any], dtype: str = "float32", data=None) -> tir.Buffer:
        """
        Create a contiguous row-major tir.Buffer for a tensor with the given shape.
        
        This overrides the base proxy to compute contiguous row-major strides automatically from `shape`
        and delegate to the base constructor.
        
        Parameters:
            shape (Tuple[Any]): Tensor dimensions (each entry is a PrimExpr or int).
            dtype (str): Element data type (default "float32").
            data: Optional backing data/pointer for the buffer; passed through unchanged.
        
        Returns:
            tir.Buffer: A TIR buffer representing the tensor with computed contiguous strides.
        """
        return super().__call__(
            shape, dtype=dtype, strides=TensorProxy._construct_strides(shape), data=data)


class StridedTensorProxy(BaseTensorProxy):
    """Main tensor proxy class for global scope buffers, with strides supported.

    This class implements the default tensor proxy with global memory scope, with the stride information required.
    """

    def __call__(self,
                 shape: Tuple[Any],
                 strides: Tuple[Any],
                 dtype: str = "float32") -> tir.Buffer:
        """
                 Create a TIR Buffer for a strided tensor view.
                 
                 Parameters:
                     shape (Tuple[Any]): Dimensions of the tensor.
                     strides (Tuple[Any]): Stride for each dimension; must have the same length as `shape`
                         and the last element must be 1 (contiguous innermost dimension).
                     dtype (str): Element data type (default "float32").
                 
                 Returns:
                     tir.Buffer: A TIR buffer describing the tensor view with the provided shape and strides.
                 
                 Raises:
                     ValueError: If `len(shape) != len(strides)` or if the last stride is not 1.
                 """
                 if len(shape) != len(strides):
            raise ValueError("Invalid shape/strides' dimensions")
        if not bool(strides[-1] == 1):
            # TODO(chenggang): shall we support non-contiguous even for the last dimension?
            raise ValueError("The stride of the last dimension must be 1 (contiguous)")
        return super().__call__(shape, dtype=dtype, strides=strides)


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

        def __getitem__(self, key) -> Any:
            ...

        def __setitem__(self, key, value) -> None:
            ...

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
        ):
            ...

        @classmethod
        def from_ptr(cls,
                     pointer_var: Var,
                     shape: Sequence[PrimExpr, ...],
                     dtype: str = "float32",
                     strides: tuple[PrimExpr, ...] = None) -> Self:
            """
                     Create a buffer view that binds a pointer variable to a TIR buffer with the given shape and optional strides.
                     
                     This classmethod calls `match_buffer(pointer_var, shape, dtype=dtype, strides=strides)` to produce a Buffer that views the memory referenced by `pointer_var`. If `strides` is provided, it is forwarded to `match_buffer` to create a strided view; otherwise no explicit strides are set.
                     
                     Parameters:
                         pointer_var: The TIR pointer/handle variable that identifies the underlying memory.
                         shape: Sequence of PrimExpr describing the buffer shape.
                         dtype: Element data type for the resulting buffer (default "float32").
                         strides: Optional sequence of PrimExpr specifying element strides for each dimension; forwarded to `match_buffer`.
                     
                     Returns:
                         A buffer object (instance of the proxy's Buffer/TIR buffer) representing the memory view bound to `pointer_var`.
                     """
                     ...

    class Tensor(BaseTensor):
        ...

    class StridedTensor(BaseTensor):
        ...

    class FragmentBuffer(BaseTensor):
        ...

    class SharedBuffer(BaseTensor):
        ...

    class LocalBuffer(BaseTensor):
        ...
else:
    Tensor = TensorProxy()  # pylint: disable=invalid-name
    StridedTensor = StridedTensorProxy()  # pylint: disable=invalid-name
    FragmentBuffer = FragmentBufferProxy()  # pylint: disable=invalid-name
    SharedBuffer = SharedBufferProxy()  # pylint: disable=invalid-name
    LocalBuffer = LocalBufferProxy()  # pylint: disable=invalid-name


def ptr(dtype: Optional[str] = None,
        storage_scope: str = "global",
        *,
        is_size_var: bool = False) -> Var:
    """
        Create a TIR Var representing a pointer (a handle).
        
        One-line summary:
            Return a TIR handle Var (or a SizeVar when is_size_var is True) that encodes a pointer's element dtype and storage scope.
        
        Parameters:
            dtype (Optional[str]): Element dtype for the pointer (e.g., "float32"); None produces an untyped handle.
            storage_scope (str): Storage scope string (e.g., "global", "local", "shared"); defaults to "global".
            is_size_var (bool): If True, return a SizeVar suitable for expressing sizes/lengths instead of a regular handle Var.
        
        Returns:
            Var: A TIR Var whose type is a handle (or a SizeVar when is_size_var is True) describing a pointer with the given dtype and storage scope.
        """
    return handle(dtype=dtype, storage_scope=storage_scope, is_size_var=is_size_var)


def make_tensor(ptr: Var,
                shape: tuple[PrimExpr, ...],
                dtype: str = "float32",
                strides: tuple[PrimExpr, ...] = None) -> tir.Buffer:
    """
                Create a tir.Buffer (tensor view) from a pointer and shape, optionally with explicit strides.
                
                Parameters:
                    ptr (Var): TIR handle/variable that represents the underlying pointer.
                    shape (tuple[PrimExpr, ...]): Shape of the tensor as a tuple of TIR primitive expressions.
                    dtype (str, optional): Element data type (default "float32").
                    strides (tuple[PrimExpr, ...], optional): Optional explicit strides; when provided, creates a strided view
                        instead of assuming contiguous row-major layout.
                
                Returns:
                    tir.Buffer: A TIR buffer that matches the given pointer, shape, dtype, and strides.
                """
                return Tensor.from_ptr(ptr, shape, dtype, strides)

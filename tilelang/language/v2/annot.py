from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from tvm import tir
from tvm.ir.expr import PrimExpr
from tvm.script.ir_builder.tir import buffer
from typing import Any, Callable, Literal, TypeVar, Generic, TYPE_CHECKING

# Python 3.9 compatibility for advanced typing features
try:
    from typing import ParamSpec, TypeVarTuple, Unpack, Self  # type: ignore[attr-defined]
except Exception:  # Python < 3.10 for ParamSpec, < 3.11 for Unpack/TypeVarTuple/Self
    from typing_extensions import ParamSpec, TypeVarTuple, Unpack, Self  # type: ignore

# Compatibility for generic alias detection across Python versions
try:
    from typing import _GenericAlias as _TypingGenericAlias  # type: ignore[attr-defined]
except Exception:
    _TypingGenericAlias = None  # type: ignore
try:
    # Builtin generic alias type for e.g. tuple[int]
    from types import GenericAlias as _TypesGenericAlias  # type: ignore[attr-defined]
except Exception:
    _TypesGenericAlias = None  # type: ignore

_GenericAliasTypes = tuple(t for t in (_TypingGenericAlias, _TypesGenericAlias) if t is not None)
if not _GenericAliasTypes:

    class _DummyGenericAlias:  # type: ignore
        pass

    _GenericAliasTypes = (_DummyGenericAlias,)  # type: ignore
from collections.abc import Sequence
from .dtypes import AnyDType
from . import dtypes as dt
import tvm.script.ir_builder.tir as tb_tir
from tvm.script.ir_builder import IRBuilder
import torch
import inspect

_Shapes = TypeVarTuple("_Shapes")
_Shape = ParamSpec("_Shape")
_Stride = ParamSpec("_Stride")
_DType = TypeVar("_DType")

Scope = Literal["global", "shared.dyn", "local", "local.fragment"]


class Annot(ABC):
    """
    Base class for tilelang kernel annotations
    Tilelang kernel annotations are used to specify how to interpret each argument of the jit kernel

    It provides 3 main functionalities:
    1. determine whether the argument is a kernel argument (i.e., needs to be passed at kernel launch time)
    2. parse the argument value into a hash key for jit caching
    3. convert the argument into a tvm tir argument (tir.Var | tir.Buffer) for prim func generation
    """

    def is_kernel_arg(self) -> bool:
        """
        Determine whether the argument is a kernel argument (i.e., needs to be passed at kernel launch time)
        """
        return False

    @abstractmethod
    def with_name(self: Self, name) -> Self:
        pass

    @abstractmethod
    def get_key_parser(self) -> Callable[[str, Any], tuple[Any, ...]]:
        """
        Return a parser function that converts the argument value into a hash key for jit caching
        """

    @abstractmethod
    def create_prim_func_arg(self, name: str, value: Any, vt: ArgVarTable) -> tir.Var | tir.Buffer:
        """
        Convert the argument into a tvm tir argument (tir.Var | tir.Buffer) for prim func generation
        """

    def promote(self) -> TIRAnnot | None:
        """
        Try to promote the annotation into a FixedAnnot if possible
        Return None if not promotable
        """
        return None


@dataclass
class ArgVarTable:
    """
    ArgVarTable is used to manage the mapping from argument names to tir.Var objects
    """

    var_tab: dict[str, tir.Var] = field(default_factory=dict)
    tmp_name_idx: int = 0

    def get_or_create_var(self, name: str, dtype: dt.dtype) -> tir.Var:
        if not name:
            name = self.create_tmp_name()
        if name not in self.var_tab:
            self.var_tab[name] = tir.Var(name, dtype)
        return self.var_tab[name]

    def create_tmp_name(self) -> str:
        name = f"varg_{self.tmp_name_idx}"
        self.tmp_name_idx += 1
        return name


@dataclass
class Value(Annot):
    kind: Literal["static", "dynamic"] = "dynamic"
    name: str | None = None
    dtype: dt.dtype | None = dt.int32
    value: int | tir.Var | None = None
    creator: Callable[[], Any] | None = None

    def is_kernel_arg(self) -> bool:
        return self.kind == "dynamic"

    @classmethod
    def from_value(cls, value: Any, prefer_name: str = None) -> Value:
        if isinstance(value, int):
            # handle A: T.Tensor[[1024, 1024], ...]
            return Value(kind="static", name=prefer_name, dtype=dt.int32, value=value)
        elif isinstance(value, float):
            return Value(kind="static", name=prefer_name, dtype=dt.float32, value=value)
        elif isinstance(value, dt.dtype):
            # handle A: T.float32
            return Value(kind="dynamic", name=prefer_name, dtype=value, value=None)
        elif isinstance(value, Value):
            # handle A: T.dyn
            return value
        elif isinstance(value, TypeVar):
            return Value(kind="static", name=value.__name__, value=None)
        elif isinstance(value, (tir.Var, PrimExpr)):
            # handle A: T.Tensor[[M, N, K], ...]
            # or primexpr annotation like A: T.Tensor[[M, N * 4 +1]]
            name = value.name if isinstance(value, tir.Var) else prefer_name
            return Value(kind="dynamic", name=name, dtype=value.dtype, value=value)
        elif value is Any or value is None or value is dt.dtype or isinstance(value, (type,) + _GenericAliasTypes):
            # A # no annotation
            # A: Any
            # A: _T
            # A: dt.dtype
            # A: tuple[...]
            return Value(kind="static", name=prefer_name, value=None)
        else:
            raise TypeError(f"Unsupported Value annotation: {value!r}, type: {type(value)}")

    def with_name(self, name: str) -> Value:
        return Value(kind=self.kind, name=self.name or name, dtype=self.dtype, value=self.value)

    def get_key_parser(self):
        if self.kind == "static":
            if self.value is not None:
                expected_value = self.value

                def key_parser(name: str, target: Any):
                    assert target == expected_value
                    return target

                return key_parser
            else:
                return lambda name, target: (target,)
        else:
            return lambda name, target: (None,)

    def parse_key(self, target: Any):
        return self.get_key_parser()(target)

    def create_prim_func_arg(self, name: str, value: Any, vt: ArgVarTable, create_arg: bool = True):
        if self.kind == "static":
            if self.value:
                assert self.value == value, f"static value mismatch for {name}: expected {self.value}, got {value}"
            return value
        else:
            name = self.name or name or vt.create_tmp_name()
            if self.value is not None:
                arg = self.value
            elif self.creator is not None:
                arg = self.creator()
            else:
                arg = vt.get_or_create_var(name, self.dtype)
            return tb_tir.arg(name, arg) if create_arg else arg

    def __repr__(self):
        if self.kind == "static":
            if self.value is not None:
                return repr(self.value)
            else:
                return (str(self.name) or "$unnamed") + "$"
        else:
            if self.value is not None:
                return repr(self.value)
            elif self.creator is not None:
                return repr(self.creator())
            else:
                return (str(self.name) or "$unnamed") + "$dyn"


def _canonicalize_dtype(val: Any) -> dt.dtype | None:
    if val == Any or val is None:
        return None
    if isinstance(val, TypeVar):
        return None
    return dt.dtype(val)


def _canonicalize_shape(shape: Sequence[Any]) -> list[Value]:
    if shape is None or shape is Any:
        return None
    return [Value.from_value(dim) for _, dim in enumerate(iterable=shape)]


def _canonicalize_strides(strides: Sequence[Any]) -> list[Value]:
    if strides is None or strides is Any:
        return None
    return [Value.from_value(dim) for _, dim in enumerate(strides)]


def _shape_with_name(shape: Sequence[Value], base_name: str) -> list[Value]:
    if shape is None:
        return None
    res = []
    for i, dim in enumerate(shape):
        dim = dim.with_name(f"{base_name}_{i}")
        res.append(dim)
    return res


def _try_convert_static_shape(shape: Sequence[Value]):
    if shape is None:
        return None
    res = []
    for s in shape:
        if s.kind == "static" and s.value is not None or s.kind == "dynamic" and s.value is not None:
            res.append(s.value)
    if len(res) == len(shape):
        return res


@dataclass
class BufferAnnot(Annot):
    shape: tuple = None
    strides: tuple = None
    dtype: dt.dtype = None

    def is_kernel_arg(self) -> bool:
        return True

    @property
    def scope(self):
        return "global"

    def __call__(
        self,
        shape: tuple[Unpack[_Shapes]],
        dtype: _DType = "float32",
        data=None,
        strides=None,
        elem_offset=None,
        scope=None,
        align=0,
        offset_factor=0,
        buffer_type="",
        axis_separators=None,
    ) -> Tensor[Callable[[Unpack[_Shapes]]], _DType]:
        return buffer(
            shape,
            dtype=dtype,
            data=data,
            strides=strides,
            elem_offset=elem_offset,
            scope=scope or self.scope,
            align=align,
            offset_factor=offset_factor,
            buffer_type=buffer_type,
            axis_separators=axis_separators,
        )

    def __getitem__(self, params):
        shape, dtype = params
        if not isinstance(shape, (tuple, list)):
            shape = (shape,)
        shape = _canonicalize_shape(shape)
        dtype = _canonicalize_dtype(dtype)
        return self.__class__(shape, strides=self.strides, dtype=dtype)

    def with_name(self, name: str):
        shape = _shape_with_name(self.shape, base_name=f"{name}_shape")
        strides = _shape_with_name(self.strides, base_name=f"{name}_stride")
        return self.__class__(shape, strides, self.dtype)

    def get_key_parser(self):
        raw_shapes = True
        if self.shape is not None:
            raw_shapes = False
            shape_len = len(self.shape)
            static_shape_idx = [i for i, dim in enumerate(self.shape) if dim.kind == "static"]
            # static_fixed_shape_idx = [i for i, dim in enumerate(self.shape) if dim.kind == 'static' and dim.value is not None]
            # static_fixed_shape_values = [dim.value for dim in self.shape if dim.kind == 'static' and dim.value is not None]
        raw_strides = True
        if self.strides is not None:
            raw_strides = False
            strides_len = len(self.strides)
            strides_shape_idx = [i for i, dim in enumerate(self.strides) if dim.kind == "static"]
            # static_fixed_strides_idx = [i for i, dim in enumerate(self.strides) if dim.kind == 'static' and dim.value is not None]
            # static_fixed_strides_values = [dim.value for dim in self.strides if dim.kind == 'static' and dim.value is not None]
        raw_dtype = True
        if self.dtype is not None:
            raw_dtype = False
            expected_dtype = self.dtype

        def key_parser(name: str, target: Any):
            if isinstance(target, torch.Tensor):
                shape = tuple(target.shape)
                strides = tuple(target.stride())
                dtype = dt.dtype(target.dtype)
            elif isinstance(target, tir.Buffer):
                shape = tuple(target.shape)
                strides = tuple(target.strides)
                dtype = dt.dtype(target.dtype)
            else:
                raise TypeError(
                    f"Unsupported buffer argument type for argument `{name}`: expected a `torch.Tensor` or `tir.Buffer`, got {type(target)}"
                )
            if not raw_shapes:
                assert len(shape) == shape_len
                shape = tuple(shape[i] for i in static_shape_idx)
                # shape_fixed = tuple(shape[i] for i in static_fixed_shape_idx)
                # assert shape_fixed == static_fixed_shape_values, f"shape mismatch"
            if not raw_strides:
                assert len(strides) == strides_len
                strides = tuple(strides[i] for i in strides_shape_idx)
                # strides_fixed = tuple(strides[i] for i in static_fixed_strides_idx)
                # assert strides_fixed == static_fixed_strides_values
            if not raw_dtype:
                dtype = dt.dtype(dtype)
                if dtype != expected_dtype:
                    raise TypeError(f"Tensor dtype mismatch for argument `{name}`, expected {expected_dtype}, got {dtype}")
            return shape, strides, dtype

        return key_parser

    def parse_key(self, target: Any):
        return self.get_key_parser()(target)

    @staticmethod
    def match_shape(shape: tuple[Value, ...], target_shape: tuple[int, ...], vt: ArgVarTable):
        if shape is None:
            return target_shape
        args = []
        for s, target in zip(shape, target_shape):
            args.append(s.create_prim_func_arg(s.name, target, vt, create_arg=False))
        return args

    def create_prim_func_arg(self, name: str, value: Any, vt: ArgVarTable):
        if isinstance(value, tir.Buffer):
            shape = value.shape
            strides = value.strides
            dtype = value.dtype
        elif isinstance(value, torch.Tensor):
            shape = value.shape
            strides = value.stride()
            dtype = dt.dtype(value.dtype)
        else:
            raise TypeError(f"Unsupported buffer argument type: {type(value)}")
        shape = self.match_shape(self.shape, shape, vt)
        strides = self.match_shape(self.strides, strides, vt)
        arg = buffer(shape, dtype=self.dtype or dtype, strides=strides, scope=self.scope)
        return tb_tir.arg(name, arg)

    def promote(self):
        shape = _try_convert_static_shape(self.shape)
        strides = _try_convert_static_shape(self.strides)
        if shape is not None and strides is not None and self.dtype is not None:
            buf = buffer(shape, self.dtype, strides=strides, scope=self.scope)
            return TIRAnnot(data=buf)


class TensorAnnot(BufferAnnot):
    @staticmethod
    def _construct_strides(shape: tuple[Any]):
        s, strides = 1, [1]
        for dim in shape[:0:-1]:
            s *= dim
            strides.append(s)
        return tuple(reversed(strides))

    def __call__(
        self,
        shape: tuple[Unpack[_Shapes]],
        dtype: _DType = "float32",
        data=None,
        strides=None,
        elem_offset=None,
        scope=None,
        align=0,
        offset_factor=0,
        buffer_type="",
        axis_separators=None,
    ):
        if isinstance(shape, (int, PrimExpr)):
            shape = (shape,)
        strides = strides or self._construct_strides(shape)
        return super().__call__(
            shape=shape,
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

    def promote(self):
        shape = _try_convert_static_shape(self.shape)
        if shape is not None and self.dtype is not None:
            strides = self._construct_strides(shape)
            buf = buffer(shape, self.dtype, strides=strides, scope=self.scope)
            return TIRAnnot(data=buf)


class StridedTensorAnnot(BufferAnnot):
    def __call__(
        self,
        shape,
        strides,
        dtype: _DType = "float32",
        data=None,
        elem_offset=None,
        scope=None,
        align=0,
        offset_factor=0,
        buffer_type="",
        axis_separators=None,
    ):
        return super().__call__(
            shape=shape,
            strides=strides,
            dtype=dtype,
            data=data,
            elem_offset=elem_offset,
            scope=scope,
            align=align,
            offset_factor=offset_factor,
            buffer_type=buffer_type,
            axis_separators=axis_separators,
        )

    def __getitem__(self, params):
        shape, strides, dtype = params
        shape = _canonicalize_shape(shape)
        strides = _canonicalize_strides(strides)
        dtype = _canonicalize_dtype(dtype)
        return StridedTensorAnnot(shape, strides, dtype)


class FragmentBufferAnnot(BufferAnnot):
    @property
    def scope(self):
        return "local.fragment"


class SharedBufferAnnot(BufferAnnot):
    @property
    def scope(self):
        return "shared.dyn"


class LocalBufferAnnot(BufferAnnot):
    @property
    def scope(self):
        return "local"


class DynAnnot(Value):
    """
    Dynamic variable annotation represents a tvm tir.Var argument
    """

    def __call__(self, dtype: AnyDType = dt.float32, name: str | None = None) -> DynAnnot:
        return tir.Var(name, dtype)

    def __getitem__(self, params):
        if not isinstance(params, tuple):
            params = (params,)
        dtype = None
        if len(params) == 1:
            (name,) = params
        if len(params) == 2:
            dtype, name = params
        dtype = _canonicalize_dtype(dtype) or dt.int32
        return DynAnnot(kind="dynamic", dtype=dtype, name=name)


@dataclass
class DTypeAnnot(Annot):
    """
    Data type annotation ensures automatically conversion from AnyDType to dtype
    >>> def foo(A: T.dtype): print(A)
    >>> foo(torch.float32)
    dtype('float32')
    >>> foo(T.float32)
    dtype('float32')
    >>> foo('float32')
    dtype('float32')
    """

    name: str | None = None

    def is_kernel_arg(self) -> bool:
        return False

    def with_name(self, name):
        return DTypeAnnot(name=name)

    def get_key_parser(self):
        return lambda name, value: (dt.dtype(value),)

    def create_prim_func_arg(self, name, value, vt):
        return dt.dtype(value)

    def __repr__(self):
        return self.name + "$dtype"


@dataclass
class TIRAnnot(Annot):
    """
    TIR annotation is used to directly pass tir.Buffer or tir.Var as kernel arguments
    >>> def foo(A: T.Buffer((128,), T.float32)): ...
    """

    data: tir.Buffer | tir.Var

    def is_kernel_arg(self) -> bool:
        return True

    def get_key_parser(self):
        return lambda name, value: (None,)

    def create_prim_func_arg(self, name, value, vt):
        return tb_tir.arg(name, self.data)

    def with_name(self, name: str):
        IRBuilder.name(name, self.data)
        return self

    def __repr__(self):
        return repr(self.data)


if TYPE_CHECKING:

    class Buffer(Generic[_Shape, _DType]):
        def __init__(
            shape: tuple[Unpack[_Shapes]],
            dtype: _DType = "float32",
            data=None,
            strides=None,
            elem_offset=None,
            scope=None,
            align=0,
            offset_factor=0,
            buffer_type="",
            axis_separators=None,
        ) -> Buffer[Callable[[Unpack[_Shapes]]], _DType]: ...

        @property
        def shape(self: Buffer[Callable[[Unpack[_Shapes]]], _DType]) -> tuple[Unpack[_Shapes]]: ...

        @property
        def dtype(self: Buffer[Callable[[Unpack[_Shapes]]], _DType]) -> dt.dtype[_DType]: ...

        @property
        def strides(self) -> tuple[tir.PrimExpr]: ...

        def scope(self) -> Scope: ...

    class Tensor(Generic[_Shape, _DType], Buffer[_Shape, _DType]):
        def __new__(
            shape: tuple[Unpack[_Shapes]],
            dtype: _DType = "float32",
            data=None,
            strides=None,
            elem_offset=None,
            scope=None,
            align=0,
            offset_factor=0,
            buffer_type="",
            axis_separators=None,
        ) -> Tensor[Callable[[Unpack[_Shapes]]], _DType]: ...

    class StridedTensor(Generic[_Shape, _Stride, _DType], Buffer[_Shape, _DType]):
        def __new__(
            shape: tuple[Unpack[_Shapes]],
            strides=None,
            dtype: _DType = "float32",
            data=None,
            elem_offset=None,
            scope=None,
            align=0,
            offset_factor=0,
            buffer_type="",
            axis_separators=None,
        ) -> Tensor[Callable[[Unpack[_Shapes]]], _DType]: ...

    class FragmentBuffer(Generic[_Shape, _DType], Buffer[_Shape, _DType]):
        pass

    class LocalBuffer(Generic[_Shape, _DType], Buffer[_Shape, _DType]):
        pass

    class SharedBuffer(Generic[_Shape, _DType], Buffer[_Shape, _DType]):
        pass

    class dyn(tir.Var):
        def __new__(cls, dtype: _DType = "float32", name: str | None = None) -> dyn[_DType]: ...

        @property
        def dtype(self: dyn[_DType]) -> dt.dtype[_DType]: ...

else:
    Buffer = BufferAnnot()
    Tensor = TensorAnnot()
    StridedTensor = StridedTensorAnnot()
    FragmentBuffer = FragmentBufferAnnot()
    SharedBuffer = SharedBufferAnnot()
    LocalBuffer = LocalBufferAnnot()
    dyn = DynAnnot()


@dataclass
class FuncAnnot:
    sig: inspect.Signature
    arg_names: list[str]
    annots: dict[str, Annot]
    arg_parser: dict[str, Callable[[Any], tuple[Any, ...]]]
    ker_arg_names: list[str]

    @classmethod
    def from_sig_annots(cls, sig: inspect.Signature, func_annots: dict[str, Any]) -> FuncAnnot:
        annots = {}
        arg_parser = {}
        ker_arg_names = []
        for param in sig.parameters.values():
            name = param.name
            annot = func_annots.get(name, Value("static", name))
            if not isinstance(annot, Annot):
                if not isinstance(annot, type) and callable(annot):
                    annot = annot()
                if annot is dt.dtype:
                    annot = DTypeAnnot(name=name)
                elif isinstance(annot, (tir.Buffer, tir.Var)):
                    annot = TIRAnnot(data=annot)
                else:
                    annot = Value(kind="static", name=name)
            annot = annot.promote() or annot
            annots[name] = annot.with_name(name)
            if annot.is_kernel_arg():
                ker_arg_names.append(name)
            arg_parser[name] = annot.get_key_parser()
        arg_names = list(sig.parameters.keys())
        return FuncAnnot(sig, arg_names, annots, arg_parser, ker_arg_names)

    def parse_key(self, *args, **kws):
        """
        Parse arguments and generates the cache key for jit caching
        """
        args = {name: arg for name, arg in zip(self.arg_names, args)}
        arg_dict = dict(**args, **kws)
        parsed = []
        for name, value in arg_dict.items():
            key = self.arg_parser[name](name, value)
            parsed.append((name, key))
        return tuple(sorted(parsed))

    def convert_to_kernel_args(self, *args, **kws):
        args = {name: arg for name, arg in zip(self.arg_names, args)}
        arg_dict = dict(**args, **kws)
        return [arg_dict[name] for name in self.ker_arg_names]

    def create_argument(self, name: str, value: Any, vt: ArgVarTable):
        """
        Convert the argument into a tvm tir argument (tir.Var | tir.Buffer) for prim func generation
        """
        return self.annots[name].create_prim_func_arg(name, value, vt)

    def is_all_static(self):
        """
        Check if all arguments are static (i.e., can be fully determined at compile time)
        """
        return all(isinstance(annot, TIRAnnot) for annot in self.annots.values())

    def get_all_static_args(self):
        res = {}
        for name, annot in self.annots.items():
            if isinstance(annot, TIRAnnot):
                res[name] = annot.data
        return res

    def get_compile_time_unknown_args(self):
        res = []
        for name, annot in self.annots.items():
            if not isinstance(annot, TIRAnnot):
                res.append(name)
        return res

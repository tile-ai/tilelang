"""Native tensor/scalar metadata shared by code generation, JIT, and caching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

if TYPE_CHECKING:
    import torch
    import tilelang.language as T


def _storage_dtype(value) -> tvm_ffi.dtype:
    """Preserve DLPack fields when a reflected primitive type crosses to Python."""
    if isinstance(value, ir.PrimType):
        value = value.dtype
    return value if isinstance(value, tvm_ffi.dtype) else tvm_ffi.dtype(value)


@dataclass
class KernelParam:
    """Describe one public argument without depending on a removed buffer_map.

    ``kind`` distinguishes a rank-zero tensor from a scalar. Two-argument
    construction keeps its established interpretation: an empty shape is a
    scalar, and a nonempty shape is a tensor. Native constructors set the kind
    from the parameter type, including for rank-zero tensors and raw pointers.
    """

    dtype: tvm_ffi.dtype
    shape: list[int | ir.Expr]
    kind: Literal["tensor", "scalar", "pointer"] | None = None

    def __post_init__(self):
        self.dtype = _storage_dtype(self.dtype)
        if self.kind is None:
            self.kind = "tensor" if self.shape else "scalar"
        if self.kind not in ("tensor", "scalar", "pointer"):
            raise ValueError(f"Unknown kernel parameter kind: {self.kind}")
        if self.kind != "tensor" and self.shape:
            raise ValueError("Only tensor parameters have a shape")

    @classmethod
    def from_tensor(cls, tensor: tirx.TensorType | ir.Expr):
        """Read dtype and shape from a native TensorType or tensor expression."""
        ty = tensor if isinstance(tensor, tirx.TensorType) else tensor.ty
        if not isinstance(ty, tirx.TensorType):
            raise TypeError(f"Expected a native tensor type, got {type(ty).__name__}")
        shape = []
        for dim in ty.shape:
            if isinstance(dim, ir.IntImm):
                shape.append(int(dim.value))
            elif ir.is_prim_expr(dim):
                shape.append(dim)
            else:
                raise ValueError(f"Unsupported tensor dimension: {type(dim).__name__}")
        return cls(_storage_dtype(ty.dtype), shape, "tensor")

    @classmethod
    def from_buffer(cls, buffer: tirx.TensorType | ir.Expr):
        """Retain this metadata constructor spelling for native tensor inputs."""
        return cls.from_tensor(buffer)

    @classmethod
    def from_var(cls, var: ir.Var):
        """Dispatch by the native parameter type; raw pointers have no shape."""
        if not isinstance(var, ir.Var):
            raise TypeError("Kernel parameters must be native ir.Var objects")
        if isinstance(var.ty, tirx.TensorType):
            return cls.from_tensor(var)
        if isinstance(var.ty, ir.PrimType):
            return cls(_storage_dtype(var.ty), [], "scalar")
        if isinstance(var.ty, ir.PtrType):
            return cls(tvm_ffi.dtype("handle"), [], "pointer")
        raise TypeError(f"Unsupported kernel parameter type: {var.ty}")

    def is_scalar(self) -> bool:
        return self.kind != "tensor"

    def is_unsigned(self) -> bool:
        return self.dtype.type_code == tvm_ffi.dtype("uint8").type_code

    def is_float8(self) -> bool:
        return str(self.dtype).startswith("float8")

    def is_float4(self) -> bool:
        return str(self.dtype).startswith("float4") or str(self.dtype).startswith("custom[float4_")

    def is_boolean(self) -> bool:
        return self.dtype.is_bool

    def torch_dtype(self) -> torch.dtype:
        import tilelang.language as T

        return T.dtype(self.dtype).as_torch()

    def tilelang_dtype(self) -> T.dtype:
        import tilelang.language as T

        return T.dtype(self.dtype)


@dataclass
class CompiledArtifact:
    """IR, generated source, and optional executable for a compiled kernel."""

    host_mod: ir.IRModule
    device_mod: ir.IRModule
    params: list[KernelParam] | None
    kernel_source: str
    rt_mod: tvm_ffi.Module | None = None
    target: Target | None = None
    target_host: Target | None = None


_PARAM_SCHEMA_VERSION = 2


def dump_kernel_params(params: list[KernelParam]) -> str:
    """Serialize the complete shape graph, retaining shared symbolic identities.

    The native cache format is intentionally versioned separately from the old
    TVM cache. Data is reflected IR plus scalar metadata and executes no code.
    """
    records = [{"dtype": str(_storage_dtype(p.dtype)), "shape": list(p.shape), "kind": p.kind} for p in params]
    payload = {"schema_version": _PARAM_SCHEMA_VERSION, "params": records}
    return ir.save_json(tvm_ffi.convert(payload))


def load_kernel_params(text: str) -> list[KernelParam]:
    """Read the matching native cache format; reject stale or invalid metadata."""
    payload = ir.load_json(text)
    if not isinstance(payload, tvm_ffi.Map) or payload.get("schema_version") != _PARAM_SCHEMA_VERSION:
        raise ValueError("Unsupported TileLang kernel parameter cache schema; recompile the kernel")
    params = []
    for record in payload["params"]:
        shape = []
        for dim in record["shape"]:
            if isinstance(dim, int):
                shape.append(dim)
            elif isinstance(dim, ir.IntImm):
                shape.append(int(dim.value))
            elif ir.is_prim_expr(dim):
                shape.append(dim)
            else:
                raise ValueError(f"Unsupported cached dimension: {type(dim).__name__}")
        params.append(KernelParam(tvm_ffi.dtype(str(record["dtype"])), shape, str(record["kind"])))
    return params

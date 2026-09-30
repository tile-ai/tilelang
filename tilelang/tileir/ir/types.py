"""Types and dtype registries for typed TileIR values."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any


# Storage widths for floating-point types.

_FLOAT_NAME_TO_BITS: dict[str, int] = {
    "float4_e2m1fn": 4,
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
    "float8_e8m0fnu": 8,
    "float16": 16,
    "bfloat16": 16,
    # TF32 has 19 significant bits but occupies 32 bits of storage.
    "tf32": 32,
    "float32": 32,
    "float64": 64,
}


# Dtype registry rows.

# (canonical_name, aliases, mlir_ir_type_attr_name, ct_element_attr_name)
_FLOAT_DTYPE_ROWS: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    ("float16", (), "F16Type", "Float16"),
    # tf32 stays a distinct type for Gemm operand casts, but TileLang's
    # ``tfloat32`` buffers alias to float32 STORAGE: tf32 is a compute-only
    # format — cuda_tile's elementwise ops (mulf/addf/…) reject tile<…xtf32>,
    # and the Gemm emit already down-casts f32 operands to tf32 (the
    # f32×f32→tf32 TileLang convention), so f32 storage composes with both.
    ("tf32", (), "FloatTF32Type", "TFloat32"),
    ("float32", ("tfloat32",), "F32Type", "Float32"),
    ("float64", (), "F64Type", "Float64"),
    ("bfloat16", (), "BF16Type", "BFloat16"),
    ("float8_e4m3fn", ("float8_e4m3",), "Float8E4M3FNType", "Float8E4M3FN"),
    ("float8_e5m2", (), "Float8E5M2Type", "Float8E5M2"),
    ("float8_e8m0fnu", (), "Float8E8M0FNUType", "Float8E8M0FNU"),
    ("float4_e2m1fn", (), "Float4E2M1FNType", "Float4E2M1FN"),
)

# (canonical_name, aliases, bits, ct_element_attr_name_or_None)
_INT_DTYPE_ROWS: tuple[tuple[str, tuple[str, ...], int, str | None], ...] = (
    ("bool", (), 1, None),
    ("int4", (), 4, "Int4"),
    ("int8", ("uint8",), 8, "Int8"),
    ("int16", ("uint16",), 16, "Int16"),
    ("int32", ("uint32",), 32, "Int32"),
    ("int64", ("uint64",), 64, "Int64"),
)


# DType


class DType:
    """Singleton element-type descriptor.

    Instances are created once at module import by ``_build_dtype_registry()``
    and interned in ``DTYPES``.  Do not instantiate directly — use ``dtype()``.

    Attributes
    ----------
    name : str
        Canonical dtype name (e.g. ``"float16"``, ``"int32"``).
    bitwidth : int
        Significant bit-width of the type.
    _is_float : bool
        True for float dtypes; False for integer dtypes.
    _mlir_ir_attr : str | None
        Attribute name on ``cuda_tile._mlir.ir`` used to build the MLIR type
        (float dtypes only).  E.g. ``"F16Type"``.
    _ct_attr : str | None
        Attribute name on the ``ct`` (cuda_tile._mlir) builder used to produce
        the cuda_tile element constant.  None for ``bool``.
    """

    __slots__ = ("name", "bitwidth", "_is_float", "_mlir_ir_attr", "_ct_attr")

    def __init__(
        self,
        name: str,
        bitwidth: int,
        *,
        is_float: bool,
        mlir_ir_attr: str | None,
        ct_attr: str | None,
    ) -> None:
        self.name: str = name
        self.bitwidth: int = bitwidth
        self._is_float: bool = is_float
        self._mlir_ir_attr: str | None = mlir_ir_attr
        self._ct_attr: str | None = ct_attr

    # MLIR / cuda_tile factories (require a live MLIR context)

    def mlir_type(self, ctx: Any) -> Any:
        """Return the MLIR type object for this dtype.

        Parameters
        ----------
        ctx :
            The ``cuda_tile._mlir`` module (i.e. the object that exposes both
            ``ctx.ir`` and ``ctx.<ct_attr>``).
        """
        ir = ctx.ir
        if self._is_float:
            assert self._mlir_ir_attr is not None
            return getattr(ir, self._mlir_ir_attr).get()
        # Integer: signless IntegerType of self.bitwidth
        return ir.IntegerType.get_signless(self.bitwidth)

    def ct_element(self, ctx: Any) -> Any:
        """Return the cuda_tile element wrapper for this dtype.

        ``bool`` has no CUDA Tile IR element wrapper, so it uses the i1 MLIR
        type directly.
        """
        if self._ct_attr is None:
            # bool: no dedicated ct element constant; return the i1 mlir type.
            return ctx.ir.IntegerType.get_signless(self.bitwidth)
        return getattr(ctx, self._ct_attr)

    # Dunder helpers

    def __repr__(self) -> str:
        return f"DType({self.name!r}, bitwidth={self.bitwidth})"

    def __str__(self) -> str:
        return self.name

    # DType objects are singletons — identity comparison is intentional.
    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return hash(self.name)


# Registry construction (runs once at import)


def _build_dtype_registry() -> tuple[dict[str, DType], dict[str, str]]:
    """Build DTYPES (canonical name -> DType) and _ALIAS_MAP (alias -> canonical name).

    Returns (dtypes, alias_map).
    """
    dtypes: dict[str, DType] = {}
    alias_map: dict[str, str] = {}  # alias -> canonical name

    for name, aliases, mlir_ir_attr, _ct_attr in _FLOAT_DTYPE_ROWS:
        bits = _FLOAT_NAME_TO_BITS[name]
        obj = DType(name, bits, is_float=True, mlir_ir_attr=mlir_ir_attr, ct_attr=_ct_attr)
        dtypes[name] = obj
        for alias in aliases:
            alias_map[alias] = name

    for name, aliases, bits, ct_attr in _INT_DTYPE_ROWS:
        obj = DType(name, bits, is_float=False, mlir_ir_attr=None, ct_attr=ct_attr)
        dtypes[name] = obj
        for alias in aliases:
            alias_map[alias] = name

    return dtypes, alias_map


DTYPES: dict[str, DType]
_ALIAS_MAP: dict[str, str]
DTYPES, _ALIAS_MAP = _build_dtype_registry()


def dtype(name: str) -> DType:
    """Return the ``DType`` singleton for *name* (canonical or alias).

    Guarantees: ``dtype("float16") is dtype("float16")`` — singletons.

    Raises
    ------
    KeyError
        If *name* is not a known dtype or alias.
    """
    canonical = _ALIAS_MAP.get(name, name)
    try:
        return DTYPES[canonical]
    except KeyError:
        raise KeyError(f"Unknown TileIR dtype: {name!r}") from None


def dtype_from_mlir(mlir_ty: Any) -> DType:
    """Return the ``DType`` singleton corresponding to an MLIR type object.

    This performs a linear scan over ``DTYPES``.  It is called at most once per
    buffer/op during lowering (not in a hot loop), so the scan is acceptable.
    A live MLIR context is required because MLIR type objects need ``==`` to work.

    Raises
    ------
    KeyError
        If *mlir_ty* does not match any known dtype.
    """
    # Float types are identified by their MLIR wrapper class. Integer wrappers
    # expose their storage width directly.
    for dt in DTYPES.values():
        # For integer types we can compare width without a live ctx.
        if not dt._is_float:
            # cuda_tile._mlir.ir.IntegerType exposes a .width attribute.
            try:
                if mlir_ty.width == dt.bitwidth:
                    return dt
            except AttributeError:
                pass
        else:
            # For float types we compare the class name of the mlir type object
            # with the expected ir attr name (e.g. F16Type).
            assert dt._mlir_ir_attr is not None
            if type(mlir_ty).__name__ == dt._mlir_ir_attr:
                return dt

    raise KeyError(f"No TileIR DType matches MLIR type: {mlir_ty!r}")


# MemSpace


class MemSpace(enum.Enum):
    """TileIR memory-space annotation."""

    GLOBAL = "global"
    SHARED = "shared"
    REGISTER = "register"


# Layout


class Layout:
    """Thin wrapper around ``lowering.types.TileLayout``."""

    __slots__ = ("_tile_layout",)

    def __init__(self, tile_layout: Any) -> None:
        self._tile_layout = tile_layout

    @property
    def tile_layout(self) -> Any:
        """The underlying ``TileLayout`` dataclass."""
        return self._tile_layout

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Layout):
            return NotImplemented
        return self._tile_layout == other._tile_layout

    def __hash__(self) -> int:
        return hash(self._tile_layout)

    def __repr__(self) -> str:
        return f"Layout({self._tile_layout!r})"


# TileType


@dataclass(frozen=True)
class TileType:
    """Immutable descriptor for a TileIR value type.

    Attributes
    ----------
    dtype : DType
        Element type singleton.
    shape : tuple[int, ...]
        Logical shape of the tile.  Empty tuple ``()`` denotes a scalar.
    space : MemSpace
        Memory space where the tile resides.
    layout : Layout | None
        Optional layout annotation (wraps ``TileLayout``).
    """

    dtype: DType
    shape: tuple[int, ...]
    space: MemSpace
    layout: Layout | None

    def is_scalar(self) -> bool:
        """Return True iff this is a 0-d (scalar) type."""
        return len(self.shape) == 0

    def with_shape(self, shape: tuple[int, ...]) -> TileType:
        """Return a copy with a different shape; other fields are preserved."""
        return TileType(self.dtype, tuple(shape), self.space, self.layout)

    def __repr__(self) -> str:
        return f"TileType(dtype={self.dtype}, shape={self.shape}, space={self.space}, layout={self.layout!r})"

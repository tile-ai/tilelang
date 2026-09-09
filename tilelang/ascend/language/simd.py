"""T.simd.* - Raw CCE vector intrinsics for Ascend SIMD programming.

Function names match the underlying CCE intrinsics directly.

MODE_MERGING preserves inactive lanes of the mutable destination register.
On Ascend 950, validated 8/16/32-bit operations use the CCE merging overloads;
``vdupv`` maps to CCE's vector ``vdup`` overload. Scalar BF16 ``vdup`` retains
software merging to work around CANN 9.2's inactive-lane bug. Precision-specific
SFU algorithms also retain their wrappers, including the default exact FP32
division and the ``ftz_false`` variants of ``vexp``, ``vln``, and ``vsqrt``.
"""

from tvm import tirx
from tvm.tirx import BufferLoad
from tvm.tirx.script.builder.ir import bind as _bind
from tvm.script.ir_builder import IRBuilder
from tilelang import tvm
from tilelang._typing import DType, ShapeType
from tilelang.language.allocate import alloc_var as _alloc_var, alloc_local as _alloc_local
from tilelang.language.builtin import access_ptr

_Op = tirx.op.Op.get


class SimdPair:
    """Wraps a two-result SIMD intrinsic and preserves each result dtype.

    ``a, b = ...`` emits two ``pair_get`` calls against the pair. The pair-
    producing op (``vintlv`` / ``vdintlv`` / ``vld2`` / ``vaddc``) is bound once
    at its call site by the frontend, so both ``pair_get`` calls reference a
    single bound variable rather than inlining the pair expression twice.

    ``dtype`` is a pair describing both result types, such as the
    ``(boolx256, int32x64)`` carry/result pair returned by ``vaddc``. Omitting it
    defaults both result types to the dtype of the backing TIR expression.
    """

    def __init__(self, pair, dtype=None):
        self._pair = pair
        if dtype is None:
            dtype = (pair.dtype, pair.dtype)
        if not isinstance(dtype, (tuple, list)):
            raise TypeError(f"SimdPair `dtype` must be a tuple or list, got {type(dtype).__name__}")
        if len(dtype) != 2:
            raise ValueError(f"SimdPair requires exactly two result dtypes, got {len(dtype)}")
        self._dtype = tuple(tvm.DataType(result_dtype) for result_dtype in dtype)

    @property
    def dtype(self):
        return self._dtype

    def __getitem__(self, index):
        index = _pair_index(index)
        return tirx.call_intrin(str(self._dtype[index]), _Op("tl.simd.pair_get"), self._pair, index)

    def __iter__(self):
        yield self[0]
        yield self[1]


def _pair_index(index) -> int:
    if isinstance(index, tirx.IntImm):
        index = int(index.value)
    if not isinstance(index, int) or isinstance(index, bool):
        raise TypeError(f"SIMD pair index must be the integer 0 or 1, got {type(index).__name__}")
    if index not in (0, 1):
        raise IndexError(f"SIMD pair index must be 0 or 1, got {index}")
    return index


_ELEM_BITS = {
    "float32": 32,
    "float16": 16,
    "bfloat16": 16,
    "int32": 32,
    "uint32": 32,
    "int16": 16,
    "uint16": 16,
    "int64": 64,
    "uint64": 64,
    "int8": 8,
    "uint8": 8,
    "float8_e4m3": 8,
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
    "float8_e8m0fnu": 8,
    "float4_e2m1fn": 4,
}


def _vreg_lanes(dtype_str: DType) -> int:
    key = str(dtype_str)
    if key == "bool":
        return 256
    if key not in _ELEM_BITS:
        raise ValueError(f"Unknown element dtype '{key}'. Must be one of {sorted(_ELEM_BITS)}")
    elem_bits = _ELEM_BITS[key]
    return 2048 // elem_bits


def _vec_dtype(elem_dtype: DType) -> str:
    lanes = _vreg_lanes(elem_dtype)
    return f"{elem_dtype}x{lanes}"


def pset(elem_width: int, dist: str = "PAT_ALL"):
    """Create a predicate mask: pset_bXX(dist).

    Returns a vector_bool (boolx256).
    dist: "PAT_ALL", "PAT_VL1".."PAT_VL128", "PAT_M3", "PAT_M4", "PAT_H", "PAT_Q", etc.
    """
    return tirx.call_intrin("boolx256", _Op("tl.simd.pset"), elem_width, dist)


def pge(elem_width: int, dist: str = "PAT_ALL"):
    """Create a predicate mask from pge_bXX(dist)."""
    return tirx.call_intrin("boolx256", _Op("tl.simd.pge"), elem_width, dist)


def update_mask(value, width=32):
    """Runtime tail predicate: lanes [0, value) active (b8/b16/b32)."""
    return tirx.call_intrin("boolx256", _Op("tl.simd.update_mask"), value, width)


def _mask_bits(dtype_like) -> int:
    """CCE predicate width (b8/b16/b32/b64) for a dtype's element type."""
    elem = str(dtype_like).split("x", 1)[0]
    if elem not in _ELEM_BITS:
        raise ValueError(f"Cannot infer default mask width for dtype '{dtype_like}'")
    return max(_ELEM_BITS[elem], 8)  # CCE has no pset_b4; float4 -> b8


# Auto default-mask cache: one bound pset per element-bit width, reused across maskless
# ops as long as the frame it was bound into is still an enclosing scope. The TVMScript
# parser hands out a fresh Python wrapper per frame access, so identity is compared with
# ``same_as`` (underlying-object equality), not ``id()``. Reset per SimdVF block by its frame.
_default_mask_cache: dict = {}  # elem_bits -> (declaring_frame, mask_var)


def _reset_default_mask_cache() -> None:
    _default_mask_cache.clear()


def _frame_on_stack(frame) -> bool:
    """True if ``frame`` is still an enclosing scope on the IRBuilder frame stack."""
    if frame is None or not IRBuilder.is_in_scope():
        return False
    return any(frame.same_as(f) for f in IRBuilder.current().frames)


def _default_mask_bits(bits: int):
    """Bind (once per scope) and return an all-lanes pset mask of the given predicate
    width (b8/b16/b32/b64).

    Emits ``vector_bool v = pset_bXX(PAT_ALL);`` into the current SimdVF frame so the
    Ascend codegen BindNode path renders it correctly (an inline pset is not handled in
    expression context). Repeated maskless ops reuse the same bound mask while its
    declaring scope is still open, instead of emitting a new pset each time.
    """
    cached = _default_mask_cache.get(bits)
    if cached is not None and _frame_on_stack(cached[0]):
        return cached[1]
    var = _bind(pset(bits, "PAT_ALL"))
    if IRBuilder.is_in_scope():
        frames = IRBuilder.current().frames
        _default_mask_cache[bits] = (frames[-1] if len(frames) else None, var)
    return var


def _default_mask(dtype_like):
    """Bind and return an all-lanes pset mask matching dtype_like's element width."""
    return _default_mask_bits(_mask_bits(dtype_like))


def pand(src0, src1, mask):
    return tirx.call_intrin("boolx256", _Op("tl.simd.pand"), src0, src1, mask)


def por(src0, src1, mask):  # codespell:ignore
    return tirx.call_intrin("boolx256", _Op("tl.simd.por"), src0, src1, mask)  # codespell:ignore


def pxor(src0, src1, mask):
    return tirx.call_intrin("boolx256", _Op("tl.simd.pxor"), src0, src1, mask)


def pnot(src, mask):
    return tirx.call_intrin("boolx256", _Op("tl.simd.pnot"), src, mask)


def psel(src0, src1, mask):
    return tirx.call_intrin("boolx256", _Op("tl.simd.psel"), src0, src1, mask)


def ppack(src, part=0):
    """Predicate pack 2:1 (zeroing): dst = ppack(src, LOWER/HIGHER)."""
    if isinstance(part, tirx.IntImm):
        part = int(part.value)
    if isinstance(part, int):
        part = ("LOWER", "HIGHER")[part]
    return tirx.call_intrin("boolx256", _Op("tl.simd.ppack"), src, part)


def punpack(src, part=0):
    """Predicate unpack 1:2 (zeroing): dst = punpack(src, LOWER/HIGHER)."""
    if isinstance(part, tirx.IntImm):
        part = int(part.value)
    if isinstance(part, int):
        part = ("LOWER", "HIGHER")[part]
    return tirx.call_intrin("boolx256", _Op("tl.simd.punpack"), src, part)


def pintlv(src0, src1, width=32):
    """Predicate interleave -> pair of predicates (b8/b16/b32)."""
    pair = tirx.call_intrin("boolx256", _Op("tl.simd.pintlv"), src0, src1, width)
    if not IRBuilder.is_in_scope():
        raise RuntimeError("pintlv requires an active IRBuilder to bind the pair")
    return SimdPair(_bind(pair), ("boolx256", "boolx256"))


def pdintlv(src0, src1, width=32):
    """Predicate deinterleave -> pair of predicates (b8/b16/b32)."""
    pair = tirx.call_intrin("boolx256", _Op("tl.simd.pdintlv"), src0, src1, width)
    if not IRBuilder.is_in_scope():
        raise RuntimeError("pdintlv requires an active IRBuilder to bind the pair")
    return SimdPair(_bind(pair), ("boolx256", "boolx256"))


def alloc_var(dtype: DType):
    """Allocate a single mutable SIMD register variable (return-value style).

    Uses ``local.var`` scope and behaves as one vector register value. For an
    addressable array of registers (``v[i]``), use :func:`alloc_local`.
    """
    return _alloc_var(_vec_dtype(dtype), scope="local.var")


def alloc_local(shape: ShapeType, dtype: DType):
    """Allocate an addressable array of mutable SIMD register variables.

    Uses ``local`` scope so each element is an individually addressable
    register, allowing indexed access like ``v[i]``::

        v = T.simd.alloc_local(4, "float32")
        for i in T.Unroll(4, explicit=True):
            v[i] = vld(s_ub[i * VL])
    """
    return _alloc_local(shape, _vec_dtype(dtype), scope="local")


# -- Data movement ---------------------------------------------------------------


_PREDICATE_LOAD_DISTS = ("NORM", "US", "DS")
_PREDICATE_STORE_DISTS = ("NORM", "PK")
_PREDICATE_PAYLOAD_BYTES = {
    "NORM": 32,
    "US": 16,
    "DS": 64,
    "PK": 16,
}


def _validate_predicate_dist(name: str, dist: str, allowed: tuple[str, ...]) -> None:
    if not isinstance(dist, str):
        raise TypeError(f"{name}: `dist` must be a string, got {type(dist).__name__}")
    if dist not in allowed:
        raise ValueError(f"{name}: `dist` must be one of {allowed}, got {dist!r}")


def _predicate_access_ptr(addr, access_type: str, dist: str):
    if not isinstance(addr, BufferLoad):
        return addr
    elem_dtype = str(addr.buffer.dtype)
    if elem_dtype not in _ELEM_BITS:
        raise ValueError(f"Predicate load/store does not support buffer dtype {elem_dtype!r}")
    payload_bits = _PREDICATE_PAYLOAD_BYTES[dist] * 8
    elem_bits = _ELEM_BITS[elem_dtype]
    extent = (payload_bits + elem_bits - 1) // elem_bits
    return access_ptr(addr, access_type, extent=extent)


def pld(addr, dist="NORM"):
    """Load a predicate from UB. Express address offsets in ``addr``."""
    _validate_predicate_dist("pld", dist, _PREDICATE_LOAD_DISTS)
    addr = _predicate_access_ptr(addr, "r", dist)
    return tirx.call_intrin("boolx256", _Op("tl.simd.pld"), addr, dist)


def pst(addr, src, dist="NORM"):
    """Store a predicate to UB. Express address offsets in ``addr``."""
    _validate_predicate_dist("pst", dist, _PREDICATE_STORE_DISTS)
    addr = _predicate_access_ptr(addr, "w", dist)
    return tirx.call_intrin("void", _Op("tl.simd.pst"), addr, src, dist)


def vld(addr, dist="NORM"):
    """Vector load. Returns a typed vector register.

    addr can be a BufferLoad auto-wrapped as tl.access_ptr. Express address
    offsets in ``addr``.
    """
    if isinstance(addr, BufferLoad):
        elem_dtype = str(addr.buffer.dtype)
        lanes = _vreg_lanes(elem_dtype)
        extent = lanes
        if isinstance(dist, str):
            if dist.startswith("BRC_"):
                extent = 1
            elif dist.startswith("E2B_"):
                extent = 8
            elif dist == "BLK":
                extent = 32
            elif dist == "UNPK4_B8":
                extent = lanes // 4
            elif dist.startswith("US_") or dist.startswith("UNPK_"):
                extent = lanes // 2
            elif dist.startswith("DS_"):
                extent = lanes * 2
        addr = access_ptr(addr, "r", extent=extent)
        vec_dtype = f"{elem_dtype}x{lanes}"
    else:
        inner_load = addr.args[0]
        vec_dtype = _vec_dtype(inner_load.buffer.dtype)

    return tirx.call_intrin(vec_dtype, _Op("tl.simd.vld"), addr, dist)


_VLD2_DISTS = ("DINTLV_B8", "DINTLV_B16", "DINTLV_B32")


def vld2(addr, dist="DINTLV_B16", off=None):
    """Dual-dest vector load: ``a, b = vld2(x_ub[i, col], dist=\"DINTLV_B8\")``.

    Supported dists:
      - ``DINTLV_B8``: load 512xu8/fp8 -> two 256-lane regs (even/odd bytes)
      - ``DINTLV_B16``: load 256xbf16/u16 -> two 128-lane regs
      - ``DINTLV_B32``: load 128xf32/u32 -> two 64-lane regs

    The access_ptr footprint is always 2x the single-vector width. This is an
    opaque memory load: the pair is bound once at this program point so the two
    ``pair_get`` calls from the unpack share a single load rather than issuing
    two independent loads.
    """
    if isinstance(dist, str) and dist not in _VLD2_DISTS:
        raise ValueError(f"vld2 only supports dist in {_VLD2_DISTS}, got {dist!r}")
    if isinstance(addr, BufferLoad):
        elem_dtype = str(addr.buffer.dtype)
        lanes = _vreg_lanes(elem_dtype)
        # Dual-dest deinterleave always consumes 2x vector width from UB.
        extent = lanes * 2
        addr = access_ptr(addr, "r", extent=extent)
        vec_dtype = f"{elem_dtype}x{lanes}"
    else:
        # Pre-wrapped access_ptr: this path does not auto-double the footprint
        # (same as vld). Callers must size extent to 2x vector width themselves.
        inner_load = addr.args[0]
        vec_dtype = _vec_dtype(inner_load.buffer.dtype)

    args = [addr, dist]
    if off is not None:
        args.append(off)
    pair = tirx.call_intrin(vec_dtype, _Op("tl.simd.vld2"), *args)
    # Bind at the original site (opaque load). Requires an active IRBuilder -
    # do not fall back to an unbound SimdPair (that reintroduces duplicate
    # inline loads under unpack).
    if not IRBuilder.is_in_scope():
        raise RuntimeError("vld2 requires an active IRBuilder to bind the opaque load")
    return SimdPair(_bind(pair), (vec_dtype, vec_dtype))


def vsts(addr, src, mask=None, dist="NORM_B32", extent=None):
    """Vector store to ``addr``.

    addr can be a BufferLoad auto-wrapped as tl.access_ptr. Express address
    offsets in ``addr``.
    Optional ``extent`` overrides the default access_ptr footprint (e.g. 8 for
    a dense PAT_VL8 NORM_B16 recip pack).
    """
    if mask is None:
        mask = _default_mask(src.dtype)
    if isinstance(addr, BufferLoad):
        lanes = _vreg_lanes(addr.buffer.dtype)
        if extent is None:
            extent = lanes
            if isinstance(dist, str):
                if dist.startswith("ONEPT_"):
                    extent = 1
                elif dist.startswith("PK_"):
                    extent = lanes // 2
                elif dist.startswith("PK4_"):
                    extent = lanes // 4
        addr = access_ptr(addr, "w", extent=extent)
    return tirx.call_intrin("void", _Op("tl.simd.vsts"), addr, src, mask, dist)


def vsstb(src, base, stride, mask=None, update=False):
    """Scatter-store 32B blocks with an optional POST_UPDATE pointer.

    Passing a regular buffer access performs a store and returns ``void``.
    Set ``update=True`` with the mutable handle returned by
    :func:`make_ubuf_ptr` to enable POST_UPDATE and return the advanced
    pointer, which should be assigned back to the same handle::

        dst_ptr = T.simd.make_ubuf_ptr(dst_ub[0], "bfloat16")
        dst_ptr = T.simd.vsstb(src, dst_ptr, stride, mask, update=True)
    """
    if mask is None:
        mask = _default_mask(src.dtype)
    mutable_pointer = isinstance(base, BufferLoad) and base.buffer.scope() == "local.var" and str(base.dtype) == "handle"
    if update and not mutable_pointer:
        raise ValueError("vsstb update=True base must be a pointer from make_ubuf_ptr")
    if isinstance(base, BufferLoad) and not mutable_pointer:
        base = access_ptr(base, "w", extent=_vreg_lanes(base.buffer.dtype))
    args = [src, base, stride, mask]
    if update:
        args.append("POST_UPDATE")
    return tirx.call_intrin(
        "handle" if update else "void",
        _Op("tl.simd.vsstb"),
        *args,
    )


def make_ubuf_ptr(buf_access, dtype):
    """Allocate a mutable UB pointer for POST_UPDATE :func:`vsstb` calls.

    The pointer is carried by a ``local.var`` handle buffer. Assigning the
    handle returned by :func:`vsstb` writes it back into the same mutable
    carrier::

        dst_ptr = T.simd.make_ubuf_ptr(dst_ub[0], "bfloat16")
        dst_ptr = T.simd.vsstb(src, dst_ptr, stride, mask, update=True)
    """
    if isinstance(buf_access, BufferLoad):
        buf_access = access_ptr(buf_access, "rw")
    elif str(buf_access.dtype) != "handle":
        raise ValueError("make_ubuf_ptr expects a buffer access or handle expression")

    pointer = _alloc_var("handle", buf_access, scope="local.var")
    return pointer


# -- Binary arithmetic ------------------------------------------------------------


def vadd(src0, src1, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vadd"), src0, src1, mask, mode)


def vaddc(src0, src1, mask=None):
    """Add int32/uint32 vectors without carry-in and return ``(carry, result)``.

    ``carry`` is a ``boolx256`` predicate register and ``result`` has the same
    dtype as the inputs. The underlying Ascend ``vaddc`` instruction only
    supports full-register ``int32x64`` and ``uint32x64`` operands.
    """
    if mask is None:
        mask = _default_mask(src0.dtype)

    pair = tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vaddc"), src0, src1, mask)
    if not IRBuilder.is_in_scope():
        raise RuntimeError("vaddc requires an active IRBuilder to bind the pair")
    return SimdPair(_bind(pair), ("boolx256", src0.dtype))


def vsubc(src0, src1, mask=None):
    """Subtract int32/uint32 without carry-in and return ``(carry, result)``.

    ``carry`` is 1 where the subtraction completes without borrow.
    """
    if mask is None:
        mask = _default_mask(src0.dtype)

    pair = tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vsubc"), src0, src1, mask)
    if not IRBuilder.is_in_scope():
        raise RuntimeError("vsubc requires an active IRBuilder to bind the pair")
    return SimdPair(_bind(pair), ("boolx256", src0.dtype))


def vaddcs(src0, src1, carrysrcp, mask=None):
    """Add int32/uint32 with carry-in predicate and return ``(carry, result)``."""
    if mask is None:
        mask = _default_mask(src0.dtype)

    pair = tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vaddcs"), src0, src1, carrysrcp, mask)
    if not IRBuilder.is_in_scope():
        raise RuntimeError("vaddcs requires an active IRBuilder to bind the pair")
    return SimdPair(_bind(pair), ("boolx256", src0.dtype))


def vsubcs(src0, src1, carrysrcp, mask=None):
    """Subtract int32/uint32 with carry-in predicate and return ``(carry, result)``."""
    if mask is None:
        mask = _default_mask(src0.dtype)

    pair = tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vsubcs"), src0, src1, carrysrcp, mask)
    if not IRBuilder.is_in_scope():
        raise RuntimeError("vsubcs requires an active IRBuilder to bind the pair")
    return SimdPair(_bind(pair), ("boolx256", src0.dtype))


def vmull(src0, src1, mask=None):
    """Widening 32x32->64 multiply returning ``(lo, hi)`` (int32/uint32)."""
    if mask is None:
        mask = _default_mask(src0.dtype)

    pair = tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vmull"), src0, src1, mask)
    if not IRBuilder.is_in_scope():
        raise RuntimeError("vmull requires an active IRBuilder to bind the pair")
    return SimdPair(_bind(pair), (src0.dtype, src0.dtype))


def vsub(src0, src1, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vsub"), src0, src1, mask, mode)


def vmul(src0, src1, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vmul"), src0, src1, mask, mode)


def vmula(dst, src0, src1, mask=None, mode="MODE_ZEROING"):
    """Fused multiply-add: dst = dst + src0 * src1."""
    if mask is None:
        mask = _default_mask(dst.dtype)
    if not isinstance(dst, BufferLoad):
        raise ValueError(
            "vmula: `dst` must be a buffer access (e.g. c[0]) allocated with "
            "T.simd.alloc_local()/alloc_var(), not an immutable register value. "
            f"Got {type(dst).__name__} {dst}."
        )
    dst = access_ptr(dst, "rw")
    return tirx.call_intrin("void", _Op("tl.simd.vmula"), dst, src0, src1, mask, mode)


def vmadd(dst, src0, src1, mask=None, mode="MODE_ZEROING"):
    """Fused multiply-add: dst = dst * src0 + src1."""
    if mask is None:
        mask = _default_mask(dst.dtype)
    if not isinstance(dst, BufferLoad):
        raise ValueError(
            "vmadd: `dst` must be a buffer access (e.g. c[0]) allocated with "
            "T.simd.alloc_local()/alloc_var(), not an immutable register value. "
            f"Got {type(dst).__name__} {dst}."
        )
    dst = access_ptr(dst, "rw")
    return tirx.call_intrin("void", _Op("tl.simd.vmadd"), dst, src0, src1, mask, mode)


def vaxpy(dst, src, scalar, mask=None, mode="MODE_ZEROING"):
    """Fused scalar multiply-add: dst = src * scalar + dst."""
    if mask is None:
        mask = _default_mask(dst.dtype)
    if not isinstance(dst, BufferLoad):
        raise ValueError(
            "vaxpy: `dst` must be a buffer access (e.g. c[0]) allocated with "
            "T.simd.alloc_local()/alloc_var(), not an immutable register value. "
            f"Got {type(dst).__name__} {dst}."
        )
    dst = access_ptr(dst, "rw")
    return tirx.call_intrin("void", _Op("tl.simd.vaxpy"), dst, src, scalar, mask, mode)


# Per-op SFU precision selectors (mirrors the l2_cache_ctrl pattern in
# tilelang/language/copy_op.py: user-facing string aliases are normalized to
# integer codes here, the annotation carries only the int, and codegen reads
# the int -- no alias table lives in C++).
# "intrinsic" is the cross-function alias for the bare hardware SFU.
_PRECISION_MODES = {
    # Per-op precision selectors, 3 aliases each:
    #   "ftz_true"            - cross-function: bare hardware SFU (FTZ_TRUE)
    #   semantic alias        - "exact" (vdiv) / "ftz_false" (unary SFUs)
    #   CANN tier name        - <op>_<N>ulp_ftz_<mode> (kernel_reg_compute_utils.h)
    "vdiv": {
        "ftz_true": "hw",
        "exact": "exact",
        "vdiv_0ulp_ftz_true": "exact",
    },
    "vexp": {
        "ftz_true": "hw",
        "ftz_false": "ftz_false",
        "vexp_1ulp_ftz_false": "ftz_false",
    },
    "vln": {
        "ftz_true": "hw",
        "ftz_false": "ftz_false",
        "vln_1ulp_ftz_false": "ftz_false",
    },
    "vsqrt": {
        "ftz_true": "hw",
        "ftz_false": "ftz_false",
        "vsqrt_0ulp_ftz_false": "ftz_false",
    },
}


def _histogram_bin(bin):  # noqa: A002
    if isinstance(bin, tirx.IntImm):
        bin = int(bin.value)
    if not isinstance(bin, int) or isinstance(bin, bool):
        raise TypeError(f"Histogram bin must be the integer 0 or 1, got {bin!r}")
    if bin not in (0, 1):
        raise ValueError(f"Histogram bin must be 0 or 1, got {bin}")
    return bin


def _histogram(op_name, dst, src, mask, bin):  # noqa: A002
    if not isinstance(dst, BufferLoad):
        raise ValueError(
            f"{op_name}: `dst` must be a buffer access (e.g. hist[0]) allocated with "
            "T.simd.alloc_local()/alloc_var(), not an immutable register value. "
            f"Got {type(dst).__name__} {dst}."
        )
    if str(dst.dtype) != "uint16x128":
        raise TypeError(f"{op_name}: `dst` must be a uint16 vector, got {dst.dtype}")
    if str(src.dtype) != "uint8x256":
        raise TypeError(f"{op_name}: `src` must be a uint8 vector, got {src.dtype}")
    if mask is None:
        mask = _default_mask(src.dtype)
    dst = access_ptr(dst, "rw")
    return tirx.call_intrin("void", _Op(f"tl.simd.{op_name}"), dst, src, mask, _histogram_bin(bin))


def dhistv2(dst, src, mask=None, bin=0):  # noqa: A002
    """Accumulate a frequency histogram of a uint8 vector into a uint16 vector.

    ``bin=0`` counts values in ``[0, 127]`` and ``bin=1`` counts values in
    ``[128, 255]``. The destination register is updated in place.
    """
    return _histogram("dhistv2", dst, src, mask, bin)


def chistv2(dst, src, mask=None, bin=0):  # noqa: A002
    """Accumulate a cumulative histogram of a uint8 vector into a uint16 vector.

    ``bin=0`` returns cumulative counts through values ``[0, 127]`` and
    ``bin=1`` returns cumulative counts through values ``[128, 255]``. The
    destination register is updated in place.
    """
    return _histogram("chistv2", dst, src, mask, bin)


# Canonical -> integer code passed through the "precision" annotation
# (0=hw, 1=exact, 2=ftz_false), like l2_cache_ctrl's int values.
_PRECISION_CODE = {"hw": 0, "exact": 1, "ftz_false": 2}


def _normalize_precision(value, op):
    """Validate a precision selector for `op`; return the canonical code.

    Aliases (e.g. "intrinsic", "vdiv_0ulp_ftz_true") are accepted;
    unknown names raise here so typos fail at trace time instead of silently
    mapping to bare SFU.
    """
    key = str(value).lower()
    table = _PRECISION_MODES.get(op)
    if table is None:
        raise ValueError(f"precision= not supported for {op}")
    if key not in table:
        raise ValueError(f"precision {value!r} not valid for {op}. Valid: {sorted(table)}")
    return _PRECISION_CODE[table[key]]


def _precision_annotation(op, precision, dtype=None):
    """Validate a precision selector and return the annotation map.

    Precise paths are float32-only: the C++ wrappers (``vdiv_0ulp_ftz_true``,
    ``*_1ulp_ftz_false``) hardcode fp32 constants/overloads, and non-fp32 always
    uses the hardware instruction.
    """
    code = _normalize_precision(precision, op)
    if code != 0 and dtype is not None and tvm.DataType(dtype).with_lanes(1) != "float32":
        raise ValueError(
            f"{op}: precision={precision!r} requires float32 operands (got {dtype}); non-fp32 always uses the hardware instruction"
        )
    return {"precision": tirx.IntImm("int32", code)}


def vdiv(src0, src1, mask=None, mode="MODE_ZEROING", precision=None):
    """Divide vectors, optionally selecting the SFU implementation.

    ``precision=None`` follows ``tl.enable_fast_math`` (precise fp32
    division when fast math is off, hardware instruction otherwise).
    Non-fp32 division always uses the hardware instruction.

    ``precision`` selects the implementation per op:

    - ``'ftz_true'``: bare hardware SFU (flush-to-zero semantics)
    - ``'exact'`` (alias ``'vdiv_0ulp_ftz_true'``): correctly-rounded fp32
      division (CANN DivAlgo::PRECISION_0ULP_FTZ_TRUE / DivPrecisionImpl)

    Requires float32 for precise paths.
    """
    if mask is None:
        mask = _default_mask(src0.dtype)
    annotations = None
    if precision is not None:
        annotations = _precision_annotation("vdiv", precision, dtype=src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vdiv"), src0, src1, mask, mode, annotations=annotations)


def vmax(src0, src1, mask=None, mode="MODE_ZEROING"):  # noqa: A001
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vmax"), src0, src1, mask, mode)


def vmin(src0, src1, mask=None, mode="MODE_ZEROING"):  # noqa: A001
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vmin"), src0, src1, mask, mode)


def vand(src0, src1, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vand"), src0, src1, mask, mode)


def vor(src0, src1, mask=None, mode="MODE_ZEROING"):  # codespell:ignore
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vor"), src0, src1, mask, mode)  # codespell:ignore


def vxor(src0, src1, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vxor"), src0, src1, mask, mode)


def vshl(src0, src1, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vshl"), src0, src1, mask, mode)


def vshr(src0, src1, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vshr"), src0, src1, mask, mode)


# -- Unary ------------------------------------------------------------------------


def vexp(src, mask=None, mode="MODE_ZEROING", precision=None):
    if mask is None:
        mask = _default_mask(src.dtype)
    annotations = None if precision is None else _precision_annotation("vexp", precision, dtype=src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vexp"), src, mask, mode, annotations=annotations)


def vln(src, mask=None, mode="MODE_ZEROING", precision=None):
    if mask is None:
        mask = _default_mask(src.dtype)
    annotations = None if precision is None else _precision_annotation("vln", precision, dtype=src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vln"), src, mask, mode, annotations=annotations)


def vsqrt(src, mask=None, mode="MODE_ZEROING", precision=None):
    if mask is None:
        mask = _default_mask(src.dtype)
    annotations = None if precision is None else _precision_annotation("vsqrt", precision, dtype=src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vsqrt"), src, mask, mode, annotations=annotations)


def vabs(src, mask=None, mode="MODE_ZEROING"):  # noqa: A001
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vabs"), src, mask, mode)


def vneg(src, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vneg"), src, mask, mode)


def vrelu(src, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vrelu"), src, mask, mode)


def vlrelu(src, alpha, mask=None):
    """Leaky ReLU with scalar slope (f16/f32): dst = src >= 0 ? src : alpha * src."""
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vlrelu"), src, alpha, mask)


def vprelu(src0, src1, mask=None):
    """Parametric ReLU with per-lane slope vector (f16/f32)."""
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vprelu"), src0, src1, mask)


def vnot(src, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vnot"), src, mask, mode)


# -- Broadcast --------------------------------------------------------------------


def vdup(src, dtype_str: DType, mask=None, mode="MODE_ZEROING"):
    """Broadcast scalar to all lanes: dst = vdup(scalar, dtype_str, mask).

    dtype_str specifies the target vector element type (e.g. "float32").
    """
    if mask is None:
        mask = _default_mask(dtype_str)
    vec_dtype = _vec_dtype(dtype_str)
    return tirx.call_intrin(vec_dtype, _Op("tl.simd.vdup"), src, mask, mode)


def vdupv(src, mask=None, pos="POS_LOWEST", mode="MODE_ZEROING"):
    """Broadcast lane N of src vector to all lanes: dst = vdupv(src, mask, pos).

    pos: "POS_LOWEST" (lane 0) or "POS_HIGHEST" (lane N).
    """
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vdupv"), src, mask, pos, mode)


# -- Cross-lane reduction ---------------------------------------------------------


def vcpadd(src, mask=None, mode="MODE_ZEROING"):
    """Pairwise adjacent-lane add.

    The sums of adjacent source lanes are packed into the low half of the
    result vector. Supported types: float32, float16.
    """
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vcpadd"), src, mask, mode)


def vcadd(src, mask=None, mode="MODE_ZEROING"):
    """Pairwise add reduction: dst = vcadd(src, mask)."""
    if mask is None:
        mask = _default_mask(src.dtype)
    src_elem = str(src.dtype).split("x", 1)[0]
    out_dtype = {
        "int16": _vec_dtype("int32"),
        "uint16": _vec_dtype("uint32"),
    }.get(src_elem, str(src.dtype))
    return tirx.call_intrin(out_dtype, _Op("tl.simd.vcadd"), src, mask, mode)


def vcmax(src, mask=None, mode="MODE_ZEROING"):
    """Pairwise max reduction: dst = vcmax(src, mask)."""
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vcmax"), src, mask, mode)


def vcmin(src, mask=None, mode="MODE_ZEROING"):
    """Pairwise min reduction: dst = vcmin(src, mask)."""
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vcmin"), src, mask, mode)


def vcgadd(src, mask=None, mode="MODE_ZEROING"):
    """Grouped add reduction: dst = vcgadd(src, mask)."""
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vcgadd"), src, mask, mode)


def vcgmax(src, mask=None, mode="MODE_ZEROING"):
    """Grouped max reduction: dst = vcgmax(src, mask)."""
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vcgmax"), src, mask, mode)


def vcgmin(src, mask=None, mode="MODE_ZEROING"):
    """Grouped min reduction: dst = vcgmin(src, mask)."""
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vcgmin"), src, mask, mode)


def vsqz(src, mask=None, mode="MODE_STORED"):
    """Squeeze selected lanes toward the low lanes: dst = vsqz(src, mask)."""
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vsqz"), src, mask, mode)


def vusqz(mask, dtype="int32"):
    """Per-lane exclusive prefix count of mask (s8/s16/s32)."""
    return tirx.call_intrin(_vec_dtype(dtype), _Op("tl.simd.vusqz"), mask)


# -- Index ramp / compare ---------------------------------------------------------


def vci(index, dtype_str: DType, order="INC_ORDER"):
    """Index ramp: dst = vci(index, dtype_str, order).

    dst[lane] = index + lane (INC_ORDER) / index - lane (DEC_ORDER).
    dtype_str is the destination vector element type (e.g. "int32", "float32").
    """
    vec_dtype = _vec_dtype(dtype_str)
    return tirx.call_intrin(vec_dtype, _Op("tl.simd.vci"), index, order)


def vcmp(src0, src1, mask=None, op="eq"):
    """Elementwise compare -> vector_bool: dst = vcmp_<op>(src0, src1, mask).

    op: eq/ne/gt/ge/lt/le.
    """
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin("boolx256", _Op("tl.simd.vcmp"), src0, src1, mask, op)


def vcmps(src, scalar, mask=None, op="lt"):
    """Compare vector vs scalar -> vector_bool: dst = vcmps_<op>(src, scalar, mask).

    op: eq/ne/gt/ge/lt/le.
    """
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin("boolx256", _Op("tl.simd.vcmps"), src, scalar, mask, op)


# -- Register permutation ---------------------------------------------------------


def vintlv(src0, src1):
    """Interleave two vector registers.

    Unpack: a, b = vintlv(x, y)

    The pair is bound once at the call site so the unpack's two ``pair_get``
    calls share a single permutation instruction instead of inlining the pair
    expression twice.
    """
    pair = tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vintlv"), src0, src1)
    if not IRBuilder.is_in_scope():
        raise RuntimeError("vintlv requires an active IRBuilder to bind the pair")
    return SimdPair(_bind(pair), (src0.dtype, src0.dtype))


def vdintlv(src0, src1):
    """De-interleave two vector registers.

    Unpack: a, b = vdintlv(x, y)

    The pair is bound once at the call site so the unpack's two ``pair_get``
    calls share a single permutation instruction instead of inlining the pair
    expression twice.
    """
    pair = tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vdintlv"), src0, src1)
    if not IRBuilder.is_in_scope():
        raise RuntimeError("vdintlv requires an active IRBuilder to bind the pair")
    return SimdPair(_bind(pair), (src0.dtype, src0.dtype))


def pair_get(pair, index):
    """Extract element from a pair: reg = pair_get(pair, 0) or pair_get(pair, 1)."""
    if isinstance(pair, SimdPair):
        return pair[index]
    index = _pair_index(index)
    return tirx.call_intrin(str(pair.dtype), _Op("tl.simd.pair_get"), pair, index)


def vpack(src, part=0):
    """Pack wider lanes to narrower lanes: dst = vpack(src, LOWER/HIGHER).

    Supports u32->u16 and u16->u8 (needed for dense UE8M0 scale packing).
    """
    if isinstance(part, tirx.IntImm):
        part = int(part.value)
    if isinstance(part, int):
        part = ("LOWER", "HIGHER")[part]
    src_elem = str(src.dtype).split("x", 1)[0]
    src_lanes = src.dtype.lanes
    pack_map = {
        "float32": "float16",
        "int32": "int16",
        "uint32": "uint16",
        "uint16": "uint8",
        "int16": "int8",
    }
    dst_elem = pack_map.get(src_elem, src_elem)
    vec_dtype = f"{dst_elem}x{src_lanes * 2}"
    return tirx.call_intrin(vec_dtype, _Op("tl.simd.vpack"), src, part)


def vunpack(src, part=0):
    """Widen half of src: u8->u16, s8->s16, u16->u32, s16->s32."""
    if isinstance(part, tirx.IntImm):
        part = int(part.value)
    if isinstance(part, int):
        part = ("LOWER", "HIGHER")[part]
    src_elem = str(src.dtype).split("x", 1)[0]
    widen_map = {"uint8": "uint16", "int8": "int16", "uint16": "uint32", "int16": "int32"}
    if src_elem not in widen_map:
        raise ValueError(f"vunpack does not support {src_elem}")
    return tirx.call_intrin(_vec_dtype(widen_map[src_elem]), _Op("tl.simd.vunpack"), src, part)


def vgatherb(base, index, mask=None):
    """Gather 32B blocks from base using vector_u32 block offsets. Returns a vector."""
    if isinstance(base, BufferLoad):
        elem_dtype = base.buffer.dtype
        base = access_ptr(base, "r", extent=_vreg_lanes(elem_dtype))
        vec_dtype = _vec_dtype(elem_dtype)
    else:
        inner_load = base.args[0]
        vec_dtype = _vec_dtype(inner_load.buffer.dtype)
    args = [base, index]
    if mask is not None:
        args.append(mask)
    return tirx.call_intrin(vec_dtype, _Op("tl.simd.vgatherb"), *args)


def vgather2(base, index, mask=None):
    """Gather elements from base using per-lane offsets."""
    if isinstance(base, BufferLoad):
        elem_dtype = str(base.buffer.dtype)
        base = access_ptr(base, "r", extent=_vreg_lanes(elem_dtype))
    else:
        inner_load = base.args[0]
        elem_dtype = str(inner_load.buffer.dtype)
    if mask is None:
        # gather reads source elements, so the predicate aligns to the source (low-lane)
        # width -- e.g. an int8 gather widens the output to int16 but masks with asc_create_mask_b8.
        mask = _default_mask_bits(_mask_bits(elem_dtype))
    out_elem = {
        "int8": "int16",
        "uint8": "uint16",
    }.get(elem_dtype, elem_dtype)
    return tirx.call_intrin(
        _vec_dtype(out_elem),
        _Op("tl.simd.vgather2"),
        base,
        index,
        mask,
    )


def vscatter(src, base, index, mask=None):
    """Scatter-store: base[index[lane]] = src[lane].

    Write-side counterpart to vgatherb/vgather2. ``index`` is a vector register of
    per-lane element offsets (uint32 for 32-bit elements, uint16 for 8/16-bit).
    """
    if mask is None:
        mask = _default_mask(src.dtype)
    if isinstance(base, BufferLoad):
        base = access_ptr(base, "w", extent=_vreg_lanes(base.buffer.dtype))
    return tirx.call_intrin("void", _Op("tl.simd.vscatter"), src, base, index, mask)


# -- Type conversion --------------------------------------------------------------


def vexpdif(src0, src1, mask=None):
    """Fused exp-sub: dst = exp(src0 - src1).

    The same-width form supports matching float32 vectors. Widening float16
    inputs to float32 requires separate even/odd results and is not represented
    by this API.
    """
    src0_dtype = str(src0.dtype)
    src1_dtype = str(src1.dtype)
    if src0_dtype != src1_dtype or src0_dtype.split("x", 1)[0] != "float32":
        raise TypeError("T.simd.vexpdif requires matching float32 vectors")
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(src0_dtype, _Op("tl.simd.vexpdif"), src0, src1, mask)


def vabsdif(src0, src1, mask=None, mode="MODE_ZEROING"):
    """Fused abs-sub: dst = vabsdif(src0, src1, mask, mode).

    Computes dst = abs(src0 - src1) in a single instruction.
    Supported types: float32, float16.
    """
    if mask is None:
        mask = _default_mask(src0.dtype)
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vabsdif"), src0, src1, mask, mode)


def vcvt(
    src,
    target_dtype: DType,
    mask=None,
    round="ROUND_R",
    sat=True,
    part=0,
    mode="MODE_ZEROING",
):
    """Vector type conversion between float32, float16, bfloat16, float8, float4, and integers.

    Parameters
    ----------
    src : VReg
        Source vector register holding the input element values.
    target_dtype : T.dtype
        Destination element type, e.g. "float32", "float16", "bfloat16", "float8_e4m3",
        "float8_e5m2", "float4_e2m1fn", "int32", "int16", "uint16", "int8", "uint8", "int64".
    mask : VReg (bool)
        Predicate mask; only lanes where mask[i] is true participate and are written.
        Defaults to an all-lanes mask matching the narrower (low-lane) element: the
        source width when widening, the target width when narrowing.
    round : str
        IEEE-754 rounding mode. Valid values:
        - ``"ROUND_R"`` (default): Round to nearest, ties to even (banker's rounding).
        - ``"ROUND_A"``: Round away from zero.
        - ``"ROUND_F"``: Round toward +inf (ceiling).
        - ``"ROUND_C"``: Round toward -inf (floor).
        - ``"ROUND_Z"``: Round toward zero (truncation).
        - ``"ROUND_O"``: Round to odd (only for f32->f16).
        - ``"ROUND_H"``: Round half away from zero (only for hif8 conversions).
        Not all modes are valid for every conversion pair; unused modes are ignored when absent.
    sat : bool
        saturation mode for narrow/overflow-prone conversions.
        - ``True`` (default): Saturate to target range on overflow.
        - ``False``: No sat; out-of-range values wrap.
        Appears in: all float->fp8, f32->f16/bf16, all float->int, integer narrowing.
        Absent from: widening conversions, f16->bf16, s16->f16, s32->f32 (no overflow possible).
    part : int
        Sub-register half/quarter selector for widening/narrowing conversions.
        For even/odd 2-way splits: ``"PART_EVEN"`` (0), ``"PART_ODD"`` (1).
        For fp8/fp4/int4 4-way splits: ``"PART_P0"`` (0), ``"PART_P1"`` (1), ``"PART_P2"`` (2), ``"PART_P3"`` (3).
        Not needed for same-width conversions (e.g. f16<->bf16, s32->f32, f32->s32, f16->s16).
    mode : str
        Write mode for masked-off lanes.
        - ``"MODE_ZEROING"`` (default): Inactive lanes are set to zero.
        - ``"MODE_MERGING"``: Inactive lanes preserve their prior value (only on mask-less int->int paths; 920R1 only for mask paths).

    Returns
    -------
    VReg
        Destination vector with the converted elements.

    Supported conversion pairs (simplified)
    ----------------------------------------
    * Float->Int: f32->s64/s32/s16, f16->s32/s16/s8/u8, bf16->s32.
    * Float->Float: f32->f16/bf16/fp8, f16->fp8/bf16/f32, bf16->fp8/f16/f32/fp4.
    * Int->Float: s16/s32/s64->f16/f32, s8/u8->f16.
    * Int->Int: Most s/u{8,16,32,64} widening/narrowing pairs.
    """

    if mask is None:
        # vcvt masks by the narrower (low-lane) element: source when widening, target
        # when narrowing. So the predicate width is min(src, target), e.g. u8<->u16 both
        # use asc_create_mask_b8 and u16<->u32 both use asc_create_mask_b16.
        mask = _default_mask_bits(min(_mask_bits(src.dtype), _mask_bits(target_dtype)))

    target_dtype = str(target_dtype)
    src_dtype = str(src.dtype).split("x", 1)[0]
    fp16_dtypes = ("float16", "bfloat16")
    fp8_dtypes = ("float8_e4m3", "float8_e4m3fn", "float8_e5m2")
    fp4_dtypes = ("float4_e2m1fn",)

    rounds = ("ROUND_R", "ROUND_A", "ROUND_F", "ROUND_C", "ROUND_Z", "ROUND_O", "ROUND_H")
    if round not in rounds:
        raise ValueError(f"Invalid round '{round}'. Must be one of {rounds}")

    sat = "RS_ENABLE" if sat else "RS_DISABLE"

    part_x2 = ("PART_EVEN", "PART_ODD")
    part_x4 = ("PART_P0", "PART_P1", "PART_P2", "PART_P3")
    if isinstance(part, tirx.IntImm):
        part = int(part.value)

    modes = ("MODE_ZEROING", "MODE_MERGING")
    if mode not in modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of {modes}")

    vec_dtype = _vec_dtype(target_dtype)

    def call(*args):
        return tirx.call_intrin(vec_dtype, _Op("tl.simd.vcvt"), src, mask, *args)

    # =========================================================================
    # 1. Float to Int
    # =========================================================================

    if src_dtype == "float32":
        if target_dtype == "int64":
            return call(round, sat, part_x2[part], mode)
        if target_dtype == "int32":
            return call(round, sat, mode)
        if target_dtype == "int16":
            return call(round, sat, part_x2[part], mode)

    if src_dtype == "float16":
        if target_dtype == "int32":
            return call(round, part_x2[part], mode)
        if target_dtype == "int16":
            return call(round, sat, mode)
        if target_dtype in ("int8", "uint8"):
            return call(round, sat, part_x2[part], mode)

    if src_dtype == "bfloat16" and target_dtype == "int32":
        return call(round, sat, part_x2[part], mode)

    # =========================================================================
    # 2. Float to Float
    # =========================================================================

    if src_dtype == "float32":
        if target_dtype in fp16_dtypes:
            return call(round, sat, part_x2[part], mode)
        if target_dtype in fp8_dtypes:
            return call(round, sat, part_x4[part], mode)

    if src_dtype == "float16":
        if target_dtype in fp8_dtypes:
            return call(round, sat, part_x2[part], mode)
        if target_dtype == "bfloat16":
            return call(round, mode)
        if target_dtype == "float32":
            return call(part_x2[part], mode)

    if src_dtype == "bfloat16":
        if target_dtype in fp8_dtypes:
            return call(round, sat, part_x2[part], mode)
        if target_dtype == "float16":
            return call(sat, round, mode)
        if target_dtype == "float32":
            return call(part_x2[part], mode)
        if target_dtype in fp4_dtypes:
            return call(round, part_x4[part], mode)

    if src_dtype in fp8_dtypes and target_dtype == "float32":
        return call(part_x4[part], mode)

    if src_dtype in fp4_dtypes and target_dtype == "bfloat16":
        return call(part_x4[part], mode)

    # =========================================================================
    # 3. Int to Float
    # =========================================================================

    if src_dtype in ("int8", "uint8") and target_dtype == "float16":
        return call(part_x2[part], mode)

    if src_dtype == "int16":
        if target_dtype == "float16":
            return call(round, mode)
        if target_dtype == "float32":
            return call(part_x2[part], mode)

    if src_dtype == "int32" and target_dtype == "float32":
        return call(round, mode)

    if src_dtype == "int64" and target_dtype == "float32":
        return call(round, part_x2[part], mode)

    # =========================================================================
    # 4. Int to Int
    # =========================================================================

    if src_dtype == "int32":
        if target_dtype == "int64":
            return call(part_x2[part], mode)
        if target_dtype in ("int16", "uint16"):
            return call(sat, part_x2[part], mode)
        if target_dtype == "uint8":
            return call(sat, part_x4[part], mode)

    if src_dtype == "uint32":
        if target_dtype in ("int16", "uint16"):
            return call(sat, part_x2[part], mode)
        if target_dtype == "uint8":
            return call(sat, part_x4[part], mode)

    if src_dtype == "int16":
        if target_dtype in ("int32", "uint32"):
            return call(part_x2[part], mode)
        if target_dtype == "uint8":
            return call(sat, part_x2[part], mode)

    if src_dtype == "uint16":
        if target_dtype == "uint32":
            return call(part_x2[part], mode)
        if target_dtype == "uint8":
            return call(sat, part_x2[part], mode)

    if src_dtype == "int8":
        if target_dtype == "int16":
            return call(part_x2[part], mode)
        if target_dtype == "int32":
            return call(part_x4[part], mode)

    if src_dtype == "uint8":
        if target_dtype == "uint16":
            return call(part_x2[part], mode)
        if target_dtype == "uint32":
            return call(part_x4[part], mode)

    if src_dtype == "int64" and target_dtype == "int32":
        return call(sat, part_x2[part], mode)

    raise ValueError(f"Unsupported vcvt conversion {src_dtype}->{target_dtype}")


# -- Special ----------------------------------------------------------------------


def vsel(src0, src1, mask):
    """Bitwise select: mask ? src0 : src1."""
    return tirx.call_intrin(str(src0.dtype), _Op("tl.simd.vsel"), src0, src1, mask)


def vselr(src, index):
    """Select lanes from src using per-lane indices."""
    src_dtype = str(src.dtype)
    index_dtype = str(index.dtype)
    # The index must be an unsigned integer vector.
    if index_dtype.startswith("int"):
        index = tirx.reinterpret("u" + index_dtype, index)
    # vselr supports f16/bf16 natively but not f32: reinterpret f32 to the same-width unsigned integer, select, then reinterpret back.
    if src_dtype.startswith("float32"):
        uint_dtype = src_dtype.replace("float", "uint")
        src = tirx.reinterpret(uint_dtype, src)
        res = tirx.call_intrin(uint_dtype, _Op("tl.simd.vselr"), src, index)
        return tirx.reinterpret(src_dtype, res)
    return tirx.call_intrin(src_dtype, _Op("tl.simd.vselr"), src, index)


def vmaxs(src, scalar, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vmaxs"), src, scalar, mask, mode)


def vmins(src, scalar, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vmins"), src, scalar, mask, mode)


def vmuls(src, scalar, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vmuls"), src, scalar, mask, mode)


def vadds(src, scalar, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vadds"), src, scalar, mask, mode)


def vshls(src, scalar, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vshls"), src, scalar, mask, mode)


def vshrs(src, scalar, mask=None, mode="MODE_ZEROING"):
    if mask is None:
        mask = _default_mask(src.dtype)
    return tirx.call_intrin(str(src.dtype), _Op("tl.simd.vshrs"), src, scalar, mask, mode)


def mem_bar(mem_type):
    """Memory barrier: mem_bar(mem_type), e.g. mem_bar(VST_VLD)."""
    return tirx.call_intrin("void", _Op("tl.simd.mem_bar"), mem_type)

"""
Explicit-width and read-only-cache loads and stores for the CuTeDSL backend.

Mirrors ``tl::load_global_*`` / ``tl::store_global_*`` / ``tl::load_shared_*`` /
``tl::store_shared_*`` in ``src/tl_templates/cuda/copy.h`` and ``copy_sm100.h``,
and CUDA's ``__ldg``.
"""

__all__ = [
    "__ldg",
    "ldg32",
    "ldg64",
    "ldg128",
    "ldg256",
    "lds32",
    "lds64",
    "lds128",
    "stg32",
    "stg64",
    "stg128",
    "stg256",
    "sts32",
    "sts64",
    "sts128",
]

import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
from cutlass._mlir_helpers.arith import ArithValue
from cutlass.base_dsl.typing import (
    BFloat16,
    Boolean,
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Integer,
    Numeric,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
)
from cutlass.cute.tensor import TensorSSA
from cutlass.cutlass_dsl import dsl_user_op

from .atomic import _arch_at_least


def _as_numeric(val):
    if isinstance(val, ArithValue):
        return Numeric.from_mlir_type(val.type)(val)
    return val


def _pred_operand(pred, *, loc=None, ip=None):
    """``(int)pred_guard`` of the ``bool pred`` parameter: 1 if ``pred`` is non-zero, else 0."""
    pred = _as_numeric(pred)
    if isinstance(pred, (bool, int, float)):
        return Int32(1 if pred else 0).ir_value(loc=loc, ip=ip)
    if not isinstance(pred, Boolean):
        pred = pred != type(pred)(0)
    return arith.extui(Int32.mlir_type, Boolean(pred).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)


def _global_addr(ptr, op):
    if ptr.memspace not in (cute.AddressSpace.gmem, cute.AddressSpace.generic):
        raise TypeError(f"T.{op} expects a global memory pointer, but got a {ptr.memspace} pointer")
    return ptr


def _zeros(ty, n, *, loc=None, ip=None):
    zero = arith.constant(ty, 0, loc=loc, ip=ip)
    return [zero] * n


def _vector_result(words, ty, n, *, loc=None, ip=None):
    vec = vector.from_elements(ir.VectorType.get([n], ty), words, loc=loc, ip=ip)
    if ty != Uint32.mlir_type:
        vec = llvm.bitcast(ir.VectorType.get([n * ty.width // 32], Uint32.mlir_type), vec, loc=loc, ip=ip)
        n = n * ty.width // 32
    return TensorSSA(vec, (n,), Uint32)


def _words(value, ty, n, op, *, loc=None, ip=None):
    """Split a stored value into ``n`` words of type ``ty`` (its bits, low lane first)."""
    if isinstance(value, cute.Tensor):
        value = value.load()
    if not isinstance(value, TensorSSA):
        raise TypeError(f"T.{op} expects a vector value, but got {type(value).__name__}")
    vec = value.ir_value(loc=loc, ip=ip)
    bits = vec.type.shape[0] * vec.type.element_type.width
    if bits != n * ty.width:
        raise TypeError(f"T.{op} expects a {n * ty.width}-bit value, but got a {bits}-bit value")
    vec_ty = ir.VectorType.get([n], ty)
    if vec.type != vec_ty:
        vec = llvm.bitcast(vec_ty, vec, loc=loc, ip=ip)
    return [vector.extract(vec, [], [i], loc=loc, ip=ip) for i in range(n)]


def _word32(value, op, *, loc=None, ip=None):
    value = _as_numeric(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return Uint32(value & 0xFFFFFFFF).ir_value(loc=loc, ip=ip)
    if isinstance(value, Integer) and type(value).width == 32:
        return value.ir_value(loc=loc, ip=ip)
    raise TypeError(f"T.{op} expects a 32-bit integer value, but got {type(value).__name__}")


def _asm(res, operands, asm, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        asm,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _extract(res, ty, n, *, loc=None, ip=None):
    if n == 1:
        return [res]
    return [llvm.extractvalue(ty, res, [i], loc=loc, ip=ip) for i in range(n)]


def _global_load_asm(n, vec, wide=False):
    """Body of ``tl::global_load`` (or the 256-bit ``global_load_256``) for ``n`` result registers."""
    mov = "mov.b64" if wide else "mov.b32"
    lines = ["{", "  .reg .pred p;", f"  setp.ne.b32 p, ${n + 1}, 0;"]
    lines += [f"  {mov} ${i}, ${n + 2 + i};" for i in range(n)]
    if vec is None:
        lines.append(f"  @p ld.global.u32 $0, [${n}];")
    else:
        regs = ", ".join(f"${i}" for i in range(n))
        lines.append(f"  @p ld.global.{vec} {{{regs}}}, [${n}];")
    lines.append("}")
    return "\n".join(lines) + "\n"


@dsl_user_op
def _global_load(ptr, pred, n, vec, op, *, loc=None, ip=None):
    i32 = Int32.mlir_type
    ptr = _global_addr(ptr, op)
    res_ty = i32 if n == 1 else llvm.StructType.get_literal([i32] * n)
    res = _asm(
        res_ty,
        [ptr.llvm_ptr, _pred_operand(pred, loc=loc, ip=ip)] + _zeros(i32, n, loc=loc, ip=ip),
        _global_load_asm(n, vec),
        ",".join(["=r"] * n + ["l", "r"] + ["r"] * n),
        loc=loc,
        ip=ip,
    )
    words = _extract(res, i32, n, loc=loc, ip=ip)
    if n == 1:
        return Uint32(words[0])
    return _vector_result(words, i32, n, loc=loc, ip=ip)


def ldg32(ptr, pred=True):
    """``tl::load_global_32[_conditional]``: ``@p ld.global.u32``; 0 when ``pred`` is false."""
    return _global_load(ptr, pred, 1, None, "ldg32")


def ldg64(ptr, pred=True):
    """``tl::load_global_64[_conditional]``: ``@p ld.global.v2.u32``; zeros when ``pred`` is false."""
    return _global_load(ptr, pred, 2, "v2.u32", "ldg64")


def ldg128(ptr, pred=True):
    """``tl::load_global_128[_conditional]``: ``@p ld.global.v4.u32``; zeros when ``pred`` is false."""
    return _global_load(ptr, pred, 4, "v4.u32", "ldg128")


_LDG256_SPLIT = """{
  .reg .pred p;
  setp.ne.b32 p, $9, 0;
  mov.b32 $0, $10;
  mov.b32 $1, $11;
  mov.b32 $2, $12;
  mov.b32 $3, $13;
  mov.b32 $4, $14;
  mov.b32 $5, $15;
  mov.b32 $6, $16;
  mov.b32 $7, $17;
  @p ld.global.v4.u32 {$0, $1, $2, $3}, [$8];
  @p ld.global.v4.u32 {$4, $5, $6, $7}, [$18];
}
"""


@dsl_user_op
def _ldg256(ptr, pred, *, loc=None, ip=None):
    ptr = _global_addr(ptr, "ldg256")
    pred_op = _pred_operand(pred, loc=loc, ip=ip)
    if _arch_at_least(100):
        i64 = Int64.mlir_type
        res = _asm(
            llvm.StructType.get_literal([i64] * 4),
            [ptr.llvm_ptr, pred_op] + _zeros(i64, 4, loc=loc, ip=ip),
            _global_load_asm(4, "v4.u64", wide=True),
            "=l,=l,=l,=l,l,r,l,l,l,l",
            loc=loc,
            ip=ip,
        )
        return _vector_result(_extract(res, i64, 4, loc=loc, ip=ip), i64, 4, loc=loc, ip=ip)
    i32 = Int32.mlir_type
    hi = cute.recast_ptr(ptr, dtype=Uint8, loc=loc, ip=ip) + 16
    res = _asm(
        llvm.StructType.get_literal([i32] * 8),
        [ptr.llvm_ptr, pred_op] + _zeros(i32, 8, loc=loc, ip=ip) + [hi.llvm_ptr],
        _LDG256_SPLIT,
        ",".join(["=r"] * 8 + ["l", "r"] + ["r"] * 8 + ["l"]),
        loc=loc,
        ip=ip,
    )
    return _vector_result(_extract(res, i32, 8, loc=loc, ip=ip), i32, 8, loc=loc, ip=ip)


def ldg256(ptr, pred=True):
    """``tl::load_global_256[_conditional]``: ``@p ld.global.v4.u64`` on sm_100+, else two
    ``@p ld.global.v4.u32``; zeros when ``pred`` is false."""
    return _ldg256(ptr, pred)


def _global_store_asm(n, vec, wide=False):
    """Body of ``tl::global_store`` (or ``global_store_256``) for ``n`` source registers."""
    lines = ["{", "  .reg .pred p;", f"  setp.ne.b32 p, ${n + 1}, 0;"]
    if vec is None:
        lines.append("  @p st.global.u32 [$0], $1;")
    else:
        regs = ", ".join(f"${i + 1}" for i in range(n))
        lines.append(f"  @p st.global.{vec} [$0], {{{regs}}};")
    lines.append("}")
    return "\n".join(lines) + "\n"


@dsl_user_op
def _global_store(ptr, words, pred, vec, op, wide=False, *, loc=None, ip=None):
    ptr = _global_addr(ptr, op)
    reg = "l" if wide else "r"
    _asm(
        None,
        [ptr.llvm_ptr] + words + [_pred_operand(pred, loc=loc, ip=ip)],
        _global_store_asm(len(words), vec),
        ",".join(["l"] + [reg] * len(words) + ["r"]),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg32(ptr, value, pred=True, *, loc=None, ip=None):
    """``tl::store_global_32[_conditional]``: ``@p st.global.u32``."""
    _global_store(ptr, [_word32(value, "stg32", loc=loc, ip=ip)], pred, None, "stg32", loc=loc, ip=ip)


@dsl_user_op
def stg64(ptr, value, pred=True, *, loc=None, ip=None):
    """``tl::store_global_64[_conditional]``: ``@p st.global.v2.u32``."""
    words = _words(value, Int32.mlir_type, 2, "stg64", loc=loc, ip=ip)
    _global_store(ptr, words, pred, "v2.u32", "stg64", loc=loc, ip=ip)


@dsl_user_op
def stg128(ptr, value, pred=True, *, loc=None, ip=None):
    """``tl::store_global_128[_conditional]``: ``@p st.global.v4.u32``."""
    words = _words(value, Int32.mlir_type, 4, "stg128", loc=loc, ip=ip)
    _global_store(ptr, words, pred, "v4.u32", "stg128", loc=loc, ip=ip)


_STG256_SPLIT = """{
  .reg .pred p;
  setp.ne.b32 p, $5, 0;
  @p st.global.v4.u32 [$0], {$1, $2, $3, $4};
  @p st.global.v4.u32 [$6], {$7, $8, $9, $10};
}
"""


@dsl_user_op
def stg256(ptr, value, pred=True, *, loc=None, ip=None):
    """``tl::store_global_256[_conditional]``: ``@p st.global.v4.u64`` on sm_100+, else two
    ``@p st.global.v4.u32``."""
    ptr = _global_addr(ptr, "stg256")
    if _arch_at_least(100):
        words = _words(value, Int64.mlir_type, 4, "stg256", loc=loc, ip=ip)
        _global_store(ptr, words, pred, "v4.u64", "stg256", wide=True, loc=loc, ip=ip)
        return
    words = _words(value, Int32.mlir_type, 8, "stg256", loc=loc, ip=ip)
    hi = cute.recast_ptr(ptr, dtype=Uint8, loc=loc, ip=ip) + 16
    _asm(
        None,
        [ptr.llvm_ptr] + words[:4] + [_pred_operand(pred, loc=loc, ip=ip), hi.llvm_ptr] + words[4:],
        _STG256_SPLIT,
        "l,r,r,r,r,r,l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _word_view(ptr, n, *, loc=None, ip=None):
    """``*reinterpret_cast<uintN *>(ptr)``: ``n`` 32-bit words aligned to the access width."""
    addr = cute.recast_ptr(ptr, dtype=Uint32, loc=loc, ip=ip)
    addr = cute.make_ptr(Uint32, addr.toint(loc=loc, ip=ip), addr.memspace, assumed_align=4 * n, loc=loc, ip=ip)
    return cute.make_tensor(addr, cute.make_layout((n,)), loc=loc, ip=ip)


def lds32(ptr):
    """``tl::load_shared_32``: a plain 32-bit load."""
    return Uint32(_word_view(ptr, 1)[0])


def lds64(ptr):
    """``tl::load_shared_64``: a plain 64-bit (``uint2``) load."""
    return _word_view(ptr, 2).load()


def lds128(ptr):
    """``tl::load_shared_128``: a plain 128-bit (``uint4``) load."""
    return _word_view(ptr, 4).load()


def sts32(ptr, value):
    """``tl::store_shared_32``: a plain 32-bit store."""
    _word_view(ptr, 1)[0] = Uint32(_word32(value, "sts32"))


def _sts(ptr, value, n, op):
    words = _words(value, Int32.mlir_type, n, op)
    view = _word_view(ptr, n)
    view.store(TensorSSA(vector.from_elements(ir.VectorType.get([n], Uint32.mlir_type), words), (n,), Uint32))


def sts64(ptr, value):
    """``tl::store_shared_64``: a plain 64-bit (``uint2``) store."""
    _sts(ptr, value, 2, "sts64")


def sts128(ptr, value):
    """``tl::store_shared_128``: a plain 128-bit (``uint4``) store."""
    _sts(ptr, value, 4, "sts128")


# CUDA's __ldg overloads by element type: (PTX type, register class, Numeric
# type of the register, volatile). 8-bit loads go through a 32-bit register;
# the __half / __nv_bfloat16 overloads are the only non-volatile ones.
_LDG_FORMS = {
    Int8: ("s8", "r", Int32, True),
    Uint8: ("u8", "r", Int32, True),
    Int16: ("s16", "h", Int16, True),
    Uint16: ("u16", "h", Int16, True),
    Int32: ("s32", "r", Int32, True),
    Uint32: ("u32", "r", Int32, True),
    Int64: ("s64", "l", Int64, True),
    Uint64: ("u64", "l", Int64, True),
    Float32: ("f32", "f", Float32, True),
    Float64: ("f64", "d", Float64, True),
    Float16: ("b16", "h", Int16, False),
    BFloat16: ("b16", "h", Int16, False),
}


@dsl_user_op
def __ldg(ptr, dtype, *, loc=None, ip=None):
    """
    CUDA ``__ldg``: a non-coherent ``ld.global.nc`` load of one ``dtype`` element.

    ``dtype`` is passed explicitly because CuTe pointers do not keep the
    signedness that selects between the ``sN`` and ``uN`` overloads.
    """
    ptr = _global_addr(ptr, "__ldg")
    form = _LDG_FORMS.get(dtype)
    if form is None:
        raise TypeError(f"T.__ldg does not support {dtype.__name__} elements")
    if ptr.dtype.width != dtype.width:
        raise TypeError(f"T.__ldg of {dtype.__name__} got a pointer to {ptr.dtype.__name__}")
    ptx_ty, reg, reg_ty, volatile = form
    res = llvm.inline_asm(
        reg_ty.mlir_type,
        [ptr.llvm_ptr],
        f"ld.global.nc.{ptx_ty} $0, [$1];",
        f"={reg},l",
        has_side_effects=volatile,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if reg_ty.width != dtype.width:
        res = arith.trunci(dtype.mlir_type, res, loc=loc, ip=ip)
    elif reg_ty.mlir_type != dtype.mlir_type:
        res = arith.bitcast(dtype.mlir_type, res, loc=loc, ip=ip)
    return dtype(res)

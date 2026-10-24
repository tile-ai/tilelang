"""FP8 (e4m3 / e5m2) elementwise arithmetic for the CuTeDSL backend.

No PTX instruction computes on e4m3/e5m2 registers on sm_89/sm_90/sm_120, and
CuTeDSL cannot lower ``arith`` ops on FP8 MLIR types. Each operation is therefore
one inline PTX block on packed registers: widen with ``cvt.rn.f16x2.{e4m3x2,e5m2x2}``,
compute in ``f16x2``, and narrow with ``cvt.rn.satfinite``. Every FP8 value is exact
in f16, and f16 carries more than ``2p + 2`` significand bits for both formats, so
rounding the f16 result once more to FP8 gives the correctly rounded FP8 result.
Division has no f16 instruction and multiplies by ``rcp.approx.ftz.f32`` (relative
error about 2^-22; FP8 magnitudes never reach f32 subnormals): a quotient of two
FP8 values that is not exactly an FP8 rounding midpoint lies at least about 2^-9
(relative) away from one, so the approximation still rounds to the correctly
rounded FP8 quotient.

``max``/``min`` use the ``.NaN`` forms (NaN propagates, ``+0 > -0``), matching the
``arith.maximumf``/``minimumf`` semantics CuTeDSL uses for other float types.
"""

from __future__ import annotations

__all__ = [
    "fp8_binary",
]

import operator

from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
from cutlass.base_dsl.typing import Float16, Float8E4M3FN, Float8E5M2, Numeric
from cutlass.cute.tensor import TensorSSA
from cutlass.cute.typing import Tensor

_PTX_TYPE = {Float8E4M3FN: "e4m3x2", Float8E5M2: "e5m2x2"}
_F16X2_OP = {
    "add": "add.rn.f16x2",
    "sub": "sub.rn.f16x2",
    "mul": "mul.rn.f16x2",
    "max": "max.NaN.f16x2",
    "min": "min.NaN.f16x2",
}
_COMPARE = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def _fp8_type(*args):
    for arg in args:
        dtype = arg.dtype if isinstance(arg, TensorSSA) else type(arg) if isinstance(arg, Numeric) else None
        if dtype in _PTX_TYPE:
            return dtype
    raise TypeError(f"expected an e4m3/e5m2 FP8 operand, got {[type(a).__name__ for a in args]}")


def _int(bits):
    return ir.IntegerType.get_signless(bits)


def _pair_ops(kind, fmt, ins, out, tag, single_lane=False):
    """PTX computing two FP8 lanes: b16 operands ``ins`` -> b16 ``out``. With
    ``single_lane`` only the low lane is meaningful (the cost differs only for div)."""
    wide = [f"w{tag}{i}" for i in range(len(ins))]
    regs = [f".reg .b32 {', '.join(wide)};"]
    lines = [f"cvt.rn.f16x2.{fmt} {w}, {src};" for w, src in zip(wide, ins)]
    if kind == "div":
        half = [f"h{tag}{i}" for i in range(4)]
        single = [f"f{tag}{i}" for i in range(4)]
        regs += [f".reg .b16 {', '.join(half)};", f".reg .f32 {', '.join(single)};"]
        lines += [f"mov.b32 {{{half[0]}, {half[1]}}}, {wide[0]};", f"mov.b32 {{{half[2]}, {half[3]}}}, {wide[1]};"]
        lines += [f"cvt.f32.f16 {s}, {h};" for s, h in zip(single, half) if not (single_lane and s in (single[1], single[3]))]
        for lane in (0,) if single_lane else (0, 1):
            num, den = single[lane], single[lane + 2]
            lines += [f"rcp.approx.ftz.f32 {den}, {den};", f"mul.rn.f32 {num}, {num}, {den};"]
        lines.append(f"cvt.rn.satfinite.{fmt}.f32 {out}, {single[0] if single_lane else single[1]}, {single[0]};")
        return regs, lines
    lines.append(f"{_F16X2_OP[kind]} {wide[0]}, {wide[0]}, {wide[1]};")
    lines.append(f"cvt.rn.satfinite.{fmt}.f16x2 {out}, {wide[0]};")
    return regs, lines


def _packed_asm(kind, fmt, n_args, lanes):
    """One inline PTX block for 1 or 2 (b16 registers) or 4 (b32 registers) FP8 lanes."""
    if lanes <= 2:
        regs, lines = _pair_ops(kind, fmt, [f"${i + 1}" for i in range(n_args)], "$0", "a", single_lane=lanes == 1)
        return "{ " + " ".join(regs + lines) + " }", "=h," + ",".join("h" * n_args)
    lo = [f"l{i}" for i in range(n_args + 1)]
    hi = [f"u{i}" for i in range(n_args + 1)]
    regs = [f".reg .b16 {', '.join(lo + hi)};"]
    lines = [f"mov.b32 {{{lo[i + 1]}, {hi[i + 1]}}}, ${i + 1};" for i in range(n_args)]
    for half, tag in ((lo, "a"), (hi, "b")):
        r, body = _pair_ops(kind, fmt, half[1:], half[0], tag)
        regs += r
        lines += body
    lines.append(f"mov.b32 $0, {{{lo[0]}, {hi[0]}}};")
    return "{ " + " ".join(regs + lines) + " }", "=r," + ",".join("r" * n_args)


def _vector_of(value, fp8, lanes):
    if isinstance(value, TensorSSA):
        return value.ir_value()
    scalar = fp8(value).ir_value()
    return vector.broadcast(ir.VectorType.get([lanes], scalar.type), scalar)


def _chunks(lanes):
    off = 0
    while off < lanes:
        width = 4 if lanes - off >= 4 else 2 if lanes - off >= 2 else 1
        yield off, width
        off += width


def _apply_packed(kind, args):
    fp8 = _fp8_type(*args)
    fmt = _PTX_TYPE[fp8]
    vectors = [a for a in args if isinstance(a, TensorSSA)]
    if not vectors:
        values = [arith.extui(_int(16), arith.bitcast(_int(8), fp8(a).ir_value())) for a in args]
        asm, constraints = _packed_asm(kind, fmt, len(args), 1)
        res = llvm.inline_asm(_int(16), values, asm, constraints)
        return fp8(arith.bitcast(fp8.mlir_type, arith.trunci(_int(8), res)))

    shape = vectors[0].shape
    lanes = vectors[0].ir_value().type.shape[0]
    values = [_vector_of(a, fp8, lanes) for a in args]
    result = values[0]
    for off, width in _chunks(lanes):
        if width == 1:
            elems = [arith.extui(_int(16), arith.bitcast(_int(8), vector.extract(v, [], [off]))) for v in values]
            asm, constraints = _packed_asm(kind, fmt, len(args), 1)
            res = llvm.inline_asm(_int(16), elems, asm, constraints)
            result = vector.insert(arith.bitcast(fp8.mlir_type, arith.trunci(_int(8), res)), result, [], [off])
            continue
        chunk_type = ir.VectorType.get([width], fp8.mlir_type)
        reg = _int(8 * width)
        parts = [llvm.bitcast(reg, vector.extract_strided_slice(chunk_type, v, [off], [width], [1])) for v in values]
        asm, constraints = _packed_asm(kind, fmt, len(args), width)
        res = llvm.inline_asm(reg, parts, asm, constraints)
        result = vector.insert_strided_slice(llvm.bitcast(chunk_type, res), result, [off], [1])
    return TensorSSA(result, shape, fp8)


def _widen(value, fp8):
    """FP8 -> Float16 through ``cvt.rn.f16x2`` (exact), for comparisons."""
    fmt = _PTX_TYPE[fp8]
    asm = f"cvt.rn.f16x2.{fmt} $0, $1;"
    f16x2 = ir.VectorType.get([2], ir.F16Type.get())
    if not isinstance(value, TensorSSA):
        packed = arith.extui(_int(16), arith.bitcast(_int(8), fp8(value).ir_value()))
        res = llvm.bitcast(f16x2, llvm.inline_asm(_int(32), [packed], asm, "=r,h"))
        return Float16(vector.extract(res, [], [0]))
    src = value.ir_value()
    lanes = src.type.shape[0]
    result = vector.broadcast(ir.VectorType.get([lanes], ir.F16Type.get()), arith.constant(ir.F16Type.get(), 0.0))
    for off in range(0, lanes, 2):
        if off + 1 < lanes:
            chunk = vector.extract_strided_slice(ir.VectorType.get([2], fp8.mlir_type), src, [off], [2], [1])
            packed = llvm.bitcast(_int(16), chunk)
        else:
            packed = arith.extui(_int(16), arith.bitcast(_int(8), vector.extract(src, [], [off])))
        res = llvm.bitcast(f16x2, llvm.inline_asm(_int(32), [packed], asm, "=r,h"))
        if off + 1 < lanes:
            result = vector.insert_strided_slice(res, result, [off], [1])
        else:
            result = vector.insert(vector.extract(res, [], [0]), result, [], [off])
    return TensorSSA(result, value.shape, Float16)


def _operand(value, lanes):
    """Normalize one operand to a TensorSSA of ``lanes`` lanes (or keep a scalar).

    The codegen keeps FP8 vectors that are not a whole number of 32-bit words
    (e.g. ``float8x2``) in rmem tensors, and pads some of them to the next word,
    so a tensor operand is loaded and cut back to the logical lane count.
    """
    if isinstance(value, Tensor):
        value = value.load()
    if lanes is None or not isinstance(value, TensorSSA):
        return value
    vec = value.ir_value()
    if vec.type.shape[0] == lanes:
        return value
    elem = vec.type.element_type
    sliced = vector.extract_strided_slice(ir.VectorType.get([lanes], elem), vec, [0], [lanes], [1])
    return TensorSSA(sliced, (lanes,), value.dtype)


def fp8_binary(op, a, b, lanes=None):
    """``a <op> b`` for FP8 operands: ``add``/``sub``/``mul``/``div``/``max``/``min``
    return FP8; comparison operators (``<``, ``==``, ...) return Boolean.

    ``lanes`` is the logical vector width; it is needed when an operand is an
    rmem tensor padded past that width."""
    a = _operand(a, lanes)
    b = _operand(b, lanes)
    if op in _COMPARE:
        fp8 = _fp8_type(a, b)
        return _COMPARE[op](_widen(a, fp8), _widen(b, fp8))
    return _apply_packed(op, (a, b))

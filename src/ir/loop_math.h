/*!
 * \file tl/ir/loop_math.h
 * \brief Finite iteration arithmetic for TileLang analyses and unrolling.
 */
#ifndef TILELANG_IR_LOOP_MATH_H_
#define TILELANG_IR_LOOP_MATH_H_

#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/stmt.h>

namespace tvm::tl {

// Layout and pipeline domains use end - begin only for unit-step loops.
// Keep this precondition explicit rather than interpreting every native For
// as a dense iteration domain.
inline PrimExpr LoopExtent(const ForObj *loop) {
  TVM_FFI_CHECK(prim::IsOne(loop->step), ValueError)
      << "TileLang parallel layout lowering requires a unit-step loop";
  return loop->end - loop->begin;
}

// Native For always tests var < end. An increasing update justifies a
// half-open domain only when that update cannot wrap before reaching end.
inline bool CanUseLoopBounds(const PrimExpr &begin, const PrimExpr &end,
                             const PrimExpr &step, sym::AnalyzerObj *analyzer) {
  using namespace tvm::prim;
  const auto type = end.ty();
  if (!type.IsScalar() || !type.MatchesCode(kDLInt, kDLUInt) ||
      type != step.ty() || type != begin.ty() ||
      SideEffect(begin) > CallEffectKind::kPure ||
      SideEffect(end) > CallEffectKind::kPure ||
      SideEffect(step) > CallEffectKind::kPure ||
      !analyzer->CanProve(step > IntImm(type, 0)))
    return false;
  if (IsOne(step))
    return true;
  const int bits = type.bits() - (type.MatchesCode(kDLInt) ? 1 : 0);
  ffi::BigInt maximum = (ffi::BigInt(1) << bits) - 1;
  return analyzer->CanProve(end <=
                            IntImm(type, maximum) - (step - IntImm(type, 1)));
}

inline int ConstantUnrollTripCount(const ForObj *loop,
                                   const sym::Analyzer &analyzer) {
  auto step = analyzer->Simplify(loop->step);
  const auto *stride = step.as<IntImmObj>();
  if (!stride || stride->value == 0)
    return -1;
  auto begin = analyzer->Simplify(loop->begin);
  auto end = analyzer->Simplify(loop->end);
  const auto *first = begin.as<IntImmObj>();
  const auto *last = end.as<IntImmObj>();
  ffi::BigInt distance;
  if (first && last) {
    distance = last->value - first->value;
  } else {
    if (loop->var.ty().MatchesCode(kDLUInt))
      return -1;
    auto extent = analyzer->Simplify(end - begin);
    const auto *literal = extent.as<IntImmObj>();
    if (!literal)
      return -1;
    distance = literal->value;
  }
  if (distance <= 0)
    return 0;
  if (stride->value < 0)
    return -1;
  ffi::BigInt count = (distance - 1) / stride->value + 1;
  if (first && last) {
    auto type = loop->var.ty();
    bool is_signed = type.MatchesCode(kDLInt);
    ffi::BigInt limit = ffi::BigInt(1) << (type.bits() - (is_signed ? 1 : 0));
    ffi::BigInt next = first->value + count * stride->value;
    if (next < (is_signed ? -limit : ffi::BigInt(0)) || next >= limit)
      return -1;
  }
  return count.as<int>().value_or(-1);
}

inline PrimExpr UnrolledLoopValue(const ForObj *loop, int ordinal,
                                  const sym::Analyzer &analyzer) {
  auto begin = analyzer->Simplify(loop->begin);
  auto step = analyzer->Simplify(loop->step);
  if (const auto *first = begin.as<IntImmObj>()) {
    if (const auto *stride = step.as<IntImmObj>())
      return IntImm(loop->var.ty(),
                    first->value + ffi::BigInt(ordinal) * stride->value);
  }
  return loop->begin + IntImm(loop->var.ty(), ordinal) * loop->step;
}

} // namespace tvm::tl

#endif // TILELANG_IR_LOOP_MATH_H_

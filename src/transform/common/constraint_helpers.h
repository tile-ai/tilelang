#ifndef TILELANG_TRANSFORM_CONSTRAINT_HELPERS_H_
#define TILELANG_TRANSFORM_CONSTRAINT_HELPERS_H_

#include "compat/analyzer.h"
#include "compat/context.h"
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/with_context.h>

#include <cstdint>
#include <vector>

namespace tvm {
namespace tl::arith {
namespace detail {

enum class CompareKind { kEQ, kLT, kLE, kGT, kGE };

inline void AppendFloorDivConstraints(const prim::FloorDivNode *div,
                                      const IntImm &constant, CompareKind kind,
                                      std::vector<PrimExpr> *out) {
  auto divisor = div->b.as<IntImm>();
  if (!divisor.has_value() || (*divisor)->value <= 0)
    return;

  PrimType dtype = div->a.ty();
  ffi::BigInt lo_value = constant->value * (*divisor)->value;
  ffi::BigInt hi_value = (constant->value + 1) * (*divisor)->value;

  bool is_signed = dtype.MatchesCode(DLDataTypeCode::kDLInt);
  ffi::BigInt limit = ffi::BigInt(1) << (dtype.bits() - is_signed);
  ffi::BigInt minimum = is_signed ? -limit : ffi::BigInt(0);
  if (lo_value < minimum || lo_value >= limit || hi_value < minimum ||
      hi_value >= limit)
    return;
  PrimExpr lo = IntImm(dtype, lo_value);
  PrimExpr hi = IntImm(dtype, hi_value);

  switch (kind) {
  case CompareKind::kEQ:
    out->push_back(div->a >= lo);
    out->push_back(div->a < hi);
    break;
  case CompareKind::kLT:
    out->push_back(div->a < lo);
    break;
  case CompareKind::kLE:
    out->push_back(div->a < hi);
    break;
  case CompareKind::kGT:
    out->push_back(div->a >= hi);
    break;
  case CompareKind::kGE:
    out->push_back(div->a >= lo);
    break;
  }
}

inline CompareKind InvertCompare(CompareKind kind) {
  switch (kind) {
  case CompareKind::kEQ:
    return CompareKind::kEQ;
  case CompareKind::kLT:
    return CompareKind::kGT;
  case CompareKind::kLE:
    return CompareKind::kGE;
  case CompareKind::kGT:
    return CompareKind::kLT;
  case CompareKind::kGE:
    return CompareKind::kLE;
  }
  return CompareKind::kEQ;
}

inline void CollectFloorDivConstraintsFromCompare(const PrimExpr &lhs,
                                                  const PrimExpr &rhs,
                                                  CompareKind kind,
                                                  std::vector<PrimExpr> *out) {
  if (const auto *div = lhs.as<prim::FloorDivNode>()) {
    if (auto value = rhs.as<IntImm>(); value.has_value())
      AppendFloorDivConstraints(div, *value, kind, out);
  }
  if (const auto *div = rhs.as<prim::FloorDivNode>()) {
    if (auto value = lhs.as<IntImm>(); value.has_value()) {
      AppendFloorDivConstraints(div, *value, InvertCompare(kind), out);
    }
  }
}

inline void CollectDerivedConstraintFacts(const PrimExpr &condition,
                                          std::vector<PrimExpr> *out) {
  if (const auto *and_node = condition.as<prim::AndNode>()) {
    CollectDerivedConstraintFacts(and_node->a, out);
    CollectDerivedConstraintFacts(and_node->b, out);
    return;
  }
  if (const auto *and_node = condition.as<prim::BitwiseAndNode>()) {
    if (and_node->a.ty().MatchesElementType(DLDataTypeCode::kDLBool, 8) &&
        and_node->b.ty().MatchesElementType(DLDataTypeCode::kDLBool, 8)) {
      CollectDerivedConstraintFacts(and_node->a, out);
      CollectDerivedConstraintFacts(and_node->b, out);
      return;
    }
  }
  if (const auto *eq = condition.as<prim::EQNode>()) {
    CollectFloorDivConstraintsFromCompare(eq->a, eq->b, CompareKind::kEQ, out);
  } else if (const auto *lt = condition.as<prim::LTNode>()) {
    CollectFloorDivConstraintsFromCompare(lt->a, lt->b, CompareKind::kLT, out);
  } else if (const auto *le = condition.as<prim::LENode>()) {
    CollectFloorDivConstraintsFromCompare(le->a, le->b, CompareKind::kLE, out);
  } else if (const auto *gt = condition.as<prim::GTNode>()) {
    CollectFloorDivConstraintsFromCompare(gt->a, gt->b, CompareKind::kGT, out);
  } else if (const auto *ge = condition.as<prim::GENode>()) {
    CollectFloorDivConstraintsFromCompare(ge->a, ge->b, CompareKind::kGE, out);
  }
}

inline void EnterConstraintFacts(WithGroup<ConstraintContext> *constraints,
                                 sym::AnalyzerObj *analyzer,
                                 const PrimExpr &condition) {
  sym::Analyzer analyzer_ref = ffi::GetRef<sym::Analyzer>(analyzer);
  constraints->Emplace(analyzer_ref, condition);
  std::vector<PrimExpr> derived;
  CollectDerivedConstraintFacts(condition, &derived);
  for (const PrimExpr &fact : derived) {
    constraints->Emplace(analyzer_ref, fact);
  }
}

} // namespace detail
} // namespace tl::arith
} // namespace tvm

#endif

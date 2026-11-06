/*!
 * \file ascend/codegen/vector_index.cc
 * \brief Contiguous vector-index analysis for CCE and PTO memory emission.
 */
#include "vector_index.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/prim/op.h>

#include <optional>

#include "ir/analyzer.h"

namespace tvm::codegen::ascend {
namespace {

// These total integer constructors can be inspected without duplicating a
// memory read, a call, or a potentially trapping divide. In particular, a cast
// from a floating-point expression is not an integer indexing proof.
bool IsPureIntegerIndex(const PrimExpr &index) {
  if (!index.ty().MatchesCode(kDLInt) && !index.ty().MatchesCode(kDLUInt))
    return false;
  if (index.as<IntImmObj>() || index.as<VarObj>())
    return true;
  if (const auto *ramp = index.as<prim::RampObj>())
    return IsPureIntegerIndex(ramp->base) && IsPureIntegerIndex(ramp->stride);
  if (const auto *broadcast = index.as<prim::BroadcastObj>())
    return IsPureIntegerIndex(broadcast->value);
  if (const auto *cast = index.as<prim::CastObj>())
    return IsPureIntegerIndex(cast->value);
  if (const auto *shuffle = index.as<prim::ShuffleObj>()) {
    for (const PrimExpr &vector : shuffle->vectors)
      if (!IsPureIntegerIndex(vector))
        return false;
    for (const PrimExpr &position : shuffle->indices)
      if (!position.as<IntImmObj>())
        return false;
    return true;
  }
#define TL_ASCEND_INDEX_PURE_BINARY(Op)                                        \
  if (const auto *binary = index.as<prim::Op##Obj>())                          \
  return IsPureIntegerIndex(binary->a) && IsPureIntegerIndex(binary->b)
  TL_ASCEND_INDEX_PURE_BINARY(Add);
  TL_ASCEND_INDEX_PURE_BINARY(Sub);
  TL_ASCEND_INDEX_PURE_BINARY(Mul);
#undef TL_ASCEND_INDEX_PURE_BINARY
  return false;
}

// Native layout lowering can retain Broadcast/Cast/Shuffle arithmetic around
// an authored Ramp. Recover scalar lanes from those pure integer constructors
// without asking the scalar Analyzer to simplify a vector, and without
// evaluating a tensor load or a stateful call once per lane.
std::optional<PrimExpr> IndexLane(const PrimExpr &index, int lane,
                                  const sym::Analyzer &analyzer) {
  if (index.ty().IsScalar())
    return analyzer->Simplify(index);
  if (const auto *ramp = index.as<prim::RampObj>())
    return analyzer->Simplify(
        ramp->base + ramp->stride * prim::MakeConst(ramp->base.ty(), lane));
  if (const auto *broadcast = index.as<prim::BroadcastObj>())
    return analyzer->Simplify(broadcast->value);
  if (const auto *cast = index.as<prim::CastObj>()) {
    auto value = IndexLane(cast->value, lane, analyzer);
    if (value)
      return analyzer->Simplify(
          prim::Cast(index.ty().WithLanes(1), *value, index->loc));
    return std::nullopt;
  }
  if (const auto *shuffle = index.as<prim::ShuffleObj>()) {
    if (lane < 0 || static_cast<size_t>(lane) >= shuffle->indices.size())
      return std::nullopt;
    const auto *position = shuffle->indices[lane].as<IntImmObj>();
    if (!position)
      return std::nullopt;
    auto wanted = position->value.as<int64_t>();
    if (!wanted || *wanted < 0)
      return std::nullopt;
    for (const PrimExpr &vector : shuffle->vectors) {
      if (!vector.ty().IsScalar() && !vector.ty().IsFixedLengthVector())
        return std::nullopt;
      if (*wanted < vector.ty().lanes())
        return IndexLane(vector, static_cast<int>(*wanted), analyzer);
      *wanted -= vector.ty().lanes();
    }
    return std::nullopt;
  }
#define TL_ASCEND_INDEX_BINARY(Op)                                             \
  if (const auto *binary = index.as<prim::Op##Obj>()) {                        \
    auto lhs = IndexLane(binary->a, lane, analyzer);                           \
    auto rhs = IndexLane(binary->b, lane, analyzer);                           \
    if (lhs && rhs)                                                            \
      return analyzer->Simplify(prim::Op(*lhs, *rhs, index->loc));             \
    return std::nullopt;                                                       \
  }
  TL_ASCEND_INDEX_BINARY(Add)
  TL_ASCEND_INDEX_BINARY(Sub)
  TL_ASCEND_INDEX_BINARY(Mul)
#undef TL_ASCEND_INDEX_BINARY
  return std::nullopt;
}

} // namespace

// HoistBroadcastValues materializes scalar constants in native Bind nodes.
// Record only constant-valued immutable definitions; the emitted Bind stays
// in place. Never substitute a memory snapshot or a call into later accesses.
void BindIndexConstant(const BindObj *bind, const sym::Analyzer &analyzer) {
  auto value = bind->value.as<PrimExpr>();
  if (!value || !value->ty().IsScalar() || !IsPureIntegerIndex(*value))
    return;
  PrimExpr constant = analyzer->Simplify(*value);
  if (constant.as<IntImmObj>())
    tl::BindAnalyzer(analyzer, bind->var, constant);
}

PrimExpr NormalizeVectorIndex(const PrimExpr &index,
                              const sym::Analyzer &analyzer) {
  if (!index.ty().IsFixedLengthVector() || !IsPureIntegerIndex(index))
    return index;
  if (const auto *ramp = index.as<prim::RampObj>()) {
    PrimExpr base = analyzer->Simplify(ramp->base);
    PrimExpr stride = analyzer->Simplify(ramp->stride);
    if (ffi::StructuralEqual()(base, ramp->base) &&
        ffi::StructuralEqual()(stride, ramp->stride))
      return index;
    return prim::Ramp(base, stride, ramp->lanes, index->loc);
  }
  const int lanes = index.ty().lanes();
  auto base = IndexLane(index, 0, analyzer);
  auto next = IndexLane(index, 1, analyzer);
  if (!base || !next)
    return index;
  PrimExpr stride = analyzer->Simplify(*next - *base);
  for (int lane = 2; lane < lanes; ++lane) {
    auto value = IndexLane(index, lane, analyzer);
    if (!value ||
        !analyzer->CanProveEqual(
            *value, *base + stride * prim::MakeConst(base->ty(), lane)))
      return index;
  }
  return prim::Ramp(*base, stride, prim::MakeConst(PrimType::Int(32), lanes),
                    index->loc);
}

} // namespace tvm::codegen::ascend

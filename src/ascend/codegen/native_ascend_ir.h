/*!
 * \file ascend/codegen/native_ascend_ir.h
 * \brief Native region metadata shared by the Ascend CCE and PTO emitters.
 */
#ifndef TILELANG_ASCEND_CODEGEN_NATIVE_ASCEND_IR_H_
#define TILELANG_ASCEND_CODEGEN_NATIVE_ASCEND_IR_H_

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/region.h>

#include <functional>
#include <optional>

#include "ascend/target_utils.h"
#include "backend/common/codegen/codegen_ir_utils.h"
#include "ir/analyzer.h"

namespace tvm::codegen {

struct AscendBlockInfo {
  ffi::String name;
  ffi::Map<ffi::String, ffi::Any> annotations;
};

inline std::optional<AscendBlockInfo>
DecodeAscendBlock(const RegionStmtObj *region) {
  if (!region || region->op->name != "tl.attr")
    return std::nullopt;
  auto attribute = DecodeTileLangAttribute(region);
  if (attribute.key != "tl.sblock")
    return std::nullopt;
  TVM_FFI_CHECK(region->body_params.empty() && region->result_vars.empty(),
                ValueError)
      << "Ascend block metadata cannot introduce hidden region bindings";
  auto fields = attribute.value.cast<ffi::Map<ffi::String, ffi::Any>>();
  AscendBlockInfo info{CodegenAnnotationString(fields.at("name")), {}};
  if (auto annotations = fields.Get("annotations"))
    info.annotations =
        annotations.value().cast<ffi::Map<ffi::String, ffi::Any>>();
  return info;
}

inline int64_t AscendAnnotationInt(const ffi::Any &value) {
  if (const auto *literal = value.as<IntImmObj>())
    return CodegenInt(literal);
  return value.cast<int64_t>();
}

// Walk native statements/expressions, including explicit annotation operands.
// This is an emitter query and does not reconstruct legacy TIR node objects.
inline void
VisitAscendIR(ffi::AnyView root,
              const std::function<void(const ffi::ObjectRef &)> &visit) {
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      root,
      [&](const StmtObj *stmt) -> ffi::Expected<ffi::WalkResult> {
        visit(ffi::GetRef<Stmt>(stmt));
        return ffi::WalkResult::Advance();
      },
      [&](const ExprObj *expr) -> ffi::Expected<ffi::WalkResult> {
        visit(ffi::GetRef<Expr>(expr));
        return ffi::WalkResult::Advance();
      });
}

inline bool IsAscendSS(const VarObj *tensor) {
  const std::string scope = CodegenStorageScope(ffi::GetRef<Var>(tensor));
  return scope == "shared.ss" || scope == "shared.ss.dyn";
}

inline PtrType AscendPointerType(const Var &var) {
  if (auto tensor = var.as<tirx::TensorVar>())
    return PtrType(tensor.value().dtype(), tensor.value().scope());
  return var->ty.as_or_throw<PtrType>();
}

inline bool IsAscendPointer(const Expr &expr) {
  return expr->ty.as<PtrTypeObj>() != nullptr;
}

inline PrimType AscendPointerElementType(const Expr &expr) {
  const auto *pointer = expr->ty.as<PtrTypeObj>();
  TVM_FFI_CHECK(pointer, TypeError)
      << "Ascend memory operation expects a typed pointer, got " << expr->ty;
  return pointer->element_type.as_or_throw<PrimType>();
}

// These total integer constructors can be inspected without duplicating a
// memory read, a call, or a potentially trapping divide. In particular, a cast
// from a floating-point expression is not an integer indexing proof.
inline bool IsPureAscendIntegerIndex(const PrimExpr &index) {
  if (!index.ty().MatchesCode(kDLInt) && !index.ty().MatchesCode(kDLUInt))
    return false;
  if (index.as<IntImmObj>() || index.as<VarObj>())
    return true;
  if (const auto *ramp = index.as<prim::RampObj>())
    return IsPureAscendIntegerIndex(ramp->base) &&
           IsPureAscendIntegerIndex(ramp->stride);
  if (const auto *broadcast = index.as<prim::BroadcastObj>())
    return IsPureAscendIntegerIndex(broadcast->value);
  if (const auto *cast = index.as<prim::CastObj>())
    return IsPureAscendIntegerIndex(cast->value);
  if (const auto *shuffle = index.as<prim::ShuffleObj>()) {
    for (const PrimExpr &vector : shuffle->vectors)
      if (!IsPureAscendIntegerIndex(vector))
        return false;
    for (const PrimExpr &position : shuffle->indices)
      if (!position.as<IntImmObj>())
        return false;
    return true;
  }
#define TL_ASCEND_INDEX_PURE_BINARY(Op)                                        \
  if (const auto *binary = index.as<prim::Op##Obj>())                          \
  return IsPureAscendIntegerIndex(binary->a) &&                                \
         IsPureAscendIntegerIndex(binary->b)
  TL_ASCEND_INDEX_PURE_BINARY(Add);
  TL_ASCEND_INDEX_PURE_BINARY(Sub);
  TL_ASCEND_INDEX_PURE_BINARY(Mul);
#undef TL_ASCEND_INDEX_PURE_BINARY
  return false;
}

// HoistBroadcastValues materializes scalar constants in native Bind nodes.
// Record only constant-valued immutable definitions; the emitted Bind stays
// in place. Never substitute a memory snapshot or a call into later accesses.
inline void BindAscendIndexConstant(const BindObj *bind,
                                    const sym::Analyzer &analyzer) {
  auto value = bind->value.as<PrimExpr>();
  if (!value || !value->ty().IsScalar() || !IsPureAscendIntegerIndex(*value))
    return;
  PrimExpr constant = analyzer->Simplify(*value);
  if (constant.as<IntImmObj>())
    tl::BindAnalyzer(analyzer, bind->var, constant);
}

// Native layout lowering can retain Broadcast/Cast/Shuffle arithmetic around
// an authored Ramp. Recover scalar lanes from those pure integer constructors
// without asking the scalar Analyzer to simplify a vector, and without
// evaluating a tensor load or a stateful call once per lane.
inline std::optional<PrimExpr> AscendIndexLane(const PrimExpr &index, int lane,
                                               const sym::Analyzer &analyzer) {
  if (index.ty().IsScalar())
    return analyzer->Simplify(index);
  if (const auto *ramp = index.as<prim::RampObj>())
    return analyzer->Simplify(
        ramp->base + ramp->stride * prim::MakeConst(ramp->base.ty(), lane));
  if (const auto *broadcast = index.as<prim::BroadcastObj>())
    return analyzer->Simplify(broadcast->value);
  if (const auto *cast = index.as<prim::CastObj>()) {
    auto value = AscendIndexLane(cast->value, lane, analyzer);
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
        return AscendIndexLane(vector, static_cast<int>(*wanted), analyzer);
      *wanted -= vector.ty().lanes();
    }
    return std::nullopt;
  }
#define TL_ASCEND_INDEX_BINARY(Op)                                             \
  if (const auto *binary = index.as<prim::Op##Obj>()) {                        \
    auto lhs = AscendIndexLane(binary->a, lane, analyzer);                     \
    auto rhs = AscendIndexLane(binary->b, lane, analyzer);                     \
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

inline PrimExpr NormalizeAscendVectorIndex(const PrimExpr &index,
                                           const sym::Analyzer &analyzer) {
  if (!index.ty().IsFixedLengthVector() || !IsPureAscendIntegerIndex(index))
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
  auto base = AscendIndexLane(index, 0, analyzer);
  auto next = AscendIndexLane(index, 1, analyzer);
  if (!base || !next)
    return index;
  PrimExpr stride = analyzer->Simplify(*next - *base);
  for (int lane = 2; lane < lanes; ++lane) {
    auto value = AscendIndexLane(index, lane, analyzer);
    if (!value ||
        !analyzer->CanProveEqual(
            *value, *base + stride * prim::MakeConst(base->ty(), lane)))
      return index;
  }
  return prim::Ramp(*base, stride, prim::MakeConst(PrimType::Int(32), lanes),
                    index->loc);
}

} // namespace tvm::codegen
#endif // TILELANG_ASCEND_CODEGEN_NATIVE_ASCEND_IR_H_

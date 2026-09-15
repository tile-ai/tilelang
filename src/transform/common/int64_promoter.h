/*!
 * \file int64_promoter.h
 * \brief Helper rewriter that promotes sub-64-bit integer expressions to int64.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_INT64_PROMOTER_H_
#define TVM_TL_TRANSFORM_COMMON_INT64_PROMOTER_H_

#include <tvm/ir/cast.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include "../../../3rdparty/tvm/src/tirx/ir/data_type_rewriter.h"
#include "../../op/builtin.h"

namespace tvm {
namespace tl {

/*!
 * \brief Promote integer variables, immediates, and casts to int64.
 *
 * Used by passes that need to widen index expressions to avoid overflow.
 */
class Int64Promoter : public tirx::IndexDataTypeRewriter {
public:
  using Parent = tirx::IndexDataTypeRewriter;

  PrimExpr VisitExpr_(const tirx::VarNode *op) final {
    PrimExpr var = Parent::VisitExpr_(op);
    if (var.dtype().is_int() && var.dtype().bits() < 64) {
      return tvm::cast(DataType::Int(64), var);
    }
    return var;
  }

  PrimExpr VisitExpr_(const tirx::IntImmNode *op) final {
    if (op->dtype.is_int() && op->dtype.bits() < 64) {
      return IntImm(DataType::Int(64), op->value);
    }
    return ffi::GetRef<PrimExpr>(op);
  }

  PrimExpr VisitExpr_(const tirx::CastNode *op) final {
    if (op->dtype.is_int() && op->dtype.bits() < 64) {
      return tvm::cast(DataType::Int(64), op->value);
    }
    return ffi::GetRef<PrimExpr>(op);
  }

  tirx::Stmt VisitStmt_(const tirx::BufferStoreNode *op) final {
    auto node = Downcast<tirx::BufferStore>(Parent::VisitStmt_(op));
    return std::move(node);
  }

  PrimExpr VisitExpr_(const tirx::BufferLoadNode *op) final {
    auto node = Downcast<tirx::BufferLoad>(Parent::VisitExpr_(op));
    return std::move(node);
  }

  PrimExpr VisitExpr_(const tirx::CallNode *op) final {
    // tl.magic_div has a fixed 32-bit contract (uint32 mul-high semantics).
    // Promoting its operands or result would break the bit-pattern of the
    // magic multiplier and leave widen/narrow pairs in device code, so the
    // whole call subtree is left untouched; the surrounding index expression
    // widens the int32 quotient with an explicit cast if needed.
    if (op->op.same_as(tl::magic_div()) || op->op.same_as(tl::magic_mod())) {
      return ffi::GetRef<PrimExpr>(op);
    }
    return Parent::VisitExpr_(op);
  }
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_INT64_PROMOTER_H_

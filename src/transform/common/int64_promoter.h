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
    if (op->dtype.is_int() && op->dtype.bits() < 64) {
      return tvm::cast(DataType::Int(64), ffi::GetRef<tirx::Var>(op));
    }
    return ffi::GetRef<PrimExpr>(op);
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
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_INT64_PROMOTER_H_

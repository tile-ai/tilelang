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
    // Magic constants and the divisor retain their fixed int32 contract, but
    // index legalization must still widen the dividend before its arithmetic
    // can overflow. Codegen selects the uint32 mul-high fast path only when
    // the widened value fits its contract and otherwise uses an int64 floor
    // div/mod fallback.
    if (op->op.same_as(tl::magic_div()) ||
        op->op.same_as(tl::magic_div_with_validity()) ||
        op->op.same_as(tl::magic_mod()) ||
        op->op.same_as(tl::magic_mod_with_validity()) ||
        op->op.same_as(tl::magic_mod_from_quotient())) {
      // The caller has already determined that this index needs widening.
      // Thread, loop and shape variables can overflow just like scalar params.
      ffi::Array<PrimExpr> args = op->args;
      args.Set(0, VisitExpr(op->args[0]));
      return tirx::Call(DataType::Int(64), op->op, args, op->annotations,
                        op->span);
    }
    return Parent::VisitExpr_(op);
  }
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_INT64_PROMOTER_H_

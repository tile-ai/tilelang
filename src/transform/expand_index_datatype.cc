// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

#include <tvm/runtime/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/data_type_rewriter.h>
#include <tvm/tir/op.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;
class ExpandDataTypeRewriter : public IndexDataTypeRewriter
{
public:
    using Parent = IndexDataTypeRewriter;
    ExpandDataTypeRewriter() = default;

    Stmt operator()(Stmt s) { return VisitStmt(s); }

protected:
    using Parent::VisitExpr_;
    using Parent::VisitStmt_;

    PrimExpr VisitExpr_(const VarNode *op) final
    {
        if (op->dtype.is_int() && op->dtype.bits() < 64)
        {
            DataType new_dtype = DataType::Int(64);
            if (!var_remap_.count(op))
            {
                var_remap_[op] = Var(op->name_hint, new_dtype);
            }
        }
        return Parent::VisitExpr_(op);
    }

    PrimExpr VisitExpr_(const IntImmNode *op) final
    {
        if (is_enabled_ && op->dtype.is_int() && op->dtype.bits() < 64)
        {
            return IntImm(DataType::Int(64), op->value);
        }
        return GetRef<PrimExpr>(op);
    }

    PrimExpr VisitExpr_(const CastNode *op) final
    {
        if (is_enabled_ && op->dtype.is_int() && op->dtype.bits() < 64)
        {
            PrimExpr value = VisitExpr(op->value);
            return Cast(DataType::Int(64), value);
        }
        return Parent::VisitExpr_(op);
    }

    Stmt VisitStmt_(const BufferStoreNode *op) final
    {
        // Force indices to be int64
        bool is_enabled = is_enabled_;
        is_enabled_ = true;
        auto node = Downcast<BufferStore>(Parent::VisitStmt_(op));
        is_enabled_ = is_enabled;
        return std::move(node);
    }

    PrimExpr VisitExpr_(const BufferLoadNode *op) final
    {
        // Force indices to be int64
        bool is_enabled = is_enabled_;
        is_enabled_ = true;
        auto node = Downcast<BufferLoad>(Parent::VisitExpr_(op));
        is_enabled_ = is_enabled;
        return std::move(node);
    }
};

tvm::transform::Pass ExpandIndexDataType() {
    using namespace tir::transform;
    auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
        auto* n = f.CopyOnWrite();
        n->body = ExpandDataTypeRewriter()(std::move(n->body));
        return f;
    };
    return CreatePrimFuncPass(pass_func, 0, "tl.ExpandIndexDataType", {});
}

TVM_REGISTER_GLOBAL("tl.transform.ExpandIndexDataType")
    .set_body_typed(ExpandIndexDataType);

}  // namespace tir
}  // namespace tvm

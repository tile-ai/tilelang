#include "compat/analyzer.h"
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "op/prim.h"
#include "transform/common/data_type_rewriter.h"
#include "transform/tirx_passes.h"

namespace tvm::tl {
using namespace tvm::tirx;
using tvm::tirx::transform::CreateFunctionPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
using namespace tvm::prim;

using sym::Analyzer;
using sym::ConstIntBound;

class DataTypeVisitor final : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;

  explicit DataTypeVisitor(int target_bits)
      : bits_(target_bits), target_bits_(target_bits) {}

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView expr) final {
    if (expr == nullptr)
      return std::nullopt;
    auto prim_expr = expr.as<PrimExpr>();
    if (!prim_expr) {
      return StmtExprVisitor::Visit(expr);
    }
    PrimExpr e = prim_expr.value();
    PrimType e_ty = e.ty();
    if (e_ty.MatchesCode(DLDataTypeCode::kDLInt)) {
      int bits = max_bits_;
      if (bound_.find(e) == bound_.end()) {
        tl::arith::ConstIntBoundOf(analyzer_, e, &bound_);
      }
      ConstIntBound bound = bound_.at(e);
      int64_t ubound = static_cast<int64_t>(
          max_value(PrimType::Int(target_bits_)).as_or_throw<IntImm>()->value);
      int64_t lbound = static_cast<int64_t>(
          min_value(PrimType::Int(target_bits_)).as_or_throw<IntImm>()->value);
      if (e_ty.bits() <= target_bits_ ||
          (bound->max_value <= ubound && bound->min_value >= lbound)) {
        bits = target_bits_;
      }
      int tmp = bits > bits_ ? bits : bits_;
      std::swap(bits_, tmp);
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(expr));
      std::swap(bits_, tmp);
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit(expr));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadNode *op) {
    int tmp = bits_;
    bits_ = target_bits_;
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    bits_ = tmp;
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForNode *op) {
    tl::arith::Bind(analyzer_, op->var, Range(op->begin, op->end));
    vextent_.insert_or_assign(op->var.as<VarNode>(), op->end.ty());
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtNode *op) {
    if (op->op.same_as(tirx::launch_thread_op())) {
      PrimVar var = op->body_params[0].as_or_throw<PrimVar>();
      PrimExpr extent = op->args[1].as_or_throw<PrimExpr>();
      tl::arith::Bind(analyzer_, var,
                      Range::FromMinExtent(IntImm(extent.ty(), 0), extent));
      vextent_.insert_or_assign(var.get(), extent.ty());
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const VarNode *op) {
    if (auto vextent_it = vextent_.find(op); vextent_it != vextent_.end()) {

      int bits = std::min(vextent_it->second.bits(), bits_);
      PrimType op_ty = op->ty.as_or_throw<PrimType>();
      if (auto it = vmap.find(op); it == vmap.end()) {
        vmap.emplace(op, op_ty.WithBits(bits));
      } else {

        it->second = op_ty.WithBits(std::max(it->second.bits(), bits));
      }
    }
    return StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const IntImmNode *op) {
    PrimType op_ty = op->ty.as_or_throw<PrimType>();
    if (op_ty.MatchesCode(DLDataTypeCode::kDLInt)) {

      int bits = std::min(op_ty.bits(), bits_);
      if (auto it = vmap.find(op); it == vmap.end()) {
        vmap.emplace(op, op_ty.WithBits(bits));
      } else {
        it->second = op_ty.WithBits(std::max(it->second.bits(), bits));
      }
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const prim::CastNode *op) {
    PrimType op_ty = op->ty.as_or_throw<PrimType>();
    if (op_ty.MatchesCode(DLDataTypeCode::kDLInt)) {

      int bits = std::min(op_ty.bits(), bits_);
      if (auto it = vmap.find(op); it == vmap.end()) {
        vmap.emplace(op, op_ty.WithBits(bits));
      } else {
        it->second = op_ty.WithBits(std::max(it->second.bits(), bits));
      }
    }
    return StmtExprVisitor::Visit_(op);
  }

  std::unordered_map<const ExprNode *, PrimType> vmap;

protected:
  sym::Analyzer analyzer_;

private:
  static constexpr const int max_bits_ = 64;

  int bits_;

  int target_bits_;

  std::unordered_map<const VarNode *, PrimType> vextent_;

  sym::ConstIntBoundAnalyzer::BoundMapType bound_;
};

class NarrowDataTypeRewriter : public IndexDataTypeRewriter {
public:
  using IndexDataTypeRewriter::Mutate;
  using IndexDataTypeRewriter::Mutate_;
  using Parent = IndexDataTypeRewriter;
  explicit NarrowDataTypeRewriter(int target_bits)
      : visitor_(ffi::make_object<DataTypeVisitor>(target_bits)) {}

  Stmt Rewrite(Stmt s) {
    visitor_->Visit(s);
    for (auto i = visitor_->vmap.begin(), last = visitor_->vmap.end();
         i != last;) {
      PrimExpr e = ffi::GetRef<Expr>(i->first).as_or_throw<PrimExpr>();
      if (e.ty() == i->second) {
        i = visitor_->vmap.erase(i);
      } else {
        if (auto var = e.as<Var>()) {
          VarRemapSet(var.value(), Var(var.value()->name, i->second));
        }
        ++i;
      }
    }
    return Mutate(s, InplaceMode::kDisallow).ValueOrUnchanged(s);
  }

protected:
  UnchangedOr<PrimExpr> Mutate_(const IntImmNode *op,
                                InplaceMode inplace_mode) final {
    if (is_enabled_) {
      if (visitor_->vmap.find(op) != visitor_->vmap.end()) {
        return IntImm(visitor_->vmap.at(op), op->value);
      }
    }
    return Parent::Mutate_(op, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::CastNode *op,
                                InplaceMode inplace_mode) final {
    if (is_enabled_ && visitor_->vmap.find(op) != visitor_->vmap.end()) {
      PrimExpr e = Parent::Mutate_(op, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op));
      const prim::CastNode *new_op = e.as<prim::CastNode>();
      TVM_FFI_ICHECK(new_op != nullptr) << "Expected type to be CastNode"
                                        << ", but get " << e->GetTypeKey();
      PrimExpr new_value = new_op->value;
      PrimType cast_type = visitor_->vmap.at(op);
      if (new_value.ty() != cast_type) {
        new_value = prim::Cast(cast_type, new_value);
      }
      return new_value;
    }
    return Parent::Mutate_(op, inplace_mode);
  }

#define TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(OP, FUNC)                  \
  UnchangedOr<PrimExpr> Mutate_(const OP *op, InplaceMode inplace_mode) {      \
    auto a_result = this->Mutate(op->a, inplace_mode);                         \
    bool a_unchanged = a_result.UnchangedOrSameAs(op->a);                      \
    PrimExpr a = std::move(a_result).ValueOrUnchanged(op->a);                  \
    auto b_result = this->Mutate(op->b, inplace_mode);                         \
    bool b_unchanged = b_result.UnchangedOrSameAs(op->b);                      \
    PrimExpr b = std::move(b_result).ValueOrUnchanged(op->b);                  \
    if (a_unchanged && b_unchanged && a.ty() == b.ty()) {                      \
      return ffi::Unchanged();                                                 \
    } else {                                                                   \
      if (a.ty() != b.ty()) {                                                  \
        bool is_enabled = is_enabled_;                                         \
        is_enabled_ = true;                                                    \
        PrimExpr lhs =                                                         \
            this->Mutate(op->a, inplace_mode).ValueOrUnchanged(op->a);         \
        PrimExpr rhs =                                                         \
            this->Mutate(op->b, inplace_mode).ValueOrUnchanged(op->b);         \
        is_enabled_ = is_enabled;                                              \
        return FUNC(lhs, rhs);                                                 \
      } else {                                                                 \
        return FUNC(a, b);                                                     \
      }                                                                        \
    }                                                                          \
  }

  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::AddNode, operator+);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::SubNode, operator-);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::MulNode, operator*);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::DivNode, div);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::ModNode, truncmod);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::FloorDivNode, floordiv);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::FloorModNode, floormod);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::MinNode, min);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::MaxNode, max);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::EQNode, operator==);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::NENode, operator!=);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::LENode, operator<=);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::LTNode, operator<);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::GTNode, operator>);
  TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH(prim::GENode, operator>=);

#undef TVM_DEFINE_BIOP_EXPR_MUTATE_WITH_TYPE_MATCH

private:
  ffi::ObjectPtr<DataTypeVisitor> visitor_;
};

Stmt NarrowDataType(Stmt stmt, int target_bits) {
  return ffi::make_object<NarrowDataTypeRewriter>(target_bits)->Rewrite(stmt);
}

namespace tirx_passes {

Pass NarrowDataType(int target_bits) {
  auto pass_func = [target_bits](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value())
      return f;
    auto *n = f.CopyOnWrite();
    n->body = ffi::make_object<NarrowDataTypeRewriter>(target_bits)
                  ->Rewrite(n->body.value());
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tl.NarrowDataType");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl._tirx_passes.NarrowDataType", NarrowDataType);
}

} // namespace tirx_passes
} // namespace tvm::tl

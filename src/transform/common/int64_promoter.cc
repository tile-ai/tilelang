#include "int64_promoter.h"

#include "compat/expr.h"
#include <tvm/tirx/data_type_rewriter.h>

namespace tvm::tl {
namespace {

class IndexPromoter : public tirx::IndexDataTypeRewriter {
public:
  using IndexDataTypeRewriter::Mutate;

private:
  ffi::UnchangedOr<Expr> Mutate_(const VarNode *node, ffi::InplaceMode) final {
    Var variable = ffi::GetRef<Var>(node);
    if (auto scalar = variable.as<PrimVar>()) {
      if (auto binding = bindings_.Get(scalar.value()))
        return binding.value();
      PrimType type = scalar.value().ty();
      if (type.MatchesCode(kDLInt) && type.bits() < 64) {
        return prim::cast(scalar.value(), type.WithBits(64));
      }
    }
    return ffi::Unchanged();
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const IntImmNode *node,
                                     ffi::InplaceMode) final {
    PrimExpr expression = ffi::GetRef<PrimExpr>(node);
    PrimType type = expression.ty();
    if (type.MatchesCode(kDLInt) && type.bits() < 64) {
      return IntImm(type.WithBits(64), node->value, node->loc);
    }
    return ffi::Unchanged();
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const prim::CastNode *node,
                                     ffi::InplaceMode) final {
    PrimExpr original = ffi::GetRef<PrimExpr>(node);
    PrimExpr preserved = Substitute(original, preserved_bindings_);
    PrimType type = original.ty();
    if (type.MatchesCode(kDLInt) && type.bits() < 64) {
      return prim::cast(preserved, type.WithBits(64));
    }
    if (preserved.same_as(original))
      return ffi::Unchanged();
    return preserved;
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const prim::LetNode *node,
                                     ffi::InplaceMode mode) final {
    PrimExpr value = Mutate(node->value, mode).ValueOrUnchanged(node->value);
    PrimVar original = node->var.as_or_throw<PrimVar>();
    PrimVar variable = value.ty() == original.ty()
                           ? original
                           : PrimVar(original->name, value.ty(), original->loc);
    auto previous = bindings_.Get(original);
    auto previous_preserved = preserved_bindings_.Get(original);
    bindings_.Set(original, variable);
    preserved_bindings_.Set(original, prim::cast(variable, original.ty()));
    PrimExpr body = Mutate(node->body, mode).ValueOrUnchanged(node->body);
    if (previous.has_value()) {
      bindings_.Set(original, previous.value());
      preserved_bindings_.Set(original, previous_preserved.value());
    } else {
      bindings_.erase(original);
      preserved_bindings_.erase(original);
    }
    return prim::Let(variable, value, body, node->loc);
  }

  ffi::Map<PrimVar, PrimExpr> bindings_;
  ffi::Map<PrimVar, PrimExpr> preserved_bindings_;
};

} // namespace

PrimExpr Int64Promoter::operator()(const PrimExpr &expression) const {
  auto promoter = ffi::make_object<IndexPromoter>();
  return promoter->Mutate(expression).ValueOrUnchanged(expression);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.PromoteIndexToInt64",
      [](PrimExpr expression) { return Int64Promoter()(expression); });
}

} // namespace tvm::tl

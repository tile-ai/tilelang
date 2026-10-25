#include "compat/expr.h"
#include "compat/visit.h"
#include "op/builtin.h"
#include "op/builtin_registry.h"

#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {

namespace {

class OpaqueScopeLowerer : public tirx::StmtExprMutator {
private:
  Stmt Initialize(const tirx::TensorVar &tensor, const PrimExpr &value,
                  const Location &location) {
    TVM_FFI_CHECK(tensor.scope().name() == "local" ||
                      tensor.scope().name() == "local.var",
                  ValueError)
        << "Local variable initialization requires local storage";
    ffi::Array<PrimExpr> indices;
    for (const auto &extent : tensor.shape()) {
      TVM_FFI_CHECK(analyzer_->CanProve(extent == 1), ValueError)
          << "Local variable initialization requires a single-element tensor";
      indices.push_back(MakeConst(extent.ty(), 0));
    }
    return TensorStore(tensor, indices, Cast(tensor.dtype(), value), location);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BindNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt original = ffi::GetRef<Stmt>(node);
    auto bind = StmtExprMutator::Mutate_(node, mode)
                    .ValueOrUnchanged(original)
                    .as_or_throw<Bind>();
    auto tensor = bind->var.as<tirx::TensorVar>();
    const auto *allocation = bind->value.as<CallNode>();
    if (!tensor || !allocation ||
        !allocation->op.same_as(tirx::alloc_tensor_op()) ||
        !allocation->attrs) {
      return bind;
    }
    auto attributes = allocation->attrs.value().as_or_throw<DictAttrs>();
    auto initializer = attributes.GetAttr<PrimExpr>(attr::kLocalVarInit);
    if (!initializer) {
      return bind;
    }
    auto values = attributes->dict;
    values.erase(attr::kLocalVarInit);
    Call allocate(allocation->ty, allocation->op, allocation->args,
                  DictAttrs(values), allocation->ty_args, allocation->loc);
    return SeqStmt({Bind(tensor.value(), allocate, bind->loc),
                    Initialize(tensor.value(), initializer.value(), bind->loc)},
                   bind->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const EvaluateNode *node,
                                 ffi::InplaceMode mode) final {
    const auto *call = node->value.as<CallNode>();
    const auto *operation = call ? call->op.as<OpNode>() : nullptr;
    if (!operation || operation->name != "tl.local_var_init_hint") {
      return StmtExprMutator::Mutate_(node, mode);
    }
    TVM_FFI_CHECK_EQ(call->args.size(), 2, ValueError)
        << "Local initialization requires a tensor and a value";
    auto tensor = call->args[0].as_or_throw<tirx::TensorVar>();
    auto value = Mutate(call->args[1], mode)
                     .ValueOrUnchanged(call->args[1])
                     .as_or_throw<PrimExpr>();
    return Initialize(tensor, value, node->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt original = ffi::GetRef<Stmt>(node);
    auto region = StmtExprMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(original)
                      .as_or_throw<RegionStmt>();
    TVM_FFI_CHECK(!region->attrs->dict.count(attr::kLocalVarInit), ValueError)
        << "Local initializers must belong to their native allocation or "
           "initialization hint";
    if (region->op->name == "tl.scope") {
      return RegionStmt(tirx::scope_op(), region->args, region->body_params,
                        region->attrs, region->body, region->result_vars,
                        region->loc);
    }
    return region;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    bool unannotated = node->annotations.empty() ||
                       (node->annotations.size() == 1 &&
                        node->annotations.count("pragma_unroll_explicit"));
    bool exits = false;
    PostOrderVisit(node->body, [&](const ffi::ObjectRef &object) {
      exits |= object.as<BreakNode>() || object.as<ContinueNode>();
      if (const auto *call = object.as<CallNode>()) {
        exits |= call->op.same_as(loop_break());
      }
    });
    if (!unannotated || exits ||
        SideEffect(node->begin) != CallEffectKind::kPure ||
        SideEffect(node->end) != CallEffectKind::kPure ||
        SideEffect(node->step) != CallEffectKind::kPure ||
        !analyzer_->CanProve(node->step > 0) ||
        !analyzer_->CanProve(node->begin < node->end) ||
        !analyzer_->CanProve(node->begin + node->step >= node->end)) {
      return StmtExprMutator::Mutate_(node, mode);
    }
    Stmt body = Substitute(
        node->body, ffi::Map<PrimVar, PrimExpr>{{node->var, node->begin}});
    body = Mutate(body, mode).ValueOrUnchanged(body);
    return RegionStmt(tirx::scope_op(), {}, {}, DictAttrs(), SeqStmt(body), {},
                      node->loc);
  }

  sym::Analyzer analyzer_;
};

} // namespace

tvm::transform::Pass LowerOpaqueBlock() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    TVM_FFI_CHECK(!function->attrs->dict.count(attr::kLocalVarInit), ValueError)
        << "Local initializers must belong to their native allocation or "
           "initialization hint";
    if (function->body) {
      auto lowerer = ffi::make_object<OpaqueScopeLowerer>();
      SeqStmt original = function->body.value();
      function.CopyOnWrite()->body =
          lowerer->Mutate(original).ValueOrUnchanged(original);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LowerOpaqueBlock");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterBuiltin<2>("tl.local_var_init_hint", CallEffectKind::kOpaque);
  OpDef("tl.local_var_init_hint")
      .set_attr(tvm::op_attr::kFixedReturnType, TupleType(ffi::Array<Type>{}));
  ffi::reflection::GlobalDef().def("tl.transform.LowerOpaqueBlock",
                                   LowerOpaqueBlock);
}

} // namespace tvm::tl

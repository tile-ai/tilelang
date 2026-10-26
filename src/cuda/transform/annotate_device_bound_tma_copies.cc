#include "compat/metadata.h"
#include "compat/visit.h"
#include "cuda/op/builtin.h"
#include "op/copy.h"
#include "op/utils.h"

#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>

namespace tvm::tl {

namespace {

using Bindings = std::unordered_map<const VarNode *, Expr>;

class DeviceBoundAddress : public tvm::tl::StmtExprVisitor {
public:
  explicit DeviceBoundAddress(const Bindings &bindings) : bindings_(bindings) {}

  bool Check(const tirx::TensorVar &tensor) {
    Visit(tensor);
    return found_;
  }

private:
  ffi::Optional<ffi::VisitInterrupt> Visit_(const VarNode *variable) final {
    auto found = bindings_.find(variable);
    if (found == bindings_.end() || !visited_.insert(variable).second) {
      return std::nullopt;
    }
    if (!variable->ty.as<tirx::TensorTypeNode>()) {
      found_ = true;
      return std::nullopt;
    }
    const auto *call = found->second.as<CallNode>();
    if (!call || !call->op.same_as(tirx::decl_tensor_op())) {
      found_ = true;
      return std::nullopt;
    }
    return Visit(call->args[0]);
  }

  const Bindings &bindings_;
  std::unordered_set<const VarNode *> visited_;
  bool found_ = false;
};

class DeviceBoundTmaCopyAnnotator : public tvm::tl::StmtExprMutator {
public:
  explicit DeviceBoundTmaCopyAnnotator(Bindings bindings)
      : bindings_(std::move(bindings)) {}

private:
  ffi::UnchangedOr<Expr> Mutate_(const CallNode *node,
                                 ffi::InplaceMode inplace_mode) final {
    Expr original = ffi::GetRef<Expr>(node);
    Call call = tvm::tl::StmtExprMutator::Mutate_(node, inplace_mode)
                    .ValueOrUnchanged(original)
                    .as_or_throw<Call>();
    auto op = call->op.as<Op>();
    if (!op || (op.value()->name != "tl.tileop.copy" &&
                op.value()->name != "tl.tileop.async_copy" &&
                op.value()->name != "tl.tileop.tma_copy")) {
      return call.same_as(original) ? ffi::UnchangedOr<Expr>(ffi::Unchanged())
                                    : call;
    }
    TileOperator tile_op = ParseOperator(call);
    const auto *copy = tile_op.as<CopyNode>();
    ICHECK(copy);
    auto is_device_bound = [&](const tirx::TensorVar &tensor) {
      return IsGlobalBuffer(tensor) &&
             ffi::make_object<DeviceBoundAddress>(bindings_)->Check(tensor);
    };
    if (!is_device_bound(copy->src) && !is_device_bound(copy->dst)) {
      return call.same_as(original) ? ffi::UnchangedOr<Expr>(ffi::Unchanged())
                                    : call;
    }
    BlockAnnotations annotations = copy->annotations;
    annotations.Set(attr::kTmaDescriptorBaseIsDeviceBound, true);
    return Call(call->ty, call->op, call->args, DictAttrs(annotations),
                call->ty_args, call->loc);
  }

  Bindings bindings_;
};

} // namespace

tvm::transform::Pass AnnotateDeviceBoundTmaCopies() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (function->body) {
      Bindings bindings;
      PostOrderVisit(function->body.value(), [&](const ffi::ObjectRef &object) {
        if (const auto *bind = object.as<BindNode>()) {
          bindings.emplace(bind->var.get(), bind->value);
        }
      });
      auto annotator =
          ffi::make_object<DeviceBoundTmaCopyAnnotator>(std::move(bindings));
      SeqStmt original = function->body.value();
      auto body = annotator->Mutate(original)
                      .ValueOrUnchanged(original)
                      .as_or_throw<SeqStmt>();
      if (!body.same_as(original)) {
        function.CopyOnWrite()->body = std::move(body);
      }
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.AnnotateDeviceBoundTmaCopies");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.cuda.transform.AnnotateDeviceBoundTmaCopies",
      AnnotateDeviceBoundTmaCopies);
}

} // namespace tvm::tl

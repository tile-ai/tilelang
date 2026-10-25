/*!
 * \file mark_scalar_dcache_bypass.cc
 * \brief Rewrite scalar BufferLoad/BufferStore on global buffers that have
 * writes (either scalar BufferStore or MTE copy) into explicit
 * ascend_read/write_gm_bypass_dcache Call nodes.
 *
 * Pure-read global buffers keep their normal BufferLoad (dcache path).
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "ascend/op/builtin.h"
#include "compat/visit.h"

namespace tvm::tl {
namespace {

struct GlobalWrites {
  std::unordered_map<const VarNode *, const VarNode *> aliases;
  std::unordered_set<const VarNode *> written;

  const VarNode *Root(const Expr &expression) const {
    Expr current = expression;
    while (true) {
      if (auto load = current.as<TensorLoadNode>()) {
        current = load->source;
      } else if (auto call = current.as<CallNode>();
                 call && !call->args.empty() &&
                 (call->op.same_as(tirx::tensor_data_ptr_op()) ||
                  call->op.same_as(tirx::ptr_byte_offset_op()) ||
                  call->op.same_as(tirx::reinterpret_op()) ||
                  call->op.same_as(tirx::decl_tensor_op()) ||
                  call->op.same_as(tirx::address_of_op()) ||
                  call->op.same_as(access_ptr()))) {
        current = call->args[0];
      } else {
        break;
      }
    }
    auto variable = current.as<VarNode>();
    auto found = aliases.find(variable);
    return found == aliases.end() ? variable : found->second;
  }

  bool IsGlobal(const VarNode *variable) const {
    if (!variable)
      return false;
    if (auto tensor = variable->ty.as<tirx::TensorTypeNode>())
      return tensor->scope.name() == "global";
    if (auto pointer = variable->ty.as<PtrTypeNode>())
      return pointer->scope.name() == "global";
    return false;
  }

  void Collect(const SeqStmt &body) {
    PostOrderVisit(body, [&](const ffi::ObjectRef &object) {
      if (auto binding = object.as<BindNode>()) {
        if (binding->var->ty.as<tirx::TensorTypeNode>() ||
            binding->var->ty.as<PtrTypeNode>()) {
          if (auto root = Root(binding->value))
            aliases[binding->var.get()] = root;
        }
      }
      if (auto store = object.as<TensorStoreNode>()) {
        auto root = Root(store->dest);
        if (IsGlobal(root))
          written.insert(root);
      }
      if (auto call = object.as<CallNode>();
          call && !call->args.empty() &&
          (call->op.same_as(ascend_copy_ubuf_to_gm()) ||
           call->op.same_as(ascend_copy_matrix_cc_to_gm()))) {
        auto root = Root(call->args[0]);
        if (IsGlobal(root))
          written.insert(root);
      }
    });
  }
};

class ScalarDcacheBypassRewriter : public tirx::StmtExprMutator {
public:
  explicit ScalarDcacheBypassRewriter(GlobalWrites writes)
      : writes_(std::move(writes)) {}

protected:
  ffi::UnchangedOr<Expr> Mutate_(const CallNode *node,
                                 ffi::InplaceMode mode) final {
    if (!node->args.empty() && (node->op.same_as(access_ptr()) ||
                                node->op.same_as(tirx::address_of_op()))) {
      if (const auto *load = node->args[0].as<TensorLoadNode>()) {
        ffi::Array<Expr> arguments;
        arguments.push_back(StmtExprMutator::Mutate_(load, mode)
                                .ValueOrUnchanged(ffi::GetRef<PrimExpr>(load)));
        for (size_t index = 1; index < node->args.size(); ++index) {
          arguments.push_back(
              Mutate(node->args[index]).ValueOrUnchanged(node->args[index]));
        }
        auto result = ffi::GetRef<Call>(node);
        result.CopyOnWrite()->args = arguments;
        return result;
      }
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    bool previous = inside_vf_;
    auto name = node->attrs.GetAttr<ffi::String>("name_hint").value_or("");
    inside_vf_ = inside_vf_ || name == "SIMT_VF" || name == "SIMD_VF";
    auto result = StmtExprMutator::Mutate_(node, mode);
    inside_vf_ = previous;
    return result;
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *node,
                                     ffi::InplaceMode mode) final {
    PrimExpr original = ffi::GetRef<PrimExpr>(node);
    auto value =
        StmtExprMutator::Mutate_(node, mode).ValueOrUnchanged(original);
    auto load = value.as_or_throw<TensorLoad>();
    if (load->indices.size() != 1 || !load.ty().IsScalar() ||
        !ShouldBypass(load->source))
      return value;
    Call address(std::nullopt, tirx::address_of_op(), {load});
    return Call(load.ty(), ascend_read_gm_bypass_dcache(), {address})
        .as_or_throw<PrimExpr>();
  }

  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt original = ffi::GetRef<Stmt>(node);
    auto store = StmtExprMutator::Mutate_(node, mode)
                     .ValueOrUnchanged(original)
                     .as_or_throw<TensorStore>();
    if (store->indices.size() != 1 || !store->value.ty().IsScalar() ||
        !ShouldBypass(store->dest))
      return store;
    TensorLoad destination(store->dest, store->indices, store->loc);
    Call address(std::nullopt, tirx::address_of_op(), {destination});
    return Evaluate(Call(PrimType::Void(), ascend_write_gm_bypass_dcache(),
                         {address, store->value}),
                    store->loc);
  }

private:
  bool ShouldBypass(const Expr &tensor) const {
    return !inside_vf_ && writes_.written.count(writes_.Root(tensor));
  }
  GlobalWrites writes_;
  bool inside_vf_{false};
};
} // namespace

tvm::transform::Pass MarkScalarDcacheBypass() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (!function->body)
      return function;
    GlobalWrites writes;
    writes.Collect(function->body.value());
    if (writes.written.empty())
      return function;
    auto rewriter =
        ffi::make_object<ScalarDcacheBypassRewriter>(std::move(writes));
    SeqStmt original = function->body.value();
    function.CopyOnWrite()->body =
        rewriter->Mutate(original).ValueOrUnchanged(original);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.MarkScalarDcacheBypass");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.MarkScalarDcacheBypass",
                                   MarkScalarDcacheBypass);
}
} // namespace tvm::tl

#include "compat/expr.h"
#include "cuda/op/builtin.h"

#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
using namespace ffi;
using namespace tirx;

namespace {

class SharedBarrierRewriter : public StmtExprMutator {
public:
  explicit SharedBarrierRewriter(bool disable_shuffle)
      : disable_shuffle_(disable_shuffle) {}

private:
  UnchangedOr<Expr> Mutate_(const VarNode *node, InplaceMode mode) final {
    if (auto tensor = GetRef<Var>(node).as<TensorVar>()) {
      if (auto replacement = tensors_.Get(tensor.value()))
        return replacement.value();
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Expr> Mutate_(const CallNode *node, InplaceMode mode) final {
    Expr result = StmtExprMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(GetRef<Expr>(node));
    if (node->op.same_as(access_ptr()) || node->op.same_as(address_of_op()) ||
        node->op.same_as(tensor_data_ptr_op())) {
      Call call = result.as_or_throw<Call>();
      return Call(std::nullopt, call->op, call->args, call->attrs,
                  call->ty_args, call->loc);
    }
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode mode) final {
    auto previous = threads_;
    if (node->op.same_as(launch_thread_op())) {
      String tag = node->args[0].as_or_throw<StringImm>()->value;
      if (tag == "threadIdx.x" || tag == "threadIdx.y" ||
          tag == "threadIdx.z") {
        threads_.Set(tag, node->body_params[0].as_or_throw<PrimVar>());
      }
    }
    auto result = StmtExprMutator::Mutate_(node, mode);
    threads_ = previous;
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *node, InplaceMode mode) final {
    auto tensor = node->var.as<TensorVar>();
    if (!tensor)
      return StmtExprMutator::Mutate_(node, mode);
    String scope = tensor.value().scope().name();
    if (scope != "shared.barrier" && scope != "shared.cluster_barrier") {
      return StmtExprMutator::Mutate_(node, mode);
    }
    TVM_FFI_CHECK(threads_.count("threadIdx.x"), ValueError)
        << "Shared barriers must be allocated inside a CUDA thread launch";
    const auto *allocation = node->value.as<CallNode>();
    TVM_FFI_CHECK(allocation && allocation->op.same_as(alloc_tensor_op()),
                  ValueError)
        << "Shared barriers require an explicit allocation";
    TVM_FFI_CHECK(tensor.value().dtype() == PrimType::UInt(64) &&
                      tensor.value().shape().size() == 1,
                  ValueError)
        << "Shared barrier storage must be a rank-one uint64 tensor";
    auto attributes =
        allocation->attrs.value_or(DictAttrs()).as_or_throw<DictAttrs>();
    auto counts = attributes.GetAttr<Array<PrimExpr>>("barrier_init");
    auto extent = GetConstInt(tensor.value().shape()[0]);
    TVM_FFI_CHECK(counts && extent && *extent > 0 &&
                      static_cast<size_t>(*extent) == counts.value().size(),
                  ValueError)
        << "Barrier initialization requires one arrive count per slot";
    TensorType type(tensor.value().shape(), tensor.value().dtype(),
                    runtime::MemoryScope::Get("shared"),
                    std::max(8, tensor.value().ty()->data_alignment),
                    tensor.value().ty()->layout, node->loc);
    TensorVar physical(tensor.value()->name, type, node->loc);
    tensors_.Set(tensor.value(), physical);
    attributes.CopyOnWrite()->dict.erase("barrier_init");
    Array<Stmt> initialization;
    for (size_t slot = 0; slot < counts.value().size(); ++slot) {
      PrimExpr count = counts.value()[slot];
      if (auto constant = GetConstInt(count)) {
        TVM_FFI_CHECK(*constant > 0, ValueError)
            << "Barrier arrive counts must be positive";
      }
      initialization.push_back(Evaluate(
          Call(PrimType::Void(), mbarrier_init(),
               {TensorLoad(physical, {MakeConst(PrimType::Int(32), slot)}),
                count})));
    }
    PrimExpr leader = Call(PrimType::Bool(), tl_shuffle_elect(), {PrimExpr(0)})
                          .as_or_throw<PrimExpr>();
    if (disable_shuffle_) {
      leader = MakeConst(PrimType::Bool(), true);
      for (const auto &[tag, variable] : threads_)
        leader = leader && variable == MakeConst(variable.ty(), 0);
    }
    Array<Stmt> statements{
        Bind(physical,
             Call(std::nullopt, alloc_tensor_op(), allocation->args, attributes,
                  {type}, allocation->loc),
             node->loc),
        If(leader, SeqStmt(initialization)),
        Evaluate(Call(PrimType::Void(), ptx_fence_barrier_init(), {})),
        Evaluate(
            Call(std::nullopt, gpu_storage_sync_op(),
                 {StringImm(scope == "shared.cluster_barrier" ? "cluster"
                                                              : "shared")}))};
    return SeqStmt(statements, node->loc);
  }

  Map<TensorVar, TensorVar> tensors_;
  Map<String, PrimVar> threads_;
  bool disable_shuffle_;
};

} // namespace

namespace transform {
tvm::transform::Pass LowerSharedBarrier() {
  auto pass =
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext context) -> Optional<tirx::Function> {
    if (function->body) {
      auto rewriter = make_object<SharedBarrierRewriter>(
          context->GetConfig<bool>(kDisableShuffleElect).value_or(false));
      SeqStmt body = function->body.value();
      function.CopyOnWrite()->body =
          rewriter->Mutate(body).ValueOrUnchanged(body).as_or_throw<SeqStmt>();
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.cuda.LowerSharedBarrier");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.cuda.transform.LowerSharedBarrier",
                              LowerSharedBarrier);
}
} // namespace transform

} // namespace tvm::tl

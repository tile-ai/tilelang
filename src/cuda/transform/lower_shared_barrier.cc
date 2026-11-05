/*!
 * \file lower_shared_barrier.cc
 * \brief Initialize native shared barrier definitions within their owning
 * scope.
 */
#include "cuda/op/builtin.h"
#include "transform/common/native_layout_utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <vector>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

class SharedBarrierLowerer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;
  explicit SharedBarrierLowerer(bool disable_elect)
      : disable_elect_(disable_elect) {}

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *region,
                            InplaceMode mode) final {
    auto saved = thread_x_;
    if (region->op.same_as(launch_thread_op()) &&
        region->args[0].as_or_throw<StringImm>()->value == "threadIdx.x")
      thread_x_ = region->body_params[0].as_or_throw<PrimVar>();
    auto result = StmtExprMutator::Mutate_(region, mode);
    thread_x_ = std::move(saved);
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *sequence,
                            InplaceMode mode) final {
    struct Pending {
      TensorVar tensor;
      size_t position;
      Optional<Array<PrimExpr>> counts;
    };
    Array<Stmt> output;
    std::vector<Pending> pending;
    std::unordered_map<const VarObj *, size_t> indices;
    auto flush = [&]() {
      if (pending.empty())
        return;
      TVM_FFI_CHECK(thread_x_, ValueError)
          << "LowerSharedBarrier requires an enclosing threadIdx.x region";
      Array<Stmt> initializers;
      bool cluster = false;
      for (const Pending &entry : pending) {
        TVM_FFI_CHECK(entry.counts, ValueError)
            << "barrier_init is missing for native barrier "
            << entry.tensor->name;
        auto old = output[entry.position].as_or_throw<Bind>();
        auto call = old->value.as_or_throw<Call>();
        auto attrs = call->attrs ? call->attrs.value().as_or_throw<DictAttrs>()
                                 : DictAttrs();
        auto fields = attrs->dict;
        fields.Set("tl.barrier_initialized", true);
        output.Set(entry.position,
                   Bind(old->var,
                        Call(call->ty, call->op, call->args, DictAttrs(fields),
                             call->ty_args, call->loc),
                        old->loc));
        cluster |= entry.tensor.scope().name() == "shared.cluster_barrier";
        for (size_t i = 0; i < entry.counts.value().size(); ++i)
          initializers.push_back(Evaluate(
              Call(PrimType::Void(), ptx_init_barrier_thread_count(),
                   {TensorLoad(entry.tensor, {IntImm(PrimType::Int(32), i)}),
                    entry.counts.value()[i]},
                   {}, {}, old->loc),
              old->loc));
      }
      if (!initializers.empty()) {
        const Location loc = initializers.front()->loc;
        PrimExpr elected =
            disable_elect_
                ? prim::EQ(*thread_x_, IntImm(thread_x_.value().ty(), 0))
                : Call(PrimType::Bool(), tl_shuffle_elect(),
                       {IntImm(PrimType::Int(32), 0)}, {}, {}, loc)
                      .as_or_throw<PrimExpr>();
        output.push_back(
            If(elected, SeqStmt(initializers, loc), std::nullopt, loc));
        output.push_back(Evaluate(
            Call(PrimType::Void(), ptx_fence_barrier_init(), {}, {}, {}, loc),
            loc));
        output.push_back(Evaluate(
            Call(PrimType::Void(), gpu_storage_sync_op(),
                 {StringImm(cluster ? "cluster" : "shared")}, {}, {}, loc),
            loc));
      }
      pending.clear();
      indices.clear();
    };

    for (const Stmt &statement : sequence->seq) {
      if (const auto *binding = statement.as<BindObj>()) {
        const auto *call = binding->value.as<CallObj>();
        auto tensor = binding->var.as<TensorVar>();
        if (tensor && call && call->op.same_as(alloc_tensor_op()) &&
            (tensor->scope().name() == "shared.barrier" ||
             tensor->scope().name() == "shared.cluster_barrier")) {
          bool initialized =
              call->attrs && call->attrs.value()
                                 .as_or_throw<DictAttrs>()
                                 .GetAttr<bool>("tl.barrier_initialized")
                                 .value_or(false);
          if (!initialized) {
            indices.emplace(tensor->get(), pending.size());
            pending.push_back({tensor.value(), output.size(), std::nullopt});
          }
          output.push_back(statement);
          continue;
        }
      }
      const auto *evaluate = statement.as<EvaluateObj>();
      const auto *call = evaluate ? evaluate->value.as<CallObj>() : nullptr;
      if (call && call->op.same_as(Op::Get("tl.annotation"))) {
        auto fields = ReadNativeAnnotations(call);
        if (auto values = fields.Get("barrier_init")) {
          for (const auto &[key, value] : values->cast<Map<Any, Any>>()) {
            TensorVar tensor = key.cast<TensorVar>();
            auto found = indices.find(tensor.get());
            TVM_FFI_CHECK(found != indices.end(), ValueError)
                << "barrier_init must follow its allocation in the same "
                   "lexical sequence";
            auto &entry = pending[found->second];
            TVM_FFI_CHECK(!entry.counts, ValueError)
                << "Duplicate barrier_init for " << tensor->name;
            TVM_FFI_CHECK_EQ(tensor.shape().size(), 1U, ValueError);
            const auto *extent = tensor.shape()[0].as<IntImmObj>();
            TVM_FFI_CHECK(extent, ValueError)
                << "A shared barrier allocation must have a constant size";
            Array<PrimExpr> counts;
            for (const Any &item : value.cast<Array<Any>>()) {
              PrimExpr count = item.cast<PrimExpr>();
              TVM_FFI_CHECK(count.ty().MatchesCode(kDLInt) ||
                                count.ty().MatchesCode(kDLUInt),
                            TypeError)
                  << "A barrier arrival count must be an integer";
              if (const auto *constant = count.as<IntImmObj>())
                TVM_FFI_CHECK(constant->value > 0, ValueError)
                    << "A barrier arrival count must be positive";
              counts.push_back(count);
            }
            TVM_FFI_CHECK_EQ(counts.size(), extent->value.as<size_t>().value(),
                             ValueError)
                << "The number of arrival counts must match the barrier size";
            entry.counts = counts;
          }
          fields.erase("barrier_init");
          if (!fields.empty())
            output.push_back(MakeNativeAnnotation(fields, statement->loc));
          continue;
        }
      }
      // Preserve allocation and initialization order across every executable
      // statement. Consecutive declarations are initialized together, while a
      // nested scope receives its own synchronization instead of inheriting a
      // sibling's cluster-barrier state.
      flush();
      output.push_back(Mutate(statement, mode).ValueOrUnchanged(statement));
    }
    flush();
    return SeqStmt(output, sequence->loc);
  }

private:
  Optional<PrimVar> thread_x_;
  bool disable_elect_;
};
} // namespace

namespace transform {
tvm::transform::Pass LowerSharedBarrier() {
  auto rewrite = [](tirx::Function function, IRModule,
                    tvm::transform::PassContext context) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto lowerer = make_object<SharedBarrierLowerer>(
        context->GetConfig<bool>(kDisableShuffleElect).value_or(false));
    SeqStmt body = function->body;
    function.CopyOnWrite()->body = lowerer->Mutate(body).ValueOrUnchanged(body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(rewrite, 0,
                                             "tl.LowerSharedBarrier");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.LowerSharedBarrier",
                                   LowerSharedBarrier);
}
} // namespace transform
} // namespace tvm::tl

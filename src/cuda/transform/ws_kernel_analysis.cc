#include "ws_kernel_analysis.h"
#include "ir/annotations.h"
#include "transform/common/warp_specialize.h"
#include "ws_analysis.h"
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl {
using namespace tirx;
using namespace ffi;
bool IsWSAllocation(const BindObj *bind) {
  const auto *call = bind->value.as<CallObj>();
  return bind->var.as<TensorVar>() && call &&
         call->op.same_as(alloc_tensor_op());
}

void MergeWSAnnotations(Map<String, Any> *into, const Map<String, Any> &from) {
  for (const auto &[key, value] : from)
    into->Set(key, value);
}

WSKernelAnalysis AnalyzeWSKernel(SeqStmt body,
                                 const PipelineMemoryAnalysis *memory) {
  WSKernelAnalysis kernel;
  kernel.loc = body->loc;
  kernel.memory = memory;
  while (body.size() == 1) {
    const auto *region = body[0].as<RegionStmtObj>();
    if (!region)
      break;
    bool peel = false;
    if (region->op.same_as(launch_thread_op())) {
      auto tag = region->args[0].as_or_throw<StringImm>()->value;
      peel = tag == "threadIdx.y" || tag == "threadIdx.z";
    } else if (auto data = WSAttribute(region)) {
      peel = data.value().at("key").cast<String>() == "tl.kernel";
      if (peel)
        MergeWSAnnotations(&kernel.annotations,
                           data.value().at("value").cast<Map<String, Any>>());
    }
    if (!peel)
      break;
    kernel.wrappers.push_back(GetRef<RegionStmt>(region));
    body = region->body;
  }
  // Kernel attributes may nest after allocations (for example swizzle and
  // minimum-block annotations). Read schedule/layout metadata through those
  // transparent scopes, keeping the wrappers for the materializer to rebuild.
  // A named operation or a defining region is not a metadata boundary.
  std::function<SeqStmt(const SeqStmt &)> extract = [&](const SeqStmt &body) {
    Array<Stmt> executable;
    for (const Stmt &statement : body->seq) {
      if (const auto *bind = statement.as<BindObj>();
          bind && IsWSAllocation(bind)) {
        auto tensor = bind->var.as_or_throw<TensorVar>();
        kernel.allocations.push_back(tensor);
        kernel.definitions.Set(tensor, GetRef<Bind>(bind));
        continue;
      }
      if (const auto *bind = statement.as<BindObj>()) {
        const auto *call = bind->value.as<CallObj>();
        auto root = memory->Root(bind->var);
        if (call && call->op.same_as(decl_tensor_op()) && root &&
            root.value().as<TensorVar>() &&
            kernel.definitions.count(root.value().as_or_throw<TensorVar>()) &&
            SideEffect(bind->value) <= CallEffectKind::kPure) {
          // Views are explicit native definitions but correspond to no runtime
          // operation in the source DSL. Preserve their one evaluation before
          // role partitioning and shift the captured pointer at each access.
          kernel.views.push_back(GetRef<Bind>(bind));
          continue;
        }
      }
      if (const auto *evaluate = statement.as<EvaluateObj>();
          evaluate && !WSNextOpId(statement)) {
        if (const auto *call = evaluate->value.as<CallObj>();
            IsAnnotationCall(call)) {
          MergeWSAnnotations(&kernel.annotations, ReadAnnotations(call));
          continue;
        }
      }
      if (const auto *region = statement.as<RegionStmtObj>()) {
        auto attribute = WSAttribute(region);
        if (attribute && region->body_params.empty() &&
            region->result_vars.empty() &&
            attribute.value().at("key").cast<String>() != kWSOpIdKey) {
          executable.push_back(RegionStmt(
              region->op, region->args, region->body_params, region->attrs,
              extract(region->body), region->result_vars, region->loc));
          continue;
        }
      }
      executable.push_back(statement);
    }
    return SeqStmt(executable, body->loc);
  };
  kernel.body = extract(body);
  return kernel;
}

SeqStmt RebuildWSKernel(const WSKernelAnalysis &kernel, Stmt body,
                        const Map<TensorVar, TensorVar> &replacements,
                        const Array<TensorVar> &extra_allocations) {
  Array<Stmt> statements;
  for (const TensorVar &old : kernel.allocations) {
    auto replacement = replacements.Get(old);
    if (replacement) {
      const auto *call = kernel.definitions.at(old)->value.as<CallObj>();
      statements.push_back(
          Bind(*replacement,
               Call((*replacement).ty(), alloc_tensor_op(), {}, call->attrs,
                    {(*replacement).ty()}, (*replacement)->loc),
               (*replacement)->loc));
    } else
      statements.push_back(kernel.definitions.at(old));
  }
  for (const auto &tensor : extra_allocations)
    statements.push_back(
        Bind(tensor,
             Call((tensor).ty(), alloc_tensor_op(), {}, std::nullopt,
                  {(tensor).ty()}, (tensor)->loc),
             (tensor)->loc));
  for (const auto &view : kernel.views) {
    auto remapper = make_object<StmtExprMutator>();
    for (const auto &[old, tensor] : replacements)
      remapper->VarRemapSet(old, tensor);
    Expr value = remapper->Mutate(view->value).ValueOrUnchanged(view->value);
    statements.push_back(Bind(view->var, value, view->loc));
  }
  if (!kernel.annotations.empty())
    statements.push_back(MakeAnnotation(kernel.annotations, kernel.loc));
  statements.push_back(body);
  SeqStmt result = SeqStmt(statements, kernel.loc);
  for (auto it = kernel.wrappers.rbegin(); it != kernel.wrappers.rend(); ++it) {
    auto region = *it;
    result = SeqStmt(RegionStmt(region->op, region->args, region->body_params,
                                region->attrs, result, region->result_vars,
                                region->loc));
  }
  return result;
}

// MaterializeKernelLaunch has already established native thread definitions.
// Rewrite one x launch at a time; y/z definitions and kernel metadata stay in
// their lexical regions. A declined automatic schedule returns its exact input.
tirx::Function RewriteWSKernels(tirx::Function function,
                                const WSKernelRewrite &rewrite) {
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return function;
  class Rewriter : public StmtExprMutator {
  public:
    using StmtExprMutator::Mutate;
    using StmtExprMutator::Mutate_;
    Rewriter(tirx::Function function, WSKernelRewrite rewrite)
        : target_(function->GetAttr<Target>(tvm::attr::kTarget).value()),
          memory_(function), rewrite_(std::move(rewrite)) {}
    UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
      if (!op->op.same_as(launch_thread_op()) ||
          op->args[0].as_or_throw<StringImm>()->value != "threadIdx.x")
        return StmtExprMutator::Mutate_(op, mode);
      auto kernel = AnalyzeWSKernel(op->body, &memory_);
      kernel.threads = op->args[1].as_or_throw<PrimExpr>();
      auto result =
          rewrite_(std::move(kernel), op->body_params[0].as_or_throw<PrimVar>(),
                   target_);
      if (!result)
        return ffi::Unchanged();
      auto args = op->args;
      if (result.value().threads)
        args.Set(1, IntImm(op->args[1]->ty.as_or_throw<PrimType>(),
                           result.value().threads.value()));
      return RegionStmt(op->op, args, op->body_params, op->attrs,
                        result.value().body, op->result_vars, op->loc);
    }

  private:
    Target target_;
    PipelineMemoryAnalysis memory_;
    WSKernelRewrite rewrite_;
  };
  TVM_FFI_CHECK(function->GetAttr<Target>(tvm::attr::kTarget), ValueError)
      << "Warp specialization requires a bound target";
  auto mutator = make_object<Rewriter>(function, rewrite);
  auto body = function->body;
  auto result = mutator->Mutate(body).ValueOrUnchanged(body);
  if (!result.same_as(body))
    function.CopyOnWrite()->body = result;
  return function;
}

} // namespace tvm::tl

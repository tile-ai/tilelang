#include "compat/metadata.h"
/*!
 * \file normalize_control_flow_for_schedule.cc
 * \brief Normalize control flow ahead of AutoSchedule.
 *
 * Three rewrites, applied in this order:
 *   1. while-rewrite: `while cond: BODY` becomes a bounded serial for loop
 *      `for w in range(0, 3): if (not cond): T.loop_break(); BODY`, tagged with
 * a `synthetic_while` annotation. This lets AutoSchedule's For->ControlNode
 *      machinery software-pipeline / multi-buffer the while body.
 * RestoreWhileLoops turns the tagged for back into `while(true)` after
 * scheduling.
 *   2. if-condition-extract: hoist any complex `if` condition (including the
 *      `not cond` guard introduced above) into a temporary Bind variable so
 *      AutoSchedule sees a flat scalar, while keeping the if-then-else as a
 * single node (both branches preserved).
 *   3. loop-bound-extract: hoist serial/unrolled loop min, extent, and step
 *      expressions that access buffers into preceding Bind statements. This
 *      preserves their evaluate-on-entry semantics and gives dependency and
 *      lifetime analysis an ordinary Task endpoint for each access.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <numeric>

#include "ascend/transform/auto_schedule/kernel_rewriter.h"
#include "compat/expr.h"
#include "compat/visit.h"

namespace tvm::tl {
namespace {

class ControlFlowNormalizer : public tvm::tl::StmtExprMutator {
protected:
  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    if (node->attrs->dict.count("tl.ascend_stage") ||
        node->attrs->dict.count("tilelang_assume") ||
        node->attrs->dict.count("tl.assume_requires_runtime_check"))
      return tvm::tl::StmtExprMutator::Mutate_(node, mode);
    return ffi::Unchanged{};
  }

  ffi::UnchangedOr<Stmt> Mutate_(const WhileNode *node,
                                 ffi::InplaceMode) final {
    SeqStmt body = Mutate(node->body).ValueOrUnchanged(node->body);
    int64_t max_stages = 1;
    PostOrderVisit(body, [&](const ffi::ObjectRef &object) {
      if (auto loop = object.as<ForNode>()) {
        if (auto stages = loop->annotations.Get("num_stages")) {
          auto count = GetIntAnnotation(*stages);
          TVM_FFI_CHECK(count && *count > 0, ValueError)
              << "num_stages must be a positive constant integer";
          max_stages = std::max(max_stages, *count);
        }
      }
    });
    int64_t extent = 1;
    for (int64_t count = 1; count <= max_stages; ++count) {
      int64_t multiplier = count / std::gcd(extent, count);
      TVM_FFI_CHECK(extent <= std::numeric_limits<int32_t>::max() / multiplier,
                    ValueError)
          << "While scheduling period exceeds int32";
      extent *= multiplier;
    }
    extent = std::max<int64_t>(extent, 2);
    if (GetConstInt(node->condition) != 1) {
      If guard(!node->condition, SeqStmt(Break(node->loc)), std::nullopt,
               node->loc);
      Stmt normalized_guard = Mutate(Stmt(guard)).ValueOrUnchanged(Stmt(guard));
      body = SeqStmt({normalized_guard, body}, node->loc);
    }
    PrimVar counter("__while_w_" + std::to_string(while_counter_++),
                    PrimType::Int(32));
    return For(counter, IntImm::Int32(0), IntImm::Int32(extent),
               IntImm::Int32(1), ForKind::kDefault, body,
               {{"synthetic_while", 1}}, node->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const IfNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt original = ffi::GetRef<Stmt>(node);
    auto conditional = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                           .ValueOrUnchanged(original)
                           .as_or_throw<If>();
    if (conditional->condition.as<VarNode>())
      return conditional;
    PrimVar snapshot("__cond_" + std::to_string(condition_counter_++),
                     PrimType::Bool());
    Bind binding(snapshot, conditional->condition, node->loc);
    conditional.CopyOnWrite()->condition = snapshot;
    return SeqStmt({binding, conditional}, node->loc);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    if (node->kind != ForKind::kDefault && node->kind != ForKind::kUnrolled) {
      return ffi::Unchanged{};
    }
    Stmt original = ffi::GetRef<Stmt>(node);
    auto loop = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                    .ValueOrUnchanged(original)
                    .as_or_throw<For>();
    ffi::Array<Stmt> snapshots;
    auto snapshot = [&](PrimExpr expression) -> PrimExpr {
      if (SideEffect(expression) == CallEffectKind::kPure)
        return expression;
      PrimVar value("__loop_bound_" + std::to_string(bound_counter_++),
                    expression.ty());
      snapshots.push_back(Bind(value, expression, node->loc));
      return value;
    };
    auto writer = loop.CopyOnWrite();
    writer->begin = snapshot(writer->begin);
    writer->end = snapshot(writer->end);
    writer->step = snapshot(writer->step);
    if (snapshots.empty())
      return loop;
    snapshots.push_back(loop);
    return SeqStmt(snapshots, node->loc);
  }

private:
  int condition_counter_{0};
  int while_counter_{0};
  int bound_counter_{0};
};

} // namespace

tvm::transform::Pass NormalizeControlFlowForSchedule() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    return ascend::RewriteTilelangKernels(
        function, "NormalizeControlFlowForSchedule",
        [](const ascend::TilelangKernelContext &context) {
          auto rewriter = ffi::make_object<ControlFlowNormalizer>();
          RegionStmt root = context.root;
          root.CopyOnWrite()->body =
              rewriter->Mutate(root->body).ValueOrUnchanged(root->body);
          return root;
        },
        false);
  };
  return tirx::transform::CreateFunctionPass(
      pass, 0, "tl.NormalizeControlFlowForSchedule");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.transform.NormalizeControlFlowForSchedule",
      NormalizeControlFlowForSchedule);
}
} // namespace tvm::tl

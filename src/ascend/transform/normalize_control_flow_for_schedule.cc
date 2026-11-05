/*!
 * \file normalize_control_flow_for_schedule.cc
 * \brief Normalize native control flow ahead of Ascend AutoSchedule.
 *
 * While conditions remain at the top of every logical iteration. Complex if
 * conditions and memory-dependent loop bounds become flat SSA Bind snapshots;
 * the native For stores an exclusive end, not the former (min, extent) pair.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <string>
#include <utility>

#include "ascend/transform/auto_schedule/kernel_rewriter.h"
#include "op/native_utils.h"

namespace tvm::tl {
using namespace tirx;
using namespace ascend;
using namespace ffi;
using namespace prim;

namespace {
class ControlFlowNormalizer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  static SeqStmt Rewrite(const SeqStmt &body) {
    auto rewriter = make_object<ControlFlowNormalizer>();
    return rewriter->Mutate(body).ValueOrUnchanged(body);
  }

private:
  int counter_{0};
  int while_counter_{0};
  int loop_bound_counter_{0};

  static int64_t MaxNumStages(const Stmt &body) {
    int64_t result = 1;
    VisitNativePostOrder(body, [&](const ObjectRef &node) {
      if (const auto *loop = node.as<ForObj>()) {
        if (auto value = ReadLoopAnnotations(loop).Get("num_stages")) {
          auto stages = value->try_cast<PrimExpr>();
          auto integer = stages ? TryConstInt(*stages) : std::nullopt;
          TVM_FFI_CHECK(integer && *integer > 0, ValueError)
              << "num_stages must be a positive constant integer";
          result = std::max(result, *integer);
        }
      }
    });
    return result;
  }

  // The finite surrogate is an analysis domain, not a bound on execution.
  // A multiple of all possible version counts removes enclosing-loop terms
  // from while-local ring subscripts. Check the LCM before narrowing to i32.
  UnchangedOr<Stmt> Mutate_(const WhileObj *op, InplaceMode) final {
    SeqStmt body = Mutate(op->body).ValueOrUnchanged(op->body);
    int64_t extent = 1;
    for (int64_t stages = 1, max = MaxNumStages(body); stages <= max;
         ++stages) {
      int64_t factor = extent / std::gcd(extent, stages);
      TVM_FFI_CHECK_LE(factor, std::numeric_limits<int32_t>::max() / stages,
                       ValueError)
          << "Synthetic while version period exceeds int32";
      extent = factor * stages;
    }
    extent = std::max<int64_t>(extent, 2);
    PrimVar var("__while_w_" + std::to_string(while_counter_++),
                PrimType::Int(32), op->loc);
    if (!IsOne(op->condition)) {
      If guard(Not(op->condition), SeqStmt({Break(op->loc)}, op->loc),
               std::nullopt, op->loc);
      Stmt normalized = Mutate(guard).ValueOrUnchanged(guard);
      Array<Stmt> statements{normalized};
      for (const Stmt &stmt : body->seq)
        statements.push_back(stmt);
      body = SeqStmt(std::move(statements), op->body->loc);
    }
    return For(var, IntImm::Int32(0), IntImm::Int32(extent), IntImm::Int32(1),
               ForKind::kDefault, std::move(body),
               {{"synthetic_while", IntImm::Int32(1)}}, op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const IfObj *op, InplaceMode mode) final {
    If branch = StmtExprMutator::Mutate_(op, mode)
                    .ValueOrUnchanged(GetRef<Stmt>(op))
                    .as_or_throw<If>();
    if (branch->condition.as<VarObj>())
      return branch;
    PrimVar condition("__cond_" + std::to_string(counter_++),
                      branch->condition.ty(), branch->condition->loc);
    If normalized(condition, branch->then_case, branch->else_case, branch->loc);
    return SeqStmt({Bind(condition, branch->condition, branch->condition->loc),
                    std::move(normalized)},
                   branch->loc);
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode) final {
    Array<Stmt> statements;
    for (const Stmt &stmt : op->seq) {
      Stmt rewritten = Mutate(stmt).ValueOrUnchanged(stmt);
      if (const auto *sequence = rewritten.as<SeqStmtObj>()) {
        for (const Stmt &item : sequence->seq)
          statements.push_back(item);
      } else {
        statements.push_back(rewritten);
      }
    }
    return SeqStmt(std::move(statements), op->loc);
  }

  // Only serial/unrolled loops belong to the scheduler's control-flow spine.
  // Parallel/vectorized loops and block regions are atomic task boundaries.
  UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode) final {
    if (op->kind != ForKind::kDefault && op->kind != ForKind::kUnrolled)
      return Unchanged();
    Array<Stmt> prefix;
    auto snapshot = [&](const PrimExpr &expr) -> PrimExpr {
      if (SideEffect(expr) <= CallEffectKind::kPure)
        return expr;
      PrimVar var("__loop_bound_" + std::to_string(loop_bound_counter_++),
                  expr.ty(), expr->loc);
      prefix.push_back(Bind(var, expr, expr->loc));
      return var;
    };
    // Preserve argument evaluation order and entry-time snapshots even if the
    // body updates a tensor that occurs in the begin/end/step expressions.
    PrimExpr begin = snapshot(op->begin);
    PrimExpr end = snapshot(op->end);
    PrimExpr step = snapshot(op->step);
    SeqStmt body = Mutate(op->body).ValueOrUnchanged(op->body);
    For loop(op->var, std::move(begin), std::move(end), std::move(step),
             op->kind, std::move(body), op->annotations, op->loc);
    if (prefix.empty())
      return loop;
    prefix.push_back(loop);
    return SeqStmt(std::move(prefix), op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    if (NativeBlockName(op))
      return Unchanged();
    return StmtExprMutator::Mutate_(op, mode);
  }
};
} // namespace

tvm::transform::Pass NormalizeControlFlowForSchedule() {
  auto pass = [](tirx::Function f, const IRModule &,
                 const tvm::transform::PassContext &) {
    return RewriteTilelangKernels(
        std::move(f), "NormalizeControlFlowForSchedule",
        [](const TilelangKernelContext &context) {
          RegionStmt root = context.root;
          root.CopyOnWrite()->body = ControlFlowNormalizer::Rewrite(root->body);
          return root;
        },
        false);
  };
  return tirx::transform::CreateFunctionPass(
      pass, 0, "tl.NormalizeControlFlowForSchedule");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.NormalizeControlFlowForSchedule",
                              NormalizeControlFlowForSchedule);
}
} // namespace tvm::tl

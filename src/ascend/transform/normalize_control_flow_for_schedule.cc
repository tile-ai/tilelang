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
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <utility>

#include "ascend/transform/auto_schedule/kernel_rewriter.h"
#include "ascend/transform/auto_schedule/task_analysis.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ascend;
using namespace ffi;

class ControlFlowNormalizer : public StmtExprMutator {
public:
  static Stmt Rewrite(const Stmt &body) {
    auto rewriter = ControlFlowNormalizer();
    return rewriter(body);
  }

private:
  ControlFlowNormalizer() = default;

  // counter to generate unique name for each IfStmt
  int counter_ = 0;
  // counter to generate unique loop var for each rewritten while
  int while_counter_ = 0;
  // counter to generate unique snapshots for buffer-dependent loop bounds
  int loop_bound_counter_ = 0;

  //! \brief Check if the expression is a simple variable.
  bool IsSimpleVar(const PrimExpr &expr) {
    return expr.as<VarNode>() != nullptr;
  }

  static int64_t Gcd(int64_t a, int64_t b) {
    while (b != 0) {
      int64_t t = a % b;
      a = b;
      b = t;
    }
    return a;
  }

  // Largest `num_stages` annotation among For loops inside `stmt` (default 1).
  static int64_t MaxNumStages(const Stmt &stmt) {
    int64_t max_stages = 1;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (const auto *for_node = node.as<ForNode>()) {
        auto ns = for_node->annotations.Get("num_stages");
        if (ns.has_value()) {
          max_stages =
              std::max<int64_t>(max_stages, ns.value().cast<IntImm>()->value);
        }
      }
    });
    return max_stages;
  }

  // while cond: BODY  ->  for w in range(0, W): { if (not cond) loop_break();
  // BODY } The guard is routed back through VisitStmt so the IfThenElse visitor
  // below hoists `not cond` into a Bind variable.
  //
  // Extent W = max(2, lcm(1..max_num_stages_in_body)):
  //  - W never enters a buffer's version/flag subscript directly (the outermost
  //    loop var's extent is a discarded final multiplier in
  //    CalculateIterationCount). It does become the coefficient stride for any
  //    loop enclosing this while. Choosing W = lcm(1..max_num_stages) makes the
  //    stride M_inner * W divisible by every possible num_versions
  //    (num_versions
  //    <= num_stages <= max_num_stages), so an *enclosing* real for's loop var
  //    vanishes from the while-body version subscripts after Simplify. This
  //    keeps the while body's indices self-contained, so RestoreWhileLoops can
  //    safely substitute w:=0.
  //  - W >= 2 guarantees the cross-iteration dependency analysis sees a
  //  distinct
  //    producer/consumer iteration (w_c > w_p) for while-level dependencies.
  // num_stages is left unset (=1) so the synthetic for is not itself pipelined.
  Stmt VisitStmt_(const WhileNode *op) final {
    Var loop_var("__while_w_" + std::to_string(while_counter_++),
                 DataType::Int(32));
    Stmt visited_body = VisitStmt(op->body);
    int64_t max_stages = MaxNumStages(visited_body);
    int64_t extent = 1;
    for (int64_t m = 1; m <= max_stages; ++m) {
      extent = extent / Gcd(extent, m) * m;
    }
    extent = std::max<int64_t>(extent, 2);
    Stmt body;
    if (is_one(op->condition)) {
      body = visited_body;
    } else {
      Stmt guard = VisitStmt(
          IfThenElse(logical_not(op->condition),
                     Evaluate(Call(DataType::Handle(), tl::loop_break(), {}))));
      body = SeqStmt({guard, visited_body});
    }
    Map<String, Any> annotations;
    annotations.Set("synthetic_while", Integer(1));
    return For(loop_var, IntImm(DataType::Int(32), 0),
               IntImm(DataType::Int(32), extent), ForKind::kSerial, body,
               /*thread_binding=*/std::nullopt, annotations);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = VisitExpr(op->condition);
    Stmt then_case = VisitStmt(op->then_case);
    Optional<Stmt> else_case = op->else_case;
    if (else_case.defined()) {
      else_case = VisitStmt(else_case.value());
    }

    if (IsSimpleVar(condition)) {
      // The condition is already a simple variable; keep the if as-is.
      return IfThenElse(condition, then_case, else_case);
    }

    // Extract the complex condition into a temporary bind variable so its
    // value is computed once and consumers see a flat var, but keep the
    // if-then-else as a single node. The initial scheduled-TIR builder handles
    // both branches natively.
    std::string var_name = "__cond_" + std::to_string(counter_++);
    Var cond_var(var_name, DataType::Bool());
    Stmt new_if = IfThenElse(cond_var, then_case, else_case);
    return SeqStmt({tirx::Bind(cond_var, condition), new_if});
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> seq;
    for (auto stmt : op->seq) {
      auto new_stmt = VisitStmt(stmt);
      if (!new_stmt.defined())
        continue;
      if (auto seq_node = new_stmt.as<SeqStmtNode>()) {
        seq.insert(seq.end(), seq_node->seq.begin(), seq_node->seq.end());
      } else {
        seq.push_back(new_stmt);
      }
    }
    return SeqStmt(std::move(seq));
  }

  // Only descend through the same spine AutoSchedule's IRStructure tree walks:
  // serial/unrolled ControlNode loops. The callback runs this visitor on
  // tilelang_root's body. Everything else (Parallel / Vectorized loops,
  // SIMT_VF / SIMD_VF / user blocks) becomes an opaque TaskNode leaf that is
  // never scheduled, so a
  // `while` nested inside it must NOT be rewritten (codegen emits it verbatim)
  // and its conditions need no extraction. This mirrors the TaskNode leaf
  // boundaries.
  Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled) {
      return GetRef<For>(op);
    }

    PrimExpr min = VisitExpr(op->min);
    PrimExpr extent = VisitExpr(op->extent);
    Optional<PrimExpr> step = op->step;
    if (step.defined())
      step = VisitExpr(step.value());
    Stmt body = VisitStmt(op->body);

    Array<Stmt> sequence;
    auto snapshot_buffer_access = [&](PrimExpr expr) {
      if (!HasTaskBufferAccess(Evaluate(expr)))
        return expr;
      Var snapshot("__loop_bound_" + std::to_string(loop_bound_counter_++),
                   expr.dtype());
      sequence.push_back(Bind(snapshot, std::move(expr)));
      return PrimExpr(snapshot);
    };
    min = snapshot_buffer_access(std::move(min));
    extent = snapshot_buffer_access(std::move(extent));
    if (step.defined())
      step = snapshot_buffer_access(std::move(step.value()));

    For loop = GetRef<For>(op);
    ForNode *writer = loop.CopyOnWrite();
    writer->min = std::move(min);
    writer->extent = std::move(extent);
    writer->step = std::move(step);
    writer->body = std::move(body);
    if (sequence.empty())
      return loop;
    sequence.push_back(std::move(loop));
    return SeqStmt::Flatten(sequence);
  }

  Stmt VisitStmt_(const SBlockNode *op) final { return GetRef<SBlock>(op); }
};

using namespace tirx::transform;
tvm::transform::Pass NormalizeControlFlowForSchedule() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return RewriteTilelangKernels(
        std::move(f), "NormalizeControlFlowForSchedule",
        [](const TilelangKernelContext &context) {
          SBlock root = context.root;
          root.CopyOnWrite()->body = ControlFlowNormalizer::Rewrite(root->body);
          return root;
        },
        /*require_kernel=*/false);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.NormalizeControlFlowForSchedule",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.NormalizeControlFlowForSchedule",
                        NormalizeControlFlowForSchedule);
}

} // namespace tl
} // namespace tvm

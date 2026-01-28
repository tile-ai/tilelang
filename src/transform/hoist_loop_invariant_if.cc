/*!
 * \file hoist_loop_invariant_if.cc
 * \brief Hoist loop-invariant if statements out of loops
 *
 * This pass detects if statements inside loops where the condition is
 * loop-invariant (does not depend on the loop variable and the buffers
 * used in the condition are not modified within the loop body), and
 * hoists them outside the loop.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <unordered_set>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Collect all buffers/vars that are written to in a statement
 */
class WriteCollector : public StmtExprVisitor {
public:
  std::unordered_set<const VarNode *> written_vars;
  std::unordered_set<const BufferNode *> written_buffers;

  static void Collect(const Stmt &stmt,
                      std::unordered_set<const VarNode *> &written_vars,
                      std::unordered_set<const BufferNode *> &written_buffers) {
    WriteCollector collector;
    collector.VisitStmt(stmt);
    written_vars = std::move(collector.written_vars);
    written_buffers = std::move(collector.written_buffers);
  }

private:
  void VisitStmt_(const BufferStoreNode *op) final {
    written_buffers.insert(op->buffer.get());
    written_vars.insert(op->buffer->data.get());
    StmtExprVisitor::VisitStmt_(op);
  }
};

/*!
 * \brief Check if an expression reads from any of the given buffers/vars
 */
class ReadChecker : public ExprVisitor {
public:
  bool reads_written_buffer = false;
  const std::unordered_set<const VarNode *> &written_vars;
  const std::unordered_set<const BufferNode *> &written_buffers;

  ReadChecker(const std::unordered_set<const VarNode *> &written_vars,
              const std::unordered_set<const BufferNode *> &written_buffers)
      : written_vars(written_vars), written_buffers(written_buffers) {}

  static bool
  Check(const PrimExpr &expr,
        const std::unordered_set<const VarNode *> &written_vars,
        const std::unordered_set<const BufferNode *> &written_buffers) {
    ReadChecker checker(written_vars, written_buffers);
    checker.VisitExpr(expr);
    return checker.reads_written_buffer;
  }

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    if (written_buffers.count(op->buffer.get()) ||
        written_vars.count(op->buffer->data.get())) {
      reads_written_buffer = true;
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode *op) final {
    if (written_vars.count(op)) {
      reads_written_buffer = true;
    }
  }
};

/*!
 * \brief Check if an expression uses any of the given loop variables
 */
class LoopVarChecker : public ExprVisitor {
public:
  bool uses_loop_var = false;
  const std::unordered_set<const VarNode *> &loop_vars;

  explicit LoopVarChecker(const std::unordered_set<const VarNode *> &loop_vars)
      : loop_vars(loop_vars) {}

  static bool Check(const PrimExpr &expr,
                    const std::unordered_set<const VarNode *> &loop_vars) {
    LoopVarChecker checker(loop_vars);
    checker.VisitExpr(expr);
    return checker.uses_loop_var;
  }

private:
  void VisitExpr_(const VarNode *op) final {
    if (loop_vars.count(op)) {
      uses_loop_var = true;
    }
  }
};

/*!
 * \brief Information about a hoistable if statement
 */
struct HoistableIf {
  PrimExpr condition;
  bool has_else;
};

/*!
 * \brief Collect if statements that can be hoisted from a loop body
 */
class HoistableIfCollector : public StmtVisitor {
public:
  std::vector<HoistableIf> hoistable_ifs;
  const std::unordered_set<const VarNode *> &loop_vars;
  const std::unordered_set<const VarNode *> &written_vars;
  const std::unordered_set<const BufferNode *> &written_buffers;
  int depth = 0;

  HoistableIfCollector(
      const std::unordered_set<const VarNode *> &loop_vars,
      const std::unordered_set<const VarNode *> &written_vars,
      const std::unordered_set<const BufferNode *> &written_buffers)
      : loop_vars(loop_vars), written_vars(written_vars),
        written_buffers(written_buffers) {}

  static std::vector<HoistableIf>
  Collect(const Stmt &stmt,
          const std::unordered_set<const VarNode *> &loop_vars,
          const std::unordered_set<const VarNode *> &written_vars,
          const std::unordered_set<const BufferNode *> &written_buffers) {
    HoistableIfCollector collector(loop_vars, written_vars, written_buffers);
    collector.VisitStmt(stmt);
    return std::move(collector.hoistable_ifs);
  }

private:
  void VisitStmt_(const IfThenElseNode *op) final {
    // Check if this if's condition is loop-invariant
    bool uses_loop_var = LoopVarChecker::Check(op->condition, loop_vars);
    bool reads_written =
        ReadChecker::Check(op->condition, written_vars, written_buffers);

    if (!uses_loop_var && !reads_written) {
      // This if can be hoisted
      hoistable_ifs.push_back({op->condition, op->else_case.defined()});
    }

    // Continue visiting
    StmtVisitor::VisitStmt_(op);
  }

  // Don't descend into nested loops - they have their own scope
  void VisitStmt_(const ForNode *op) final {
    // Skip nested loops
  }
};

/*!
 * \brief Remove if statements with the given condition from a statement
 *        by replacing them with just their then/else body
 */
class IfRemover : public StmtExprMutator {
public:
  const PrimExpr &target_condition;
  bool condition_is_true;

  IfRemover(const PrimExpr &target_condition, bool condition_is_true)
      : target_condition(target_condition),
        condition_is_true(condition_is_true) {}

  static Stmt Remove(const Stmt &stmt, const PrimExpr &target_condition,
                     bool condition_is_true) {
    IfRemover remover(target_condition, condition_is_true);
    return remover(stmt);
  }

private:
  Stmt VisitStmt_(const IfThenElseNode *op) final {
    if (ExprDeepEqual()(op->condition, target_condition)) {
      // Replace this if with just the appropriate body
      if (condition_is_true) {
        return this->VisitStmt(op->then_case);
      } else {
        if (op->else_case.defined()) {
          return this->VisitStmt(op->else_case.value());
        } else {
          return Evaluate(0);
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // Allow recursion into For loops - we need to remove the if inside the loop
  // body. This is safe because HoistableIfCollector only collects conditions
  // from the current loop level (it skips nested loops), so ExprDeepEqual won't
  // match conditions in nested loops.
  Stmt VisitStmt_(const ForNode *op) final {
    Stmt body = this->VisitStmt(op->body);
    if (body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    }
    return For(op->loop_var, op->min, op->extent, op->kind, body,
               op->thread_binding, op->annotations);
  }
};

/*!
 * \brief Main rewriter that hoists loop-invariant if statements
 */
class HoistLoopInvariantIfRewriter : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    f.CopyOnWrite()->body = HoistLoopInvariantIfRewriter::Apply(f->body);
    return f;
  }

  static Stmt Apply(Stmt stmt) {
    auto rewriter = HoistLoopInvariantIfRewriter();
    return rewriter(stmt);
  }

private:
  HoistLoopInvariantIfRewriter() = default;

  // Track loop variables in the current scope
  std::unordered_set<const VarNode *> current_loop_vars;

  Stmt VisitStmt_(const ForNode *op) final {
    // First, recursively process the body
    current_loop_vars.insert(op->loop_var.get());
    Stmt body = this->VisitStmt(op->body);
    current_loop_vars.erase(op->loop_var.get());

    // Collect writes in the loop body
    std::unordered_set<const VarNode *> written_vars;
    std::unordered_set<const BufferNode *> written_buffers;
    WriteCollector::Collect(body, written_vars, written_buffers);

    // Collect hoistable ifs
    std::unordered_set<const VarNode *> loop_vars = {op->loop_var.get()};
    auto hoistable_ifs = HoistableIfCollector::Collect(
        body, loop_vars, written_vars, written_buffers);

    if (hoistable_ifs.empty()) {
      // No if to hoist, just return the updated for loop
      if (body.same_as(op->body)) {
        return ffi::GetRef<Stmt>(op);
      }
      return For(op->loop_var, op->min, op->extent, op->kind, body,
                 op->thread_binding, op->annotations);
    }

    // Hoist the if statements one by one (from innermost to outermost)
    Stmt result = For(op->loop_var, op->min, op->extent, op->kind, body,
                      op->thread_binding, op->annotations);

    for (const auto &hoistable : hoistable_ifs) {
      // Create the hoisted if statement
      // Remove the if from the loop body (replacing with then case)
      Stmt true_body = IfRemover::Remove(result, hoistable.condition,
                                         /*condition_is_true=*/true);

      if (hoistable.has_else) {
        // Also need to handle the else case
        Stmt false_body = IfRemover::Remove(result, hoistable.condition,
                                            /*condition_is_true=*/false);
        result = IfThenElse(hoistable.condition, true_body, false_body);
      } else {
        result = IfThenElse(hoistable.condition, true_body);
      }
    }

    return result;
  }
};

PrimFunc HoistLoopInvariantIfSubstitute(PrimFunc &f) {
  return HoistLoopInvariantIfRewriter::Substitute(f);
}

Stmt ApplyHoistLoopInvariantIf(Stmt stmt) {
  return HoistLoopInvariantIfRewriter::Apply(stmt);
}

using namespace tir::transform;
tvm::transform::Pass HoistLoopInvariantIf() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return HoistLoopInvariantIfRewriter::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.HoistLoopInvariantIf", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.HoistLoopInvariantIf",
                        HoistLoopInvariantIf);
}

} // namespace tl
} // namespace tvm

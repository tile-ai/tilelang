/*!
 * \file loop_unswitching.cc
 * \brief Loop Unswitching: Hoist loop-invariant if statements out of loops
 *
 * Transformation:
 *   for i in range(n):        if cond:
 *       if cond:         =>       for i in range(n): A(i)
 *           A(i)               else:
 *       else:                     for i in range(n): B(i)
 *           B(i)
 *
 * A condition is loop-invariant iff:
 *   1. It does not use the loop variable
 *   2. It does not read buffers written inside the loop
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"

#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Collect buffer data vars that are written in a statement
 *
 * Handles:
 *   - BufferStore
 *   - tvm_access_ptr with write flag (rw_mask & 2)
 *   - address_of(BufferLoad) as call argument (conservative)
 */
class WrittenVarCollector : public StmtExprVisitor {
public:
  std::unordered_set<const VarNode *> written;

  void VisitStmt_(const BufferStoreNode *op) final {
    written.insert(op->buffer->data.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      // tvm_access_ptr(dtype, data, offset, extent, rw_mask)
      ICHECK_EQ(op->args.size(), 5U);
      const VarNode *buf = op->args[1].as<VarNode>();
      ICHECK(buf) << "tvm_access_ptr data argument must be a Var";
      const IntImmNode *flag = op->args[4].as<IntImmNode>();
      // Conservative: assume write if flag is non-constant
      bool maybe_write = !flag || (flag->value & 2);
      if (maybe_write) {
        written.insert(buf);
      }
    } else if (op->op.same_as(builtin::address_of())) {
      // address_of(BufferLoad) - conservatively treat as write
      ICHECK_EQ(op->args.size(), 1U);
      const auto *load = op->args[0].as<BufferLoadNode>();
      ICHECK(load) << "address_of argument must be a BufferLoad";
      written.insert(load->buffer->data.get());
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

/*!
 * \brief Check if an expression reads any written buffer
 *
 * Also handles Let-bound variables that are bound to BufferLoad expressions.
 */
class WrittenBufferReadChecker : public ExprVisitor {
public:
  bool reads_written = false;
  const std::unordered_set<const VarNode *> &written_vars;
  const std::unordered_map<const VarNode *, PrimExpr> *let_bindings;

  explicit WrittenBufferReadChecker(
      const std::unordered_set<const VarNode *> &written,
      const std::unordered_map<const VarNode *, PrimExpr> *bindings = nullptr)
      : written_vars(written), let_bindings(bindings) {}

  void VisitExpr_(const BufferLoadNode *op) final {
    if (written_vars.count(op->buffer->data.get())) {
      reads_written = true;
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode *op) final {
    // Check if this var is Let-bound to a BufferLoad that reads a written
    // buffer
    if (let_bindings) {
      auto it = let_bindings->find(op);
      if (it != let_bindings->end()) {
        // Recursively check the bound expression
        VisitExpr(it->second);
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr())) {
      // tvm_access_ptr read
      ICHECK_EQ(op->args.size(), 5U);
      const VarNode *buf = op->args[1].as<VarNode>();
      ICHECK(buf) << "tvm_access_ptr data argument must be a Var";
      const IntImmNode *flag = op->args[4].as<IntImmNode>();
      bool maybe_read = !flag || (flag->value & 1);
      if (maybe_read && written_vars.count(buf)) {
        reads_written = true;
      }
    } else if (op->op.same_as(builtin::address_of())) {
      // address_of(BufferLoad) counts as reading the buffer
      ICHECK_EQ(op->args.size(), 1U);
      const auto *load = op->args[0].as<BufferLoadNode>();
      ICHECK(load) << "address_of argument must be a BufferLoad";
      if (written_vars.count(load->buffer->data.get())) {
        reads_written = true;
      }
    }
    ExprVisitor::VisitExpr_(op);
  }
};

/*!
 * \brief Check if an expression contains any CallNode
 */
class CallNodeChecker : public ExprVisitor {
public:
  bool has_call = false;

  void VisitExpr_(const CallNode *op) final {
    has_call = true;
    // No need to continue visiting once we find a call
  }
};

/*!
 * \brief Check if a statement contains any CallNode, excluding matching If
 * nodes
 *
 * Loop unswitching is unsafe when there are function calls OUTSIDE the
 * hoisted if statement, because those calls (originally executed by all
 * threads together) would be split into different code paths after
 * unswitching, potentially breaking synchronization semantics.
 *
 * Calls INSIDE the if are safe because they were already conditionally
 * executed before unswitching.
 *
 * Since we replace ALL if statements with matching conditions, we need to
 * exclude all such if statements when checking for calls.
 */
class CallCheckerExcludingIf : public StmtExprVisitor {
public:
  bool has_call = false;
  PrimExpr excluded_condition;

  void VisitStmt_(const IfThenElseNode *op) final {
    // Skip the interior of any if statement with matching condition
    if (excluded_condition.defined() &&
        StructuralEqual()(op->condition, excluded_condition)) {
      return;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    has_call = true;
    // No need to continue once we find a call
  }
};

/*!
 * \brief Check if condition or any Let-bound variable it uses depends on loop
 * var
 */
bool UsesLoopVarThroughLetBindings(
    const PrimExpr &cond, const Var &loop_var,
    const std::unordered_map<const VarNode *, PrimExpr> *let_bindings) {
  // Check if condition directly uses loop variable
  if (UsesVar(cond, [&](const VarNode *v) { return v == loop_var.get(); })) {
    return true;
  }

  // Check if any Let-bound variable used in condition has a binding that uses
  // the loop variable
  if (let_bindings) {
    bool uses_loop_var = false;
    PostOrderVisit(cond, [&](const ObjectRef &obj) {
      if (uses_loop_var)
        return;
      if (const auto *var_node = obj.as<VarNode>()) {
        auto it = let_bindings->find(var_node);
        if (it != let_bindings->end()) {
          // Check if the bound expression uses the loop variable
          if (UsesLoopVarThroughLetBindings(it->second, loop_var,
                                            let_bindings)) {
            uses_loop_var = true;
          }
        }
      }
    });
    if (uses_loop_var) {
      return true;
    }
  }
  return false;
}

/*!
 * \brief Check if a condition is loop-invariant
 */
bool IsLoopInvariant(const PrimExpr &cond, const Var &loop_var,
                     const std::unordered_set<const VarNode *> &written_vars,
                     const std::unordered_map<const VarNode *, PrimExpr>
                         *let_bindings = nullptr) {
  // Check 1: must not use loop variable (directly or through Let bindings)
  if (UsesLoopVarThroughLetBindings(cond, loop_var, let_bindings)) {
    return false;
  }

  // Check 2: must not read written buffers (including through Let bindings)
  WrittenBufferReadChecker checker(written_vars, let_bindings);
  checker(cond);
  if (checker.reads_written) {
    return false;
  }

  // Check 3: conservatively reject if condition contains any call node
  // (calls may have side effects or depend on loop-variant state)
  CallNodeChecker call_checker;
  call_checker(cond);
  return !call_checker.has_call;
}

/*!
 * \brief Replace if nodes with matching condition with their then/else branch
 *
 * When hoisting a condition out of a loop, we need to replace ALL if statements
 * with the same condition, not just the first one found. This ensures that
 * in the then-branch all matching conditions are replaced with their then-case,
 * and in the else-branch all matching conditions are replaced with their
 * else-case.
 *
 * Also removes LetStmts for variables that have been hoisted, since they are
 * now redundant (the variable is already bound outside the loop).
 */
class IfBranchReplacer : public StmtExprMutator {
public:
  PrimExpr hoisted_condition;
  bool take_then;
  std::unordered_set<const VarNode *> hoisted_vars;

  IfBranchReplacer(
      const PrimExpr &condition, bool take_then,
      const std::vector<std::pair<Var, PrimExpr>> &hoisted_let_bindings)
      : hoisted_condition(condition), take_then(take_then) {
    for (const auto &binding : hoisted_let_bindings) {
      hoisted_vars.insert(binding.first.get());
    }
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    // Replace if the condition is structurally equal to the hoisted condition
    if (StructuralEqual()(op->condition, hoisted_condition)) {
      if (take_then) {
        return VisitStmt(op->then_case);
      } else {
        return op->else_case.defined() ? VisitStmt(op->else_case.value())
                                       : Evaluate(0);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const LetStmtNode *op) final {
    // Remove LetStmts for hoisted variables (they are now bound outside the
    // loop)
    if (hoisted_vars.count(op->var.get())) {
      return VisitStmt(op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

/*!
 * \brief Collect Let-bound variables used in an expression
 */
class LetVarCollector : public ExprVisitor {
public:
  std::vector<std::pair<Var, PrimExpr>> used_let_bindings;
  const std::unordered_map<const VarNode *, PrimExpr> &let_bindings;
  std::unordered_set<const VarNode *> visited;

  explicit LetVarCollector(
      const std::unordered_map<const VarNode *, PrimExpr> &bindings)
      : let_bindings(bindings) {}

  void VisitExpr_(const VarNode *op) final {
    if (visited.count(op))
      return;
    auto it = let_bindings.find(op);
    if (it != let_bindings.end()) {
      visited.insert(op);
      // First recursively collect Let-bound vars used in this binding's value
      VisitExpr(it->second);
      // Then add this binding (so dependencies come first)
      used_let_bindings.push_back(
          std::make_pair(ffi::GetRef<Var>(op), it->second));
    }
  }
};

/*!
 * \brief Find first hoistable if (not descending into nested loops)
 *
 * Also tracks Let bindings where variables are bound to BufferLoad expressions.
 */
class HoistableIfFinder : public StmtVisitor {
public:
  const IfThenElseNode *found = nullptr;
  const Var &loop_var;
  const std::unordered_set<const VarNode *> &written_vars;
  std::unordered_map<const VarNode *, PrimExpr> let_bindings_;
  // Let bindings that need to be hoisted with the condition
  std::vector<std::pair<Var, PrimExpr>> hoisted_let_bindings;

  HoistableIfFinder(const Var &loop_var,
                    const std::unordered_set<const VarNode *> &written_vars)
      : loop_var(loop_var), written_vars(written_vars) {}

  void VisitStmt_(const LetStmtNode *op) final {
    // Track ALL Let bindings to detect when a condition uses a variable
    // that is defined inside the loop with a loop-variant value.
    // This is necessary because variables like i_s may be bound to expressions
    // containing the loop variable (e.g., if_then_else(...k...)), and
    // conditions using such variables should not be hoisted.
    let_bindings_[op->var.get()] = op->value;
    StmtVisitor::VisitStmt_(op);
    // Remove the binding when leaving scope
    let_bindings_.erase(op->var.get());
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    if (found)
      return;
    if (IsLoopInvariant(op->condition, loop_var, written_vars,
                        &let_bindings_)) {
      found = op;
      // Collect Let-bound variables used in the condition
      LetVarCollector collector(let_bindings_);
      collector(op->condition);
      hoisted_let_bindings = std::move(collector.used_let_bindings);
      return;
    }
    StmtVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *) final {
    // Don't descend into nested loops
  }
};

/*!
 * \brief Main pass: Loop Unswitching
 */
class LoopUnswitcher : public StmtExprMutator {
public:
  Stmt VisitStmt_(const ForNode *op) final {
    // Bottom-up: process nested structures first
    Stmt body = VisitStmt(op->body);

    // Collect written buffer vars
    WrittenVarCollector collector;
    collector(body);

    // Find hoistable if
    HoistableIfFinder finder(op->loop_var, collector.written);
    finder(body);

    if (!finder.found) {
      if (body.same_as(op->body)) {
        return ffi::GetRef<Stmt>(op);
      }
      return For(op->loop_var, op->min, op->extent, op->kind, body,
                 op->thread_binding, op->annotations);
    }

    // Check if there are any function calls OUTSIDE the hoisted if statement.
    // Calls outside the if are executed by all threads together; unswitching
    // would split them into different code paths, breaking synchronization.
    // Calls inside the if are already conditionally executed, so they're safe.
    CallCheckerExcludingIf call_checker;
    call_checker.excluded_condition = finder.found->condition;
    call_checker(body);
    if (call_checker.has_call) {
      if (body.same_as(op->body)) {
        return ffi::GetRef<Stmt>(op);
      }
      return For(op->loop_var, op->min, op->extent, op->kind, body,
                 op->thread_binding, op->annotations);
    }

    // Unswitch: create two loop versions
    const IfThenElseNode *if_node = finder.found;
    PrimExpr hoisted_condition = if_node->condition;

    Stmt then_body = IfBranchReplacer(hoisted_condition, true,
                                      finder.hoisted_let_bindings)(body);
    Stmt else_body = IfBranchReplacer(hoisted_condition, false,
                                      finder.hoisted_let_bindings)(body);

    // Create new loop_var for else_loop to maintain SSA form
    Var else_loop_var(op->loop_var->name_hint, op->loop_var->dtype);
    else_body = Substitute(else_body, {{op->loop_var, else_loop_var}});

    For then_loop(op->loop_var, op->min, op->extent, op->kind, then_body,
                  op->thread_binding, op->annotations);
    For else_loop(else_loop_var, op->min, op->extent, op->kind, else_body,
                  op->thread_binding, op->annotations);

    Stmt result = IfThenElse(if_node->condition, then_loop, else_loop);

    // Wrap with hoisted Let bindings (in reverse order so first binding is
    // outermost)
    for (auto it = finder.hoisted_let_bindings.rbegin();
         it != finder.hoisted_let_bindings.rend(); ++it) {
      result = LetStmt(it->first, it->second, result);
    }

    return result;
  }
};

// --- Public API ---

Stmt ApplyLoopUnswitching(Stmt stmt) {
  return LoopUnswitcher()(std::move(stmt));
}

using namespace tir::transform;

tvm::transform::Pass LoopUnswitching() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    bool disable_loop_unswitching =
        ctx->GetConfig<Bool>(kDisableLoopUnswitching, Bool(false)).value();
    if (disable_loop_unswitching) {
      return f;
    }
    f.CopyOnWrite()->body = ApplyLoopUnswitching(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LoopUnswitching", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LoopUnswitching", LoopUnswitching);
}

} // namespace tl
} // namespace tvm

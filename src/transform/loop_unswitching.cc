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
 */
class WrittenBufferReadChecker : public ExprVisitor {
public:
  bool reads_written = false;
  const std::unordered_set<const VarNode *> &written_vars;

  explicit WrittenBufferReadChecker(
      const std::unordered_set<const VarNode *> &written)
      : written_vars(written) {}

  void VisitExpr_(const BufferLoadNode *op) final {
    if (written_vars.count(op->buffer->data.get())) {
      reads_written = true;
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
 * \brief Check if a condition is loop-invariant
 */
bool IsLoopInvariant(const PrimExpr &cond, const Var &loop_var,
                     const std::unordered_set<const VarNode *> &written_vars) {
  // Check 1: must not use loop variable
  if (UsesVar(cond, [&](const VarNode *v) { return v == loop_var.get(); })) {
    return false;
  }

  // Check 2: must not read written buffers
  WrittenBufferReadChecker checker(written_vars);
  checker(cond);
  return !checker.reads_written;
}

/*!
 * \brief Replace a specific if node with its then/else branch
 */
class IfBranchReplacer : public StmtExprMutator {
public:
  const IfThenElseNode *target;
  bool take_then;

  IfBranchReplacer(const IfThenElseNode *target, bool take_then)
      : target(target), take_then(take_then) {}

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    if (op == target) {
      if (take_then) {
        return VisitStmt(op->then_case);
      } else {
        return op->else_case.defined() ? VisitStmt(op->else_case.value())
                                       : Evaluate(0);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

/*!
 * \brief Find first hoistable if (not descending into nested loops)
 */
class HoistableIfFinder : public StmtVisitor {
public:
  const IfThenElseNode *found = nullptr;
  const Var &loop_var;
  const std::unordered_set<const VarNode *> &written_vars;

  HoistableIfFinder(const Var &loop_var,
                    const std::unordered_set<const VarNode *> &written_vars)
      : loop_var(loop_var), written_vars(written_vars) {}

  void VisitStmt_(const IfThenElseNode *op) final {
    if (found)
      return;
    if (IsLoopInvariant(op->condition, loop_var, written_vars)) {
      found = op;
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

    // Unswitch: create two loop versions
    const IfThenElseNode *if_node = finder.found;

    Stmt then_body = IfBranchReplacer(if_node, true)(body);
    Stmt else_body = IfBranchReplacer(if_node, false)(body);

    // Create new loop_var for else_loop to maintain SSA form
    Var else_loop_var(op->loop_var->name_hint, op->loop_var->dtype);
    else_body = Substitute(else_body, {{op->loop_var, else_loop_var}});

    For then_loop(op->loop_var, op->min, op->extent, op->kind, then_body,
                  op->thread_binding, op->annotations);
    For else_loop(else_loop_var, op->min, op->extent, op->kind, else_body,
                  op->thread_binding, op->annotations);

    return IfThenElse(if_node->condition, then_loop, else_loop);
  }
};

// --- Public API ---

Stmt ApplyLoopUnswitching(Stmt stmt) {
  return LoopUnswitcher()(std::move(stmt));
}

using namespace tir::transform;

tvm::transform::Pass LoopUnswitching() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
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

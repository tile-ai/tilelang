/*!
 * \file restore_while_loops.cc
 * \brief Restore `while` loops that NormalizeControlFlowForSchedule turned into
 *        bounded serial for loops so AutoSchedule could pipeline their bodies.
 *
 * For every For tagged with the `synthetic_while` annotation, rewrite
 *   `for w in range(0, W): BODY`
 * back into
 *   `while(true): BODY[w := counter]`
 * The real termination test is the `if (not cond): loop_break()` guard that
 * NormalizeControlFlowForSchedule placed at the top of BODY; it is preserved.
 *
 * Most synthetic loop variables simplify away from version and flag ring
 * indices because the synthetic for is not itself pipelined. If w occurs,
 * simplify the body before checking for residual uses so those terms do not
 * create unnecessary counters. A frontend buffer-version override can
 * nevertheless make a write-first while-local buffer multi-versioned at this
 * level. When the simplified body still uses w, retain a monotonically
 * increasing local counter so version and flag ring indices survive while
 * restoration.
 */

#include <string>
#include <utility>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class ExpressionSimplifier : public StmtExprMutator {
public:
  static Stmt Simplify(Stmt stmt) {
    ExpressionSimplifier simplifier;
    return simplifier(std::move(stmt));
  }

private:
  PrimExpr VisitExpr(const PrimExpr &expr) final {
    PrimExpr visited = StmtExprMutator::VisitExpr(expr);
    return analyzer_.Simplify(visited);
  }

  arith::Analyzer analyzer_;
};

class WhileLoopRestorer : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    auto rewriter = WhileLoopRestorer();
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  WhileLoopRestorer() = default;

  Stmt VisitStmt_(const ForNode *op) final {
    // Recurse first so nested synthetic-while fors are restored bottom-up.
    For for_node = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    auto tag = for_node->annotations.Get("synthetic_while");
    if (!tag.has_value()) {
      return for_node;
    }
    Var loop_var = for_node->loop_var;
    auto uses_loop_var = [loop_var](const Stmt &stmt) {
      return tirx::UsesVar(stmt, [loop_var](const VarNode *var) {
        return loop_var.same_as(GetRef<Var>(var));
      });
    };
    Stmt body = for_node->body;
    if (!uses_loop_var(body)) {
      return While(IntImm(DataType::Bool(), 1), body);
    }
    body = ExpressionSimplifier::Simplify(std::move(body));
    if (!uses_loop_var(body)) {
      return While(IntImm(DataType::Bool(), 1), body);
    }

    PrimExpr zero_index = IntImm(DataType::Int(32), 0);
    Buffer counter =
        decl_buffer({IntImm(DataType::Int(32), 1)}, loop_var.dtype(),
                    loop_var->name_hint + "_counter", "local.var");
    PrimExpr counter_value = BufferLoad(counter, {zero_index});
    body = tirx::Substitute(std::move(body),
                            Map<Var, PrimExpr>{{loop_var, counter_value}});
    PrimExpr next_value = counter_value + make_const(loop_var.dtype(), 1);
    Stmt increment = BufferStore(counter, next_value, {zero_index});
    Stmt loop = While(IntImm(DataType::Bool(), 1),
                      SeqStmt({body, std::move(increment)}));
    Stmt initialize = BufferStore(counter, for_node->min, {zero_index});
    return SeqStmt(
        {AllocBuffer(counter), std::move(initialize), std::move(loop)});
  }
};

using namespace tirx::transform;
tvm::transform::Pass RestoreWhileLoops() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return WhileLoopRestorer::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.RestoreWhileLoops", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.RestoreWhileLoops", RestoreWhileLoops);
}

} // namespace tl
} // namespace tvm

/*!
 * \file restore_while_loops.cc
 * \brief Restore native while loops after the Ascend scheduler's finite model.
 *
 * A residual synthetic iteration variable is captured once on loop entry from
 * a private counter. Advance that counter before running the body, so native
 * Continue follows the same ring-index progression as the surrogate For.
 * Break exits immediately and never needs a synthetic intrinsic marker.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>

#include "ir/analyzer.h"
#include "op/native_utils.h"

namespace tvm::tl {
using namespace tirx;
using namespace ffi;
using namespace prim;

namespace {
class ExpressionSimplifier : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  UnchangedOr<Any> Mutate(AnyView value, InplaceMode mode) final {
    auto result = StmtExprMutator::Mutate(value, mode);
    bool unchanged = result.IsUnchanged();
    Any rewritten = unchanged ? Any(value) : std::move(result).ValueUnchecked();
    if (auto expr = rewritten.try_cast<PrimExpr>()) {
      PrimExpr simplified = analyzer_->Simplify(*expr);
      if (!simplified.same_as(*expr))
        return Any(simplified);
    }
    if (unchanged)
      return Unchanged();
    return rewritten;
  }

private:
  sym::Analyzer analyzer_{CreateTileLangAnalyzer()};
};

class WhileLoopRestorer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const ForObj *op, InplaceMode mode) final {
    For loop = StmtExprMutator::Mutate_(op, mode)
                   .ValueOrUnchanged(GetRef<Stmt>(op))
                   .as_or_throw<For>();
    if (!loop->annotations.count("synthetic_while"))
      return loop;
    TVM_FFI_CHECK(IsOne(loop->step), ValueError)
        << "Synthetic while restoration requires a unit-step analysis loop";
    auto uses_index = [&](const Stmt &body) {
      return UsesAnyVar(
          body, [&](const VarObj *var) { return var == loop->var.get(); });
    };
    SeqStmt body = loop->body;
    if (uses_index(body)) {
      auto simplify = make_object<ExpressionSimplifier>();
      body = simplify->Mutate(body).ValueOrUnchanged(body);
    }
    if (!uses_index(body))
      return While(IntImm(PrimType::Bool(), 1), body, loop->loc);

    PrimType index_type = loop->var.ty();
    TensorType counter_type({IntImm::Int32(1)}, index_type,
                            runtime::MemoryScope("local.var"));
    TensorVar counter(loop->var->name + "_counter", counter_type, loop->loc);
    PrimVar snapshot(loop->var->name + "_iteration", index_type, loop->loc);
    auto replace = make_object<StmtExprMutator>();
    replace->VarRemapSet(loop->var, snapshot);
    body = replace->Mutate(body).ValueOrUnchanged(body);
    Array<Stmt> iteration{
        Bind(snapshot, TensorLoad(counter, {IntImm::Int32(0)}, loop->loc),
             loop->loc),
        TensorStore(counter, {IntImm::Int32(0)},
                    snapshot + IntImm(index_type, 1), loop->loc)};
    for (const Stmt &stmt : body->seq)
      iteration.push_back(stmt);
    return SeqStmt(
        {Bind(counter,
              Call(counter_type, alloc_tensor_op(), {}, {}, {counter_type},
                   loop->loc),
              loop->loc),
         TensorStore(counter, {IntImm::Int32(0)}, loop->begin, loop->loc),
         While(IntImm(PrimType::Bool(), 1),
               SeqStmt(std::move(iteration), body->loc), loop->loc)},
        loop->loc);
  }
};
} // namespace

tvm::transform::Pass RestoreWhileLoops() {
  auto pass = [](tirx::Function f, const IRModule &,
                 const tvm::transform::PassContext &) {
    if ((!f->GetAttr(tvm::attr::kDeclaration).value_or(false))) {
      auto restorer = make_object<WhileLoopRestorer>();
      f.CopyOnWrite()->body =
          restorer->Mutate(f->body).ValueOrUnchanged(f->body);
    }
    return f;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.RestoreWhileLoops");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.RestoreWhileLoops",
                              RestoreWhileLoops);
}
} // namespace tvm::tl

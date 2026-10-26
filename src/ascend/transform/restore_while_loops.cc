#include "compat/analyzer.h"
#include "compat/metadata.h"
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

#include <tvm/ffi/reflection/registry.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "compat/expr.h"
#include "compat/visit.h"

namespace tvm::tl {
namespace {

class ExpressionSimplifier : public tvm::tl::StmtExprMutator {
public:
  using tvm::tl::StmtExprMutator::Mutate;
  ffi::UnchangedOr<ffi::Any> Mutate(ffi::AnyView input,
                                    ffi::InplaceMode mode) final {
    auto result = tvm::tl::StmtExprMutator::Mutate(input, mode);
    if (def_region_kind() != kTVMFFIDefRegionKindNone || !input.as<PrimExpr>())
      return result;
    auto expression =
        std::move(result).ValueOrUnchanged(input).as_or_throw<PrimExpr>();
    return tl::arith::Simplify(analyzer_, expression);
  }

private:
  sym::Analyzer analyzer_;
};

class ContinueCounter : public tvm::tl::StmtExprMutator {
public:
  explicit ContinueCounter(Stmt increment) : increment_(std::move(increment)) {}

protected:
  ffi::UnchangedOr<Stmt> Mutate_(const ContinueNode *node,
                                 ffi::InplaceMode) final {
    return SeqStmt({increment_, ffi::GetRef<Stmt>(node)});
  }
  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *, ffi::InplaceMode) final {
    return ffi::Unchanged{};
  }
  ffi::UnchangedOr<Stmt> Mutate_(const WhileNode *, ffi::InplaceMode) final {
    return ffi::Unchanged{};
  }

private:
  Stmt increment_;
};

class WhileLoopRestorer : public tvm::tl::StmtExprMutator {
protected:
  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt original = ffi::GetRef<Stmt>(node);
    auto loop = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                    .ValueOrUnchanged(original)
                    .as_or_throw<For>();
    if (!loop->annotations.count("synthetic_while"))
      return loop;
    auto uses_counter = [&](const Stmt &body) {
      bool found = false;
      PostOrderVisit(body, [&](const ffi::ObjectRef &object) {
        if (auto variable = object.as<VarNode>())
          found = found || variable == loop->var.get();
      });
      return found;
    };
    auto simplifier = ffi::make_object<ExpressionSimplifier>();
    SeqStmt body = simplifier->Mutate(loop->body).ValueOrUnchanged(loop->body);
    auto condition = IntImm(PrimType::Bool(), 1);
    if (!uses_counter(body))
      return While(condition, body, loop->loc);
    tirx::TensorVar counter(
        loop->var->name + "_counter",
        tirx::TensorType({PrimExpr(1)}, loop->var.ty(),
                         runtime::MemoryScope("local.var")));
    TensorLoad value(counter, {PrimExpr(0)});
    Stmt substituted = tl::Substitute(body, {{loop->var, value}});
    TensorStore increment(counter, {PrimExpr(0)}, value + loop->step);
    auto continue_counter = ffi::make_object<ContinueCounter>(increment);
    substituted =
        continue_counter->Mutate(substituted).ValueOrUnchanged(substituted);
    Bind allocation(counter, Call(std::nullopt, tirx::alloc_tensor_op(), {},
                                  std::nullopt, {counter.ty()}));
    return SeqStmt(
        {allocation, TensorStore(counter, {PrimExpr(0)}, loop->begin),
         While(condition, SeqStmt({substituted, increment}), loop->loc)});
  }
};
} // namespace

tvm::transform::Pass RestoreWhileLoops() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (!function->body)
      return function;
    auto rewriter = ffi::make_object<WhileLoopRestorer>();
    SeqStmt original = function->body.value();
    function.CopyOnWrite()->body =
        rewriter->Mutate(original).ValueOrUnchanged(original);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.RestoreWhileLoops");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.RestoreWhileLoops",
                                   RestoreWhileLoops);
}
} // namespace tvm::tl

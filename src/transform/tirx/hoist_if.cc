#include "compat/metadata.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tirx_compat/transform.h"

#include "transform/tirx_passes.h"

namespace tvm::tl::tirx_passes {
using namespace tvm::tirx;
using tvm::tirx::transform::CreateFunctionPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
namespace {

class IfHoister : public tvm::tl::StmtExprMutator {
public:
  explicit IfHoister(bool allow_nontrivial_else)
      : allow_nontrivial_else_(allow_nontrivial_else) {}

  using tvm::tl::StmtExprMutator::Mutate;
  using tvm::tl::StmtExprMutator::Mutate_;

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input,
                               InplaceMode inplace_mode) final {

    if (input.as<ExprNode>())
      return ffi::Unchanged();
    if (input.as<StmtNode>() && !input.as<ForNode>() && !input.as<IfNode>() &&
        !input.as<SeqStmtNode>()) {

      return scopes_.WithNewScope([&] {
        return tvm::tl::StmtExprMutator::Mutate(input, inplace_mode);
      });
    }
    return tvm::tl::StmtExprMutator::Mutate(input, inplace_mode);
  }

private:
  struct LoopState {

    int else_depth_at_entry;

    std::vector<std::pair<PrimExpr, bool>> conditions;
  };

  struct ScopeState {

    std::vector<LoopState *> loops;

    std::unordered_map<const VarNode *, size_t> loop_depths;
  };

  UnchangedOr<Stmt> Mutate_(const SeqStmtNode *op,
                            InplaceMode inplace_mode) final {

    if (op->seq.size() == 1)
      return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);
    return scopes_.WithNewScope(
        [&] { return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode); });
  }

  size_t FindLiftDestination(const PrimExpr &condition) const {
    const auto &scope = scopes_.Current();

    size_t destination = 0;
    std::unordered_set<const ExprNode *> visited;
    auto advance = [&](const ExprNode *op) -> ffi::Expected<ffi::WalkResult> {
      return visited.insert(op).second ? ffi::WalkResult::Advance()
                                       : ffi::WalkResult::Skip();
    };
    auto division =
        [&](const ExprNode *op,
            const PrimExpr &divisor) -> ffi::Expected<ffi::WalkResult> {
      const auto *value = divisor.as<IntImmNode>();
      if (!value || value->value <= 0)
        return ffi::WalkResult::Interrupt();
      return advance(op);
    };
    auto blocked = ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
        condition,
        [&](const VarNode *op) -> ffi::Expected<ffi::WalkResult> {
          auto it = scope.loop_depths.find(op);
          if (it != scope.loop_depths.end())
            destination = std::max(destination, it->second + 1);

          if (destination == scope.loops.size())
            return ffi::WalkResult::Interrupt();

          return ffi::WalkResult::Skip();
        },
        [](const TypeNode *) -> ffi::Expected<ffi::WalkResult> {
          return ffi::WalkResult::Skip();
        },
        [](const TensorLoadNode *) -> ffi::Expected<ffi::WalkResult> {
          return ffi::WalkResult::Interrupt();
        },
        [&](const prim::DivNode *op) { return division(op, op->b); },
        [&](const prim::ModNode *op) { return division(op, op->b); },
        [&](const prim::FloorDivNode *op) { return division(op, op->b); },
        [&](const prim::FloorModNode *op) { return division(op, op->b); },
        [](const prim::LShiftNode *) -> ffi::Expected<ffi::WalkResult> {
          return ffi::WalkResult::Interrupt();
        },
        [](const prim::RShiftNode *) -> ffi::Expected<ffi::WalkResult> {
          return ffi::WalkResult::Interrupt();
        },
        [&](const prim::CastNode *op) -> ffi::Expected<ffi::WalkResult> {
          if (ffi::GetRef<PrimExpr>(op).ty().MatchesCode(kDLInt, kDLUInt) &&
              !op->value.ty().MatchesCode(kDLInt, kDLUInt))
            return ffi::WalkResult::Interrupt();
          return advance(op);
        },
        [&](const CallNode *op) -> ffi::Expected<ffi::WalkResult> {
          if (!op->op.same_as(prim::likely_op()))
            return ffi::WalkResult::Interrupt();
          return advance(op);
        },
        advance);
    return blocked.has_value() ? scope.loops.size() : destination;
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *op,
                            InplaceMode inplace_mode) final {

    if (op->op.same_as(launch_thread_op()))
      return ffi::Unchanged();
    return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> Mutate_(const ForNode *op, InplaceMode inplace_mode) final {

    if (op->kind != ForKind::kDefault ||
        SideEffect(op->begin) > CallEffectKind::kReadState ||
        SideEffect(op->end) > CallEffectKind::kReadState ||
        SideEffect(op->step) > CallEffectKind::kReadState) {
      return scopes_.WithNewScope(
          [&] { return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode); });
    }
    auto &scope = scopes_.Current();

    if (active_loops_++ == 0)
      split_in_nest_ = false;

    LoopState loop{else_depth_, {}};
    const VarNode *var = op->var.get();

    scope.loop_depths[var] = scope.loops.size();
    scope.loops.push_back(&loop);
    Stmt stmt = tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(op));

    scope.loops.pop_back();
    scope.loop_depths.erase(var);

    --active_loops_;

    for (auto it = loop.conditions.rbegin(); it != loop.conditions.rend();
         ++it) {

      const auto &[condition, preserve_else] = *it;
      stmt = preserve_else ? If(condition, stmt, SeqStmt(stmt))
                           : If(condition, stmt);
    }
    return stmt;
  }

  UnchangedOr<Stmt> Mutate_(const IfNode *op, InplaceMode inplace_mode) final {
    auto &scope = scopes_.Current();

    if (!scope.loops.empty() &&
        SideEffect(op->condition) > CallEffectKind::kReadState) {
      return scopes_.WithNewScope(
          [&] { return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode); });
    }
    bool has_else = op->else_case.has_value();
    if (!scope.loops.empty()) {
      size_t destination = FindLiftDestination(op->condition);
      if (destination < scope.loops.size()) {

        bool preserve_else =
            has_else ||
            else_depth_ > scope.loops[destination]->else_depth_at_entry;
        auto *loop = scope.loops[destination];

        if (!preserve_else || (allow_nontrivial_else_ && !split_in_nest_)) {

          loop->conditions.emplace_back(op->condition, preserve_else);

          split_in_nest_ |= preserve_else;
        }
      }
    }

    else_depth_ += has_else;
    auto result = tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);

    else_depth_ -= has_else;
    return result;
  }

  int else_depth_{0};
  bool allow_nontrivial_else_;

  int active_loops_{0};

  bool split_in_nest_{false};

  ScopeStack<ScopeState> scopes_;
};

} // namespace

Pass HoistIf(bool allow_nontrivial_else) {
  auto pass_func = [allow_nontrivial_else](Function f, IRModule m,
                                           PassContext ctx) {
    if (!f->body.has_value())
      return f;
    auto *n = f.CopyOnWrite();
    Stmt body = ffi::make_object<IfHoister>(allow_nontrivial_else)
                    ->Mutate(n->body.value(), InplaceMode::kAllow)
                    .ValueOrUnchanged(n->body.value());
    n->body = tl::ConvertSSA(std::move(body));
    return f;
  };
  return tvm::transform::Sequential(
      {CreateFunctionPass(pass_func, 0, "tl.InsertHoistedIf"),
       tirx_passes::StmtSimplify(), tirx_passes::RemoveNoOp()},
      "tl.HoistIf");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl._tirx_passes.HoistIf", HoistIf);
}

} // namespace tvm::tl::tirx_passes

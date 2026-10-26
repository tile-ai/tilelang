#include "compat/analyzer.h"
#include "op/prim.h"
#include "transform/common/loop_aware_mutator.h"

#include "cuda/transform/ptx_async_copy_injector.h"

#include "compat/expr.h"
#include "cuda/op/builtin.h"
#include "op/utils.h"
#include "transform/loop_vectorize.h"

#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <vector>

namespace tvm::tl {
using namespace ffi;
using namespace tirx;

namespace {

class PTXAsyncCopyInjector : public LoopAwareMutator {
public:
  PTXAsyncCopyInjector(sym::Analyzer analyzer, bool explicit_async)
      : LoopAwareMutator(analyzer), explicit_async_(explicit_async) {}

  Stmt Rewrite(const Stmt &statement) {
    return Mutate(statement).ValueOrUnchanged(statement).as_or_throw<Stmt>();
  }

  PTXAsyncCopyInjectResult Run(const Stmt &body) {
    Stmt result = Rewrite(body);
    if (pending_.open && !explicit_async_) {
      Array<Stmt> sequence{result};
      AppendVisibility(&sequence, pending_.uncommitted);
      result = SeqStmt(sequence);
    }
    return {result, injected_};
  }

private:
  struct SyncState {
    bool open{false};
    bool uncommitted{false};
  };

  struct CopyRegion {
    bool pure{true};
    int commits{0};
    int waits{0};
  };

  struct VectorLoop {
    PrimVar variable;
    PrimExpr begin;
    int extent;
  };

  UnchangedOr<Stmt> Mutate_(const ForNode *node, InplaceMode mode) final {
    int saved_lanes = vector_lanes_;
    bool pushed = false;
    if (node->kind == ForKind::kVectorized) {
      auto extent =
          GetConstInt(tl::arith::Simplify(analyzer_, node->end - node->begin));
      TVM_FFI_CHECK(extent && *extent >= 0, ValueError)
          << "Vectorized async copies require a constant loop extent";
      if (*extent > 1 &&
          *extent <= std::numeric_limits<int>::max() / vector_lanes_ &&
          IsOne(node->step)) {
        vector_lanes_ *= static_cast<int>(*extent);
        vector_loops_.push_back(
            {node->var, node->begin, static_cast<int>(*extent)});
        pushed = true;
      }
    }
    auto result = LoopAwareMutator::Mutate_(node, mode);
    if (pushed)
      vector_loops_.pop_back();
    vector_lanes_ = saved_lanes;
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtNode *node, InplaceMode mode) final {
    if (explicit_async_)
      return LoopAwareMutator::Mutate_(node, mode);
    Array<Stmt> statements;
    SyncState state = pending_;
    for (const Stmt &statement : node->seq) {
      pending_ = {};
      Stmt rewritten = Rewrite(statement);
      CopyRegion region = AnalyzeRegion(rewritten);
      if (state.open && !region.pure) {
        AppendVisibility(&statements, state.uncommitted);
        state = {};
      }
      if (state.open && state.uncommitted && region.waits > 0) {
        statements.push_back(Commit());
        state.uncommitted = false;
      }
      statements.push_back(rewritten);
      state.open = state.open || pending_.open;
      state.uncommitted = state.uncommitted || pending_.uncommitted;
      if (region.commits > 0)
        state.uncommitted = false;
      if (region.waits > 0)
        state = {};
    }
    pending_ = state;
    return SeqStmt(statements);
  }

  UnchangedOr<Stmt> Mutate_(const IfNode *node, InplaceMode mode) final {
    if (explicit_async_)
      return LoopAwareMutator::Mutate_(node, mode);
    SyncState before = pending_;
    SeqStmt then_case = SeqStmt(Rewrite(node->then_case));
    SyncState after_then = pending_;
    pending_ = before;
    Optional<SeqStmt> else_case;
    if (node->else_case)
      else_case = SeqStmt(Rewrite(node->else_case.value()));
    pending_ = {after_then.open || pending_.open,
                after_then.uncommitted || pending_.uncommitted};
    return If(node->condition, then_case, else_case, node->loc);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                            InplaceMode mode) final {
    TensorVar destination = node->dest.as_or_throw<TensorVar>();
    if (!IsSharedBuffer(destination))
      return LoopAwareMutator::Mutate_(node, mode);
    Optional<PrimExpr> predicate;
    const TensorLoadNode *load =
        MatchLoad(node->value.as_or_throw<PrimExpr>(), &predicate);
    if (!load)
      return LoopAwareMutator::Mutate_(node, mode);
    auto statement = TryInject(load, node, predicate);
    if (!statement)
      return LoopAwareMutator::Mutate_(node, mode);
    injected_ = true;
    if (!explicit_async_)
      pending_ = {true, true};
    return statement.value();
  }

  static const TensorLoadNode *MatchLoad(const PrimExpr &value,
                                         Optional<PrimExpr> *predicate) {
    if (const auto *load = value.as<TensorLoadNode>())
      return load;
    if (const auto *call = value.as<CallNode>();
        call && call->op.same_as(prim::if_then_else_op())) {
      if (!IsZeroBitPattern(call->args[2].as_or_throw<PrimExpr>()))
        return nullptr;
      const TensorLoadNode *load =
          MatchLoad(call->args[1].as_or_throw<PrimExpr>(), predicate);
      if (!load)
        return nullptr;
      PrimExpr condition = call->args[0].as_or_throw<PrimExpr>();
      *predicate = predicate->has_value() ? And(condition, predicate->value())
                                          : condition;
      return load;
    }
    const auto *select = value.as<SelectNode>();
    if (!select || !IsZeroBitPattern(select->false_value))
      return nullptr;
    const TensorLoadNode *load = MatchLoad(select->true_value, predicate);
    if (!load)
      return nullptr;
    *predicate = predicate->has_value()
                     ? And(select->condition, predicate->value())
                     : select->condition;
    return load;
  }

  static Optional<PrimExpr> LinearOffset(const TensorVar &tensor,
                                         const Array<PrimExpr> &indices) {
    Array<PrimExpr> offsets = tensor.ty()->ElemOffset(indices);
    if (offsets.size() != 1)
      return std::nullopt;
    return offsets[0];
  }

  static Optional<PrimExpr> VectorBase(const PrimExpr &index) {
    if (index.ty().IsScalar())
      return index;
    if (const auto *broadcast = index.as<BroadcastNode>())
      return broadcast->value;
    if (const auto *ramp = index.as<RampNode>()) {
      if (IsOne(ramp->stride))
        return ramp->base;
      return std::nullopt;
    }
    if (const auto *add = index.as<AddNode>()) {
      for (bool swap : {false, true}) {
        const PrimExpr &left = swap ? add->b : add->a;
        const PrimExpr &right = swap ? add->a : add->b;
        const auto *ramp = left.as<RampNode>();
        const auto *broadcast = right.as<BroadcastNode>();
        if (ramp && broadcast && IsOne(ramp->stride))
          return ramp->base + broadcast->value;
      }
    }
    return std::nullopt;
  }

  static Optional<Array<PrimExpr>> BaseIndices(const Array<PrimExpr> &indices) {
    Array<PrimExpr> bases;
    for (const PrimExpr &index : indices) {
      auto base = VectorBase(index);
      if (!base)
        return std::nullopt;
      bases.push_back(base.value());
    }
    return bases;
  }

  bool Contiguous(const PrimExpr &offset) {
    for (const auto &loop : vector_loops_) {
      PrimExpr previous = tl::arith::Simplify(
          analyzer_, tl::Substitute(offset, {{loop.variable, loop.begin}}));
      for (int lane = 1; lane < loop.extent; ++lane) {
        PrimExpr current = tl::arith::Simplify(
            analyzer_,
            tl::Substitute(
                offset,
                {{loop.variable,
                  loop.begin + tl::MakeConst(loop.variable.ty(), lane)}}));
        if (!tl::arith::CanProveEqual(analyzer_, current - previous,
                                      tl::MakeConst(current.ty(), 1)))
          return false;
        previous = current;
      }
    }
    return true;
  }

  Optional<Stmt> TryInject(const TensorLoadNode *load,
                           const TensorStoreNode *store,
                           const Optional<PrimExpr> &predicate) {
    TensorVar source = load->source.as_or_throw<TensorVar>();
    TensorVar destination = store->dest.as_or_throw<TensorVar>();
    if (!IsGlobalBuffer(source) || source.dtype() != destination.dtype())
      return std::nullopt;
    auto source_offset = LinearOffset(source, load->indices);
    auto destination_offset = LinearOffset(destination, store->indices);
    if (!source_offset || !destination_offset ||
        source_offset.value().ty().lanes() !=
            destination_offset.value().ty().lanes()) {
      return std::nullopt;
    }
    int index_lanes = source_offset.value().ty().lanes();
    int value_lanes = load->ty.as_or_throw<PrimType>().lanes();
    if (index_lanes > 1 && value_lanes > 1 && index_lanes != value_lanes)
      return std::nullopt;
    int elements = std::max(index_lanes, value_lanes);
    int64_t bits =
        static_cast<int64_t>(elements) * source.dtype().bits() * vector_lanes_;
    if (bits % 8 != 0 || bits > 128 || !IsValidCPAsyncTransferBytes(bits / 8))
      return std::nullopt;
    if (predicate) {
      if (!predicate.value().ty().IsScalar())
        return std::nullopt;
      for (const auto &loop : vector_loops_) {
        if (!IsExprInvariantInVectorBoundary(
                predicate.value(), loop.variable, PrimExpr(loop.extent),
                loop.extent, GetRef<sym::Analyzer>(analyzer_), loop.begin)) {
          return std::nullopt;
        }
      }
    }
    if (index_lanes == 1 && vector_lanes_ > 1 &&
        (!Contiguous(source_offset.value()) ||
         !Contiguous(destination_offset.value()))) {
      return std::nullopt;
    }
    auto source_indices = BaseIndices(load->indices);
    auto destination_indices = BaseIndices(store->indices);
    if (!source_indices || !destination_indices)
      return std::nullopt;
    Array<Expr> arguments{
        Call(std::nullopt, access_ptr(),
             {TensorLoad(destination, destination_indices.value()),
              PrimExpr(elements), PrimExpr(kAccessWrite)}),
        Call(std::nullopt, access_ptr(),
             {TensorLoad(source, source_indices.value()), PrimExpr(elements),
              PrimExpr(kAccessRead)}),
        PrimExpr(elements)};
    if (predicate)
      arguments.push_back(predicate.value());
    return Evaluate(Call(PrimType::Void(), ptx_cp_async(), arguments,
                         std::nullopt, {}, store->loc),
                    store->loc);
  }

  static Stmt Commit() {
    return Evaluate(
        Call(std::nullopt, Op::Get("tirx.ptx.cp_async_commit_group"),
             {StringImm("async"), StringImm("commit_group"), StringImm("")}));
  }

  static Stmt Wait() {
    return Evaluate(Call(std::nullopt, Op::Get("tirx.ptx.cp_async_wait_group"),
                         {PrimExpr(0), StringImm("async"),
                          StringImm("wait_group"), StringImm("")}));
  }

  static void AppendVisibility(Array<Stmt> *statements, bool commit) {
    if (commit)
      statements->push_back(Commit());
    statements->push_back(Wait());
  }

  static CopyRegion Merge(CopyRegion left, CopyRegion right) {
    return {left.pure && right.pure, left.commits + right.commits,
            left.waits + right.waits};
  }

  static CopyRegion AnalyzeRegion(const Stmt &statement) {
    CopyRegion result;
    if (const auto *sequence = statement.as<SeqStmtNode>()) {
      for (const Stmt &child : sequence->seq)
        result = Merge(result, AnalyzeRegion(child));
      return result;
    }
    if (const auto *conditional = statement.as<IfNode>()) {
      result = AnalyzeRegion(conditional->then_case);
      if (conditional->else_case)
        result = Merge(result, AnalyzeRegion(conditional->else_case.value()));
      return result;
    }
    if (const auto *evaluate = statement.as<EvaluateNode>()) {
      if (GetConstInt(evaluate->value))
        return result;
      if (const auto *call = evaluate->value.as<CallNode>()) {
        if (call->op.same_as(ptx_cp_async()))
          return result;
        if (const auto *operation = call->op.as<OpNode>()) {
          if (operation->name == "tirx.ptx.cp_async_commit_group")
            return {true, 1, 0};
          if (operation->name == "tirx.ptx.cp_async_wait_group" ||
              operation->name == "tirx.ptx.cp_async_wait_all")
            return {true, 0, 1};
        }
      }
      return {false, 0, 0};
    }
    if (const auto *loop = statement.as<ForNode>())
      return AnalyzeRegion(loop->body);
    if (const auto *region = statement.as<RegionStmtNode>()) {
      if (region->op->name == "tl.scope" || region->op->name == "tl.scope") {
        return AnalyzeRegion(region->body);
      }
    }
    return {false, 0, 0};
  }

  bool explicit_async_;
  int vector_lanes_{1};
  std::vector<VectorLoop> vector_loops_;
  SyncState pending_;
  bool injected_{false};
};

} // namespace

PTXAsyncCopyInjectResult InjectPTXAsyncCopy(const Stmt &body,
                                            bool explicit_async,
                                            Optional<sym::Analyzer> analyzer) {
  sym::Analyzer context = analyzer.value_or(sym::Analyzer());
  return make_object<PTXAsyncCopyInjector>(context, explicit_async)->Run(body);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def(
      "tl.cuda.transform.InjectPTXAsyncCopy", [](bool explicit_async) {
        return tirx::transform::CreateFunctionPass(
            [explicit_async](
                tirx::Function function, IRModule,
                tvm::transform::PassContext) -> Optional<tirx::Function> {
              if (function->body) {
                function.CopyOnWrite()->body = SeqStmt(
                    InjectPTXAsyncCopy(function->body.value(), explicit_async)
                        .stmt);
              }
              return function;
            },
            0, "tl.cuda.InjectPTXAsyncCopy");
      });
}

} // namespace tvm::tl

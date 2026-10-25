#include "thread_sync.h"
#include "compat/visit.h"
#include "iteration_private.h"
#include "memory_access.h"
#include "read_only_params.h"
#include "thread_sync_types.h"

#include <tvm/tirx/op/gpu.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <array>
#include <set>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

struct ThreadAxis {
  PrimVar variable;
  int dimension;
  int extent;
};

struct SyncContext {
  ConstrSet constraints;
  ConstrSet uniform_assumptions;
  std::vector<ThreadAxis> threads;
  std::vector<PrimExpr> conditions;
  std::vector<PrimExpr> loop_headers;
  std::vector<const ForNode *> loops;
  bool conditional_exit = false;
  bool *deferred_visibility = nullptr;
};

struct UniformWrite {
  TensorStore store;
  ConstrSet constraints;
};

struct PendingMemoryAccess {
  MemoryAccess access;
  bool warp_synchronized = false;
};

class UniformWriteCollector : public ConstrVisitor {
public:
  std::vector<UniformWrite> writes;

private:
  Optional<VisitInterrupt> Visit_(const TensorStoreNode *node) final {
    writes.push_back({GetRef<TensorStore>(node), GetConstrSet()});
    return ConstrVisitor::Visit_(node);
  }
};

const CallNode *EvaluatedCall(const Stmt &statement) {
  const auto *evaluation = statement.as<EvaluateNode>();
  return evaluation ? evaluation->value.as<CallNode>() : nullptr;
}

bool IsStorageSync(const Stmt &statement) {
  const CallNode *call = EvaluatedCall(statement);
  if (!call || !call->op.same_as(gpu_storage_sync_op()) || call->args.empty())
    return false;
  const auto *scope = call->args[0].as<StringImmNode>();
  return scope && (scope->value == "shared" || scope->value == "shared.dyn");
}

bool IsAsyncWait(const Stmt &statement) {
  const CallNode *call = EvaluatedCall(statement);
  if (!call)
    return false;
  auto operation = call->op.as<Op>();
  return operation &&
         (operation.value()->name == "tirx.ptx.cp_async_wait_group" ||
          operation.value()->name == "tirx.ptx.cp_async_wait_all");
}

bool IsWarpSync(const Stmt &statement) {
  const CallNode *call = EvaluatedCall(statement);
  if (!call || !call->op.same_as(gpu_storage_sync_op()) || call->args.empty())
    return false;
  const auto *scope = call->args[0].as<StringImmNode>();
  return scope && scope->value == "warp";
}

PrimExpr LinearThreadIndex(const SyncContext &context) {
  std::array<int, 3> extents{1, 1, 1};
  for (const ThreadAxis &thread : context.threads)
    extents[thread.dimension] = thread.extent;
  PrimExpr linear = IntImm::Int64(0);
  for (const ThreadAxis &thread : context.threads) {
    int stride = 1;
    for (int dimension = 0; dimension < thread.dimension; ++dimension)
      stride *= extents[dimension];
    linear = linear +
             Cast(PrimType::Int(64), thread.variable) * IntImm::Int64(stride);
  }
  return linear;
}

class ThreadSyncPlanner {
public:
  ThreadSyncPlanner(const tirx::Function &function,
                    ffi::Optional<ffi::String> sync_scope, bool cuda_features)
      : sync_scope_(std::move(sync_scope)), cuda_features_(cuda_features) {
    auto collector = make_object<MemoryAccessCollector>();
    collector->Visit(*function->body);
    aliases_ = std::move(collector->aliases);
    bool no_alias =
        function->GetAttr<bool>(tirx::attr::kNoAlias).value_or(true);
    if (no_alias) {
      auto indices =
          function->GetAttr<Array<IntImm>>("tl.readonly_param_indices")
              .value_or(ReadOnlyParameterIndices(function));
      for (const IntImm &index : indices) {
        auto position = index->value.as<int>();
        if (position && *position >= 0 &&
            *position < static_cast<int>(function->params.size()))
          uniform_global_.insert(function->params[*position].get());
      }
    }
    std::vector<TensorVar> locals;
    std::unordered_set<const VarNode *> escaped;
    PostOrderVisit(*function->body, [&](const ObjectRef &object) {
      if (const auto *region = object.as<RegionStmtNode>();
          region && region->op.same_as(launch_thread_op()) &&
          region->args[0].as_or_throw<StringImm>()->value.find("threadIdx.") ==
              0)
        varying_threads_.insert(region->body_params[0].get());
      if (const auto *bind = object.as<BindNode>()) {
        if (auto variable = bind->var.as<PrimVar>()) {
          if (auto value = bind->value.as<PrimExpr>())
            definitions_.Set(*variable, Substitute(*value, definitions_));
        }
        if (auto tensor = bind->var.as<TensorVar>();
            tensor && tensor.value().scope().name().find("local") == 0 &&
            aliases_.allocations.count(tensor.value().get()))
          locals.push_back(*tensor);
      }
      const auto *call = object.as<CallNode>();
      if (call && SideEffect(GetRef<Expr>(call)) > CallEffectKind::kReadState) {
        for (const Expr &argument : call->args) {
          MemoryLocation location = aliases_.Resolve(argument);
          if (location.allocation)
            escaped.insert(location.allocation.value().get());
        }
      }
      if (call && call->op.same_as(gpu_storage_sync_op()) &&
          call->args.size() >= 2) {
        if (auto identifier = call->args[1].as<PrimExpr>()) {
          if (auto value = GetConstInt(*identifier))
            used_barriers_.insert(*value);
        }
      }
    });
    auto writes = make_object<UniformWriteCollector>();
    writes->Visit(*function->body);
    bool changed = true;
    while (changed) {
      changed = false;
      for (const TensorVar &tensor : locals) {
        if (uniform_local_.count(tensor.get()) || escaped.count(tensor.get()))
          continue;
        bool has_write = false;
        bool uniform = true;
        for (const UniformWrite &write : writes->writes) {
          MemoryLocation location = aliases_.Resolve(write.store->dest);
          if (!location.allocation ||
              !location.allocation.value().same_as(tensor))
            continue;
          has_write = true;
          uniform &= IsUniform(write.store->value, true);
          for (const PrimExpr &index : write.store->indices)
            uniform &= IsUniform(index);
          for (const Constr &constraint : write.constraints.constrs_) {
            if (constraint.kind == Constr::kBindRange &&
                varying_threads_.count(constraint.var.get()))
              continue;
            uniform &= IsUniform(constraint.ToGenericConstr());
          }
        }
        if (has_write && uniform &&
            HasInitializedStorageReads(tensor, *function->body, aliases_)) {
          uniform_local_.insert(tensor.get());
          changed = true;
        }
      }
    }
  }

  SeqStmt Run(const SeqStmt &body) { return RewriteSequence(body, {}).first; }

private:
  bool UniformStorage(const MemoryLocation &location, bool allow_shared) const {
    if (!location.precise || !location.allocation)
      return false;
    if (allow_shared &&
        (location.scope == "shared" || location.scope == "shared.dyn"))
      return true;
    const VarNode *root = location.allocation.value().get();
    return uniform_global_.count(root) || uniform_local_.count(root);
  }

  bool IsUniform(const PrimExpr &expression, bool allow_shared = false) const {
    PrimExpr expanded = Substitute(expression, definitions_);
    bool uniform = true;
    PostOrderVisit(expanded, [&](const ObjectRef &object) {
      if (const auto *variable = object.as<VarNode>())
        uniform &= !varying_threads_.count(variable);
      if (const auto *load = object.as<TensorLoadNode>())
        uniform &= UniformStorage(aliases_.Resolve(load->source), allow_shared);
      if (const auto *call = object.as<CallNode>()) {
        auto operation = call->op.as<Op>();
        if (!operation) {
          uniform = false;
        } else if (operation.value()->name == "tl.tl_shuffle_elect" ||
                   operation.value()->name == "tirx.ptx.elect") {
          uniform = false;
        } else if (operation.value()->name == "tl.all_of" ||
                   operation.value()->name == "tl.any_of") {
          uniform &=
              !call->args.empty() &&
              UniformStorage(aliases_.Resolve(call->args[0]), allow_shared);
        } else {
          uniform &= Op::GetAttrMap(tvm::op_attr::kCallEffectKind)
                         .get(*operation, CallEffectKind::kOpaque) ==
                     CallEffectKind::kPure;
        }
      }
    });
    return uniform;
  }

  std::vector<MemoryAccess> Summarize(const Stmt &statement,
                                      const SyncContext &context,
                                      bool *exits = nullptr) const {
    auto collector = make_object<MemoryAccessCollector>(aliases_);
    collector->constr_stack_ = context.constraints.constrs_;
    collector->Visit(statement);
    if (exits)
      *exits = collector->exits;
    std::vector<MemoryAccess> result;
    for (MemoryAccess &access : collector->accesses) {
      if ((access.location.scope == "shared" ||
           access.location.scope == "shared.dyn") &&
          (!sync_scope_ || access.location.scope == sync_scope_.value()))
        result.push_back(std::move(access));
    }
    return result;
  }

  bool Conflict(const MemoryAccess &left, const MemoryAccess &right,
                const SyncContext &context,
                const ForNode *carried_loop = nullptr,
                bool warp_synchronized = false) const {
    if (!(left.mask & kAccessWrite) && !(right.mask & kAccessWrite))
      return false;
    if (left.atomic && right.atomic)
      return false;
    if (left.asynchronous && right.asynchronous)
      return false;
    if (context.threads.empty() ||
        aliases_.Disjoint(left.location, right.location))
      return false;
    if (!left.location.precise || !right.location.precise ||
        !left.location.allocation || !right.location.allocation ||
        !left.location.allocation.value().same_as(*right.location.allocation))
      return true;
    Map<PrimVar, PrimExpr> replacements;
    for (const ForNode *loop : context.loops) {
      if (loop != carried_loop && IsUniform(loop->begin) &&
          IsUniform(loop->end) && IsUniform(loop->step))
        replacements.Set(loop->var, loop->var);
    }
    ConstrSet other =
        right.constraints.RenameFrom("_other_thread", replacements);
    ConstrSet constraints = left.constraints.Merge(other).Normalized();
    if (carried_loop && IsUniform(carried_loop->begin) &&
        IsUniform(carried_loop->end) && IsUniform(carried_loop->step)) {
      auto later = replacements.Get(carried_loop->var);
      if (later)
        constraints.AddConstr(
            (carried_loop->step > 0 && later.value() > carried_loop->var) ||
            (carried_loop->step < 0 && later.value() < carried_loop->var));
    }
    PrimExpr other_offset = Substitute(right.location.offset, replacements);
    PrimExpr other_bytes = Substitute(right.bytes, replacements);
    constraints.AddConstr(left.location.offset < other_offset + other_bytes);
    constraints.AddConstr(other_offset < left.location.offset + left.bytes);
    PrimExpr same_thread = IntImm::Bool(true);
    for (const ThreadAxis &thread : context.threads) {
      if (!replacements.count(thread.variable))
        return true;
      same_thread =
          And(same_thread, thread.variable == replacements.at(thread.variable));
    }
    if (warp_synchronized) {
      PrimExpr linear = LinearThreadIndex(context);
      constraints.AddConstr(
          FloorDiv(linear, IntImm::Int64(32)) !=
          FloorDiv(Substitute(linear, replacements), IntImm::Int64(32)));
    }
    sym::Analyzer analyzer;
    if (analyzer->z3_prover.IsEnabled()) {
      analyzer->z3_prover.SetTimeoutMs(warp_synchronized ? 500 : 100);
      analyzer->z3_prover.SetRLimit(warp_synchronized ? 1000000 : 100000);
    }
    auto guard = constraints.Populate(analyzer);
    if (analyzer->CanProve(same_thread, sym::ProofStrength::kSymbolicBound))
      return false;
    std::set<int64_t> scales;
    auto collect_scales = [&](const ObjectRef &object) {
      const auto *multiply = object.as<MulNode>();
      if (!multiply)
        return;
      for (const PrimExpr &operand : {multiply->a, multiply->b}) {
        if (auto scale = GetConstInt(operand); scale && *scale > 1)
          scales.insert(*scale);
      }
    };
    PrimExpr left_begin = analyzer->Simplify(left.location.offset);
    PrimExpr right_begin = analyzer->Simplify(other_offset);
    PostOrderVisit(left_begin, collect_scales);
    PostOrderVisit(right_begin, collect_scales);
    PrimExpr left_last = left_begin + left.bytes - IntImm::Int64(1);
    PrimExpr right_last = right_begin + other_bytes - IntImm::Int64(1);
    for (const PrimExpr &last : {left_last, right_last}) {
      auto bound = analyzer->const_int_bound(last);
      if (bound->min_value == sym::ConstIntBound::kNegInf ||
          bound->max_value == sym::ConstIntBound::kPosInf)
        return true;
    }
    WithGroup<sym::ConstraintContext> interval_bounds;
    for (int64_t scale : scales) {
      PrimExpr divisor = IntImm::Int64(scale);
      interval_bounds.Emplace(
          analyzer, analyzer->Simplify(FloorDiv(left_begin, divisor) <=
                                       FloorDiv(right_last, divisor)));
      interval_bounds.Emplace(
          analyzer, analyzer->Simplify(FloorDiv(right_begin, divisor) <=
                                       FloorDiv(left_last, divisor)));
    }
    return !analyzer->CanProve(same_thread, sym::ProofStrength::kSymbolicBound);
  }

  bool Conflict(const std::vector<PendingMemoryAccess> &left,
                const std::vector<MemoryAccess> &right,
                const SyncContext &context,
                const ForNode *carried_loop = nullptr) const {
    for (const PendingMemoryAccess &first : left)
      for (const MemoryAccess &second : right)
        if (Conflict(first.access, second, context, carried_loop,
                     first.warp_synchronized))
          return true;
    return false;
  }

  PrimExpr Expand(const PrimExpr &expression,
                  const SyncContext &context) const {
    Map<PrimVar, PrimExpr> values;
    for (const Constr &constraint : context.constraints.constrs_) {
      if (constraint.kind == Constr::kBindValue)
        values.Set(constraint.var, Substitute(constraint.value, values));
    }
    return Substitute(expression, values);
  }

  bool DependsOnThreads(const PrimExpr &expression,
                        const SyncContext &context) const {
    auto variables = tvm::UndefinedVars(expression);
    for (const Var &variable : variables)
      for (const ThreadAxis &thread : context.threads)
        if (variable.same_as(thread.variable))
          return true;
    return false;
  }

  void ThreadConditions(const PrimExpr &condition, const SyncContext &context,
                        std::vector<PrimExpr> *conditions) const {
    PrimExpr expanded = Expand(condition, context);
    if (IsUniform(expanded))
      return;
    if (!context.uniform_assumptions.constrs_.empty() &&
        SideEffect(expanded) == CallEffectKind::kPure) {
      sym::Analyzer analyzer;
      if (analyzer->z3_prover.IsEnabled())
        analyzer->z3_prover.SetTimeoutMs(100);
      auto assumptions = context.uniform_assumptions.Populate(analyzer);
      Map<PrimVar, PrimExpr> first_thread;
      for (const ThreadAxis &thread : context.threads) {
        PrimExpr zero = MakeConst(thread.variable.ty(), 0);
        analyzer->Bind(thread.variable, Range(zero, thread.extent));
        first_thread.Set(thread.variable, zero);
      }
      if (analyzer->CanProve(expanded == Substitute(expanded, first_thread),
                             sym::ProofStrength::kSymbolicBound))
        return;
    }
    if (const auto *conjunction = expanded.as<AndNode>()) {
      ThreadConditions(conjunction->a, context, conditions);
      ThreadConditions(conjunction->b, context, conditions);
      return;
    }
    TVM_FFI_CHECK(SideEffect(expanded) == CallEffectKind::kPure, ValueError)
        << "Shared synchronization cannot depend on mutable or effectful "
           "predicates: "
        << expanded;
    TVM_FFI_CHECK(DependsOnThreads(expanded, context), ValueError)
        << "Cannot infer a uniform synchronization cohort for " << expanded;
    conditions->push_back(expanded);
  }

  bool CoversThreads(const Stmt &statement, const SyncContext &context) const {
    if (const auto *sequence = statement.as<SeqStmtNode>())
      return !sequence->seq.empty() &&
             CoversThreads(sequence->seq.back(), context);
    const CallNode *call = EvaluatedCall(statement);
    if (call && call->op.same_as(sync_grid()))
      return true;
    if (!IsStorageSync(statement))
      return false;
    if (call->args.size() == 1)
      return true;
    if (call->args.size() != 3)
      return false;
    auto expected = GetConstInt(call->args[2]);
    if (!expected)
      return false;
    std::array<int, 3> extents{1, 1, 1};
    for (const ThreadAxis &thread : context.threads)
      extents[thread.dimension] = thread.extent;
    std::vector<PrimExpr> conditions;
    for (const PrimExpr &condition : context.conditions)
      ThreadConditions(condition, context, &conditions);
    int total = extents[0] * extents[1] * extents[2];
    int count = 0;
    sym::Analyzer analyzer;
    auto assumptions = context.uniform_assumptions.Populate(analyzer);
    for (int linear = 0; linear < total; ++linear) {
      std::array<int, 3> coordinate{linear % extents[0],
                                    (linear / extents[0]) % extents[1],
                                    linear / (extents[0] * extents[1])};
      Map<PrimVar, PrimExpr> values;
      for (const ThreadAxis &thread : context.threads)
        values.Set(thread.variable, MakeConst(thread.variable.ty(),
                                              coordinate[thread.dimension]));
      bool active = true;
      for (const PrimExpr &condition : conditions) {
        auto predicate =
            GetConstInt(analyzer->Simplify(Substitute(condition, values)));
        if (!predicate)
          return false;
        active &= *predicate != 0;
      }
      count += active;
    }
    return count == *expected;
  }

  Stmt MakeSync(const SyncContext &context, Location location,
                Optional<String> explicit_scope = std::nullopt,
                bool defer_partial = false) {
    TVM_FFI_CHECK(!context.conditional_exit, ValueError)
        << "Cannot insert a shared barrier after a potentially divergent early "
           "exit";
    std::array<int, 3> extents{1, 1, 1};
    for (const ThreadAxis &thread : context.threads)
      extents[thread.dimension] = thread.extent;
    int total = extents[0] * extents[1] * extents[2];
    TVM_FFI_CHECK(total > 0 && total <= 1024, ValueError)
        << "Thread synchronization needs a static block of at most 1024 "
           "threads";
    std::vector<PrimExpr> conditions;
    for (const PrimExpr &condition : context.conditions)
      ThreadConditions(condition, context, &conditions);
    sym::Analyzer analyzer;
    auto assumptions = context.uniform_assumptions.Populate(analyzer);
    std::vector<bool> participants(total);
    std::array<int, 3> lower{extents[0], extents[1], extents[2]};
    std::array<int, 3> upper{0, 0, 0};
    std::vector<Map<PrimVar, PrimExpr>> substitutions;
    int count = 0;
    for (int linear = 0; linear < total; ++linear) {
      std::array<int, 3> coordinate{linear % extents[0],
                                    (linear / extents[0]) % extents[1],
                                    linear / (extents[0] * extents[1])};
      Map<PrimVar, PrimExpr> values;
      for (const ThreadAxis &thread : context.threads)
        values.Set(thread.variable, MakeConst(thread.variable.ty(),
                                              coordinate[thread.dimension]));
      bool active = true;
      for (const PrimExpr &condition : conditions) {
        PrimExpr predicate = analyzer->Simplify(Substitute(condition, values));
        auto constant = GetConstInt(predicate);
        if (!constant &&
            analyzer->CanProve(predicate, sym::ProofStrength::kSymbolicBound))
          constant = 1;
        if (!constant &&
            analyzer->CanProve(!predicate, sym::ProofStrength::kSymbolicBound))
          constant = 0;
        if (!constant && defer_partial && context.deferred_visibility) {
          *context.deferred_visibility = true;
          return SeqStmt({}, location);
        }
        TVM_FFI_CHECK(constant.has_value(), ValueError)
            << "Partial thread synchronization requires a statically known "
               "participant set";
        active &= *constant != 0;
      }
      participants[linear] = active;
      if (!active)
        continue;
      ++count;
      substitutions.push_back(values);
      for (int dimension = 0; dimension < 3; ++dimension) {
        lower[dimension] = std::min(lower[dimension], coordinate[dimension]);
        upper[dimension] =
            std::max(upper[dimension], coordinate[dimension] + 1);
      }
    }
    for (const PrimExpr &header : context.loop_headers) {
      PrimExpr expanded = Expand(header, context);
      if (IsUniform(expanded))
        continue;
      TVM_FFI_CHECK(SideEffect(expanded) == CallEffectKind::kPure, ValueError)
          << "Shared synchronization requires uniform, side-effect-free loop "
             "bounds";
      if (!substitutions.empty()) {
        PrimExpr first = Substitute(expanded, substitutions.front());
        for (const auto &values : substitutions)
          TVM_FFI_CHECK(
              analyzer->CanProveEqual(first, Substitute(expanded, values)),
              ValueError)
              << "Shared synchronization is unsafe in a thread-divergent loop";
      }
    }
    Array<Expr> arguments{
        StringImm(explicit_scope.value_or(sync_scope_.value_or("shared")))};
    if (explicit_scope && *explicit_scope == "warp") {
      TVM_FFI_CHECK(cuda_features_, ValueError)
          << "Warp synchronization requires CUDA";
      std::vector<uint32_t> masks((total + 31) / 32);
      for (int linear = 0; linear < total; ++linear)
        if (participants[linear])
          masks[linear / 32] |= uint32_t{1} << (linear % 32);
      PrimExpr mask = MakeConst(PrimType::UInt(32), masks.back());
      PrimExpr linear = LinearThreadIndex(context);
      for (int warp = static_cast<int>(masks.size()) - 2; warp >= 0; --warp)
        mask = Select(linear < IntImm::Int64((warp + 1) * 32),
                      MakeConst(PrimType::UInt(32), masks[warp]), mask);
      arguments.push_back(analyzer->Simplify(mask));
      return Evaluate(Call(std::nullopt, gpu_storage_sync_op(), arguments,
                           std::nullopt, {}, location),
                      location);
    }
    if (count != 0 && count != total) {
      TVM_FFI_CHECK(cuda_features_, ValueError)
          << "Ascend collective synchronization requires all SIMT participants";
      int rectangle =
          (upper[0] - lower[0]) * (upper[1] - lower[1]) * (upper[2] - lower[2]);
      bool whole_warps = true;
      for (int warp = 0; warp < total; warp += 32) {
        int active = 0;
        for (int lane = warp; lane < std::min(warp + 32, total); ++lane)
          active += participants[lane];
        whole_warps &= active == 0 || active == 32;
      }
      if (defer_partial && context.deferred_visibility &&
          (rectangle != count || count % 32 != 0 || !whole_warps)) {
        *context.deferred_visibility = true;
        return SeqStmt({}, location);
      }
      TVM_FFI_CHECK(rectangle == count && count % 32 == 0, ValueError)
          << "Partial shared barriers require rectangular, warp-complete "
             "participants";
      for (int warp = 0; warp < total; warp += 32) {
        int active = 0;
        for (int lane = warp; lane < std::min(warp + 32, total); ++lane)
          active += participants[lane];
        TVM_FFI_CHECK(active == 0 || active == 32, ValueError)
            << "Partial shared barriers cannot split a hardware warp";
      }
      ThreadBoundKey key{lower[0], upper[0], lower[1],
                         upper[1], lower[2], upper[2]};
      auto found = named_barriers_.find(key);
      if (found == named_barriers_.end()) {
        int identifier =
            static_cast<int>(ReservedNamedBarriers::kFirstUsedBarrier);
        while (used_barriers_.count(identifier))
          ++identifier;
        TVM_FFI_CHECK(identifier < 16, ValueError)
            << "Exhausted hardware named barrier IDs";
        used_barriers_.insert(identifier);
        found = named_barriers_.emplace(key, identifier).first;
      }
      arguments.push_back(PrimExpr(found->second));
      arguments.push_back(PrimExpr(count));
    }
    return Evaluate(Call(std::nullopt, gpu_storage_sync_op(), arguments,
                         std::nullopt, {}, location),
                    location);
  }

  std::pair<SeqStmt, std::vector<PendingMemoryAccess>>
  RewriteSequence(const SeqStmt &sequence, SyncContext context) {
    Array<Stmt> result;
    std::vector<PendingMemoryAccess> pending;
    bool *parent_visibility = context.deferred_visibility;
    bool deferred_visibility = false;
    context.deferred_visibility = &deferred_visibility;
    for (size_t position = 0; position < sequence->seq.size(); ++position) {
      const Stmt &statement = sequence->seq[position];
      bool exits = false;
      auto accesses = Summarize(statement, context, &exits);
      if ((deferred_visibility && !accesses.empty()) ||
          Conflict(pending, accesses, context)) {
        result.push_back(MakeSync(context, statement->loc));
        pending.clear();
        deferred_visibility = false;
      }
      Stmt rewritten = Rewrite(statement, context);
      result.push_back(rewritten);
      if (CoversThreads(rewritten, context)) {
        pending.clear();
        deferred_visibility = false;
      } else {
        if (cuda_features_ && IsWarpSync(rewritten)) {
          Stmt expected = MakeSync(context, statement->loc, String("warp"));
          const CallNode *actual_call = EvaluatedCall(rewritten);
          if (actual_call->args.size() == 2 &&
              sym::Analyzer()->CanProveEqual(
                  actual_call->args[1].as_or_throw<PrimExpr>(),
                  EvaluatedCall(expected)->args[1].as_or_throw<PrimExpr>()))
            for (PendingMemoryAccess &access : pending)
              access.warp_synchronized = true;
        }
        for (const MemoryAccess &access : accesses)
          pending.push_back({access});
      }
      if (cuda_features_ && IsAsyncWait(statement) &&
          !context.threads.empty()) {
        if (position + 1 == sequence->seq.size() ||
            !CoversThreads(sequence->seq[position + 1], context))
          result.push_back(
              MakeSync(context, statement->loc, std::nullopt, true));
        if (!deferred_visibility)
          pending.clear();
      }
      if (const auto *bind = statement.as<BindNode>()) {
        auto variable = bind->var.as<PrimVar>();
        auto value = bind->value.as<PrimExpr>();
        if (variable && value)
          context.constraints.AddConstr(*variable, *value);
      }
      Optional<PrimExpr> premise;
      if (const auto *assertion = statement.as<AssertStmtNode>())
        premise = assertion->condition;
      if (const auto *call = EvaluatedCall(statement);
          call && call->op.same_as(prim::assume_op()))
        premise = call->args[0].as_or_throw<PrimExpr>();
      if (premise) {
        PrimExpr expanded = Expand(*premise, context);
        if (SideEffect(expanded) == CallEffectKind::kPure) {
          context.constraints.AddConstr(expanded);
          if (DependsOnThreads(expanded, context))
            context.conditions.push_back(expanded);
          else
            context.uniform_assumptions.AddConstr(expanded);
        }
      }
      context.conditional_exit |= exits;
    }
    if (deferred_visibility) {
      TVM_FFI_CHECK(parent_visibility, ValueError)
          << "Async-copy visibility requires a convergent synchronization "
             "point";
      *parent_visibility = true;
    }
    return {SeqStmt(result, sequence->loc), std::move(pending)};
  }

  Stmt Rewrite(const Stmt &statement, SyncContext context) {
    if (cuda_features_ && (IsStorageSync(statement) || IsWarpSync(statement)) &&
        EvaluatedCall(statement)->args.size() == 1)
      return MakeSync(
          context, statement->loc,
          EvaluatedCall(statement)->args[0].as_or_throw<StringImm>()->value);
    if (const auto *call = EvaluatedCall(statement);
        call && call->op.same_as(masked_store_op())) {
      auto accesses = Summarize(statement, context);
      for (const MemoryAccess &read : accesses)
        for (const MemoryAccess &write : accesses)
          if ((read.mask & kAccessRead) && (write.mask & kAccessWrite))
            TVM_FFI_CHECK(!Conflict(read, write, context), ValueError)
                << "Cross-thread in-place masked stores require a separate, "
                   "mask-preserving snapshot before synchronization";
    }
    if (const auto *store = statement.as<TensorStoreNode>()) {
      auto accesses = Summarize(statement, context);
      bool read_write_conflict = false;
      for (const MemoryAccess &read : accesses)
        for (const MemoryAccess &write : accesses)
          if ((read.mask & kAccessRead) && (write.mask & kAccessWrite))
            read_write_conflict |= Conflict(read, write, context);
      if (read_write_conflict) {
        PrimVar value("shared_store_value",
                      store->value.ty().as_or_throw<PrimType>());
        Array<Stmt> prefix{Bind(value, store->value)};
        Array<PrimExpr> indices;
        for (const PrimExpr &index : store->indices) {
          PrimVar coordinate("shared_store_index", index.ty());
          prefix.push_back(Bind(coordinate, index));
          indices.push_back(coordinate);
        }
        prefix.push_back(MakeSync(context, store->loc));
        prefix.push_back(TensorStore(store->dest, indices, value, store->loc));
        return SeqStmt(prefix, store->loc);
      }
    }
    if (auto sequence = statement.as<SeqStmt>())
      return RewriteSequence(*sequence, context).first;
    if (auto region = statement.as<RegionStmt>()) {
      const RegionStmtNode *node = region.value().get();
      if (node->op.same_as(launch_thread_op())) {
        PrimVar variable = node->body_params[0].as_or_throw<PrimVar>();
        PrimExpr extent = node->args[1].as_or_throw<PrimExpr>();
        context.constraints.AddConstr(variable, Range(0, extent));
        auto tag = node->args[0].as_or_throw<StringImm>()->value;
        if (tag.find("threadIdx.") == 0) {
          auto size = GetConstInt(extent);
          TVM_FFI_CHECK(size && *size > 0 && *size <= 1024, ValueError)
              << "ThreadSync requires static thread extents";
          int dimension = tag.data()[tag.size() - 1] - 'x';
          TVM_FFI_CHECK(dimension >= 0 && dimension < 3, ValueError)
              << "Invalid thread dimension";
          context.threads.push_back(
              {variable, dimension, static_cast<int>(*size)});
        }
      }
      RegionStmt rewritten = *region;
      if (node->attrs.GetAttr<bool>("tl.ws_concurrent").value_or(false)) {
        Array<Stmt> roles;
        for (const Stmt &role : node->body->seq) {
          roles.push_back(Rewrite(role, context));
        }
        rewritten.CopyOnWrite()->body = SeqStmt(roles, node->body->loc);
        return rewritten;
      }
      rewritten.CopyOnWrite()->body =
          RewriteSequence(node->body, context).first;
      return rewritten;
    }
    if (auto conditional = statement.as<If>()) {
      const IfNode *node = conditional.value().get();
      SyncContext joined = context;
      bool deferred_visibility = false;
      context.deferred_visibility = &deferred_visibility;
      SyncContext otherwise = context;
      context.constraints.AddConstr(node->condition);
      context.conditions.push_back(node->condition);
      otherwise.constraints.AddConstr(!node->condition);
      otherwise.conditions.push_back(!node->condition);
      If rewritten = *conditional;
      rewritten.CopyOnWrite()->then_case =
          RewriteSequence(node->then_case, context).first;
      if (node->else_case)
        rewritten.CopyOnWrite()->else_case =
            RewriteSequence(*node->else_case, otherwise).first;
      if (deferred_visibility) {
        bool exits = false;
        Summarize(statement, joined, &exits);
        joined.conditional_exit |= exits;
        return SeqStmt(
            {rewritten, MakeSync(joined, node->loc, std::nullopt, true)},
            node->loc);
      }
      return rewritten;
    }
    if (auto loop = statement.as<For>()) {
      const ForNode *node = loop.value().get();
      context.constraints.AddConstr(
          node->var,
          Range(Min(node->begin, node->end), Max(node->begin, node->end) + 1));
      context.constraints.AddConstr(
          (node->step > 0 && node->var >= node->begin &&
           node->var < node->end) ||
          (node->step < 0 && node->var <= node->begin &&
           node->var > node->end));
      context.constraints.AddConstr(
          FloorMod(node->var - node->begin, Max(node->step, -node->step)) == 0);
      context.loops.push_back(node);
      context.loop_headers.insert(context.loop_headers.end(),
                                  {node->begin, node->end, node->step});
      bool exits = false;
      auto accesses = Summarize(node->body, context, &exits);
      auto [body, pending] = RewriteSequence(node->body, context);
      if (Conflict(pending, accesses, context, node)) {
        context.conditional_exit |= exits;
        Array<Stmt> statements = body->seq;
        statements.push_back(MakeSync(context, node->loc));
        body = SeqStmt(statements, body->loc);
      }
      For rewritten = *loop;
      rewritten.CopyOnWrite()->body = body;
      return rewritten;
    }
    if (auto loop = statement.as<While>()) {
      const WhileNode *node = loop.value().get();
      context.loop_headers.push_back(node->condition);
      bool exits = false;
      auto accesses = Summarize(node->body, context, &exits);
      auto [body, pending] = RewriteSequence(node->body, context);
      if (Conflict(pending, accesses, context)) {
        context.conditional_exit |= exits;
        Array<Stmt> statements = body->seq;
        statements.push_back(MakeSync(context, node->loc));
        body = SeqStmt(statements, body->loc);
      }
      While rewritten = *loop;
      rewritten.CopyOnWrite()->body = body;
      return rewritten;
    }
    return statement;
  }

  ffi::Optional<ffi::String> sync_scope_;
  bool cuda_features_;
  MemoryAliases aliases_;
  Map<PrimVar, PrimExpr> definitions_;
  std::unordered_set<const VarNode *> varying_threads_;
  std::unordered_set<const VarNode *> uniform_global_;
  std::unordered_set<const VarNode *> uniform_local_;
  std::set<int> used_barriers_{0, 1, 2};
  std::unordered_map<ThreadBoundKey, int> named_barriers_;
};
} // namespace

SeqStmt RewriteThreadSync(const tirx::Function &function, const SeqStmt &body,
                          ffi::Optional<ffi::String> storage_scope,
                          bool cuda_features) {
  return ThreadSyncPlanner(function, std::move(storage_scope), cuda_features)
      .Run(body);
}

} // namespace tvm::tl

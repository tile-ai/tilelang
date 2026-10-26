#include "common/memory_access.h"
#include "common/storage_size.h"
#include "compat/analyzer.h"
#include "compat/metadata.h"

#include <tvm/tirx/op/region.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

struct SharedPosition {
  const SeqStmtNode *sequence;
  size_t index;
};

struct SharedAllocation {
  Bind binding;
  TensorVar tensor;
  std::vector<SharedPosition> path;
  std::vector<std::vector<SharedPosition>> uses;
  bool asynchronous = false;
};

struct GuardedCall {
  const CallNode *call = nullptr;
  Array<PrimExpr> guards;
};

GuardedCall GetGuardedCall(Stmt statement) {
  GuardedCall result;
  while (true) {
    if (const auto *sequence = statement.as<SeqStmtNode>();
        sequence && sequence->seq.size() == 1) {
      statement = sequence->seq[0];
    } else if (const auto *branch = statement.as<IfNode>();
               branch && !branch->else_case) {
      result.guards.push_back(branch->condition);
      statement = branch->then_case;
    } else {
      if (const auto *evaluate = statement.as<EvaluateNode>())
        result.call = evaluate->value.as<CallNode>();
      return result;
    }
  }
}

const CallNode *CompletedTmaStore(const SeqStmtNode *sequence, size_t index) {
  if (index + 2 >= sequence->seq.size())
    return nullptr;
  auto store = GetGuardedCall(sequence->seq[index]);
  auto arrive = GetGuardedCall(sequence->seq[index + 1]);
  auto wait = GetGuardedCall(sequence->seq[index + 2]);
  auto named = [](const CallNode *call, const char *name) {
    auto operation = call ? call->op.as<Op>() : std::nullopt;
    return operation && operation.value()->name == name;
  };
  if (!named(store.call, "tl.tma_store") ||
      !named(arrive.call, "tl.tma_store_arrive") ||
      !named(wait.call, "tl.tma_store_wait") || wait.call->args.empty() ||
      !StructuralEqual()(store.guards, arrive.guards) ||
      !StructuralEqual()(store.guards, wait.guards))
    return nullptr;
  for (const PrimExpr &guard : store.guards)
    if (SideEffect(guard) != CallEffectKind::kPure)
      return nullptr;
  const auto *remaining = wait.call->args[0].as<IntImmNode>();
  return remaining && remaining->value == 0 ? store.call : nullptr;
}

class SharedAllocationCollector : public tvm::tl::StmtExprVisitor {
public:
  explicit SharedAllocationCollector(MemoryAliases aliases)
      : aliases_(std::move(aliases)) {}
  std::vector<SharedAllocation> allocations;
  std::unordered_map<const SeqStmtNode *, Array<Var>> parameters;

private:
  Optional<VisitInterrupt> Visit_(const SeqStmtNode *node) final {
    path_.push_back({node, 0});
    for (size_t index = 0; index < node->seq.size(); ++index) {
      path_.back().index = index;
      const CallNode *previous_store = completed_store_;
      auto previous_completion = completion_path_;
      if (const CallNode *store =
              concurrent_ ? nullptr : CompletedTmaStore(node, index)) {
        completed_store_ = store;
        completion_path_ = path_;
        completion_path_.back().index = index + 2;
      }
      Visit(node->seq[index]);
      completed_store_ = previous_store;
      completion_path_ = std::move(previous_completion);
    }
    path_.pop_back();
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const RegionStmtNode *node) final {
    parameters.emplace(node->body.get(), node->body_params);
    bool previous = concurrent_;
    if (auto concurrent = node->attrs->dict.Get("tl.ws_concurrent"))
      concurrent_ |= GetIntAnnotation(*concurrent).value_or(0) != 0;
    auto result = tvm::tl::StmtExprVisitor::Visit_(node);
    concurrent_ = previous;
    return result;
  }

  Optional<VisitInterrupt> Visit_(const ForNode *node) final {
    parameters.emplace(node->body.get(), Array<Var>{node->var});
    return tvm::tl::StmtExprVisitor::Visit_(node);
  }

  Optional<VisitInterrupt> Visit_(const BindNode *node) final {
    auto tensor = node->var.as<TensorVar>();
    const auto *call = node->value.as<CallNode>();
    if (tensor && call && call->op.same_as(alloc_tensor_op()) &&
        (tensor.value().scope().name() == "shared" ||
         tensor.value().scope().name() == "shared.dyn")) {
      indices_.emplace(node->var.get(), allocations.size());
      allocations.push_back({GetRef<Bind>(node), *tensor, path_, {}, false});
      return Visit(node->value);
    }
    return Visit(node->value);
  }

  Optional<VisitInterrupt> Visit_(const VarNode *node) final {
    MemoryLocation location = aliases_.Resolve(GetRef<Var>(node));
    if (location.allocation) {
      auto found = indices_.find(location.allocation.value().get());
      if (found != indices_.end()) {
        allocations[found->second].uses.push_back(path_);
        if (inside_completed_store_)
          allocations[found->second].uses.push_back(completion_path_);
        allocations[found->second].asynchronous |= asynchronous_;
      }
    }
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const CallNode *node) final {
    bool saved = asynchronous_;
    bool previous_completed = inside_completed_store_;
    inside_completed_store_ |= node == completed_store_;
    asynchronous_ |= node != completed_store_ &&
                     SideEffect(GetRef<Call>(node)) == CallEffectKind::kOpaque;
    if (auto operation = node->op.as<Op>()) {
      std::string name = operation.value()->name;
      asynchronous_ |= node != completed_store_ &&
                       (name.find("tma") != std::string::npos ||
                        name.find("cp_async") != std::string::npos ||
                        name.find("wgmma") != std::string::npos ||
                        name.find("tcgen05") != std::string::npos);
    }
    auto result = tvm::tl::StmtExprVisitor::Visit_(node);
    asynchronous_ = saved;
    inside_completed_store_ = previous_completed;
    return result;
  }

  MemoryAliases aliases_;
  std::vector<SharedPosition> path_;
  std::unordered_map<const VarNode *, size_t> indices_;
  bool asynchronous_ = false;
  bool concurrent_ = false;
  bool inside_completed_store_ = false;
  const CallNode *completed_store_ = nullptr;
  std::vector<SharedPosition> completion_path_;
};

struct SharedSlot {
  PrimExpr offset;
  PrimExpr bytes;
  PrimType type;
  int alignment;
  size_t last;
  bool reusable;
};

class SharedArenaRewriter : public tvm::tl::StmtExprMutator {
public:
  std::unordered_map<const BindNode *, Stmt> replacements;
  std::unordered_map<const SeqStmtNode *, std::map<size_t, Array<Stmt>>>
      insertions;

private:
  UnchangedOr<Stmt> Mutate_(const BindNode *node, InplaceMode mode) final {
    if (auto found = replacements.find(node); found != replacements.end())
      return found->second;
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }

  UnchangedOr<Stmt> Mutate_(const SeqStmtNode *node, InplaceMode mode) final {
    Array<Stmt> result;
    auto found = insertions.find(node);
    for (size_t index = 0; index < node->seq.size(); ++index) {
      if (found != insertions.end()) {
        if (auto entries = found->second.find(index);
            entries != found->second.end())
          result.insert(result.end(), entries->second.begin(),
                        entries->second.end());
      }
      result.push_back(
          Mutate(node->seq[index], mode).ValueOrUnchanged(node->seq[index]));
    }
    return SeqStmt(result, node->loc);
  }
};

class SharedArenaPlanner {
public:
  SharedArenaPlanner(tirx::Function function, bool aggressive, int alignment,
                     bool disable_reuse)
      : function_(function), aggressive_(aggressive), alignment_(alignment),
        disable_reuse_(disable_reuse) {
    TVM_FFI_CHECK(alignment > 0 && alignment <= (1 << 27) &&
                      (alignment & (alignment - 1)) == 0,
                  ValueError)
        << "Shared arena alignment must be a positive power of two";
  }

  SeqStmt Run(bool merge_static) {
    auto accesses = make_object<MemoryAccessCollector>();
    accesses->Visit(*function_->body);
    auto collector = make_object<SharedAllocationCollector>(accesses->aliases);
    collector->Visit(*function_->body);
    auto rewriter = make_object<SharedArenaRewriter>();
    for (const std::string &scope : {"shared.dyn", "shared"}) {
      if (scope == "shared" && !merge_static)
        continue;
      std::vector<const SharedAllocation *> allocations;
      for (const SharedAllocation &allocation : collector->allocations)
        if (allocation.tensor.scope().name() == scope)
          allocations.push_back(&allocation);
      if (allocations.size() > 1)
        Plan(scope, allocations, *collector, rewriter.get());
    }
    SeqStmt body = *function_->body;
    return rewriter->Mutate(body).ValueOrUnchanged(body);
  }

private:
  void Plan(const std::string &scope,
            std::vector<const SharedAllocation *> allocations,
            const SharedAllocationCollector &collector,
            SharedArenaRewriter *rewriter) {
    std::vector<SharedPosition> common = allocations.front()->path;
    for (const SharedAllocation *allocation : allocations) {
      size_t shared = 0;
      while (shared < common.size() && shared < allocation->path.size() &&
             common[shared].sequence == allocation->path[shared].sequence) {
        common[shared].index =
            std::min(common[shared].index, allocation->path[shared].index);
        ++shared;
      }
      common.resize(shared);
    }
    TVM_FFI_CHECK(!common.empty(), ValueError)
        << "Shared allocations need a common lexical scope";
    const SeqStmtNode *anchor = common.back().sequence;
    size_t insertion = common.back().index;
    auto use_interval = [anchor](const SharedAllocation *allocation) {
      std::pair<size_t, size_t> interval{anchor->seq.size(), 0};
      for (const auto &path : allocation->uses) {
        for (const SharedPosition &position : path) {
          if (position.sequence == anchor) {
            interval.first = std::min(interval.first, position.index);
            interval.second = std::max(interval.second, position.index);
            break;
          }
        }
      }
      return interval;
    };
    std::stable_sort(
        allocations.begin(), allocations.end(),
        [&](const SharedAllocation *left, const SharedAllocation *right) {
          return use_interval(left).first < use_interval(right).first;
        });
    std::unordered_set<const VarNode *> available;
    for (const Var &parameter : function_->params)
      available.insert(parameter.get());
    for (const SharedPosition &position : common) {
      if (auto parameters = collector.parameters.find(position.sequence);
          parameters != collector.parameters.end())
        for (const Var &parameter : parameters->second)
          available.insert(parameter.get());
      for (size_t index = 0; index < position.index; ++index)
        if (const auto *binding = position.sequence->seq[index].as<BindNode>())
          available.insert(binding->var.get());
    }
    std::unordered_map<std::string, int> explicit_alignments;
    if (auto annotations =
            function_->GetAttr<Map<String, Any>>(kSmemAlignmentMap)) {
      for (const auto &[name, value] : *annotations) {
        auto bytes = GetIntAnnotation(value);
        TVM_FFI_CHECK(bytes && *bytes > 0 && *bytes <= (1 << 27), ValueError)
            << "Invalid shared alignment";
        explicit_alignments.emplace(std::string(name),
                                    static_cast<int>(*bytes));
      }
    }
    sym::Analyzer analyzer;
    PrimExpr total = IntImm::Int64(0);
    int arena_alignment = alignment_;
    std::vector<SharedSlot> slots;
    std::vector<PrimExpr> offsets;
    for (const SharedAllocation *allocation : allocations) {
      const auto *call = allocation->binding->value.as<CallNode>();
      TVM_FFI_CHECK(call->args.empty(), ValueError)
          << "Placed shared allocations must be lowered before arena merging";
      TVM_FFI_CHECK(
          !call->attrs ||
              call->attrs.value().as_or_throw<DictAttrs>()->dict.empty(),
          ValueError)
          << "Shared allocation annotations must be lowered before arena "
             "merging";
      PrimExpr bytes = tl::arith::Simplify(
          analyzer, GetBufferStorageSizeBytes(allocation->tensor));
      TVM_FFI_CHECK(SideEffect(bytes) == CallEffectKind::kPure, ValueError)
          << "Shared allocation sizes must be side-effect-free";
      for (const Var &dependency : tvm::UndefinedVars(bytes))
        TVM_FFI_CHECK(available.count(dependency.get()), ValueError)
            << "Cannot hoist shared arena size across the definition of "
            << dependency->name;
      int alignment =
          std::max(alignment_, allocation->tensor.ty()->data_alignment);
      if (auto explicit_alignment =
              explicit_alignments.find(std::string(allocation->tensor->name));
          explicit_alignment != explicit_alignments.end())
        alignment = std::max(alignment, explicit_alignment->second);
      TVM_FFI_CHECK(alignment > 0 && alignment <= (1 << 27) &&
                        (alignment & (alignment - 1)) == 0,
                    ValueError)
          << "Shared allocation alignment must be a power of two";
      arena_alignment = std::max(arena_alignment, alignment);
      auto [first, last] = use_interval(allocation);
      bool reusable = !disable_reuse_ && !allocation->asynchronous;
      auto slot = slots.end();
      if (!disable_reuse_) {
        slot = std::find_if(
            slots.begin(), slots.end(), [&](const SharedSlot &candidate) {
              return candidate.reusable && candidate.last < first &&
                     candidate.alignment >= alignment &&
                     (aggressive_ ||
                      candidate.type.WithLanes(1) ==
                          allocation->tensor.dtype().WithLanes(1)) &&
                     tl::arith::CanProve(analyzer, bytes <= candidate.bytes);
            });
      }
      if (slot != slots.end()) {
        offsets.push_back(slot->offset);
        slot->last = last;
        slot->reusable = reusable;
      } else {
        PrimExpr offset = tl::arith::Simplify(
            analyzer, FloorDiv(total + IntImm::Int64(alignment - 1),
                               IntImm::Int64(alignment)) *
                          IntImm::Int64(alignment));
        slots.push_back({offset, bytes, allocation->tensor.dtype(), alignment,
                         last, reusable});
        offsets.push_back(offset);
        total = tl::arith::Simplify(analyzer, offset + bytes);
      }
    }
    TensorType arena_type({total}, PrimType::UInt(8),
                          runtime::MemoryScope::Get(scope), arena_alignment);
    TensorVar arena("merged_" + scope, arena_type);
    rewriter->insertions[anchor][insertion].push_back(
        Bind(arena, Call(arena_type, alloc_tensor_op(), {}, DictAttrs(),
                         {arena_type})));
    for (size_t index = 0; index < allocations.size(); ++index) {
      const SharedAllocation &allocation = *allocations[index];
      Expr pointer = Call(allocation.tensor.ty()->DataPtrType(),
                          ptr_byte_offset_op(), {arena.data(), offsets[index]});
      rewriter->replacements.emplace(
          allocation.binding.get(),
          Bind(allocation.tensor,
               Call(allocation.tensor.ty(), decl_tensor_op(), {pointer},
                    std::nullopt, {allocation.tensor.ty()}),
               allocation.binding->loc));
    }
  }

  tirx::Function function_;
  bool aggressive_;
  int alignment_;
  bool disable_reuse_;
};
} // namespace

namespace transform {
tvm::transform::Pass
MergeSharedMemoryAllocations(bool enable_aggressive_merge = false,
                             int align_bytes = 16, bool disable_reuse = false) {
  auto pass =
      [=](tirx::Function function, IRModule,
          tvm::transform::PassContext context) -> Optional<tirx::Function> {
    if (function->body) {
      SeqStmt body = SharedArenaPlanner(function, enable_aggressive_merge,
                                        align_bytes, disable_reuse)
                         .Run(context->GetConfig<bool>("tirx.merge_static_smem")
                                  .value_or(false));
      function.CopyOnWrite()->body = body;
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.MergeSharedMemoryAllocations");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.MergeSharedMemoryAllocations",
                                   MergeSharedMemoryAllocations);
}
} // namespace transform
} // namespace tvm::tl

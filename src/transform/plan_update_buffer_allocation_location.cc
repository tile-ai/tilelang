#include "compat/metadata.h"
#include "compat/visit.h"
#include "op/utils.h"
#include "transform/common/iteration_private.h"

#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

struct Position {
  const SeqStmtNode *sequence;
  size_t index;
  bool operator==(const Position &) const = default;
};

using Path = std::vector<Position>;

struct Allocation {
  Bind binding;
  TensorVar tensor;
  Path origin;
  std::vector<Path> uses;
  bool pinned = false;
  std::vector<Bind> aliases;
  std::unordered_set<const VarNode *> available;
};

class AllocationUses : public tvm::tl::StmtExprVisitor {
public:
  AllocationUses(bool follow_aliases, const Array<Var> &parameters)
      : follow_aliases_(follow_aliases) {
    for (const Var &parameter : parameters)
      defined_.insert(parameter.get());
  }
  std::vector<Allocation> allocations;
  std::unordered_map<const SeqStmtNode *, const StmtNode *> owners;
  MemoryAliases aliases;

private:
  Optional<VisitInterrupt> Visit_(const SeqStmtNode *node) final {
    path_.push_back({node, 0});
    for (size_t index = 0; index < node->seq.size(); ++index) {
      path_.back().index = index;
      Visit(node->seq[index]);
    }
    path_.pop_back();
    return std::nullopt;
  }

  Optional<VisitInterrupt> Visit_(const BindNode *node) final {
    aliases.Bind(node);
    auto tensor = node->var.as<TensorVar>();
    const auto *call = node->value.as<CallNode>();
    if (tensor && call && call->op.same_as(alloc_tensor_op())) {
      bool pinned = IsGlobalBuffer(*tensor) ||
                    tensor->scope().name() == "shared.barrier" ||
                    tensor->scope().name() == "shared.cluster_barrier" ||
                    tensor->scope().name() == "shared.tmem" ||
                    tensor->scope().name() == "metal.cooperative_tensor";
      if (call->attrs) {
        auto attributes = call->attrs.value().as<DictAttrs>();
        pinned |= !attributes || !attributes.value()->dict.empty();
      }
      for (const PrimExpr &extent : tensor->shape()) {
        pinned |= SideEffect(extent) != CallEffectKind::kPure;
      }
      indices_.emplace(tensor->get(), allocations.size());
      allocations.push_back(
          {GetRef<Bind>(node), *tensor, path_, {}, pinned, {}, defined_});
      defined_.insert(node->var.get());
      return Visit(node->value);
    }
    if (follow_aliases_ && (tensor || node->var->ty.as<PtrTypeNode>())) {
      MemoryLocation location = aliases.Resolve(node->var);
      auto found = location.allocation
                       ? indices_.find(location.allocation.value().get())
                       : indices_.end();
      if (found != indices_.end() && location.precise &&
          SideEffect(node->value) == CallEffectKind::kPure) {
        size_t position = found->second;
        Allocation &allocation = allocations[position];
        bool movable = true;
        for (const Var &variable : UndefinedVars(node->value)) {
          auto origin = indices_.find(variable.get());
          movable &= allocation.available.count(variable.get()) ||
                     (origin != indices_.end() && origin->second == position);
        }
        if (movable) {
          allocation.aliases.push_back(GetRef<Bind>(node));
          indices_.emplace(node->var.get(), position);
          defined_.insert(node->var.get());
          return std::nullopt;
        }
      }
    }
    defined_.insert(node->var.get());
    bool previous = address_use_;
    address_use_ |=
        node->var->ty.as<PtrTypeNode>() || node->var->ty.as<TensorTypeNode>();
    auto result = Visit(node->value);
    address_use_ = previous;
    return result;
  }

  Optional<VisitInterrupt> Visit_(const VarNode *node) final {
    if (auto found = indices_.find(node); found != indices_.end()) {
      Allocation &allocation = allocations[found->second];
      allocation.uses.push_back(path_);
      allocation.pinned |= address_use_;
    }
    return tvm::tl::StmtExprVisitor::Visit_(node);
  }

  Optional<VisitInterrupt> Visit_(const CallNode *node) final {
    bool previous = address_use_;
    auto operation = node->op.as<Op>();
    address_use_ |=
        node->op.same_as(address_of_op()) ||
        node->op.same_as(tensor_data_ptr_op()) ||
        (operation && operation.value()->name.find("tl.tileop.") == 0);
    auto result = tvm::tl::StmtExprVisitor::Visit_(node);
    address_use_ = previous;
    return result;
  }

  Optional<VisitInterrupt> Visit_(const ForNode *node) final {
    owners.emplace(node->body.get(), node);
    defined_.insert(node->var.get());
    return tvm::tl::StmtExprVisitor::Visit_(node);
  }

  Optional<VisitInterrupt> Visit_(const WhileNode *node) final {
    owners.emplace(node->body.get(), node);
    return tvm::tl::StmtExprVisitor::Visit_(node);
  }

  Optional<VisitInterrupt> Visit_(const IfNode *node) final {
    owners.emplace(node->then_case.get(), node);
    if (node->else_case)
      owners.emplace(node->else_case.value().get(), node);
    return tvm::tl::StmtExprVisitor::Visit_(node);
  }

  Optional<VisitInterrupt> Visit_(const RegionStmtNode *node) final {
    owners.emplace(node->body.get(), node);
    return tvm::tl::StmtExprVisitor::Visit_(node);
  }

  Path path_;
  bool address_use_ = false;
  bool follow_aliases_;
  std::unordered_set<const VarNode *> defined_;
  std::unordered_map<const VarNode *, size_t> indices_;
};

Path CommonSite(const std::vector<Path> &uses) {
  if (uses.empty())
    return {};
  Path result = uses.front();
  for (const Path &use : uses) {
    size_t shared = 0;
    while (shared < result.size() && shared < use.size() &&
           result[shared].sequence == use[shared].sequence) {
      ++shared;
      if (result[shared - 1].index != use[shared - 1].index)
        break;
    }
    result.resize(shared);
    if (result.empty())
      return {};
    result.back().index = std::min(result.back().index, use[shared - 1].index);
  }
  return result;
}

bool CrossesExecutionBoundary(const StmtNode *owner, const TensorVar &tensor,
                              bool cpu, const MemoryAliases &aliases) {
  Stmt statement = GetRef<Stmt>(owner);
  if (const auto *region = statement.as<RegionStmtNode>()) {
    return region->op->name == "tirx.device_entry" ||
           region->op->name == "tirx.device_scope";
  }
  if (const auto *loop = statement.as<ForNode>()) {
    if (loop->kind == ForKind::kVectorized)
      return true;
    if (!IsIterationPrivateStorage(tensor, loop->body, aliases))
      return true;
  }
  if (const auto *loop = statement.as<WhileNode>()) {
    if (!IsIterationPrivateStorage(tensor, loop->body, aliases))
      return true;
    bool used_in_condition = false;
    PostOrderVisit(loop->condition, [&](const ObjectRef &node) {
      used_in_condition |= node.same_as(tensor);
    });
    if (used_in_condition)
      return true;
  }
  return !cpu && IsSharedBuffer(tensor) &&
         (statement.as<ForNode>() || statement.as<WhileNode>() ||
          statement.as<IfNode>());
}

class AllocationPlacer : public tvm::tl::StmtExprMutator {
public:
  explicit AllocationPlacer(const AllocationUses &analysis, bool cpu) {
    for (const Allocation &allocation : analysis.allocations) {
      if (allocation.pinned || allocation.origin.empty())
        continue;
      Path site = CommonSite(allocation.uses);
      if (site.size() < allocation.origin.size())
        continue;
      size_t depth = allocation.origin.size() - 1;
      if (site[depth].sequence != allocation.origin.back().sequence ||
          site[depth].index < allocation.origin.back().index)
        continue;
      for (size_t index = depth + 1; index < site.size(); ++index) {
        auto owner = analysis.owners.find(site[index].sequence);
        if (owner != analysis.owners.end() &&
            CrossesExecutionBoundary(owner->second, allocation.tensor, cpu,
                                     analysis.aliases)) {
          site.resize(index);
          break;
        }
      }
      Position position = site.back();
      if (position == allocation.origin.back())
        continue;
      insertions_[position.sequence][position.index].push_back(
          allocation.binding);
      removed_.insert(allocation.binding.get());
      for (const Bind &alias : allocation.aliases) {
        insertions_[position.sequence][position.index].push_back(alias);
        removed_.insert(alias.get());
      }
    }
  }

private:
  UnchangedOr<Stmt> Mutate_(const SeqStmtNode *node, InplaceMode mode) final {
    Array<Stmt> statements;
    auto sequence = insertions_.find(node);
    for (size_t index = 0; index < node->seq.size(); ++index) {
      if (sequence != insertions_.end()) {
        if (auto bindings = sequence->second.find(index);
            bindings != sequence->second.end()) {
          for (const Bind &binding : bindings->second)
            statements.push_back(binding);
        }
      }
      if (!removed_.count(node->seq[index].get())) {
        statements.push_back(
            Mutate(node->seq[index], mode).ValueOrUnchanged(node->seq[index]));
      }
    }
    return SeqStmt(statements, node->loc);
  }

  std::unordered_map<const SeqStmtNode *,
                     std::unordered_map<size_t, std::vector<Bind>>>
      insertions_;
  std::unordered_set<const StmtNode *> removed_;
};
} // namespace

namespace transform {
tvm::transform::Pass PlanAndUpdateBufferAllocationLocation() {
  return tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule,
         tvm::transform::PassContext) -> Optional<tirx::Function> {
        if (!function->body)
          return function;
        auto target = function->GetAttr(tvm::attr::kTarget);
        bool cpu = target && (target.value()->kind->name == "c" ||
                              target.value()->kind->name == "llvm");
        auto analysis =
            make_object<AllocationUses>(cpu, UndefinedVars(*function->body));
        analysis->Visit(*function->body);
        auto placer = make_object<AllocationPlacer>(*analysis, cpu);
        SeqStmt body = *function->body;
        function.CopyOnWrite()->body =
            placer->Mutate(body).ValueOrUnchanged(body);
        return function;
      },
      0, "tl.PlanAndUpdateBufferAllocationLocation");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def(
      "tl.transform.PlanAndUpdateBufferAllocationLocation",
      PlanAndUpdateBufferAllocationLocation);
}
} // namespace transform
} // namespace tvm::tl

/*!
 * \file annotate_multi_buffer_eligible.cc
 * \brief Scheduled-TIR analysis: for each For loop, decide which on-chip
 *        buffers can be multi-buffered and record the set on the For's
 *        annotations under "multi_buffer_eligible".
 *
 *  Runs after NormalizeControlFlowForSchedule and MaterializeScheduleUnits,
 *  then decodes the shared IRStructure. Buffer-dependent loop bounds are
 *  therefore explicit tasks, while manual T.Stage requests are available when
 *  recovering the logical write-before-read order. Materialization flattens
 *  IfThenElse into guarded sibling nodes, so write-first analysis reasons
 *  about guard equivalence and implication instead of relying on nested
 *  statement shape.
 *
 *  Automatic claims form the deepest disjoint loop frontier that completely
 *  covers a storage's ordinary accesses; owner-external fills may initialize
 *  every physical version in ordinary modes. Ordinary owners must write the
 *  storage before any read; write-counter owners may retain an earlier value,
 *  but their updates must still be write-first. If descendant loops cover
 *  only part of an epoch (for example, a row-writing loop followed by a
 *  whole-buffer consumer), the claim is
 *  promoted to a write-first ancestor. Independent sibling loops may
 *  therefore become multiple owners of one storage.
 *
 *  Frontend may pre-set the annotation. A storage named by any explicit claim
 *  is excluded from automatic owner inference across the whole kernel; the
 *  explicit claim is preserved unless the storage is already manually
 *  versioned or pinned to one version without an explicit mode. Other buffers
 *  proven eligible are added automatically.
 *  T.annotate_buffer_versions({buf: 1}) removes the storage from eligibility;
 *  (1, mode) retains eligibility.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ascend/transform/auto_schedule/kernel_rewriter.h"
#include "ascend/transform/auto_schedule/multi_buffer.h"
#include "ascend/transform/auto_schedule/scheduled_tir.h"
#include "ascend/transform/auto_schedule/write_first_analysis.h"
#include "ascend/transform/buffer_version.h"
#include "op/builtin.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ascend;
using namespace ffi;

namespace {

using StorageSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using ControlStorageClaims =
    std::unordered_map<ControlNode *, std::vector<Var>>;

Array<Var> NormalizeEligibleStorages(const Any &annotation) {
  Array<Var> result;
  StorageSet seen;
  for (const Any &item : Downcast<Array<Any>>(annotation)) {
    Var storage;
    if (auto buffer = item.try_cast<Buffer>()) {
      storage = buffer.value()->data;
    } else if (auto var = item.try_cast<Var>()) {
      storage = var.value();
    } else {
      LOG(FATAL) << "'" << kMultiBufferEligible
                 << "' entries must be Buffer or Var objects, got " << item;
    }
    if (seen.insert(storage).second)
      result.push_back(storage);
  }
  return result;
}

// ---------------------------------------------------------------------------
// MultiBufferOwnerPlanner: for each storage, choose the deepest set of
// disjoint structural loops that covers every ordinary access. Independently
// rewritable fills may remain outside the frontier in ordinary modes. A loop
// is selected only when its descendants do not already provide complete
// coverage and its stage-ordered body satisfies the mode's write-first rule.
// ---------------------------------------------------------------------------
class MultiBufferOwnerPlanner {
public:
  static ControlStorageClaims
  Plan(const std::vector<std::shared_ptr<IRStructure>> &root,
       const StorageSet &excluded_storages, const StorageSet &write_counters,
       const L0StorageGroups &groups) {
    MultiBufferOwnerPlanner planner;
    StorageSet seen;
    for (const auto &node : root) {
      for (const Var &storage : node->GetOnChipStorages()) {
        Var representative = groups.Representative(storage);
        if (!seen.insert(representative).second ||
            excluded_storages.count(representative))
          continue;
        Array<Var> members = groups.Members(storage);
        bool write_counter = write_counters.count(representative);
        CoveragePlan plan = planner.PlanList(root, members, write_counter);
        ICHECK(!write_counter || (plan.IsComplete() && !plan.owners.empty()))
            << "Cannot infer complete loop owners for write_counter storage "
            << storage->name_hint
            << "; put every access, including fills, inside a loop and write "
               "each new value before updating it, or select the update "
               "scope explicitly with multi_buffer_eligible";
        if (!plan.IsComplete())
          continue;
        for (ControlNode *owner : plan.owners)
          for (const Var &member : members)
            planner.claims_[owner].push_back(member);
      }
    }
    for (const Var &storage : write_counters)
      ICHECK(excluded_storages.count(storage) ||
             seen.count(groups.Representative(storage)))
          << "write_counter storage " << storage->name_hint
          << " has no accesses from which to infer an owner loop";
    return std::move(planner.claims_);
  }

private:
  // Ordered from least to most restrictive; sibling composition takes max.
  enum class Coverage {
    kUntouched,
    kCovered,
    kNeedsOwner,
  };

  struct CoveragePlan {
    Coverage coverage{Coverage::kUntouched};
    std::vector<ControlNode *> owners;

    bool IsComplete() const {
      return coverage == Coverage::kUntouched || coverage == Coverage::kCovered;
    }

    bool CanClaimHere() const {
      // An ownerless covered subtree contains only independently broadcastable
      // fills. Keep those fills outside the owner frontier so they initialize
      // every physical version instead of advancing one version per loop
      // iteration.
      return coverage == Coverage::kNeedsOwner;
    }

    void AddUncoveredAccess() { coverage = Coverage::kNeedsOwner; }

    void Merge(const CoveragePlan &other) {
      coverage = std::max(coverage, other.coverage);
      owners.insert(owners.end(), other.owners.begin(), other.owners.end());
    }
  };

  ControlStorageClaims claims_;

  static CoveragePlan PlanLeaf(const TaskNode *task, const Array<Var> &members,
                               bool write_counter) {
    CoveragePlan result;
    for (const Var &storage : members) {
      if (!task->TouchesStorage(storage))
        continue;
      // A write-counter fill creates a value and must belong to an owner;
      // it cannot initialize every slot outside the generation protocol.
      bool broadcast_fill = !write_counter && members.size() == 1 &&
                            !task->GuardsTouchStorage(storage) &&
                            CanBroadcastFillToStorage(task->stmt, storage);
      if (!broadcast_fill)
        result.AddUncoveredAccess();
      else if (result.coverage == Coverage::kUntouched)
        result.coverage = Coverage::kCovered;
    }
    return result;
  }

  CoveragePlan PlanList(const std::vector<std::shared_ptr<IRStructure>> &nodes,
                        const Array<Var> &members, bool write_counter) {
    CoveragePlan result;
    for (const auto &node : nodes)
      result.Merge(PlanNode(node.get(), members, write_counter));
    return result;
  }

  CoveragePlan PlanNode(IRStructure *node, const Array<Var> &members,
                        bool write_counter) {
    if (node->IsTask())
      return PlanLeaf(static_cast<const TaskNode *>(node), members,
                      write_counter);

    auto *control = static_cast<ControlNode *>(node);
    CoveragePlan result = PlanList(control->children, members, write_counter);
    if (result.CanClaimHere()) {
      // Write clocks allow read-only owners and retained-value reads, but an
      // update cannot start with RMW. A rejected inner loop can still belong
      // to an outer owner that defines the new value before that loop.
      // Check every bound plane independently with the same mode policy.
      bool write_first = true;
      for (const Var &storage : members) {
        AccessOrder order = IRWriteFirstClassifier::Classify(
            control->children, storage, control->GetLoopBodyContext(),
            /*writes_only=*/write_counter);
        write_first &= order != AccessOrder::kReadFirst;
      }
      // A write owner cannot exit before publishing its update. An enclosing
      // loop may still own a complete update containing this inner break.
      if (write_first && !(write_counter && control->BodyHasLoopBreak())) {
        result.coverage = Coverage::kCovered;
        result.owners = {control};
      }
    }
    for (const Var &storage : members) {
      if (control->task->TouchesStorage(storage))
        result.AddUncoveredAccess();
    }
    return result;
  }
};

// Normalize explicit claims and add the automatic owner frontier directly to
// the decoded ControlNodes. Encoding preserves schedule-unit stages and guards.
class MultiBufferAnnotator {
public:
  static void Rewrite(ScheduledTIR *scheduled_tir, StorageSet manual_buffers,
                      const L0StorageGroups &groups) {
    ICHECK(scheduled_tir != nullptr);
    // This pass precedes AutoSchedule: these counts are frontend overrides,
    // not solver-selected versions. Only a bare frontend 1 means opt-out.
    StorageSet single_version_buffers;
    StorageSet excluded_storages = manual_buffers;
    for (const auto &[storage, versions] :
         scheduled_tir->metadata.buffer_versions) {
      if (versions == 1 &&
          !scheduled_tir->metadata.buffer_version_modes.count(storage)) {
        single_version_buffers.insert(storage);
        excluded_storages.insert(storage);
      }
    }
    CollectExplicitStorages(scheduled_tir->tree, &excluded_storages);
    StorageSet write_counters;
    for (const auto &[storage, mode] :
         scheduled_tir->metadata.buffer_version_modes) {
      if (mode != "write_counter")
        continue;
      ICHECK(scheduled_tir->metadata.buffer_versions.count(storage))
          << "write_counter requires an explicit version count for "
          << storage->name_hint << "; use (num_versions, \"write_counter\")";
      ICHECK(!manual_buffers.count(storage))
          << "write_counter cannot be combined with manual physical versions "
             "for "
          << storage->name_hint;
      write_counters.insert(storage);
    }
    StorageSet expanded;
    for (const Var &storage : excluded_storages)
      for (const Var &member : groups.Members(storage))
        expanded.insert(member);
    excluded_storages = std::move(expanded);
    ControlStorageClaims claims = MultiBufferOwnerPlanner::Plan(
        scheduled_tir->tree, excluded_storages, write_counters, groups);
    MultiBufferAnnotator annotator(std::move(claims), std::move(manual_buffers),
                                   std::move(single_version_buffers), groups);
    annotator.RewriteNodes(scheduled_tir->tree);
  }

private:
  MultiBufferAnnotator(ControlStorageClaims claims, StorageSet manual_buffers,
                       StorageSet single_version_buffers,
                       L0StorageGroups groups)
      : claims_(std::move(claims)), groups_(std::move(groups)),
        manual_buffers_(std::move(manual_buffers)),
        single_version_buffers_(std::move(single_version_buffers)) {}

  ControlStorageClaims claims_;
  L0StorageGroups groups_;
  StorageSet manual_buffers_;
  StorageSet single_version_buffers_;
  StorageSet warned_excluded_storages_;

  static void CollectExplicitStorages(
      const std::vector<std::shared_ptr<IRStructure>> &nodes,
      StorageSet *storages) {
    ICHECK(storages != nullptr);
    for (const auto &node : nodes) {
      if (!node->IsControl())
        continue;
      const auto *control = static_cast<const ControlNode *>(node.get());
      if (auto annotation =
              control->control->annotations.Get(kMultiBufferEligible)) {
        for (const Var &storage :
             NormalizeEligibleStorages(annotation.value())) {
          storages->insert(storage);
        }
      }
      CollectExplicitStorages(control->children, storages);
    }
  }

  void RewriteNodes(std::vector<std::shared_ptr<IRStructure>> &nodes) {
    for (const auto &node : nodes) {
      if (!node->IsControl())
        continue;
      auto *control = static_cast<ControlNode *>(node.get());
      RewriteNodes(control->children);
      RewriteControl(control);
    }
  }

  void RewriteControl(ControlNode *control) {
    For loop = control->control;
    Array<Var> eligible;
    StorageSet already;
    if (auto annotation = loop->annotations.Get(kMultiBufferEligible)) {
      for (const Var &storage : NormalizeEligibleStorages(annotation.value())) {
        if (manual_buffers_.count(storage)) {
          if (warned_excluded_storages_.insert(storage).second) {
            LOG(WARNING) << "Ignoring explicit '" << kMultiBufferEligible
                         << "' claim for storage " << storage->name_hint
                         << " because it is already manually multi-buffered";
          }
          continue;
        }
        if (single_version_buffers_.count(storage)) {
          if (warned_excluded_storages_.insert(storage).second) {
            LOG(WARNING) << "Ignoring explicit '" << kMultiBufferEligible
                         << "' claim for storage " << storage->name_hint
                         << " because T.annotate_buffer_versions({buf: 1}) "
                            "disables multi-buffer eligibility; use "
                            "{buf: (1, \"auto\")} to retain eligibility with "
                            "one version";
          }
          continue;
        }
        for (const Var &member : groups_.Members(storage))
          if (already.insert(member).second)
            eligible.push_back(member);
      }
    }

    auto planned = claims_.find(control);
    if (planned != claims_.end()) {
      for (const Var &storage : planned->second) {
        for (const Var &member : groups_.Members(storage))
          if (already.insert(member).second)
            eligible.push_back(member);
      }
    }

    loop.CopyOnWrite()->annotations.Set(kMultiBufferEligible, eligible);
    control->control = std::move(loop);
  }
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass AnnotateMultiBufferEligible() {
  auto pass_func = [=](PrimFunc f, const IRModule &, const PassContext &) {
    return RewriteTilelangKernels(
        std::move(f), "AnnotateMultiBufferEligible",
        [](const TilelangKernelContext &context) {
          ScheduledTIR scheduled_tir =
              DecodeScheduledTIR(context.root, context.outer_ctx);
          L0StorageGroups groups(CollectL0SFBindings(context.root));
          auto &metadata = scheduled_tir.metadata;
          metadata.buffer_versions = ExpandL0StorageGroupValues(
              metadata.buffer_versions, groups, "version counts");
          metadata.manual_buffer_versions = ExpandL0StorageGroupValues(
              metadata.manual_buffer_versions, groups, "manual version counts");
          metadata.buffer_version_modes = ExpandL0StorageGroupValues(
              metadata.buffer_version_modes, groups, "version modes");
          StorageSet manual_buffers;
          for (const auto &[storage, _] :
               scheduled_tir.metadata.manual_buffer_versions) {
            manual_buffers.insert(storage);
          }
          MultiBufferAnnotator::Rewrite(&scheduled_tir,
                                        std::move(manual_buffers), groups);
          return EncodeScheduledTIR(std::move(scheduled_tir));
        },
        /*require_kernel=*/false);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnnotateMultiBufferEligible", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.AnnotateMultiBufferEligible",
                        AnnotateMultiBufferEligible);
}

} // namespace tl
} // namespace tvm

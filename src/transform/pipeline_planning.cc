#include "support/check.h"
#include <algorithm>
#include <deque>
#include <map>
#include <numeric>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/sym/analyzer.h>
#include <tvm/target/target.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "backend/common/target_utils.h"
#include "common/bind_utils.h"
#include "common/pipeline_access.h"
#include "common/pipeline_utils.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/operator.h"
#include "op/parallel.h"
#include "op/utils.h"
#include "tvm/ir/expr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
using namespace prim;

namespace {
Array<int64_t> ReadIntegerArray(const Any &value) {
  Array<int64_t> result;
  for (const Any &item : value.cast<Array<Any>>())
    result.push_back(GetIntegerAnnotation(item));
  return result;
}
} // namespace

class PipelinePlanningBodyAnalyzer {
public:
  PipelinePlanningBodyAnalyzer(const PipelineMemoryAnalysis *memory,
                               Target target)
      : memory_(std::move(memory)), target_(std::move(target)) {}

  std::pair<Array<TensorRegion>, Array<TensorRegion>>
  CollectStmtAccessRegions(const Stmt &stmt) const {
    auto collector =
        ffi::make_object<PipelineAccessCollector>(memory_, target_);
    collector->Visit(stmt);
    return {collector->GetReads(), collector->GetWrites()};
  }

  TensorSet CollectPipelineWriteBuffers(const Array<Stmt> &stmts) const {
    TensorSet write_buffers;
    for (const Stmt &stmt : stmts) {
      auto [_, writes] = CollectStmtAccessRegions(stmt);
      for (const TensorRegion &write : writes) {
        write_buffers.insert(RegionTensor(write));
      }
    }
    return write_buffers;
  }

  bool
  IsReplayableScalarBindStmt(const Stmt &stmt,
                             const TensorSet &pipeline_write_buffers) const {
    auto collector =
        ffi::make_object<PipelineAccessCollector>(memory_, target_);
    collector->Visit(stmt);
    return !collector->HasUnknownAccess() &&
           IsReplayablePipelineBind(
               stmt, collector->GetReads(), pipeline_write_buffers,
               [this](const TensorVar &a, const TensorVar &b) {
                 return memory_->MayAlias(a, b);
               });
  }

  struct ScheduledStmtAnalysis {
    size_t original_stmt_count{0};
    size_t stage_stmt_count{0};
    Array<Stmt> scheduled_stmts;
    std::vector<size_t> scheduled_indices;
    std::vector<size_t> scheduled_stage_indices;
    Array<int64_t> replayable_bind_mask;
  };

  ScheduledStmtAnalysis AnalyzeScheduledStmts(const Array<Stmt> &stmts) const {
    TensorSet pipeline_write_buffers = CollectPipelineWriteBuffers(stmts);
    ScheduledStmtAnalysis analysis;
    analysis.original_stmt_count = stmts.size();
    analysis.replayable_bind_mask.reserve(stmts.size());
    size_t stage_stmt_index = 0;
    for (size_t i = 0; i < stmts.size(); ++i) {
      const Stmt &stmt = stmts[i];
      if (IsPipelineDeclarationStmt(stmt)) {
        continue;
      }
      bool replayable =
          IsReplayableScalarBindStmt(stmt, pipeline_write_buffers);
      analysis.replayable_bind_mask.push_back(int64_t(replayable ? 1 : 0));
      if (replayable) {
        ++stage_stmt_index;
        continue;
      }
      analysis.scheduled_indices.push_back(i);
      analysis.scheduled_stage_indices.push_back(stage_stmt_index);
      analysis.scheduled_stmts.push_back(stmt);
      ++stage_stmt_index;
    }
    analysis.stage_stmt_count = stage_stmt_index;
    return analysis;
  }

  Array<int64_t> FilterAnnotationsForScheduledStmts(
      const Array<int64_t> &annotations,
      const ScheduledStmtAnalysis &analysis) const {
    if (annotations.size() == analysis.scheduled_stmts.size()) {
      return annotations;
    }

    Array<int64_t> filtered;
    if (annotations.size() == analysis.stage_stmt_count) {
      for (size_t index : analysis.scheduled_stage_indices) {
        filtered.push_back(annotations[index]);
      }
    } else {
      ICHECK_EQ(annotations.size(), analysis.original_stmt_count)
          << "PipelinePlanning: expected pipeline annotation size to match "
             "the scheduled statement count, executable statement count, or "
             "original statement count";
      for (size_t index : analysis.scheduled_indices) {
        filtered.push_back(annotations[index]);
      }
    }
    ICHECK_EQ(filtered.size(), analysis.scheduled_stmts.size());
    return filtered;
  }

  class SeqStmtFlattener {
  public:
    static Array<Stmt> Flatten(const Array<Stmt> &stmts) {
      Array<Stmt> flat;
      for (const Stmt &stmt : stmts) {
        if (const auto *seq = stmt.as<SeqStmtObj>()) {
          for (const Stmt &child : Flatten(seq->seq))
            flat.push_back(child);
        } else {
          flat.push_back(stmt);
        }
      }
      return flat;
    }
  };

private:
  const PipelineMemoryAnalysis *memory_{nullptr};
  Target target_{ffi::UnsafeInit{}};
};

using ScalarVarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

/*! \brief Information about a pipeline stage
 *
 * \param reads Array of buffer regions read by this stage
 * \param writes Array of buffer regions written by this stage
 * \param original_stmt_index Original position of this stage in the pipeline
 * before reordering \param order Current position of this stage in the
 * pipeline after reordering (-1 if not yet assigned) \param stage Pipeline
 * stage number this operation belongs to (-1 if not yet assigned) \param
 * copy_stage Whether this stage is a memory copy operation \param
 * last_use_stmt_index Resolved scheduling anchor for first-stage statements
 * (-1 if not yet determined), used to keep a copy and all of its transitive
 * producers before the earliest downstream non-first-stage consumer \param
 * direct_copy_last_use_stmt_index Last direct buffer consumer of a real copy
 * (-1 for non-copy statements or copies with no consumer), preserved
 * independently for async commit-group formation
 */
struct PipelineStageInfo {
  Array<TensorRegion> reads, writes;
  TensorSet imprecise_reads, imprecise_writes;
  ScalarVarSet scalar_defs;
  ScalarVarSet scalar_uses;
  int original_stmt_index{};
  int order = -1, stage = -1;
  bool copy_stage = false;
  bool tma_copy = false; // true if this copy stage uses TMA (not cp.async)
  bool conditional_execution = false;
  bool producer_for_copy = false;
  bool unknown_access = false;
  bool unknown_address = false;
  int last_use_stmt_index =
      -1; // Initialized to -1, indicating no consumers found yet
  int direct_copy_last_use_stmt_index = -1;

public:
  bool IsFirstStage() const { return copy_stage || producer_for_copy; }
  bool IsCopyStage() const { return copy_stage; }
  bool IsTmaCopy() const { return tma_copy; }
  bool IsProducerForCopy() const { return producer_for_copy; }
  bool IsLastUseStmtIndexValid() const { return last_use_stmt_index != -1; }
};

class PipelineStageAnalyzer {
public:
  PipelineStageAnalyzer(const PipelineMemoryAnalysis *memory, Target target,
                        bool use_async_copy,
                        Map<Var, sym::IntSet> thread_domains)
      : memory_(std::move(memory)), target_(std::move(target)),
        use_async_copy_(use_async_copy),
        thread_domains_(std::move(thread_domains)) {}

  class ScalarUseDefCollector : public StmtExprVisitor {
  public:
    using StmtExprVisitor::Visit;
    using StmtExprVisitor::Visit_;
    static std::pair<ScalarVarSet, ScalarVarSet> Collect(const Stmt &stmt) {
      auto collector = ffi::make_object<ScalarUseDefCollector>();
      collector->Visit(stmt);
      return {std::move(collector->scalar_defs_),
              std::move(collector->scalar_uses_)};
    }

  private:
    Optional<VisitInterrupt> Visit_(const BindObj *op) final {
      Visit(op->value);
      Visit(op->var->ty);
      scalar_defs_.insert(op->var);
      return std::nullopt;
    }
    Optional<VisitInterrupt> Visit_(const VarObj *op) final {
      scalar_uses_.insert(GetRef<Var>(op));
      Visit(op->ty);
      return std::nullopt;
    }
    ScalarVarSet scalar_defs_, scalar_uses_;
  };

  bool MayBeConditionallyExecuted(const Stmt &stmt) const {
    bool conditional = false;
    VisitNativePostOrder(stmt, [&](const ObjectRef &node) {
      if (conditional) {
        return;
      }
      if (const auto *if_then_else = node.as<IfObj>()) {
        conditional = true;
        return;
      }
    });
    return conditional;
  }

  bool IsAsyncProducerCandidate(const PipelineStageInfo &pinfo) const {
    if (pinfo.conditional_execution) {
      return false;
    }
    if (pinfo.IsTmaCopy()) {
      return false;
    }
    return pinfo.IsCopyStage();
  }

  bool IsPureCopyStmt(const Stmt &stmt) const {
    auto is_global_like_buffer = [](const TensorVar &buffer) {
      return IsGlobalBuffer(buffer) ||
             (buffer.defined() && buffer.scope().name().empty());
    };
    auto is_pure_raw_copy_value = [&](const Expr &expr,
                                      const auto &self) -> bool {
      if (const auto *load = expr.as<TensorLoadObj>()) {
        return is_global_like_buffer(load->source.as_or_throw<TensorVar>());
      }
      if (const auto *cast = expr.as<CastObj>()) {
        return self(cast->value, self);
      }
      return false;
    };

    bool saw_copy = false;
    bool saw_non_copy_tile_op = false;
    bool saw_non_copy_buffer_store = false;
    VisitNativePostOrder(stmt, [&](const ObjectRef &node) {
      if (saw_non_copy_tile_op || saw_non_copy_buffer_store) {
        return;
      }
      if (const auto *store = node.as<TensorStoreObj>()) {
        saw_copy = true;
        if ((!IsSharedBuffer(store->dest.as_or_throw<TensorVar>()) &&
             !IsLocalBuffer(store->dest.as_or_throw<TensorVar>(),
                            /*allow_var=*/true)) ||
            !is_pure_raw_copy_value(store->value, is_pure_raw_copy_value)) {
          saw_non_copy_buffer_store = true;
        }
        return;
      }
      const auto *call = node.as<CallObj>();
      if (call == nullptr) {
        return;
      }
      auto tile_op = ParseOperator(GetRef<Call>(call));
      if (!tile_op.defined()) {
        return;
      }
      if (const auto *parallel = tile_op.as<ParallelOpNode>()) {
        if (IsPureCopyStmt(parallel->GetRoot())) {
          saw_copy = true;
        } else {
          saw_non_copy_tile_op = true;
        }
        return;
      }
      if (tile_op.as<CopyNode>() || tile_op.as<Im2ColOpNode>()) {
        saw_copy = true;
      } else {
        saw_non_copy_tile_op = true;
      }
    });
    return saw_copy && !saw_non_copy_tile_op && !saw_non_copy_buffer_store;
  }

  Optional<TileOperator> GetSinglePureCopyTileOp(const Stmt &stmt) const {
    Optional<TileOperator> copy_tile_op;
    bool saw_non_copy_tile_op = false;
    bool saw_multiple_copy_ops = false;
    VisitNativePostOrder(stmt, [&](const ObjectRef &node) {
      if (saw_non_copy_tile_op || saw_multiple_copy_ops) {
        return;
      }
      const auto *call = node.as<CallObj>();
      if (call == nullptr) {
        return;
      }
      auto tile_op = ParseOperator(GetRef<Call>(call));
      if (!tile_op.defined()) {
        return;
      }
      if (tile_op.as<CopyNode>() || tile_op.as<Im2ColOpNode>()) {
        if (copy_tile_op.has_value()) {
          saw_multiple_copy_ops = true;
          copy_tile_op = Optional<TileOperator>();
        } else {
          copy_tile_op = tile_op;
        }
      } else {
        saw_non_copy_tile_op = true;
        copy_tile_op = Optional<TileOperator>();
      }
    });
    if (saw_non_copy_tile_op || saw_multiple_copy_ops) {
      return Optional<TileOperator>();
    }
    return copy_tile_op;
  }

  static bool IsGlobalLikeBuffer(const TensorVar &buffer) {
    return IsGlobalBuffer(buffer) ||
           (buffer.defined() && buffer.scope().name().empty());
  }

  void ClassifyCopyLikeStage(const Stmt &stmt, PipelineStageInfo *pinfo) const {
    ICHECK(pinfo != nullptr);
    if (pinfo->conditional_execution) {
      return;
    }

    if (pinfo->copy_stage) {
      return;
    }

    auto copy_tile_op = GetSinglePureCopyTileOp(stmt);
    if (!copy_tile_op.has_value()) {
      return;
    }

    if (const auto *copy = copy_tile_op.value().as<CopyNode>()) {
      if (!IsGlobalLikeBuffer(copy->src) || !IsSharedBuffer(copy->dst)) {
        return;
      }
      pinfo->copy_stage = true;
      return;
    }

    if (const auto *im2col = copy_tile_op.value().as<Im2ColOpNode>()) {
      if (!IsGlobalLikeBuffer(im2col->src_) || !IsSharedBuffer(im2col->dst_)) {
        return;
      }
      pinfo->copy_stage = true;
      pinfo->tma_copy = TargetIsHopper(target_);
    }
  }

  void AnalyzeCopyLastUse(
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    for (auto &pinfo : *pipeline_stage_infos) {
      if (!pinfo.IsCopyStage()) {
        continue;
      }

      for (int i = pinfo.original_stmt_index + 1;
           i < static_cast<int>(pipeline_stage_infos->size()); ++i) {
        for (const TensorRegion &read : (*pipeline_stage_infos)[i].reads) {
          if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),
                           [&](const TensorRegion &r) {
                             return memory_->MayConflict(r, read,
                                                         thread_domains_);
                           }) != pinfo.writes.end()) {
            pinfo.last_use_stmt_index = std::max(pinfo.last_use_stmt_index, i);
          }
        }

        for (const TensorRegion &write : (*pipeline_stage_infos)[i].writes) {
          if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),
                           [&](const TensorRegion &r) {
                             return memory_->MayConflict(r, write,
                                                         thread_domains_);
                           }) != pinfo.writes.end()) {
            LOG(FATAL) << "Pipeline planning error: Multiple writes to "
                          "overlapping buffer regions detected. "
                       << "Stage " << pinfo.original_stmt_index << " and stage "
                       << i << " are both writing to buffer '"
                       << RegionTensor(write)->name
                       << "' with overlapping regions. This is not supported "
                          "in pipeline planning.";
          }
        }
      }
      pinfo.direct_copy_last_use_stmt_index = pinfo.last_use_stmt_index;
    }
  }

  using ScalarDefMap =
      std::unordered_map<Var, int, ObjectPtrHash, ObjectPtrEqual>;

  ScalarDefMap BuildScalarDefMap(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos) const {
    ScalarDefMap scalar_def_to_stmt;
    for (int i = 0; i < static_cast<int>(pipeline_stage_infos.size()); ++i) {
      for (const Var &var : pipeline_stage_infos[i].scalar_defs) {
        scalar_def_to_stmt.emplace(var, i);
      }
    }
    return scalar_def_to_stmt;
  }

  std::vector<int> FindBufferProducerIndices(
      int consumer_idx,
      const std::vector<PipelineStageInfo> &pipeline_stage_infos) const {
    std::vector<int> producer_indices;
    const auto &consumer = pipeline_stage_infos[consumer_idx];
    for (int producer_idx = 0; producer_idx < consumer_idx; ++producer_idx) {
      const auto &producer = pipeline_stage_infos[producer_idx];
      bool conflicts = false;
      for (const TensorRegion &write : producer.writes) {
        for (const TensorRegion &read : consumer.reads) {
          if (memory_->MayConflict(write, read, thread_domains_)) {
            conflicts = true;
            break;
          }
        }
        if (conflicts) {
          break;
        }
      }
      if (conflicts) {
        producer_indices.push_back(producer_idx);
      }
    }
    return producer_indices;
  }

  std::vector<int> FindScalarProducerIndices(
      int consumer_idx, const ScalarDefMap &scalar_def_to_stmt,
      const std::vector<PipelineStageInfo> &pipeline_stage_infos) const {
    std::vector<int> producer_indices;
    for (const Var &var : pipeline_stage_infos[consumer_idx].scalar_uses) {
      auto it = scalar_def_to_stmt.find(var);
      if (it == scalar_def_to_stmt.end() || it->second >= consumer_idx) {
        continue;
      }
      if (std::find(producer_indices.begin(), producer_indices.end(),
                    it->second) == producer_indices.end()) {
        producer_indices.push_back(it->second);
      }
    }
    std::sort(producer_indices.begin(), producer_indices.end());
    return producer_indices;
  }

  bool PropagateProducersForCopy(
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    ScalarDefMap scalar_def_to_stmt = BuildScalarDefMap(*pipeline_stage_infos);
    std::vector<std::unordered_set<int>> anchor_candidates(
        pipeline_stage_infos->size());
    std::deque<std::pair<int, int>> worklist;

    for (int i = 0; i < static_cast<int>(pipeline_stage_infos->size()); ++i) {
      const auto &pinfo = (*pipeline_stage_infos)[i];
      if (pinfo.IsCopyStage() && pinfo.IsLastUseStmtIndexValid()) {
        anchor_candidates[i].insert(pinfo.last_use_stmt_index);
        worklist.emplace_back(i, pinfo.last_use_stmt_index);
      }
    }

    auto relax_producer = [&](int producer_idx, int root_anchor) {
      auto &producer = (*pipeline_stage_infos)[producer_idx];
      if (producer.IsTmaCopy()) {
        // A stage-0 consumer needs an explicit wait that this schedule cannot
        // insert. Let the ordinary sequential lowering provide that wait.
        return false;
      }
      if (!producer.IsCopyStage()) {
        producer.producer_for_copy = true;
      }
      if (anchor_candidates[producer_idx].insert(root_anchor).second) {
        worklist.emplace_back(producer_idx, root_anchor);
      }
      return true;
    };

    while (!worklist.empty()) {
      auto [consumer_idx, root_anchor] = worklist.front();
      worklist.pop_front();

      for (int producer_idx : FindScalarProducerIndices(
               consumer_idx, scalar_def_to_stmt, *pipeline_stage_infos)) {
        if (!relax_producer(producer_idx, root_anchor)) {
          return false;
        }
      }
      for (int producer_idx :
           FindBufferProducerIndices(consumer_idx, *pipeline_stage_infos)) {
        if (!relax_producer(producer_idx, root_anchor)) {
          return false;
        }
      }
    }

    ResolveFirstStageAnchors(anchor_candidates, pipeline_stage_infos);
    return true;
  }

  void ResolveFirstStageAnchors(
      const std::vector<std::unordered_set<int>> &anchor_candidates,
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    const int num_stmts = static_cast<int>(pipeline_stage_infos->size());
    std::vector<int> resolved_anchors(num_stmts, -1);
    for (int i = num_stmts - 1; i >= 0; --i) {
      if (anchor_candidates[i].empty()) {
        continue;
      }

      int resolved_anchor = num_stmts;
      for (int anchor : anchor_candidates[i]) {
        ICHECK_GE(anchor, 0);
        ICHECK_LT(anchor, num_stmts);
        ICHECK_GT(anchor, i)
            << "Pipeline planning error: first-stage scheduling anchors must "
               "advance toward a consumer";
        const auto &anchor_info = (*pipeline_stage_infos)[anchor];
        if (anchor_info.IsFirstStage() && !anchor_candidates[anchor].empty()) {
          ICHECK_NE(resolved_anchors[anchor], -1);
          resolved_anchor = std::min(resolved_anchor, resolved_anchors[anchor]);
        } else {
          resolved_anchor = std::min(resolved_anchor, anchor);
        }
      }
      ICHECK_LT(resolved_anchor, num_stmts);
      resolved_anchors[i] = resolved_anchor;
      (*pipeline_stage_infos)[i].last_use_stmt_index = resolved_anchor;
    }
  }

  void ValidateScalarDependencies(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos) const {
    auto scalar_def_to_stmt = BuildScalarDefMap(pipeline_stage_infos);
    for (int consumer_idx = 0;
         consumer_idx < static_cast<int>(pipeline_stage_infos.size());
         ++consumer_idx) {
      const auto &consumer = pipeline_stage_infos[consumer_idx];
      for (const Var &var : consumer.scalar_uses) {
        auto it = scalar_def_to_stmt.find(var);
        if (it == scalar_def_to_stmt.end() || it->second == consumer_idx) {
          continue;
        }
        const auto &producer = pipeline_stage_infos[it->second];
        ICHECK_EQ(producer.stage, consumer.stage)
            << "Pipeline planning error: scalar dependency from statement "
            << producer.original_stmt_index << " to statement "
            << consumer.original_stmt_index
            << " crosses pipeline stages. Scheduled scalar Bind statements "
               "must stay in the same stage as their consumers.";
        if (producer.stage == consumer.stage) {
          ICHECK_LT(producer.order, consumer.order)
              << "Pipeline planning error: scalar dependency from statement "
              << producer.original_stmt_index << " to statement "
              << consumer.original_stmt_index
              << " is reordered within the same pipeline stage.";
        }
      }
    }
  }

  bool EmitImplicitAsyncAnnotations(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos,
      Map<String, Any> *annotations) const {
    if (!TargetHasAsyncCopy(target_) || !use_async_copy_) {
      return false;
    }

    std::vector<int> async_group_ids(pipeline_stage_infos.size(), -1);
    std::vector<int> stmt_indices_by_order(pipeline_stage_infos.size());
    std::iota(stmt_indices_by_order.begin(), stmt_indices_by_order.end(), 0);
    std::stable_sort(stmt_indices_by_order.begin(), stmt_indices_by_order.end(),
                     [&](int lhs, int rhs) {
                       if (pipeline_stage_infos[lhs].order !=
                           pipeline_stage_infos[rhs].order) {
                         return pipeline_stage_infos[lhs].order <
                                pipeline_stage_infos[rhs].order;
                       }
                       return lhs < rhs;
                     });

    int next_async_group_id = 0;
    std::map<std::pair<int, int>, int> implicit_group_ids;
    for (int stmt_idx : stmt_indices_by_order) {
      const auto &pinfo = pipeline_stage_infos[stmt_idx];
      if (!IsAsyncProducerCandidate(pinfo)) {
        continue;
      }
      // Wait insertion currently handles synchronous consumers only. A copy
      // whose index reads another copy's output must take that path too.
      auto producers =
          FindBufferProducerIndices(stmt_idx, pipeline_stage_infos);
      if (std::any_of(producers.begin(), producers.end(),
                      [&](int producer_idx) {
                        return IsAsyncProducerCandidate(
                            pipeline_stage_infos[producer_idx]);
                      })) {
        continue;
      }
      auto key =
          std::make_pair(pinfo.stage, pinfo.direct_copy_last_use_stmt_index);
      auto [it, inserted] =
          implicit_group_ids.emplace(key, next_async_group_id);
      if (inserted) {
        ++next_async_group_id;
      }
      async_group_ids[stmt_idx] = it->second;
    }

    if (next_async_group_id == 0) {
      return false;
    }

    std::vector<int64_t> async_producers;
    std::vector<int64_t> async_producer_groups;
    async_producers.reserve(pipeline_stage_infos.size());
    async_producer_groups.reserve(pipeline_stage_infos.size());
    std::unordered_set<int> async_stage_ids;
    for (size_t i = 0; i < pipeline_stage_infos.size(); ++i) {
      bool is_async_producer = async_group_ids[i] != -1;
      async_producers.push_back(int64_t(is_async_producer ? 1 : 0));
      async_producer_groups.push_back(int64_t(async_group_ids[i]));
      if (is_async_producer) {
        async_stage_ids.insert(pipeline_stage_infos[i].stage);
      }
    }

    annotations->Set(kPipelineAsyncProducers, Array<int64_t>(async_producers));
    annotations->Set(kPipelineAsyncProducerGroups,
                     Array<int64_t>(async_producer_groups));

    std::vector<int> sorted_async_stage_ids(async_stage_ids.begin(),
                                            async_stage_ids.end());
    std::sort(sorted_async_stage_ids.begin(), sorted_async_stage_ids.end());
    std::vector<int64_t> async_stages;
    async_stages.reserve(sorted_async_stage_ids.size());
    for (int stage_id : sorted_async_stage_ids) {
      async_stages.push_back(int64_t(stage_id));
    }
    annotations->Set("software_pipeline_async_stages",
                     Array<int64_t>(async_stages));
    return true;
  }

  void MaybeAnnotateLegacyAsyncPipelineLoop(const Array<Stmt> &pipeline_stmts,
                                            const Array<int64_t> &order_array,
                                            const Array<int64_t> &stage_array,
                                            Map<String, Any> *annotations) {
    if (!TargetHasAsyncCopy(target_) || !use_async_copy_) {
      return;
    }
    ICHECK_EQ(pipeline_stmts.size(), order_array.size());
    ICHECK_EQ(pipeline_stmts.size(), stage_array.size());

    std::vector<PipelineStageInfo> pipeline_stage_infos;
    pipeline_stage_infos.reserve(pipeline_stmts.size());
    for (size_t i = 0; i < pipeline_stmts.size(); ++i) {
      auto pinfo = MakePipelineStageInfo(pipeline_stmts[i], i);
      ClassifyCopyLikeStage(pipeline_stmts[i], &pinfo);
      pinfo.order = static_cast<int>(order_array[i]);
      pinfo.stage = static_cast<int>(stage_array[i]);
      if (!pinfo.IsCopyStage() && !pinfo.conditional_execution &&
          pinfo.stage == 0) {
        bool reads_global = false;
        bool writes_shared = false;
        for (const TensorRegion &read : pinfo.reads) {
          if (IsGlobalLikeBuffer(RegionTensor(read))) {
            reads_global = true;
            break;
          }
        }
        for (const TensorRegion &write : pinfo.writes) {
          if (IsSharedBuffer(RegionTensor(write))) {
            writes_shared = true;
            break;
          }
        }
        if (reads_global && writes_shared) {
          pinfo.copy_stage = true;
        }
      }
      pipeline_stage_infos.push_back(std::move(pinfo));
    }

    AnalyzeCopyLastUse(&pipeline_stage_infos);
    EmitImplicitAsyncAnnotations(pipeline_stage_infos, annotations);
  }

  PipelineStageInfo MakePipelineStageInfo(Stmt stmt, int idx) {
    auto collector =
        ffi::make_object<PipelineAccessCollector>(memory_, target_);
    collector->Visit(stmt);
    PipelineStageInfo pinfo;
    pinfo.reads = std::move(collector->GetReads());
    pinfo.writes = std::move(collector->GetWrites());
    auto [scalar_defs, scalar_uses] = ScalarUseDefCollector::Collect(stmt);
    pinfo.scalar_defs = std::move(scalar_defs);
    pinfo.scalar_uses = std::move(scalar_uses);
    pinfo.original_stmt_index = idx;
    pinfo.conditional_execution = MayBeConditionallyExecuted(stmt);
    bool pure_copy_stage =
        collector->GetGlobalCopyPattern() && IsPureCopyStmt(stmt);
    pinfo.copy_stage = pure_copy_stage;
    pinfo.tma_copy = pure_copy_stage && !pinfo.conditional_execution &&
                     collector->GetTmaCopyPattern();
    pinfo.unknown_access = collector->HasUnknownAccess();
    pinfo.unknown_address = collector->HasUnknownAddress();
    if (pinfo.unknown_access && !pinfo.unknown_address) {
      // Attribute imprecision to the affected storage. A store's value may
      // read other tensors; its address must be inspected independently so a
      // known shared-memory write is not tainted by an unrelated index load.
      VisitNativePostOrder(stmt, [&](const ObjectRef &node) {
        auto inspect = [&](const TensorLoad &access, TensorSet *tensors) {
          auto address =
              ffi::make_object<PipelineAccessCollector>(memory_, target_);
          address->Visit(access);
          if (address->HasUnknownAccess())
            tensors->insert(access->source.as_or_throw<TensorVar>());
        };
        if (const auto *store = node.as<TensorStoreObj>()) {
          inspect(TensorLoad(store->dest, store->indices, store->loc),
                  &pinfo.imprecise_writes);
        } else if (const auto *load = node.as<TensorLoadObj>()) {
          inspect(GetRef<TensorLoad>(load), &pinfo.imprecise_reads);
        }
      });
    }
    ClassifyCopyLikeStage(stmt, &pinfo);
    return pinfo;
  }

private:
  const PipelineMemoryAnalysis *memory_{nullptr};
  Target target_{ffi::UnsafeInit{}};
  bool use_async_copy_{};
  Map<Var, sym::IntSet> thread_domains_;
};

class PipelinePlanner : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  explicit PipelinePlanner(bool use_async_copy)
      : use_async_copy_(use_async_copy) {}
  static Stmt Substitute(const tirx::Function &f, bool use_async_copy = true) {
    PipelineMemoryAnalysis memory(f);
    auto substituter = ffi::make_object<PipelinePlanner>(use_async_copy);
    substituter->memory_ = &memory;
    substituter->thread_domains_ = CollectPipelineThreadDomains(f->body);
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.has_value())
        << "Pipeline_Planning: Require the target attribute";
    substituter->target_ = target.value();
    return substituter->Mutate(f->body).ValueOrUnchanged(f->body);
  }

private:
  PipelineStageAnalyzer MakeStageAnalyzer() const {
    return PipelineStageAnalyzer(memory_, target_, use_async_copy_,
                                 thread_domains_);
  }

  void AnalyzeCopyLastUse(
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    MakeStageAnalyzer().AnalyzeCopyLastUse(pipeline_stage_infos);
  }

  bool PropagateProducersForCopy(
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    return MakeStageAnalyzer().PropagateProducersForCopy(pipeline_stage_infos);
  }

  void ValidateScalarDependencies(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos) const {
    MakeStageAnalyzer().ValidateScalarDependencies(pipeline_stage_infos);
  }

  void MaybeAnnotateLegacyAsyncPipelineLoop(const Array<Stmt> &pipeline_stmts,
                                            const Array<int64_t> &order_array,
                                            const Array<int64_t> &stage_array,
                                            Map<String, Any> *annotations) {
    MakeStageAnalyzer().MaybeAnnotateLegacyAsyncPipelineLoop(
        pipeline_stmts, order_array, stage_array, annotations);
  }

  void EmitImplicitAsyncAnnotations(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos,
      Map<String, Any> *annotations) const {
    MakeStageAnalyzer().EmitImplicitAsyncAnnotations(pipeline_stage_infos,
                                                     annotations);
  }

  PipelineStageInfo MakePipelineStageInfo(Stmt stmt, int idx) {
    return MakeStageAnalyzer().MakePipelineStageInfo(std::move(stmt), idx);
  }

  bool HasUnsafeImpreciseAccesses(
      const std::vector<PipelineStageInfo> &stages) const {
    auto may_alias = [&](const TensorSet &tensors,
                         const Array<TensorRegion> &regions) {
      for (const TensorVar &tensor : tensors)
        for (const TensorRegion &region : regions)
          if (memory_->MayAlias(tensor, RegionTensor(region)))
            return true;
      return false;
    };
    for (size_t i = 0; i < stages.size(); ++i) {
      const auto &stage = stages[i];
      if (stage.unknown_address)
        return true;
      if (!stage.unknown_access)
        continue;
      // Retain the fallback if an unknown effect was not attributable to a
      // tensor load/store. Opaque pointer effects cannot be pipelined safely.
      if (stage.imprecise_reads.empty() && stage.imprecise_writes.empty())
        return true;
      for (size_t j = 0; j < stages.size(); ++j) {
        if (i == j)
          continue;
        // The collector's full-axis approximation is not an in-bounds proof.
        // Use storage identities, not those ranges, to retain RAW, WAR and
        // WAW hazards across statements and iterations, including tensor
        // views and potentially aliased function arguments. Each statement's
        // own iteration order remains unchanged by the pipeline.
        if (may_alias(stage.imprecise_writes, stages[j].reads) ||
            may_alias(stage.imprecise_writes, stages[j].writes) ||
            may_alias(stage.imprecise_reads, stages[j].writes))
          return true;
      }
    }
    return false;
  }

  using ScheduledStmtAnalysis =
      PipelinePlanningBodyAnalyzer::ScheduledStmtAnalysis;
  using SeqStmtFlattener = PipelinePlanningBodyAnalyzer::SeqStmtFlattener;

  PipelinePlanningBodyAnalyzer MakeBodyAnalyzer() const {
    return PipelinePlanningBodyAnalyzer(memory_, target_);
  }

  ScheduledStmtAnalysis AnalyzeScheduledStmts(const Array<Stmt> &stmts) const {
    return MakeBodyAnalyzer().AnalyzeScheduledStmts(stmts);
  }

  Array<int64_t> FilterAnnotationsForScheduledStmts(
      const Array<int64_t> &annotations,
      const ScheduledStmtAnalysis &analysis) const {
    return MakeBodyAnalyzer().FilterAnnotationsForScheduledStmts(annotations,
                                                                 analysis);
  }

  UnchangedOr<Stmt> Mutate_(const ForObj *loop, InplaceMode mode) final {
    auto order_anno = ReadLoopAnnotations(loop).Get("tl_pipeline_order");
    auto stage_anno = ReadLoopAnnotations(loop).Get("tl_pipeline_stage");
    auto num_stages_anno = ReadLoopAnnotations(loop).Get("num_stages");
    if (order_anno && stage_anno) {
      auto order_array = ReadIntegerArray(order_anno.value());
      auto stage_array = ReadIntegerArray(stage_anno.value());

      Map<String, Any> annotations;
      for (const auto &[key, value] : loop->annotations) {
        if (key != "tl_pipeline_order" && key != "tl_pipeline_stage") {
          annotations.Set(key, value);
        }
      }
      if (TargetHasAsyncCopy(target_) && use_async_copy_) {
        // Legacy explicit stage/order annotations do not carry per-statement
        // async producer metadata yet, so keep the previous stage-level
        // behavior as a fallback for these loops.
        annotations.Set("software_pipeline_async_stages", Array<int64_t>{0});
      }
      Array<Stmt> pipeline_body_stmts = NormalizePipelineBody(loop->body);
      Array<Stmt> pipeline_stmts =
          SeqStmtFlattener::Flatten(pipeline_body_stmts);
      ScheduledStmtAnalysis analysis = AnalyzeScheduledStmts(pipeline_stmts);
      for (const Stmt &stmt : analysis.scheduled_stmts) {
        TVM_FFI_CHECK(!MakePipelineStageInfo(stmt, 0).unknown_address,
                      ValueError)
            << "PipelinePlanning: a manually scheduled pointer access needs "
               "an explicit tensor storage identity";
      }
      ICHECK(!analysis.scheduled_stmts.empty())
          << "PipelinePlanning: explicit pipeline annotations have no "
             "schedulable statements after removing replayable scalar Bind "
             "statements";
      Array<int64_t> filtered_order_array =
          FilterAnnotationsForScheduledStmts(order_array, analysis);
      Array<int64_t> filtered_stage_array =
          FilterAnnotationsForScheduledStmts(stage_array, analysis);
      annotations.Set("software_pipeline_order", filtered_order_array);
      annotations.Set("software_pipeline_stage", filtered_stage_array);
      if (pipeline_stmts.size() == pipeline_body_stmts.size()) {
        bool flatten_preserved_original_order = true;
        for (size_t i = 0; i < pipeline_stmts.size(); ++i) {
          if (!pipeline_stmts[i].same_as(pipeline_body_stmts[i])) {
            flatten_preserved_original_order = false;
            break;
          }
        }
        if (flatten_preserved_original_order &&
            std::any_of(analysis.replayable_bind_mask.begin(),
                        analysis.replayable_bind_mask.end(),
                        [](int64_t value) { return value != 0; })) {
          annotations.Set(kPipelineReplayableScalarBinds,
                          analysis.replayable_bind_mask);
        }
      }
      MaybeAnnotateLegacyAsyncPipelineLoop(analysis.scheduled_stmts,
                                           filtered_order_array,
                                           filtered_stage_array, &annotations);
      auto for_node = GetRef<For>(loop);
      auto *n = for_node.CopyOnWrite();
      n->annotations = annotations;
      n->body = MakePipelineBody(pipeline_body_stmts).as_or_throw<SeqStmt>();
      return for_node;
    }

    if (!num_stages_anno)
      return StmtExprMutator::Mutate_(loop, mode);
    int num_stages = GetIntegerAnnotation(*num_stages_anno);
    // Skip software pipelining on ROCm targets where async-copy pipelining
    // has not been validated.  Currently only gfx950 (CDNA4 / MI350) supports
    // the full HIP async-copy pipeline path.  gfx942 (CDNA3 / MI300X) has
    // async-copy hardware but the software pipeline for that target has not
    // been validated yet, so it falls back to a plain sequential loop as well.
    // RDNA targets have no async-copy support at all and also fall back.
    if (TargetIsRocm(target_) && !TargetIsGfx950(target_) && num_stages >= 1) {
      // Strip the "num_stages" annotation before recursing so that downstream
      // passes (InjectSoftwarePipeline, MultiVersionBufferRewriter, etc.) do
      // not treat this loop as pipelined.  Leaving the annotation in place
      // would cause those passes to multi-version shared buffers and inject
      // cp.async / barrier code that is incompatible with the plain sequential
      // execution path chosen here.
      auto stripped = GetRef<For>(loop);
      Map<String, Any> annotations;
      for (const auto &[key, value] : loop->annotations) {
        if (key != "num_stages") {
          annotations.Set(key, value);
        }
      }
      stripped.CopyOnWrite()->annotations = annotations;
      return StmtExprMutator::Mutate_(stripped.get(), InplaceMode::kDisallow)
          .ValueOrUnchanged(stripped);
    }
    Array<Stmt> pipeline_body_stmts = NormalizePipelineBody(loop->body);

    ICHECK(num_stages >= 1);
    ICHECK(loop->kind == ForKind::kDefault);

    // Flatten nested SeqStmts so pipeline planning can assign stages to the
    // normalized top-level statement list.
    Array<Stmt> flat_stmts = SeqStmtFlattener::Flatten(pipeline_body_stmts);
    ScheduledStmtAnalysis analysis = AnalyzeScheduledStmts(flat_stmts);
    ICHECK(!analysis.scheduled_stmts.empty())
        << "PipelinePlanning: loop has no schedulable statements after "
           "removing replayable scalar Bind statements";

    std::vector<PipelineStageInfo> pipeline_stage_infos;
    for (size_t i = 0; i < analysis.scheduled_stmts.size(); i++) {
      auto pinfo = MakePipelineStageInfo(analysis.scheduled_stmts[i], i);
      pipeline_stage_infos.push_back(std::move(pinfo));
    }

    if (HasUnsafeImpreciseAccesses(pipeline_stage_infos)) {
      For sequential = GetRef<For>(loop);
      sequential.CopyOnWrite()->annotations.erase("num_stages");
      return StmtExprMutator::Mutate_(sequential.get(), InplaceMode::kDisallow)
          .ValueOrUnchanged(sequential);
    }

    // Determine each real copy's scheduling anchor before extending the
    // producer group. Non-copy producers inherit this anchor rather than
    // replacing it with their direct consumer.
    AnalyzeCopyLastUse(&pipeline_stage_infos);

    // Follow scalar use-def and buffer RAW edges in one backward worklist.
    // This reaches mixed chains such as buffer store -> Bind -> copy and keeps
    // every transitive producer in the same first-stage scheduling group.
    if (!PropagateProducersForCopy(&pipeline_stage_infos)) {
      For sequential = GetRef<For>(loop);
      sequential.CopyOnWrite()->annotations.erase("num_stages");
      return StmtExprMutator::Mutate_(sequential.get(), InplaceMode::kDisallow)
          .ValueOrUnchanged(sequential);
    }

    // Making stages and orders
    int order_idx = 0;
    // Stage 1. Create pipeline stages and assign order
    for (auto &pinfo : pipeline_stage_infos) {
      // Skip elements that must be in first stage:
      // 1. Copy stages (with active last_use_stmt_index) - these need special
      // handling
      //    because they have consumers that depend on their data
      // 2. All Producer stages for copy stages.
      if (pinfo.IsFirstStage() && pinfo.IsLastUseStmtIndexValid()) {
        continue;
      }

      // Main logic stage assignment:
      // - Increment order index
      // - Assign to new stage (current num_stages)
      pinfo.order = order_idx++;
      pinfo.stage = num_stages;

      // Schedule copy stages that have this stage as their last consumer
      // This ensures copy operations are placed right before their final
      // consumer for optimal pipeline efficiency
      for (auto &pinfo_1 : pipeline_stage_infos) {
        if ((pinfo_1.IsFirstStage() &&
             pinfo_1.last_use_stmt_index == pinfo.original_stmt_index)) {
          pinfo_1.order = order_idx++;
          pinfo_1.stage = 0; // Copy stages are typically assigned to stage 0
        }
      }
    }

    ICHECK(size_t(order_idx) == pipeline_stage_infos.size())
        << "The number of stages should be equal to the number of pipeline "
           "stages. "
        << "Got " << order_idx << " stages and " << pipeline_stage_infos.size()
        << " pipeline stages.";

    // Step 2. if all the copy is at the end of the order, we can move these
    // copy to the beginning of the order and shrink the stage offset by 1.
    int copy_stage_at_end = [&]() {
      int copy_stage_cnt = 0;
      int copy_order_min = pipeline_stage_infos.size();
      int non_copy_order_max = 0;
      for (auto &pinfo : pipeline_stage_infos) {
        if (pinfo.IsFirstStage()) {
          copy_stage_cnt++;
          copy_order_min = std::min(copy_order_min, pinfo.order);
        } else {
          non_copy_order_max = std::max(non_copy_order_max, pinfo.order);
        }
      }
      if (copy_order_min > non_copy_order_max)
        return copy_stage_cnt;
      return -1;
    }();
    if (copy_stage_at_end > 0 && num_stages >= 2) {
      for (auto &pinfo : pipeline_stage_infos) { // move copy to the beginning
        pinfo.order =
            (pinfo.order + copy_stage_at_end) % pipeline_stage_infos.size();
        if (!pinfo.IsCopyStage() && !pinfo.IsProducerForCopy())
          pinfo.stage--;
      }
    }

    ValidateScalarDependencies(pipeline_stage_infos);

    // Finally, make the pipeline annotation
    Map<String, Any> annotations;
    for (const auto &[key, value] : loop->annotations) {
      if (key != "num_stages") {
        annotations.Set(key, value);
      }
    }
    // Preserve the original TileLang pipelining depth for downstream scheduling
    // (e.g. generated async-copy wait placement). We intentionally do NOT
    // keep the legacy key "num_stages" here because multiple downstream passes
    // (e.g. internal buffer versioning / warp specialization) treat it as an
    // active pipeline marker and do not support nested pipelines.
    annotations.Set("tl_pipelined_num_stages", int64_t(num_stages));

    std::vector<int64_t> orders, stages;
    orders.reserve(pipeline_stage_infos.size());
    stages.reserve(pipeline_stage_infos.size());
    for (auto &pinfo : pipeline_stage_infos) {
      orders.push_back(pinfo.order);
      stages.push_back(pinfo.stage);
    }

    annotations.Set("software_pipeline_stage", Array<int64_t>(stages));
    annotations.Set("software_pipeline_order", Array<int64_t>(orders));
    if (std::any_of(analysis.replayable_bind_mask.begin(),
                    analysis.replayable_bind_mask.end(),
                    [](int64_t value) { return value != 0; })) {
      annotations.Set(kPipelineReplayableScalarBinds,
                      analysis.replayable_bind_mask);
    }

    // Propagate per-statement TMA eligibility so InjectSoftwarePipeline can
    // rewrite TMA copies to use pipeline-level barrier management.
    {
      std::vector<int64_t> tma_copies;
      tma_copies.reserve(pipeline_stage_infos.size());
      bool has_tma_copy = false;
      for (auto &pinfo : pipeline_stage_infos) {
        bool IsTmaCopy = pinfo.IsTmaCopy();
        has_tma_copy = has_tma_copy || IsTmaCopy;
        tma_copies.push_back(int64_t(IsTmaCopy ? 1 : 0));
      }
      if (has_tma_copy) {
        annotations.Set(kPipelineTmaCopies, Array<int64_t>(tma_copies));
      }
    }

    EmitImplicitAsyncAnnotations(pipeline_stage_infos, &annotations);

    // Reconstruct the loop body with the flattened SeqStmt so that
    // InjectSoftwarePipeline sees the correct number of pipeline stages.
    Stmt new_body = MakePipelineBody(flat_stmts);

    return For(loop->var, loop->begin, loop->end, loop->step, loop->kind,
               new_body.as_or_throw<SeqStmt>(), annotations, loop->loc);
  }

  const PipelineMemoryAnalysis *memory_{nullptr};
  Target target_{ffi::UnsafeInit{}};
  bool use_async_copy_{};
  Map<Var, sym::IntSet> thread_domains_;
};

tvm::transform::Pass PipelinePlanning() {
  using namespace tirx::transform;
  auto pass_func = [=](tirx::Function f, const IRModule &m, PassContext ctx) {
    if (f->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return f;
    bool use_async_copy =
        ctx->GetConfig<bool>("tirx.use_async_copy").value_or(true);
    tirx::FunctionObj *fptr = f.CopyOnWrite();
    fptr->body =
        PipelinePlanner::Substitute(f, use_async_copy).as_or_throw<SeqStmt>();
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tl.PipelinePlanning");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.PipelinePlanning", PipelinePlanning);
}

} // namespace tl
} // namespace tvm

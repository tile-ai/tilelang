#ifndef TVM_TL_TRANSFORM_PIPELINE_STAGE_ANALYSIS_H_
#define TVM_TL_TRANSFORM_PIPELINE_STAGE_ANALYSIS_H_

#include "access_analysis.h"

#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <map>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../common/pipeline_utils.h"
#include "backend/common/target_utils.h"
#include "op/builtin.h"
#include "op/copy.h"
#include "op/parallel.h"
#include "op/region.h"
#include "op/utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

/*! \brief Information about a pipeline stage
 *
 * \param reads Array of buffer regions read by this stage
 * \param writes Array of buffer regions written by this stage
 * \param original_stmt_index Original position of this stage in the pipeline
 * before reordering \param order Current position of this stage in the
 * pipeline after reordering (-1 if not yet assigned) \param stage Pipeline
 * stage number this operation belongs to (-1 if not yet assigned) \param
 * copy_stage Whether this stage is a memory copy operation \param
 * last_use_stmt_index Index of the last statement (in original order) that
 * uses the results of this stage (-1 if not yet determined). This field is
 * crucial for pipeline optimization:
 * - For copy stages: indicates the index of the last statement that reads
 * from the copied data, helping determine optimal placement of copy
 * operations
 * - Used to ensure copy operations are scheduled before their consumers
 * - A value of -1 means no subsequent statement uses this stage's output
 * - This information enables better pipeline scheduling by minimizing data
 *   dependencies and maximizing parallelism
 */
struct PipelineStageInfo {
  Array<BufferRegion> reads, writes;
  std::unordered_set<const VarNode *> scalar_defs;
  std::unordered_set<const VarNode *> scalar_uses;
  int original_stmt_index{};
  int order = -1, stage = -1;
  bool copy_stage = false;
  bool tma_copy = false; // true if this copy stage uses TMA (not cp.async)
  bool conditional_execution = false;
  bool producer_for_copy = false;
  int last_use_stmt_index =
      -1; // Initialized to -1, indicating no consumers found yet

public:
  bool is_first_stage() const { return copy_stage || producer_for_copy; }
  bool is_copy_stage() const { return copy_stage; }
  bool is_tma_copy() const { return tma_copy; }
  bool is_producer_for_copy() const { return producer_for_copy; }
  bool is_last_use_stmt_index_valid() const {
    return last_use_stmt_index != -1;
  }
};

class PipelineStageAnalyzer {
public:
  PipelineStageAnalyzer(Map<Var, Buffer> buffer_data_to_buffer, Target target,
                        bool use_async_copy)
      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)),
        target_(std::move(target)), use_async_copy_(use_async_copy) {}

  class ScalarUseDefCollector : public StmtExprVisitor {
  public:
    static std::pair<std::unordered_set<const VarNode *>,
                     std::unordered_set<const VarNode *>>
    Collect(const Stmt &stmt) {
      ScalarUseDefCollector collector;
      collector(stmt);
      return {std::move(collector.scalar_defs_),
              std::move(collector.scalar_uses_)};
    }

  private:
    void VisitStmt_(const BindNode *op) final {
      this->VisitExpr(op->value);
      scalar_defs_.insert(op->var.get());
    }

    void VisitExpr_(const VarNode *op) final { scalar_uses_.insert(op); }

    std::unordered_set<const VarNode *> scalar_defs_;
    std::unordered_set<const VarNode *> scalar_uses_;
  };

  bool MayBeConditionallyExecuted(const Stmt &stmt) const {
    bool conditional = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (conditional) {
        return;
      }
      if (const auto *if_then_else = node.as<IfThenElseNode>()) {
        conditional = true;
        return;
      }
      if (const auto *realize = node.as<SBlockRealizeNode>()) {
        if (!is_one(realize->predicate)) {
          conditional = true;
        }
      }
    });
    return conditional;
  }

  bool IsAsyncProducerCandidate(const PipelineStageInfo &pinfo) const {
    if (pinfo.conditional_execution) {
      return false;
    }
    if (pinfo.is_tma_copy()) {
      return false;
    }
    return pinfo.is_copy_stage();
  }

  bool IsPureCopyStmt(const Stmt &stmt) const {
    auto is_global_like_buffer = [](const Buffer &buffer) {
      return IsGlobalBuffer(buffer) ||
             (buffer.defined() && buffer.scope().empty());
    };
    auto is_pure_raw_copy_value = [&](const PrimExpr &expr,
                                      const auto &self) -> bool {
      if (const auto *load = expr.as<BufferLoadNode>()) {
        return is_global_like_buffer(load->buffer);
      }
      if (const auto *cast = expr.as<CastNode>()) {
        return self(cast->value, self);
      }
      return false;
    };

    bool saw_copy = false;
    bool saw_non_copy_tile_op = false;
    bool saw_non_copy_buffer_store = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (saw_non_copy_tile_op || saw_non_copy_buffer_store) {
        return;
      }
      if (const auto *store = node.as<BufferStoreNode>()) {
        saw_copy = true;
        if ((!IsSharedBuffer(store->buffer) &&
             !IsLocalBuffer(store->buffer, /*allow_var=*/true)) ||
            !is_pure_raw_copy_value(store->value, is_pure_raw_copy_value)) {
          saw_non_copy_buffer_store = true;
        }
        return;
      }
      const auto *call = node.as<CallNode>();
      if (call == nullptr) {
        return;
      }
      auto tile_op = ParseOperator(GetRef<Call>(call));
      if (!tile_op.defined()) {
        return;
      }
      if (tile_op.as<RegionOpNode>()) {
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
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (saw_non_copy_tile_op || saw_multiple_copy_ops) {
        return;
      }
      const auto *call = node.as<CallNode>();
      if (call == nullptr) {
        return;
      }
      auto tile_op = ParseOperator(GetRef<Call>(call));
      if (!tile_op.defined()) {
        return;
      }
      if (tile_op.as<RegionOpNode>()) {
        return;
      }
      if (tile_op.as<CopyNode>() || tile_op.as<Im2ColOpNode>()) {
        if (copy_tile_op.defined()) {
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

  static bool IsGlobalLikeBuffer(const Buffer &buffer) {
    return IsGlobalBuffer(buffer) ||
           (buffer.defined() && buffer.scope().empty());
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
    if (!copy_tile_op.defined()) {
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
      if (!pinfo.is_first_stage()) {
        continue;
      }

      for (int i = pinfo.original_stmt_index + 1;
           i < static_cast<int>(pipeline_stage_infos->size()); ++i) {
        for (const BufferRegion &read : (*pipeline_stage_infos)[i].reads) {
          if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),
                           [&](const BufferRegion &r) {
                             return r->buffer == read->buffer &&
                                    MayConflict(r->region, read->region);
                           }) != pinfo.writes.end()) {
            pinfo.last_use_stmt_index = std::max(pinfo.last_use_stmt_index, i);
          }
        }

        if (!pinfo.is_copy_stage()) {
          continue;
        }

        for (const BufferRegion &write : (*pipeline_stage_infos)[i].writes) {
          if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),
                           [&](const BufferRegion &r) {
                             return r->buffer == write->buffer &&
                                    MayConflict(r->region, write->region);
                           }) != pinfo.writes.end()) {
            LOG(FATAL) << "Pipeline planning error: Multiple writes to "
                          "overlapping buffer regions detected. "
                       << "Stage " << pinfo.original_stmt_index << " and stage "
                       << i << " are both writing to buffer '"
                       << write->buffer->name
                       << "' with overlapping regions. This is not supported "
                          "in pipeline planning.";
          }
        }
      }
    }
  }

  void PropagateBufferProducersForCopy(
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    struct CopyStageDependencyReadsManager {
      std::vector<BufferRegion> regions;

      void AddUnique(const BufferRegion &region) {
        for (const BufferRegion &copy_read : regions) {
          if (region->buffer.same_as(copy_read->buffer)) {
            return;
          }
        }
        regions.push_back(region);
      }

      bool Contains(const BufferRegion &region) const {
        for (const BufferRegion &copy_read : regions) {
          if (region->buffer.same_as(copy_read->buffer)) {
            return true;
          }
        }
        return false;
      }

      size_t Size() const { return regions.size(); }
    };

    CopyStageDependencyReadsManager copy_stage_dependency_reads_mgr;

    for (const auto &pinfo : *pipeline_stage_infos) {
      if (pinfo.is_copy_stage()) {
        for (const BufferRegion &read : pinfo.reads) {
          copy_stage_dependency_reads_mgr.AddUnique(read);
        }
      }
    }

    const size_t max_iterations = (pipeline_stage_infos->size() * 4) + 16;
    size_t iter_count = 0;

    for (auto &pinfo : *pipeline_stage_infos) {
      if (!pinfo.is_copy_stage()) {
        continue;
      }
      auto original_copy_stmt_index = pinfo.original_stmt_index;
      bool updated = true;
      while (updated) {
        updated = false;
        for (auto &pinfo_inner : *pipeline_stage_infos) {
          if (pinfo_inner.is_copy_stage()) {
            continue;
          }
          if (pinfo_inner.original_stmt_index >= original_copy_stmt_index) {
            break;
          }

          bool should_prepare = false;
          for (const BufferRegion &write : pinfo_inner.writes) {
            if (copy_stage_dependency_reads_mgr.Contains(write)) {
              should_prepare = true;
              break;
            }
          }
          if (should_prepare && !pinfo_inner.is_producer_for_copy()) {
            pinfo_inner.producer_for_copy = true;
            updated = true;
          }
          if (should_prepare) {
            for (const BufferRegion &read : pinfo_inner.reads) {
              size_t before = copy_stage_dependency_reads_mgr.Size();
              copy_stage_dependency_reads_mgr.AddUnique(read);
              if (copy_stage_dependency_reads_mgr.Size() > before) {
                updated = true;
              }
            }
          }
        }
        iter_count++;
        if (iter_count > max_iterations) {
          LOG(FATAL)
              << "Pipeline planning: Exceeded maximum iterations ("
              << max_iterations << ") in copy stage dependency propagation. "
              << "This may indicate a cyclic or pathological dependency graph.";
        }
      }
    }
  }

  std::unordered_map<const VarNode *, int> BuildScalarDefMap(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos) const {
    std::unordered_map<const VarNode *, int> scalar_def_to_stmt;
    for (int i = 0; i < static_cast<int>(pipeline_stage_infos.size()); ++i) {
      for (const VarNode *var : pipeline_stage_infos[i].scalar_defs) {
        scalar_def_to_stmt.emplace(var, i);
      }
    }
    return scalar_def_to_stmt;
  }

  void PropagateScalarProducersForCopy(
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    auto scalar_def_to_stmt = BuildScalarDefMap(*pipeline_stage_infos);
    const size_t max_iterations = (pipeline_stage_infos->size() * 4) + 16;
    size_t iter_count = 0;
    bool updated = true;

    auto update_producer = [](PipelineStageInfo *producer,
                              int consumer_last_use) -> bool {
      if (consumer_last_use < 0) {
        return false;
      }
      bool changed = false;
      if (!producer->producer_for_copy) {
        producer->producer_for_copy = true;
        producer->last_use_stmt_index = consumer_last_use;
        changed = true;
      } else if (!producer->is_last_use_stmt_index_valid() ||
                 consumer_last_use < producer->last_use_stmt_index) {
        producer->last_use_stmt_index = consumer_last_use;
        changed = true;
      }
      return changed;
    };

    while (updated) {
      updated = false;
      for (int consumer_idx = 0;
           consumer_idx < static_cast<int>(pipeline_stage_infos->size());
           ++consumer_idx) {
        const auto &consumer = (*pipeline_stage_infos)[consumer_idx];
        if (!(consumer.is_first_stage() &&
              consumer.is_last_use_stmt_index_valid())) {
          continue;
        }
        for (const VarNode *var : consumer.scalar_uses) {
          auto it = scalar_def_to_stmt.find(var);
          if (it == scalar_def_to_stmt.end() || it->second == consumer_idx) {
            continue;
          }
          auto &producer = (*pipeline_stage_infos)[it->second];
          if (producer.is_copy_stage()) {
            continue;
          }
          updated |= update_producer(&producer, consumer.last_use_stmt_index);
        }
      }
      if (++iter_count > max_iterations) {
        LOG(FATAL) << "Pipeline planning: Exceeded maximum iterations while "
                      "propagating scalar producers for copy stages.";
      }
    }
  }

  void ValidateScalarDependencies(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos) const {
    auto scalar_def_to_stmt = BuildScalarDefMap(pipeline_stage_infos);
    for (int consumer_idx = 0;
         consumer_idx < static_cast<int>(pipeline_stage_infos.size());
         ++consumer_idx) {
      const auto &consumer = pipeline_stage_infos[consumer_idx];
      for (const VarNode *var : consumer.scalar_uses) {
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
      auto key = std::make_pair(pinfo.stage, pinfo.last_use_stmt_index);
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

    std::vector<Integer> async_producers;
    std::vector<Integer> async_producer_groups;
    async_producers.reserve(pipeline_stage_infos.size());
    async_producer_groups.reserve(pipeline_stage_infos.size());
    std::unordered_set<int> async_stage_ids;
    for (size_t i = 0; i < pipeline_stage_infos.size(); ++i) {
      bool is_async_producer = async_group_ids[i] != -1;
      async_producers.push_back(Integer(is_async_producer ? 1 : 0));
      async_producer_groups.push_back(Integer(async_group_ids[i]));
      if (is_async_producer) {
        async_stage_ids.insert(pipeline_stage_infos[i].stage);
      }
    }

    annotations->Set(kPipelineAsyncProducers, Array<Integer>(async_producers));
    annotations->Set(kPipelineAsyncProducerGroups,
                     Array<Integer>(async_producer_groups));

    std::vector<int> sorted_async_stage_ids(async_stage_ids.begin(),
                                            async_stage_ids.end());
    std::sort(sorted_async_stage_ids.begin(), sorted_async_stage_ids.end());
    std::vector<Integer> async_stages;
    async_stages.reserve(sorted_async_stage_ids.size());
    for (int stage_id : sorted_async_stage_ids) {
      async_stages.push_back(Integer(stage_id));
    }
    annotations->Set(s_tir::attr::software_pipeline_async_stages,
                     Array<Integer>(async_stages));
    return true;
  }

  void MaybeAnnotateLegacyAsyncPipelineLoop(const Array<Stmt> &pipeline_stmts,
                                            const Array<Integer> &order_array,
                                            const Array<Integer> &stage_array,
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
      pinfo.order = static_cast<int>(order_array[i]->value);
      pinfo.stage = static_cast<int>(stage_array[i]->value);
      if (!pinfo.is_copy_stage() && !pinfo.conditional_execution &&
          pinfo.stage == 0) {
        bool reads_global = false;
        bool writes_shared = false;
        for (const BufferRegion &read : pinfo.reads) {
          if (IsGlobalLikeBuffer(read->buffer)) {
            reads_global = true;
            break;
          }
        }
        for (const BufferRegion &write : pinfo.writes) {
          if (IsSharedBuffer(write->buffer)) {
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
    SBlock block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
                 /*name_hint=*/"",
                 /*body*/ std::move(stmt));
    auto collector = BufferRegionCollector(buffer_data_to_buffer_, target_);
    collector(block);
    PipelineStageInfo pinfo;
    pinfo.reads = std::move(collector.GetReads());
    pinfo.writes = std::move(collector.GetWrites());
    auto [scalar_defs, scalar_uses] =
        ScalarUseDefCollector::Collect(block->body);
    pinfo.scalar_defs = std::move(scalar_defs);
    pinfo.scalar_uses = std::move(scalar_uses);
    pinfo.original_stmt_index = idx;
    pinfo.conditional_execution = MayBeConditionallyExecuted(block->body);
    bool pure_copy_stage =
        collector.GetGlobalCopyPattern() && IsPureCopyStmt(block->body);
    pinfo.copy_stage = pure_copy_stage;
    pinfo.tma_copy = pure_copy_stage && !pinfo.conditional_execution &&
                     collector.GetTmaCopyPattern();
    ClassifyCopyLikeStage(block->body, &pinfo);
    return pinfo;
  }

private:
  Map<Var, Buffer> buffer_data_to_buffer_;
  Target target_;
  bool use_async_copy_{};
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_PIPELINE_STAGE_ANALYSIS_H_

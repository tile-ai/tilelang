#include "support/check.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/target/target.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/operator.h"
#include "../op/parallel.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "backend/common/target_utils.h"
#include "common/bind_utils.h"
#include "common/pipeline_utils.h"
#include "tvm/ir/expr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class BufferRegionCollector : public StmtExprVisitor {
public:
  BufferRegionCollector(Map<Var, Buffer> buffer_data_to_buffer, Target target);

  Array<BufferRegion> GetReads() const;
  Array<BufferRegion> GetWrites() const;
  bool GetGlobalCopyPattern() const;
  bool GetTmaCopyPattern() const;
  bool HasNonCopyTileOp() const;

private:
  static bool IsGlobalLikeBuffer(const Buffer &buffer);

  void HandleTileOp(const TileOperator &tile_op);
  void VisitStmt_(const BufferStoreNode *op) final;
  void VisitExpr_(const BufferLoadNode *op) final;
  void VisitExpr_(const CallNode *op) final;
  void VisitStmt_(const IfThenElseNode *op) final;

  Map<Var, Buffer> buffer_data_to_buffer_;
  Target target_;
  Array<BufferRegion> reads_;
  Array<BufferRegion> writes_;
  bool is_global_read_ = false;
  bool is_global_copy_pattern_ = false;
  bool is_tma_copy_ = false;
  bool has_non_copy_tile_op_ = false;
  bool within_condition_expr_ = false;
};

/*!
 * \brief Check whether two regions have intersections.
 * \param region1 The first region.
 * \param region2 The second region.
 * \return Whether region1 and region2 have intersections.
 */
bool MayConflict(const Region &region1, const Region &region2) {
  ICHECK(region1.size() == region2.size());
  for (size_t i = 0; i < region1.size(); i++) {
    Range dim1 = region1[i];
    Range dim2 = region2[i];
    auto int_set1 = arith::IntSet::FromRange(dim1);
    auto int_set2 = arith::IntSet::FromRange(dim2);
    if (arith::Intersect({int_set1, int_set2}).IsNothing()) {
      return false;
    }
  }
  return true;
}

BufferRegionCollector::BufferRegionCollector(
    Map<Var, Buffer> buffer_data_to_buffer, Target target)
    : buffer_data_to_buffer_(buffer_data_to_buffer), target_(target) {}

Array<BufferRegion> BufferRegionCollector::GetReads() const { return reads_; }

Array<BufferRegion> BufferRegionCollector::GetWrites() const { return writes_; }

bool BufferRegionCollector::GetGlobalCopyPattern() const {
  return is_global_copy_pattern_;
}

bool BufferRegionCollector::GetTmaCopyPattern() const { return is_tma_copy_; }

bool BufferRegionCollector::HasNonCopyTileOp() const {
  return has_non_copy_tile_op_;
}

bool BufferRegionCollector::IsGlobalLikeBuffer(const Buffer &buffer) {
  return IsGlobalBuffer(buffer) || (buffer.defined() && buffer.scope().empty());
}

void BufferRegionCollector::HandleTileOp(const TileOperator &tile_op) {
  if (tile_op.as<RegionOpNode>()) {
    return;
  }
  if (const auto *parallel = tile_op.as<ParallelOpNode>()) {
    BufferRegionCollector nested(buffer_data_to_buffer_, target_);
    nested(parallel->GetRoot());
    reads_.insert(reads_.end(), nested.GetReads().begin(),
                  nested.GetReads().end());
    writes_.insert(writes_.end(), nested.GetWrites().begin(),
                   nested.GetWrites().end());
    is_global_copy_pattern_ =
        is_global_copy_pattern_ || nested.GetGlobalCopyPattern();
    is_tma_copy_ = is_tma_copy_ || nested.GetTmaCopyPattern();
    has_non_copy_tile_op_ = has_non_copy_tile_op_ || nested.HasNonCopyTileOp();
    return;
  }
  AccessRegions access = tile_op->GetAccessRegions();
  reads_.insert(reads_.end(), access.reads.begin(), access.reads.end());
  writes_.insert(writes_.end(), access.writes.begin(), access.writes.end());
  if (const auto *copy = tile_op.as<CopyNode>()) {
    if (IsGlobalLikeBuffer(copy->src) && IsSharedBuffer(copy->dst)) {
      is_global_copy_pattern_ = true;
    }
  }
  // Im2Col always uses TMA on Hopper.
  if (const auto *im2col = tile_op.as<Im2ColOpNode>()) {
    if (IsGlobalLikeBuffer(im2col->src_) && IsSharedBuffer(im2col->dst_)) {
      is_global_copy_pattern_ = true;
      if (TargetIsHopper(target_)) {
        is_tma_copy_ = true;
      }
    }
    return;
  }
  if (!tile_op.as<CopyNode>()) {
    has_non_copy_tile_op_ = true;
  }
}

void BufferRegionCollector::VisitStmt_(const BufferStoreNode *op) {
  Buffer store_buffer = op->buffer;
  Array<PrimExpr> indices = op->indices;
  // convert indices to region
  Array<Range> region;
  for (const auto &index : indices) {
    region.push_back(Range::FromMinExtent(index, 1));
  }
  auto store_region = BufferRegion(store_buffer, region);
  writes_.push_back(store_region);

  is_global_read_ = false;
  this->VisitExpr(op->value);
  if (is_global_read_ && IsSharedBuffer(store_buffer)) {
    is_global_copy_pattern_ = true;
  }
  is_global_read_ = false;
}

void BufferRegionCollector::VisitExpr_(const BufferLoadNode *op) {
  auto load_buffer = op->buffer;
  Array<PrimExpr> indices = op->indices;
  // convert indices to region
  Array<Range> region;
  for (const auto &index : indices) {
    region.push_back(Range::FromMinExtent(index, 1));
  }
  auto load_region = BufferRegion(load_buffer, region);
  reads_.push_back(load_region);

  if (IsGlobalLikeBuffer(op->buffer) && !within_condition_expr_) {
    // skip condition expr of if_then_else node
    // shared[i] = T.if_then_else(global[i] < n, register_a[i], register_b[i])
    // is not a global read shared[i] = T.if_then_else(global[i] < n,
    // global_a[i], global_b[i]) is a global read
    is_global_read_ = true;
  }
}

void BufferRegionCollector::VisitExpr_(const CallNode *op) {
  if (auto tile_op = ParseOperator(GetRef<Call>(op)); tile_op.defined()) {
    HandleTileOp(tile_op);
    StmtExprVisitor::VisitExpr_(op);
    return;
  }
  if (op->op.same_as(builtin::address_of())) {
    BufferRegion buffer_region;
    if (const auto *load = op->args[0].as<BufferLoadNode>()) {
      buffer_region = BufferRegion::FullRegion(load->buffer);
    } else if (const auto *var_node = op->args[0].as<VarNode>()) {
      Var data_var = GetRef<Var>(var_node);
      auto it = buffer_data_to_buffer_.find(data_var);
      if (it != buffer_data_to_buffer_.end()) {
        buffer_region = BufferRegion::FullRegion((*it).second);
      }
    }
    if (buffer_region.defined()) {
      // because we only care about the buffer itself instead of indices
      reads_.push_back(buffer_region);
    }
  } else if (op->op.same_as(tl::access_ptr())) {
    ICHECK_EQ(op->args.size(), 3U);
    const auto *load = op->args[0].as<BufferLoadNode>();
    ICHECK(load) << "tl.access_ptr base must be a BufferLoad";
    const BufferRegion buffer_region = BufferRegion::FullRegion(load->buffer);
    const int access_mask = GetConservativeAccessMask(op->args[2]);
    // because we only care about the buffer itself instead of indices
    if (access_mask & kAccessRead) {
      reads_.push_back(buffer_region);
    }
    if (access_mask & kAccessWrite) {
      writes_.push_back(buffer_region);
    }
    for (const PrimExpr &index : load->indices) {
      this->VisitExpr(index);
    }
    if (load->predicate.defined()) {
      this->VisitExpr(load->predicate.value());
    }
    this->VisitExpr(op->args[1]);
    this->VisitExpr(op->args[2]);
  } else if (op->op.same_as(builtin::tvm_access_ptr())) {
    const VarNode *buffer_var = op->args[1].as<VarNode>();
    ICHECK(buffer_var);
    auto it = buffer_data_to_buffer_.find(GetRef<Var>(buffer_var));
    if (it != buffer_data_to_buffer_.end()) {
      const Buffer &buffer = (*it).second;
      const BufferRegion buffer_region = BufferRegion::FullRegion(buffer);
      const int access_mask = op->args.size() == 5U
                                  ? GetConservativeAccessMask(op->args[4])
                                  : kAccessReadWrite;
      // because we only care about the buffer itself instead of indices
      if (access_mask & kAccessRead) {
        reads_.push_back(buffer_region);
      }
      if (access_mask & kAccessWrite) {
        writes_.push_back(buffer_region);
      }
    }
    for (size_t i = 2; i < op->args.size(); ++i) {
      this->VisitExpr(op->args[i]);
    }
  } else if (op->op.same_as(builtin::if_then_else())) {
    within_condition_expr_ = true;
    this->VisitExpr(op->args[0]);
    within_condition_expr_ = false;
    for (auto i = 1; i < op->args.size(); i++) {
      this->VisitExpr(op->args[i]);
    }
  } else {
    StmtExprVisitor::VisitExpr_(op);
  }
}

void BufferRegionCollector::VisitStmt_(const IfThenElseNode *op) {
  within_condition_expr_ = true;
  this->VisitExpr(op->condition);
  within_condition_expr_ = false;
  this->VisitStmt(op->then_case);
  if (op->else_case.defined()) {
    within_condition_expr_ = true;
    this->VisitStmt(op->else_case.value());
    within_condition_expr_ = false;
  }
}

class PipelinePlanningBodyAnalyzer {
public:
  PipelinePlanningBodyAnalyzer(Map<Var, Buffer> buffer_data_to_buffer,
                               Target target)
      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)),
        target_(std::move(target)) {}

  std::pair<Array<BufferRegion>, Array<BufferRegion>>
  CollectStmtAccessRegions(const Stmt &stmt) const {
    SBlock block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
                 /*name_hint=*/"", /*body*/ stmt);
    auto collector = BufferRegionCollector(buffer_data_to_buffer_, target_);
    collector(block);
    return {collector.GetReads(), collector.GetWrites()};
  }

  BufferSet CollectPipelineWriteBuffers(const Array<Stmt> &stmts) const {
    BufferSet write_buffers;
    for (const Stmt &stmt : stmts) {
      auto [_, writes] = CollectStmtAccessRegions(stmt);
      for (const BufferRegion &write : writes) {
        write_buffers.insert(write->buffer);
      }
    }
    return write_buffers;
  }

  bool
  IsReplayableScalarBindStmt(const Stmt &stmt,
                             const BufferSet &pipeline_write_buffers) const {
    auto [reads, _] = CollectStmtAccessRegions(stmt);
    return IsReplayableScalarBind(stmt, reads, pipeline_write_buffers);
  }

  struct ScheduledStmtAnalysis {
    size_t original_stmt_count{0};
    size_t stage_stmt_count{0};
    Array<Stmt> scheduled_stmts;
    std::vector<size_t> scheduled_indices;
    std::vector<size_t> scheduled_stage_indices;
    Array<Integer> replayable_bind_mask;
  };

  ScheduledStmtAnalysis AnalyzeScheduledStmts(const Array<Stmt> &stmts) const {
    BufferSet pipeline_write_buffers = CollectPipelineWriteBuffers(stmts);
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
      analysis.replayable_bind_mask.push_back(Integer(replayable ? 1 : 0));
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

  Array<Integer> FilterAnnotationsForScheduledStmts(
      const Array<Integer> &annotations,
      const ScheduledStmtAnalysis &analysis) const {
    if (annotations.size() == analysis.scheduled_stmts.size()) {
      return annotations;
    }

    Array<Integer> filtered;
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

  class SeqStmtFlattener : public StmtFunctor<Array<Stmt>(const Stmt &)> {
  public:
    using Base = StmtFunctor<Array<Stmt>(const Stmt &)>;

    static Array<Stmt> Flatten(const Array<Stmt> &stmts) {
      SeqStmtFlattener flattener;
      Array<Stmt> flattened;
      for (const Stmt &stmt : stmts) {
        Array<Stmt> nested = flattener(stmt);
        flattened.insert(flattened.end(), nested.begin(), nested.end());
      }
      return flattened;
    }

    Array<Stmt> VisitStmt(const Stmt &stmt) final {
      if (!stmt.as<SeqStmtNode>()) {
        return Array<Stmt>{stmt};
      }
      return Base::VisitStmt(stmt);
    }

    Array<Stmt> VisitStmt_(const SeqStmtNode *op) final {
      Array<Stmt> flattened;
      for (const Stmt &stmt : op->seq) {
        Array<Stmt> nested = VisitStmt(stmt);
        flattened.insert(flattened.end(), nested.begin(), nested.end());
      }
      return flattened;
    }

    Array<Stmt> VisitStmtDefault_(const Object *) final {
      return Array<Stmt>();
    }
  };

private:
  Map<Var, Buffer> buffer_data_to_buffer_;
  Target target_;
};

/*! \brief Scheduling information for one top-level pipeline statement.
 *
 * PipelinePlanning does not rewrite the loop itself.  It assigns each
 * statement a logical time offset (`stage`) and an execution position
 * (`order`); InjectSoftwarePipeline later realizes that schedule by replacing
 * the original loop variable with `pipeline_time - stage`, emitting
 * prologue/steady-state/epilogue loops, and multi-versioning buffers.  It is
 * therefore essential that dependencies point from an earlier or equal stage
 * to a later stage, and that dependencies within one stage follow `order`.
 *
 * `reads`/`writes` and `scalar_defs`/`scalar_uses` form the dependency graph.
 * The classification flags control stage weight and async-copy metadata.
 * `original_stmt_index` is the position before scheduling; `order` and `stage`
 * remain -1 until assigned.  `last_use_stmt_index` is the final source-order
 * consumer of a copy, and copies with the same final consumer may share an
 * implicit async producer group.
 */
struct PipelineStageInfo {
  Array<BufferRegion> reads, writes;
  std::unordered_set<const VarNode *> scalar_defs;
  std::unordered_set<const VarNode *> scalar_uses;
  int original_stmt_index{};
  int order = -1, stage = -1;
  bool scalar_bind = false;
  bool control_stmt = false;
  bool blocks_successor = false;
  bool explicit_ptx_async = false;
  bool lightweight_stmt = false;
  bool copy_stage = false;
  bool tma_copy = false; // true if this copy stage uses TMA (not cp.async)
  bool conditional_execution = false;
  int last_use_stmt_index =
      -1; // Initialized to -1, indicating no consumers found yet

public:
  bool IsScalarBind() const { return scalar_bind; }
  bool IsZeroWeight() const {
    return scalar_bind || control_stmt || lightweight_stmt;
  }
  bool IsControlStmt() const { return control_stmt; }
  bool BlocksSuccessor() const { return blocks_successor; }
  bool IsExplicitPtxAsync() const { return explicit_ptx_async; }
  bool IsCopyStage() const { return copy_stage; }
  bool IsTmaCopy() const { return tma_copy; }
  bool AdvancesPipelineStage() const {
    return copy_stage && !conditional_execution;
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

  bool IsExplicitPtxAsyncControl(const Call &call) const {
    return call->op.same_as(tl::ptx_cp_async()) ||
           call->op.same_as(builtin::ptx_cp_async()) ||
           call->op.same_as(builtin::ptx_commit_group()) ||
           call->op.same_as(builtin::ptx_wait_group());
  }

  bool CanReorderWithSuccessor(const Call &call) const {
    return call->op.same_as(builtin::ptx_arrive_barrier()) ||
           call->op.same_as(builtin::ptx_arrive_barrier_expect_tx()) ||
           call->op.same_as(tl::ptx_arrive_cluster_barrier()) ||
           call->op.same_as(tl::tma_store_arrive()) ||
           call->op.same_as(tl::named_barrier_arrive()) ||
           call->op.same_as(tl::cluster_arrive()) ||
           call->op.same_as(tl::cluster_arrive_relaxed());
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
      if (!pinfo.IsCopyStage()) {
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

        if (!pinfo.IsCopyStage()) {
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

  /*! \brief Unified intra-iteration dependency graph used for scheduling.
   *
   * Nodes are top-level pipeline statements in source order.  Every edge is
   * directed from a smaller source index to a larger one, making source order
   * a topological order by construction.  `predecessors` drives the weighted
   * longest-path traversal, while `successors` is used for sink detection and
   * forward constraint propagation.
   *
   * `scalar_successors` is a subset of `successors`.  It is retained separately
   * because a materialized scalar value cannot cross pipeline stages: unlike a
   * Buffer, InjectSoftwarePipeline does not create a cyclic versioned register
   * for it.  The producer Bind and all of its direct users must consequently
   * have equal stages, rather than merely ordered stages.
   *
   * `same_stage_successors` additionally records Buffer anti-dependencies.
   * These read-before-write lifecycles cannot cross skewed loop iterations
   * because the injector only derives async waits from RAW dependencies.
   */
  struct PipelineDependencyDag {
    std::vector<std::vector<int>> predecessors;
    std::vector<std::vector<int>> successors;
    std::vector<std::vector<int>> scalar_successors;
    std::vector<std::vector<int>> same_stage_successors;
  };

  bool RegionsConflict(const Array<BufferRegion> &lhs,
                       const Array<BufferRegion> &rhs) const {
    for (const BufferRegion &lhs_region : lhs) {
      for (const BufferRegion &rhs_region : rhs) {
        if (!lhs_region->buffer->data.same_as(rhs_region->buffer->data)) {
          continue;
        }
        // Aliased Buffer views may describe the same allocation with different
        // ranks.  Without an index map between the views, conservatively retain
        // their source order instead of asking MayConflict to compare unlike
        // regions.
        if (lhs_region->region.size() != rhs_region->region.size() ||
            MayConflict(lhs_region->region, rhs_region->region)) {
          return true;
        }
      }
    }
    return false;
  }

  PipelineDependencyDag BuildDependencyDag(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos) const {
    const int num_statements = static_cast<int>(pipeline_stage_infos.size());
    PipelineDependencyDag dag{std::vector<std::vector<int>>(num_statements),
                              std::vector<std::vector<int>>(num_statements),
                              std::vector<std::vector<int>>(num_statements),
                              std::vector<std::vector<int>>(num_statements)};

    auto add_edge =
        [&](int src, int dst, bool scalar_edge, bool same_stage_edge) {
          ICHECK_LT(src, dst)
              << "PipelinePlanning expects dependencies to follow source order";
          auto &successors = dag.successors[src];
          if (std::find(successors.begin(), successors.end(), dst) ==
              successors.end()) {
            successors.push_back(dst);
            dag.predecessors[dst].push_back(src);
          }
          if (scalar_edge) {
            auto &scalar_successors = dag.scalar_successors[src];
            if (std::find(scalar_successors.begin(), scalar_successors.end(),
                          dst) == scalar_successors.end()) {
              scalar_successors.push_back(dst);
            }
          }
          if (same_stage_edge) {
            auto &same_stage_successors = dag.same_stage_successors[src];
            if (std::find(same_stage_successors.begin(),
                          same_stage_successors.end(),
                          dst) == same_stage_successors.end()) {
              same_stage_successors.push_back(dst);
            }
          }
        };

    // Preserve all intra-iteration buffer hazards.  RAW edges carry values;
    // WAR and WAW edges prevent the stage-based reorder from moving a later
    // write before an earlier access to an overlapping region.
    for (int dst = 0; dst < num_statements; ++dst) {
      const PipelineStageInfo &dst_info = pipeline_stage_infos[dst];
      for (int src = 0; src < dst; ++src) {
        const PipelineStageInfo &src_info = pipeline_stage_infos[src];
        bool raw = RegionsConflict(src_info.writes, dst_info.reads);
        bool war = RegionsConflict(src_info.reads, dst_info.writes);
        bool waw = RegionsConflict(src_info.writes, dst_info.writes);
        if (raw || war || waw) {
          // WAR is a cyclic Buffer lifecycle: the earlier statement consumes
          // the value entering this iteration, and the later statement
          // overwrites it for the next iteration.  A monotonic stage edge is
          // insufficient because distinct stages execute skewed iterations.
          // Keep both ends in one stage so source order supplies the required
          // read-before-write synchronization.
          add_edge(src, dst, false, war);
        }
      }
    }

    // Add scalar def-use edges.  Replayable scalar Binds have already been
    // removed by AnalyzeScheduledStmts and will be reconstructed at each use by
    // InjectSoftwarePipeline.  Thus every Bind seen here is materialized and
    // needs the stronger same-stage treatment applied below.
    auto scalar_def_to_stmt = BuildScalarDefMap(pipeline_stage_infos);
    for (int dst = 0; dst < num_statements; ++dst) {
      for (const VarNode *var : pipeline_stage_infos[dst].scalar_uses) {
        auto it = scalar_def_to_stmt.find(var);
        if (it != scalar_def_to_stmt.end() && it->second != dst) {
          add_edge(it->second, dst, true, false);
        }
      }
    }

    // Opaque control operations do not always expose complete Buffer
    // dependencies.  Preserve the predecessor edge for every control.  A
    // blocking wait additionally orders its successor, whereas a non-blocking
    // arrive/signal may be reordered with independent following work.
    for (int i = 0; i < num_statements; ++i) {
      if (!pipeline_stage_infos[i].IsControlStmt()) {
        continue;
      }
      if (i > 0) {
        add_edge(i - 1, i, false, false);
      }
      if (i + 1 < num_statements && pipeline_stage_infos[i].BlocksSuccessor()) {
        add_edge(i, i + 1, false, false);
      }
    }
    return dag;
  }

  void
  AssignStagesAndOrders(std::vector<PipelineStageInfo> *pipeline_stage_infos,
                        int max_stage, bool compact_terminal_stage) const {
    ICHECK_GE(max_stage, 0);
    const int num_statements = static_cast<int>(pipeline_stage_infos->size());
    PipelineDependencyDag dag = BuildDependencyDag(*pipeline_stage_infos);

    if (std::any_of(
            pipeline_stage_infos->begin(), pipeline_stage_infos->end(),
            [](const auto &pinfo) { return pinfo.IsExplicitPtxAsync(); })) {
      LOG(WARNING)
          << "PipelinePlanning found explicit PTX async primitives inside "
             "T.Pipelined. InjectSoftwarePipeline does not currently "
             "multi-version buffers referenced through tl.access_ptr; the "
             "control chain will be kept with its following consumer instead "
             "of being automatically overlapped.";
    }

    // The schedule is constructed in five steps:
    //
    // 1. Compute ASAP logical levels with a weighted longest path through the
    //    dependency DAG.  Only an unconditional global-to-shared copy advances
    //    the level; synchronous compute and scalar/control glue are
    //    transparent.
    // 2. Map the logical level range onto [0, max_stage], spreading short
    // chains
    //    across the requested distance and merging levels when the dependency
    //    chain is deeper than that distance.
    // 3. Put sinks in the final consumer stage, then raise stages to satisfy
    //    materialized-scalar equality and all dependency inequalities.
    // 4. For ordinary pipelines, retime the terminal stage once so that its
    //    consumers overlap the next iteration's producers and do not require an
    //    otherwise unused extra Buffer version.
    // 5. Attach opaque control chains to a later following consumer when the
    //    injector cannot safely multi-version their operands.
    // 6. Sort by (stage, source index) to obtain a deterministic order.  Since
    //    all edges follow source order and stage never decreases along an edge,
    //    this is also a stable topological order.

    // Compute ASAP logical levels on the source-order topological traversal.
    // PipelinePlanning currently overlaps only global-to-shared transfer with
    // its consumers.  Such a copy may lower to cp.async or TMA and therefore
    // advances its successors by one logical level.  Other operations,
    // including asynchronous-looking compute such as WGMMA, remain in the
    // consumer level: wait_wgmma(0) cannot select an individual MMA group, so
    // separating WGMMA from its wait would require accumulator multi-versioning
    // without preserving useful independent in-flight work.
    std::vector<int> logical_levels(num_statements, 0);
    for (int dst = 0; dst < num_statements; ++dst) {
      for (int src : dag.predecessors[dst]) {
        int edge_weight =
            (*pipeline_stage_infos)[src].AdvancesPipelineStage() ? 1 : 0;
        logical_levels[dst] =
            std::max(logical_levels[dst], logical_levels[src] + edge_weight);
      }
    }

    // A control statement belongs to the next substantial operation so that
    // waits and arrives use the same skewed loop iteration as the work they
    // guard.  A trailing control operation instead stays with its predecessor.
    for (int i = 0; i < num_statements; ++i) {
      if (!(*pipeline_stage_infos)[i].IsControlStmt()) {
        continue;
      }
      int attached_level = -1;
      for (int j = i + 1; j < num_statements; ++j) {
        if (!(*pipeline_stage_infos)[j].IsZeroWeight()) {
          attached_level = logical_levels[j];
          break;
        }
      }
      if (attached_level < 0) {
        for (int j = i - 1; j >= 0; --j) {
          if (!(*pipeline_stage_infos)[j].IsZeroWeight()) {
            attached_level = logical_levels[j];
            break;
          }
        }
      }
      if (attached_level >= 0) {
        logical_levels[i] = attached_level;
      }
    }

    // Preserve the relative DAG depth while using the full requested stage
    // distance.  Integer division intentionally coalesces adjacent logical
    // levels when max_stage is smaller than the longest dependency chain.
    int max_logical_level = 0;
    for (int level : logical_levels) {
      max_logical_level = std::max(max_logical_level, level);
    }
    for (int i = 0; i < num_statements; ++i) {
      int stage = 0;
      if (max_logical_level > 0) {
        stage = logical_levels[i] * max_stage / max_logical_level;
      }
      (*pipeline_stage_infos)[i].stage = stage;
    }

    // A statement with no in-pipeline successor cannot prepare data for any
    // later pipeline work, so executing it early only increases its live range.
    // Keep all such sinks in the final consumer stage.  This also preserves the
    // old planner's rule that statements which are not producers for another
    // pipeline operation belong to the consumer side.
    for (int i = 0; i < num_statements; ++i) {
      if (dag.successors[i].empty()) {
        (*pipeline_stage_infos)[i].stage = max_stage;
      }
    }

    // A non-replayable Bind is materialized once per logical loop iteration.
    // If its consumer had a different stage, the two statements would execute
    // with different skewed loop indices and the consumer could observe another
    // iteration's register value.  InjectSoftwarePipeline versions Buffers but
    // does not create a cyclic register buffer, so place the Bind and every
    // direct user in their maximum common stage.
    //
    // Buffer WAR pairs require the same treatment: crossing stages would turn
    // source-order read-before-write into accesses from different logical
    // iterations.  Equalizing either kind of pair may raise a producer or
    // consumer, which can in turn violate a downstream
    // `stage(dst) >= stage(src)` constraint or raise another equality group.
    // Repeating all propagations computes the least fixed point without ever
    // decreasing a stage.
    bool updated = true;
    while (updated) {
      updated = false;
      for (int src = 0; src < num_statements; ++src) {
        if (!(*pipeline_stage_infos)[src].IsScalarBind()) {
          continue;
        }
        int common_stage = (*pipeline_stage_infos)[src].stage;
        for (int dst : dag.scalar_successors[src]) {
          common_stage =
              std::max(common_stage, (*pipeline_stage_infos)[dst].stage);
        }
        if ((*pipeline_stage_infos)[src].stage != common_stage) {
          (*pipeline_stage_infos)[src].stage = common_stage;
          updated = true;
        }
        for (int dst : dag.scalar_successors[src]) {
          if ((*pipeline_stage_infos)[dst].stage != common_stage) {
            (*pipeline_stage_infos)[dst].stage = common_stage;
            updated = true;
          }
        }
      }
      for (int src = 0; src < num_statements; ++src) {
        for (int dst : dag.same_stage_successors[src]) {
          int common_stage = std::max((*pipeline_stage_infos)[src].stage,
                                      (*pipeline_stage_infos)[dst].stage);
          if ((*pipeline_stage_infos)[src].stage != common_stage) {
            (*pipeline_stage_infos)[src].stage = common_stage;
            updated = true;
          }
          if ((*pipeline_stage_infos)[dst].stage != common_stage) {
            (*pipeline_stage_infos)[dst].stage = common_stage;
            updated = true;
          }
        }
      }
      for (int src = 0; src < num_statements; ++src) {
        for (int dst : dag.successors[src]) {
          int src_stage = (*pipeline_stage_infos)[src].stage;
          if ((*pipeline_stage_infos)[dst].stage < src_stage) {
            (*pipeline_stage_infos)[dst].stage = src_stage;
            updated = true;
          }
        }
      }
    }

    // The provisional [0, max_stage] schedule has two equivalent ways to place
    // its periodic boundary.  In the unrotated form, the terminal consumers of
    // iteration k run immediately before the stage-0 producers of iteration
    // k + max_stage.  Moving that boundary past the producers yields the
    // canonical producer-first form and decreases the terminal stage by one:
    //
    //   consumer(k), producer(k + N)  <=>  producer(k + N), consumer(k + 1)
    //
    // Moving the whole terminal stage together preserves scalar equalities and
    // source order.  It is a pure boundary retiming only when the stage
    // contains consumers and outward-facing writes, but no Buffer value
    // produced there and consumed by another pipeline statement.  Such an
    // internal producer can still occur when a dependency chain is deeper than
    // max_stage and several logical levels are coalesced into the final stage.
    // Compacting in that case would reduce real producer lookahead instead of
    // eliminating an unused endpoint, so retain the provisional schedule.
    //
    // When the final stage is consumer-only, retiming removes one endpoint from
    // every Buffer live range ending there and can save one cyclic Buffer
    // version.  Apply it only once: further compaction would again reduce the
    // requested producer lookahead.  The old planner implemented the same
    // optimization only when all copy producers happened to be at the end of
    // its temporary order.
    //
    // Manual warp specialization already uses [0, num_stages - 1] as its
    // physical ring-buffer range.  Its cross-warp order is defined by explicit
    // barriers rather than statement order, so the caller disables this generic
    // retiming for manual WS.  Preserve the old num_stages == 1 behavior as
    // well; collapsing [0, 1] would remove pipelining entirely.
    bool final_stage_has_internal_buffer_producer = false;
    for (int src = 0; src < num_statements; ++src) {
      if ((*pipeline_stage_infos)[src].stage != max_stage) {
        continue;
      }
      for (int dst : dag.successors[src]) {
        if (RegionsConflict((*pipeline_stage_infos)[src].writes,
                            (*pipeline_stage_infos)[dst].reads)) {
          final_stage_has_internal_buffer_producer = true;
          break;
        }
      }
      if (final_stage_has_internal_buffer_producer) {
        break;
      }
    }
    bool terminal_stage_compacted = false;
    if (compact_terminal_stage && max_stage >= 2 &&
        !final_stage_has_internal_buffer_producer) {
      for (PipelineStageInfo &pinfo : *pipeline_stage_infos) {
        if (pinfo.stage == max_stage) {
          --pinfo.stage;
        }
      }
      terminal_stage_compacted = true;
    }

    // Opaque control statements do not always expose enough Buffer access
    // information for InjectSoftwarePipeline to version their operands.  Once
    // all dependency and terminal-stage adjustments are final, keep a control
    // chain with its following substantial statement whenever that consumer is
    // later.  This conservatively collapses explicit PTX async sequences whose
    // tl.access_ptr buffers cannot yet be multi-versioned by the injector.
    for (int i = 0; i < num_statements; ++i) {
      if (!(*pipeline_stage_infos)[i].IsControlStmt()) {
        continue;
      }
      for (int j = i + 1; j < num_statements; ++j) {
        if ((*pipeline_stage_infos)[j].IsZeroWeight()) {
          continue;
        }
        (*pipeline_stage_infos)[i].stage = std::max(
            (*pipeline_stage_infos)[i].stage, (*pipeline_stage_infos)[j].stage);
        break;
      }
    }

    // Without terminal retiming, place each early copy immediately after its
    // last consumer.  The consumer reads the old cyclic slot before the copy
    // overwrites it for a future iteration, saving one Buffer version.  This is
    // the lifecycle ordering used by the old planner, generalized to the DAG
    // schedule.  It is legal only across distinct stages; same-stage edges must
    // retain source order.  Once the terminal stage has been retimed, the
    // periodic boundary has already moved past the producer and the canonical
    // order is producer-first instead.
    std::vector<int> indices(num_statements);
    std::iota(indices.begin(), indices.end(), 0);
    if (compact_terminal_stage && !terminal_stage_compacted) {
      std::vector<int> lifecycle_order;
      std::vector<bool> deferred_copy(num_statements, false);
      lifecycle_order.reserve(num_statements);
      for (int i = 0; i < num_statements; ++i) {
        const PipelineStageInfo &pinfo = (*pipeline_stage_infos)[i];
        if (pinfo.IsCopyStage() && pinfo.last_use_stmt_index >= 0 &&
            pinfo.stage <
                (*pipeline_stage_infos)[pinfo.last_use_stmt_index].stage) {
          deferred_copy[i] = true;
          continue;
        }
        lifecycle_order.push_back(i);
        for (int copy = 0; copy < num_statements; ++copy) {
          if (deferred_copy[copy] &&
              (*pipeline_stage_infos)[copy].last_use_stmt_index == i) {
            lifecycle_order.push_back(copy);
            deferred_copy[copy] = false;
          }
        }
      }
      for (int copy = 0; copy < num_statements; ++copy) {
        ICHECK(!deferred_copy[copy])
            << "PipelinePlanning failed to place copy statement " << copy
            << " after its last consumer";
      }
      ICHECK_EQ(lifecycle_order.size(), static_cast<size_t>(num_statements));
      indices = std::move(lifecycle_order);
    } else {
      std::stable_sort(indices.begin(), indices.end(), [&](int lhs, int rhs) {
        int lhs_stage = (*pipeline_stage_infos)[lhs].stage;
        int rhs_stage = (*pipeline_stage_infos)[rhs].stage;
        return lhs_stage != rhs_stage ? lhs_stage < rhs_stage : lhs < rhs;
      });
    }
    for (int order = 0; order < num_statements; ++order) {
      (*pipeline_stage_infos)[indices[order]].order = order;
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
      if (!pinfo.IsCopyStage() && !pinfo.conditional_execution &&
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
    // Replayable Binds were filtered before this point.  A remaining Bind must
    // stay materialized and is treated as zero-weight scheduling glue.  A plain
    // BufferStore is likewise a scalar/lightweight IR operation;
    // latency-bearing copies and matrix operations are represented by tile
    // operators instead.
    pinfo.scalar_bind = block->body.as<BindNode>() != nullptr;
    pinfo.lightweight_stmt = block->body.as<BufferStoreNode>() != nullptr;

    // Opaque state-changing calls (barrier wait/arrive, async control, etc.)
    // may not report complete Buffer regions.  Classify them as control
    // statements so BuildDependencyDag preserves their source neighbors and the
    // level assignment attaches them to the substantial operation they guard.
    if (const auto *evaluate = block->body.as<EvaluateNode>()) {
      if (const auto *call_node = evaluate->value.as<CallNode>()) {
        Call call = GetRef<Call>(call_node);
        pinfo.control_stmt = !ParseOperator(call).defined() &&
                             SideEffect(call) > CallEffectKind::kReadState;
        if (pinfo.control_stmt) {
          pinfo.blocks_successor = !CanReorderWithSuccessor(call);
          pinfo.explicit_ptx_async = IsExplicitPtxAsyncControl(call);
        }
      }
    }
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

class PipelinePlanner : public StmtExprMutator {
public:
  static Stmt Substitute(const PrimFunc &f, bool use_async_copy = true) {
    PipelinePlanner substituter(use_async_copy);
    for (const auto &[_, buffer] : f->buffer_map) {
      substituter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "Pipeline_Planning: Require the target attribute";
    substituter.target_ = target.value();
    return substituter.VisitStmt(f->body);
  }

private:
  PipelinePlanner() = default;
  PipelinePlanner(bool use_async_copy) : use_async_copy_(use_async_copy) {}

  PipelineStageAnalyzer MakeStageAnalyzer() const {
    return PipelineStageAnalyzer(buffer_data_to_buffer_, target_,
                                 use_async_copy_);
  }

  void AnalyzeCopyLastUse(
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    MakeStageAnalyzer().AnalyzeCopyLastUse(pipeline_stage_infos);
  }

  void ValidateScalarDependencies(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos) const {
    MakeStageAnalyzer().ValidateScalarDependencies(pipeline_stage_infos);
  }

  void
  AssignStagesAndOrders(std::vector<PipelineStageInfo> *pipeline_stage_infos,
                        int max_stage, bool compact_terminal_stage) const {
    MakeStageAnalyzer().AssignStagesAndOrders(pipeline_stage_infos, max_stage,
                                              compact_terminal_stage);
  }

  bool HasManualWarpSpecialization(const Stmt &stmt) const {
    // T.ws() first emits the language-level "warp_specialize" AttrStmt; some
    // lowering paths replace it with kWarpSpecializationScope.  Detect both.
    // Compiler-generated producer/consumer WS runs before PipelinePlanning and
    // strips pipeline annotations after rewriting, so an annotated pipeline
    // reaching this check is the user-authored/manual WS case.
    bool found = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (const auto *attr_stmt = node.as<AttrStmtNode>()) {
        if (attr_stmt->attr_key == "warp_specialize" ||
            attr_stmt->attr_key == attr::kWarpSpecializationScope) {
          found = true;
        }
      }
    });
    return found;
  }

  void MaybeAnnotateLegacyAsyncPipelineLoop(const Array<Stmt> &pipeline_stmts,
                                            const Array<Integer> &order_array,
                                            const Array<Integer> &stage_array,
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

  using ScheduledStmtAnalysis =
      PipelinePlanningBodyAnalyzer::ScheduledStmtAnalysis;
  using SeqStmtFlattener = PipelinePlanningBodyAnalyzer::SeqStmtFlattener;

  PipelinePlanningBodyAnalyzer MakeBodyAnalyzer() const {
    return PipelinePlanningBodyAnalyzer(buffer_data_to_buffer_, target_);
  }

  ScheduledStmtAnalysis AnalyzeScheduledStmts(const Array<Stmt> &stmts) const {
    return MakeBodyAnalyzer().AnalyzeScheduledStmts(stmts);
  }

  Array<Integer> FilterAnnotationsForScheduledStmts(
      const Array<Integer> &annotations,
      const ScheduledStmtAnalysis &analysis) const {
    return MakeBodyAnalyzer().FilterAnnotationsForScheduledStmts(annotations,
                                                                 analysis);
  }

  Stmt VisitStmt_(const ForNode *loop) final {
    auto order_anno = loop->annotations.Get("tl_pipeline_order");
    auto stage_anno = loop->annotations.Get("tl_pipeline_stage");
    auto num_stages_anno = loop->annotations.Get("num_stages");
    if (order_anno && stage_anno) {
      auto order_array = Downcast<Array<Integer>>(order_anno.value());
      auto stage_array = Downcast<Array<Integer>>(stage_anno.value());

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
        annotations.Set(s_tir::attr::software_pipeline_async_stages,
                        Array<Integer>{0});
      }
      Array<Stmt> pipeline_body_stmts = NormalizePipelineBody(loop->body);
      Array<Stmt> pipeline_stmts =
          SeqStmtFlattener::Flatten(pipeline_body_stmts);
      ScheduledStmtAnalysis analysis = AnalyzeScheduledStmts(pipeline_stmts);
      ICHECK(!analysis.scheduled_stmts.empty())
          << "PipelinePlanning: explicit pipeline annotations have no "
             "schedulable statements after removing replayable scalar Bind "
             "statements";
      Array<Integer> filtered_order_array =
          FilterAnnotationsForScheduledStmts(order_array, analysis);
      Array<Integer> filtered_stage_array =
          FilterAnnotationsForScheduledStmts(stage_array, analysis);
      annotations.Set(s_tir::attr::software_pipeline_order,
                      filtered_order_array);
      annotations.Set(s_tir::attr::software_pipeline_stage,
                      filtered_stage_array);
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
                        [](const Integer &value) { return !is_zero(value); })) {
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
      n->body = MakePipelineBody(pipeline_body_stmts);
      return for_node;
    }

    if (!num_stages_anno)
      return StmtExprMutator::VisitStmt_(loop);
    int num_stages = num_stages_anno->as<IntImmNode>()->value;
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
      return StmtExprMutator::VisitStmt_(stripped.get());
    }
    Array<Stmt> pipeline_body_stmts = NormalizePipelineBody(loop->body);

    ICHECK(num_stages >= 1);
    ICHECK(loop->kind == ForKind::kSerial);

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

    // Cyclic Buffer lifecycle ordering places an early copy immediately after
    // its final consumer when the terminal stage is not retimed, so last-use
    // information must be available before assigning order.
    AnalyzeCopyLastUse(&pipeline_stage_infos);

    // Assign stages by a weighted longest-path traversal over the unified
    // buffer/scalar dependency DAG, then derive a stable topological order.
    //
    // For an ordinary compiler-inferred pipeline, num_stages denotes the
    // producer/consumer distance and the provisional annotation range is
    // [0, num_stages], i.e. up to num_stages + 1 logical time levels.  A
    // consumer-only terminal endpoint is subsequently retimed to
    // num_stages - 1 so it can share the periodic boundary without requiring an
    // extra Buffer version.
    //
    // In manual warp specialization, num_stages is additionally the physical
    // ring-buffer/barrier slot count.  T.ws producer and consumer warps execute
    // concurrently, and InjectSoftwarePipeline evaluates a stage-s statement at
    // logical iteration `pipeline_time - s`.  Allowing stage == num_stages
    // would alias stage 0 after modulo ring indexing, potentially overwriting
    // data still consumed by another warp or waiting on the wrong barrier
    // phase.  Therefore manual WS must remain in [0, num_stages - 1].
    bool manual_warp_specialization = HasManualWarpSpecialization(loop->body);
    int max_stage = manual_warp_specialization ? num_stages - 1 : num_stages;
    AssignStagesAndOrders(
        &pipeline_stage_infos, max_stage,
        /*compact_terminal_stage=*/!manual_warp_specialization);

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
    annotations.Set("tl_pipelined_num_stages", Integer(num_stages));

    std::vector<Integer> orders, stages;
    orders.reserve(pipeline_stage_infos.size());
    stages.reserve(pipeline_stage_infos.size());
    for (auto &pinfo : pipeline_stage_infos) {
      orders.push_back(pinfo.order);
      stages.push_back(pinfo.stage);
    }

    annotations.Set(s_tir::attr::software_pipeline_stage,
                    Array<Integer>(stages));
    annotations.Set(s_tir::attr::software_pipeline_order,
                    Array<Integer>(orders));
    if (std::any_of(analysis.replayable_bind_mask.begin(),
                    analysis.replayable_bind_mask.end(),
                    [](const Integer &value) { return !is_zero(value); })) {
      annotations.Set(kPipelineReplayableScalarBinds,
                      analysis.replayable_bind_mask);
    }

    // Propagate per-statement TMA eligibility so InjectSoftwarePipeline can
    // rewrite TMA copies to use pipeline-level barrier management.
    {
      std::vector<Integer> tma_copies;
      tma_copies.reserve(pipeline_stage_infos.size());
      bool has_tma_copy = false;
      for (auto &pinfo : pipeline_stage_infos) {
        bool IsTmaCopy = pinfo.IsTmaCopy();
        has_tma_copy = has_tma_copy || IsTmaCopy;
        tma_copies.push_back(Integer(IsTmaCopy ? 1 : 0));
      }
      if (has_tma_copy) {
        annotations.Set(kPipelineTmaCopies, Array<Integer>(tma_copies));
      }
    }

    EmitImplicitAsyncAnnotations(pipeline_stage_infos, &annotations);

    // Reconstruct the loop body with the flattened SeqStmt so that
    // InjectSoftwarePipeline sees the correct number of pipeline stages.
    Stmt new_body = MakePipelineBody(flat_stmts);

    return For(loop->loop_var, loop->min, loop->extent, loop->kind, new_body,
               loop->thread_binding, annotations, loop->step, loop->span);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
    return block;
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Target target_;
  bool use_async_copy_{};
};

tvm::transform::Pass PipelinePlanning() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    bool use_async_copy =
        ctx->GetConfig<Bool>("tirx.use_async_copy", Bool(true)).value();
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = PipelinePlanner::Substitute(f, use_async_copy);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PipelinePlanning", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.PipelinePlanning", PipelinePlanning);
}

} // namespace tl
} // namespace tvm

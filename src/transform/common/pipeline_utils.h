/*!
 * \file pipeline_utils.h
 * \brief Shared utilities for software-pipeline and warp-specialization passes.
 *
 * Provides:
 *  - Pipeline annotation attribute keys
 *  - GetPipelineNumStages()  — extract num_stages from loop annotations
 *  - AddReadsWritesForTileOp() — collect buffer regions touched by a tile op
 *  - ComputeThreadBounds()  — derive thread bounds from an analyzer + IterVar
 */
#ifndef TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_

#include <tvm/tir/stmt.h>

#include "../../op/atomic_add.h"
#include "../../op/atomic_reduce.h"
#include "../../op/copy.h"
#include "../../op/fill.h"
#include "../../op/finalize_reducer.h"
#include "../../op/gemm.h"
#include "../../op/gemm_sp.h"
#include "../../op/operator.h"
#include "../../op/reduce.h"

namespace tvm {
namespace tl {

using namespace tir;

// ---------------------------------------------------------------------------
// Pipeline annotation attribute keys
// ---------------------------------------------------------------------------

/*! Marks the enclosing scope with the pipeline stage count. */
static constexpr const char *kPipelineContextNumStages =
    "tl.pipeline_context_num_stages";
/*! Multi-version buffer: stage count for buffer expansion. */
static constexpr const char *kPipelineMVBContextNumStages =
    "tl.pipeline_mvb_num_stages";
/*! Multi-version buffer: per-statement stage index expression. */
static constexpr const char *kPipelineMVBStageExpr =
    "tl.pipeline_mvb_stage_expr";
/*! Multi-version buffer: per-statement parity expression. */
static constexpr const char *kPipelineMVBParityExpr =
    "tl.pipeline_mvb_parity_expr";

// ---------------------------------------------------------------------------
// GetPipelineNumStages
// ---------------------------------------------------------------------------

/*!
 * \brief Extract the pipeline stage count from a For loop's annotations.
 *
 * Checks (in order):
 *   1. "num_stages" — user-provided stage count
 *   2. "tl_pipelined_num_stages" — set by InjectSoftwarePipeline
 *   3. tir::attr::software_pipeline_stage — max(stage) + 1
 *
 * \return The stage count, or nullopt if the loop is not pipelined.
 */
inline Optional<Integer> GetPipelineNumStages(const ForNode *loop) {
  if (auto num_stages = loop->annotations.Get("num_stages")) {
    if (const auto *imm = num_stages->as<IntImmNode>()) {
      return Integer(static_cast<int>(imm->value));
    }
  }
  if (auto num_stages = loop->annotations.Get("tl_pipelined_num_stages")) {
    if (const auto *imm = num_stages->as<IntImmNode>()) {
      return Integer(static_cast<int>(imm->value));
    }
  }
  if (auto stages_anno =
          loop->annotations.Get(tir::attr::software_pipeline_stage)) {
    auto stages = Downcast<Array<Integer>>(stages_anno.value());
    int max_stage = -1;
    for (const auto &stage : stages) {
      max_stage = std::max(max_stage, static_cast<int>(stage->value));
    }
    if (max_stage >= 0) {
      return Integer(max_stage + 1);
    }
  }
  return Optional<Integer>();
}

// ---------------------------------------------------------------------------
// AddReadsWritesForTileOp
// ---------------------------------------------------------------------------

/*!
 * \brief Collect the buffer regions read/written by a tile operator.
 *
 * Handles Copy, Gemm, GemmSP, Reduce, CumSum, Fill, Atomic{Add,Max,Min},
 * and FinalizeReducer.
 */
inline void AddReadsWritesForTileOp(const TileOperator &tile_op,
                                    Array<BufferRegion> *reads,
                                    Array<BufferRegion> *writes) {
  auto add_reads = [&](const Array<BufferRegion> &regions) {
    reads->insert(reads->end(), regions.begin(), regions.end());
  };
  auto add_writes = [&](const Array<BufferRegion> &regions) {
    writes->insert(writes->end(), regions.begin(), regions.end());
  };
  if (const auto *copy = tile_op.as<CopyNode>()) {
    add_reads({BufferRegion(copy->src, copy->src_range)});
    add_writes({BufferRegion(copy->dst, copy->dst_range)});
    return;
  }
  if (const auto *gemm = tile_op.as<GemmNode>()) {
    add_reads({gemm->aRegion_, gemm->bRegion_});
    if (!is_one(gemm->clearAccum_)) {
      add_reads({gemm->cRegion_});
    }
    add_writes({gemm->cRegion_});
    return;
  }
  if (const auto *gemm_sp = tile_op.as<GemmSPNode>()) {
    add_reads({gemm_sp->aRegion_, gemm_sp->bRegion_, gemm_sp->eRegion_});
    if (!gemm_sp->clearAccum_) {
      add_reads({gemm_sp->cRegion_});
    }
    add_writes({gemm_sp->cRegion_});
    return;
  }
  if (const auto *reduce = tile_op.as<ReduceOpNode>()) {
    add_reads({reduce->srcRegion_});
    if (!reduce->clear) {
      add_reads({reduce->dstRegion_});
    }
    add_writes({reduce->dstRegion_});
    return;
  }
  if (const auto *cumsum = tile_op.as<CumSumOpNode>()) {
    add_reads({cumsum->srcRegion_});
    add_writes({cumsum->dstRegion_});
    return;
  }
  if (const auto *fill = tile_op.as<FillNode>()) {
    add_writes({BufferRegion(fill->dst, fill->region)});
    return;
  }
  auto handle_atomic = [&](const auto *atomic) {
    if (atomic->src.defined()) {
      add_reads({BufferRegion(atomic->src, atomic->src_range)});
    }
    BufferRegion dst_region(atomic->dst, atomic->dst_range);
    add_reads({dst_region});
    add_writes({dst_region});
  };
  if (const auto *atomic = tile_op.as<AtomicAddNode>()) {
    handle_atomic(atomic);
    return;
  }
  if (const auto *atomic = tile_op.as<AtomicMaxNode>()) {
    handle_atomic(atomic);
    return;
  }
  if (const auto *atomic = tile_op.as<AtomicMinNode>()) {
    handle_atomic(atomic);
    return;
  }
  if (const auto *finalize = tile_op.as<FinalizeReducerOpNode>()) {
    BufferRegion region = BufferRegion::FullRegion(finalize->reducer);
    add_reads({region});
    add_writes({region});
  }
}

// ---------------------------------------------------------------------------
// ComputeThreadBounds
// ---------------------------------------------------------------------------

/*!
 * \brief Compute the thread index bounds from an IterVar and an analyzer.
 *
 * \return Range covering the thread index, or [0, 1) if no bound is known.
 */
inline Range ComputeThreadBounds(const IterVar &thread_var,
                                 const arith::Analyzer &analyzer) {
  if (thread_var.defined() &&
      analyzer.const_int_bound.IsBound(thread_var->var)) {
    auto const_int_bound = analyzer.const_int_bound(thread_var);
    auto min_value = const_int_bound->min_value;
    auto max_value = const_int_bound->max_value;
    auto extent = max_value - min_value + 1;
    auto dtype = thread_var->var.dtype();
    return Range::FromMinExtent(IntImm(dtype, min_value),
                                IntImm(dtype, extent));
  }
  return Range::FromMinExtent(0, 1);
}

}  // namespace tl
}  // namespace tvm

#endif  // TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_

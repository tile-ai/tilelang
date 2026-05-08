/*!
 * \file pipeline_utils.h
 * \brief Shared utilities for software-pipeline and warp-specialization passes.
 *
 * Provides:
 *  - Pipeline annotation attribute keys
 *  - GetPipelineNumStages()  — extract num_stages from loop annotations
 *  - ComputeThreadBounds()  — derive thread bounds from an analyzer + IterVar
 */
#ifndef TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_

#include <tvm/tir/stmt.h>

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
/*! Per-statement TMA copy flag (1 = TMA eligible, 0 = not). */
static constexpr const char *kPipelineTmaCopies =
    "software_pipeline_tma_copies";
/*! Per-statement async producer flag (1 = async copy producer, 0 = not). */
static constexpr const char *kPipelineAsyncProducers =
    "software_pipeline_async_producers";
/*! Per-statement async producer group id (-1 = not an async producer). */
static constexpr const char *kPipelineAsyncProducerGroups =
    "software_pipeline_async_producer_groups";

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

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_

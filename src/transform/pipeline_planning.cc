#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/parallel.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "common/pipeline_utils.h"
#include "pipeline/access_analysis.h"
#include "pipeline/body_analysis.h"
#include "pipeline/stage_analysis.h"
#include <algorithm>
#include <utility>
#include <vector>

#include "backend/common/target_utils.h"
#include "tvm/ir/expr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

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

  void PropagateBufferProducersForCopy(
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    MakeStageAnalyzer().PropagateBufferProducersForCopy(pipeline_stage_infos);
  }

  void PropagateScalarProducersForCopy(
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    MakeStageAnalyzer().PropagateScalarProducersForCopy(pipeline_stage_infos);
  }

  void ValidateScalarDependencies(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos) const {
    MakeStageAnalyzer().ValidateScalarDependencies(pipeline_stage_infos);
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

    // Some statements before a copy are not copy operations themselves, but
    // they prepare buffers that the copy must read.  A common example is
    // producer-side initialization before a conditional or partial copy:
    //
    //   fill(shared, 0)        // writes shared
    //   copy(global, shared)   // may rely on the initialized values
    //
    // If the copy is moved to the producer side, the fill must move with it;
    // otherwise the copy could observe an uninitialized or wrong shared-buffer
    // value.  PropagateBufferProducersForCopy computes a buffer-level backward
    // dependency closure from copy-stage reads to earlier non-copy writes and
    // marks those statements as `producer_for_copy`.  They then participate in
    // the producer-stage scheduling just like the copy stages they prepare.
    PropagateBufferProducersForCopy(&pipeline_stage_infos);

    // Analysis use-def chain to determine last_use_stmt_index for copy
    // operations This step is critical for pipeline optimization as it
    // identifies the index of the last statement that consumes data produced by
    // copy stages, enabling optimal placement of copy operations in the
    // pipeline schedule.
    AnalyzeCopyLastUse(&pipeline_stage_infos);

    PropagateScalarProducersForCopy(&pipeline_stage_infos);

    // Making stages and orders
    int order_idx = 0;
    // Stage 1. Create pipeline stages and assign order
    for (auto &pinfo : pipeline_stage_infos) {
      // Skip elements that must be in first stage:
      // 1. Copy stages (with active last_use_stmt_index) - these need special
      // handling
      //    because they have consumers that depend on their data
      // 2. All Producer stages for copy stages.
      if (pinfo.is_first_stage() && pinfo.is_last_use_stmt_index_valid()) {
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
        if ((pinfo_1.is_first_stage() &&
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
        if (pinfo.is_first_stage()) {
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
        if (!pinfo.is_copy_stage() && !pinfo.is_producer_for_copy())
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
        bool is_tma_copy = pinfo.is_tma_copy();
        has_tma_copy = has_tma_copy || is_tma_copy;
        tma_copies.push_back(Integer(is_tma_copy ? 1 : 0));
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
               loop->thread_binding, annotations);
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

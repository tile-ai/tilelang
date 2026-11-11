/*!
 * \file pipeline_utils.h
 * \brief Shared utilities for software-pipeline and warp-specialization passes.
 *
 * Provides:
 *  - Pipeline annotation attribute keys
 *  - GetPipelineNumStages()  — extract num_stages from loop annotations
 *  - IsPipelineDeclarationStmt() — identify non-stage buffer declarations
 *  - ComputeThreadBounds()  — derive thread bounds from an analyzer + IterVar
 */
#ifndef TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_

#include "support/check.h"
#include <deque>
#include <vector>

#include "../../op/builtin.h"
#include "../../op/gemm.h"
#include "../../op/gemm_sp.h"
#include "../../op/operator.h"
#include "../../op/utils.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace tl {

using namespace tirx;

// ---------------------------------------------------------------------------
// Pipeline annotation attribute keys
// ---------------------------------------------------------------------------

/*! Per-statement TMA copy flag (1 = TMA eligible, 0 = not). */
static constexpr const char *kPipelineTmaCopies =
    "software_pipeline_tma_copies";
/*! Per-statement async producer flag (1 = async copy producer, 0 = not). */
static constexpr const char *kPipelineAsyncProducers =
    "software_pipeline_async_producers";
/*! Per-statement async producer group id (-1 = not an async producer). */
static constexpr const char *kPipelineAsyncProducerGroups =
    "software_pipeline_async_producer_groups";
/*! Per-original-statement replayable scalar Bind flag (1 = replayable). */
static constexpr const char *kPipelineReplayableScalarBinds =
    "software_pipeline_replayable_scalar_binds";

/*! \brief Whether a flat TIRX statement declares pipeline-local buffer storage.
 *
 * Flat TIRX represents buffer allocations/declarations as standalone
 * statements. They must stay in the loop body so later rewrites can preserve
 * storage, but they are declarations rather than executable pipeline stages.
 */
inline bool IsPipelineDeclarationStmt(const Stmt &stmt) {
  return stmt.as<AllocBufferNode>() != nullptr ||
         stmt.as<DeclBufferNode>() != nullptr;
}

/*! \brief Return the top-level statement stream for a pipeline loop body.
 *
 * IfStmtBinding may preserve an if-body that starts with Bind as a single
 * IfThenElse, so a valid pipeline body is not necessarily a SeqStmt. Keep the
 * downstream planning code Array-based and only reconstruct SeqStmt when there
 * are multiple children.
 */
inline ffi::Array<Stmt> NormalizePipelineBody(const Stmt &body) {
  if (const auto *seq = body.as<SeqStmtNode>()) {
    return seq->seq;
  }
  return ffi::Array<Stmt>{body};
}

/*! \brief Build a valid TIRX statement from a pipeline statement stream. */
inline Stmt MakePipelineBody(const ffi::Array<Stmt> &stmts) {
  ICHECK(!stmts.empty());
  return stmts.size() == 1 ? stmts[0] : SeqStmt(stmts);
}

/*! \brief Collect the domains of materialized thread indices.
 *
 * Union repeated bindings of the same Var so a narrower sibling scope cannot
 * hide accesses from a wider scope. Unmaterialized launch placeholders have no
 * domain here and remain conservative in the region conflict check.
 */
inline ffi::Map<Var, arith::IntSet>
CollectPipelineThreadDomains(const Stmt &body) {
  ffi::Map<Var, arith::IntSet> domains;
  auto record_domain = [&](const Var &var, const Range &range) {
    arith::IntSet domain = arith::IntSet::FromRange(range);
    if (auto previous = domains.Get(var)) {
      domain = arith::Union({previous.value(), domain});
    }
    domains.Set(var, domain);
  };
  PostOrderVisit(body, [&](const ffi::ObjectRef &node) {
    if (const auto *loop = node.as<ForNode>()) {
      if (loop->thread_binding.defined()) {
        record_domain(loop->loop_var,
                      Range::FromMinExtent(loop->min, loop->extent));
      }
    } else if (const auto *attr = node.as<AttrStmtNode>()) {
      if (attr->attr_key == tirx::attr::thread_extent ||
          attr->attr_key == s_tir::attr::virtual_thread) {
        IterVar thread = Downcast<IterVar>(attr->node);
        record_domain(
            thread->var,
            Range::FromMinExtent(make_zero(attr->value.dtype()), attr->value));
      }
    }
  });
  return domains;
}

/*! \brief Conservatively compare pipeline accesses across threads.
 *
 * Expand each access independently over the thread domains before comparing
 * it with the other access. A shared thread Var must not cancel between two
 * regions, since the accesses can execute in different threads. Bounds that
 * remain symbolic and distinct views of one allocation stay conservative.
 */
inline bool
PipelineRegionsMayConflict(const BufferRegion &lhs, const BufferRegion &rhs,
                           const ffi::Map<Var, arith::IntSet> &thread_domains) {
  if (!lhs->buffer->data.same_as(rhs->buffer->data)) {
    return false;
  }
  if (!lhs->buffer.same_as(rhs->buffer) ||
      lhs->region.size() != rhs->region.size()) {
    return true;
  }
  arith::Analyzer analyzer;
  auto provably_disjoint = [&](const arith::IntSet &a, const arith::IntSet &b) {
    if (a.IsNothing() || b.IsNothing()) {
      return true;
    }
    return a.HasLowerBound() && a.HasUpperBound() && b.HasLowerBound() &&
           b.HasUpperBound() && as_const_int(analyzer.Simplify(a.min())) &&
           as_const_int(analyzer.Simplify(a.max())) &&
           as_const_int(analyzer.Simplify(b.min())) &&
           as_const_int(analyzer.Simplify(b.max())) &&
           arith::Intersect({a, b}).IsNothing();
  };
  if (!lhs->buffer->strides.empty()) {
    // Logical coordinates need not be injective for an explicitly strided
    // buffer. Compare physical offsets after expanding the thread domains.
    if (!lhs->buffer->axis_separators.empty()) {
      return true;
    }
    auto physical_region = [&](const BufferRegion &region) {
      ffi::Map<Var, arith::IntSet> domains = thread_domains;
      ffi::Array<PrimExpr> indices;
      ffi::Array<arith::IntSet> bounds =
          arith::EvalSet(region->region, thread_domains);
      for (size_t i = 0; i < region->region.size(); ++i) {
        const Range &range = region->region[i];
        Var index("region_index", range->min.dtype());
        domains.Set(index, bounds[i]);
        indices.push_back(index);
      }
      return arith::EvalSet(region->buffer->ElemOffset(indices)[0], domains);
    };
    return !provably_disjoint(physical_region(lhs), physical_region(rhs));
  }
  ffi::Array<arith::IntSet> lhs_bounds =
      arith::EvalSet(lhs->region, thread_domains);
  ffi::Array<arith::IntSet> rhs_bounds =
      arith::EvalSet(rhs->region, thread_domains);
  for (size_t i = 0; i < lhs->region.size(); ++i) {
    if (provably_disjoint(lhs_bounds[i], rhs_bounds[i])) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// GetPipelineNumStages
// ---------------------------------------------------------------------------

/*!
 * \brief Extract the pipeline stage count from a For loop's annotations.
 *
 * Checks (in order):
 *   1. "num_stages" — user-provided stage count
 *   2. "tl_pipelined_num_stages" — set by InjectSoftwarePipeline
 *   3. s_tir::attr::software_pipeline_stage — max(stage) + 1
 *
 * \return The stage count, or nullopt if the loop is not pipelined.
 */
inline ffi::Optional<Integer> GetPipelineNumStages(const ForNode *loop) {
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
          loop->annotations.Get(s_tir::attr::software_pipeline_stage)) {
    auto stages = Downcast<ffi::Array<Integer>>(stages_anno.value());
    int max_stage = -1;
    for (const auto &stage : stages) {
      max_stage = std::max(max_stage, static_cast<int>(stage->value));
    }
    if (max_stage >= 0) {
      return Integer(max_stage + 1);
    }
  }
  return ffi::Optional<Integer>();
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

/*! \brief Shared GEMM operands kept live by completion waits in a statement
 * list.
 *
 * GEMM issue reads and completion reads have different purposes: the former
 * anchor async-copy waits, the latter prevent reuse while Tensor Cores still
 * read the operands. Keep the two separate. The source must obey the existing
 * GEMM/barrier contract; this analysis does not validate barrier phases.
 */
inline std::vector<ffi::Array<BufferRegion>>
CollectAsyncGemmCompletionReads(const ffi::Array<Stmt> &stmts) {
  std::vector<ffi::Array<BufferRegion>> reads(stmts.size());
  struct Tcgen05Group {
    BufferLoad barrier;
    ffi::Array<BufferRegion> operands;
  };
  struct WgmmaGroup {
    ffi::Array<BufferRegion> operands;
    bool conditional;
  };
  std::vector<Tcgen05Group> tcgen05;
  ffi::Array<BufferRegion> uncommitted;
  std::deque<WgmmaGroup> wgmma;
  arith::Analyzer analyzer;

  auto barrier_load = [](const PrimExpr &expr) -> ffi::Optional<BufferLoad> {
    if (auto load = expr.as<BufferLoad>()) {
      return load;
    }
    if (const auto *call = expr.as<CallNode>();
        call && call->op.same_as(access_ptr()) && !call->args.empty()) {
      return call->args[0].as<BufferLoad>();
    }
    return std::nullopt;
  };
  auto same_barrier = [&](const BufferLoad &a, const BufferLoad &b) {
    if (!a->buffer.same_as(b->buffer) ||
        a->indices.size() != b->indices.size()) {
      return false;
    }
    for (size_t j = 0; j < a->indices.size(); ++j) {
      bool mutable_index = false;
      for (const PrimExpr &index : {a->indices[j], b->indices[j]}) {
        PostOrderVisit(index, [&](const ffi::ObjectRef &node) {
          mutable_index |= node.as<BufferLoadNode>() != nullptr;
        });
      }
      if (mutable_index ||
          !analyzer.CanProveEqual(a->indices[j], b->indices[j])) {
        return false;
      }
    }
    return true;
  };
  auto append = [](ffi::Array<BufferRegion> *dst,
                   const ffi::Array<BufferRegion> &src) {
    dst->insert(dst->end(), src.begin(), src.end());
  };
  auto retire_wgmma = [&](size_t i, int64_t retain, bool conditional) {
    // Conditional groups need not have been committed. Only definitely
    // executed newer groups count towards a partial wait's retained suffix.
    int64_t newer = 0;
    size_t first_retained = wgmma.size();
    for (size_t j = wgmma.size(); j > 0; --j) {
      if (newer < retain) {
        first_retained = j - 1;
      }
      newer += !wgmma[j - 1].conditional;
    }
    for (size_t j = 0; j < first_retained; ++j) {
      append(&reads[i], wgmma[j].operands);
    }
    if (!conditional) {
      wgmma.erase(wgmma.begin(), wgmma.begin() + first_retained);
    }
  };

  for (size_t i = 0; i < stmts.size(); ++i) {
    bool conditional = false;
    ffi::Map<Var, arith::IntSet> local_domains;
    PostOrderVisit(stmts[i], [&](const ffi::ObjectRef &node) {
      conditional |= node.as<IfThenElseNode>() || node.as<ForNode>() ||
                     node.as<WhileNode>();
      if (const auto *loop = node.as<ForNode>()) {
        local_domains.Set(loop->loop_var,
                          arith::IntSet::FromRange(
                              Range::FromMinExtent(loop->min, loop->extent)));
      }
      if (const auto *block = node.as<SBlockRealizeNode>()) {
        conditional |= !is_one(block->predicate);
      }
    });
    PostOrderVisit(stmts[i], [&](const ffi::ObjectRef &node) {
      const auto *call = node.as<CallNode>();
      if (!call || !call->op.as<OpNode>()) {
        return;
      }
      if (call->op.same_as(mbarrier_wait_parity())) {
        if (auto barrier = barrier_load(call->args[0])) {
          for (auto it = tcgen05.begin(); it != tcgen05.end();) {
            if (same_barrier(it->barrier, barrier.value())) {
              append(&reads[i], it->operands);
              if (!conditional) {
                it = tcgen05.erase(it);
                continue;
              }
            }
            ++it;
          }
        }
        return;
      }
      if (call->op.as<OpNode>()->name == "tl.tcgen05_mma_arrive") {
        if (auto barrier = barrier_load(call->args[0])) {
          tcgen05.push_back({barrier.value(), uncommitted});
          uncommitted = {};
        }
        return;
      }
      if (call->op.same_as(wait_wgmma()) ||
          call->op.as<OpNode>()->name == "tl.warpgroup_wait") {
        const int64_t *retain = as_const_int(call->args[0]);
        ICHECK(retain && *retain >= 0) << "WGMMA wait count must be constant";
        retire_wgmma(i, *retain, conditional);
        return;
      }
      TileOperator op = ParseOperator(ffi::GetRef<Call>(call));
      if (!op.defined()) {
        return;
      }
      bool is_tcgen05 = false;
      bool is_wgmma = false;
      int wait = 0;
      ffi::Optional<BufferLoad> barrier;
      if (const auto *gemm = op.as<GemmNode>()) {
        is_tcgen05 = gemm->isTcgen05_;
        is_wgmma = !is_tcgen05 && (gemm->isWgmma_ || gemm->wgWait_ != 0);
        wait = gemm->wgWait_;
        if (gemm->mbar_.defined()) {
          barrier = gemm->mbar_;
        }
      } else if (const auto *gemm = op.as<GemmSPNode>()) {
        is_tcgen05 = gemm->isTcgen05_;
        is_wgmma = !is_tcgen05 && (gemm->isWgmma_ || gemm->wg_wait != 0);
        wait = gemm->wg_wait;
      }
      if (!is_tcgen05 && !is_wgmma) {
        return;
      }
      ffi::Array<BufferRegion> operands;
      for (const BufferRegion &region : op->GetAccessRegions().reads) {
        if (IsSharedBuffer(region->buffer)) {
          Region bounds;
          auto sets = arith::EvalSet(region->region, local_domains);
          for (size_t j = 0; j < sets.size(); ++j) {
            bounds.push_back(sets[j].CoverRange(
                Range::FromMinExtent(0, region->buffer->shape[j])));
          }
          operands.push_back(BufferRegion(region->buffer, bounds));
        }
      }
      if (is_tcgen05) {
        append(&uncommitted, operands);
        if (barrier) {
          tcgen05.push_back({barrier.value(), uncommitted});
          uncommitted = {};
        }
      } else if (wait == 0) {
        retire_wgmma(i, 0, conditional);
      } else {
        wgmma.push_back({operands, conditional});
        if (wait > 0) {
          retire_wgmma(i, wait, conditional);
        }
      }
    });
  }
  // An operation not completed in this list remains live at its boundary.
  // In particular a conditional wait cannot shorten an unconditional issue.
  if (!reads.empty()) {
    append(&reads.back(), uncommitted);
    for (const auto &group : tcgen05) {
      append(&reads.back(), group.operands);
    }
    for (const auto &group : wgmma) {
      append(&reads.back(), group.operands);
    }
  }
  return reads;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_

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

/*! \brief Keep shared GEMM operands live through their completion waits.
 *
 * Only straight-line, iteration-local async GEMM protocols are supported.
 * The source must obey the GEMM/barrier contract (including barrier phases).
 * Issue reads still anchor copy waits; completion reads only prevent reuse.
 */
inline std::vector<ffi::Array<BufferRegion>>
CollectAsyncGemmCompletionReads(const ffi::Array<Stmt> &stmts,
                                std::vector<size_t> *events = nullptr) {
  std::vector<ffi::Array<BufferRegion>> reads(stmts.size());
  std::deque<ffi::Array<BufferRegion>> wgmma;
  ffi::Array<BufferRegion> uncommitted, committed;
  ffi::Optional<BufferLoad> commit_barrier;
  bool has_async = false, nested_event = false;
  arith::Analyzer analyzer;
  auto append = [](ffi::Array<BufferRegion> *dst,
                   const ffi::Array<BufferRegion> &src) {
    dst->insert(dst->end(), src.begin(), src.end());
  };
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
  auto commit = [&](const BufferLoad &barrier) {
    if (commit_barrier) {
      LOG(FATAL) << "Async GEMM pipeline requires a completion wait before "
                    "the next TCGEN05 commit";
    }
    commit_barrier = barrier;
    committed = uncommitted;
    uncommitted = {};
  };
  for (size_t i = 0; i < stmts.size(); ++i) {
    bool control_flow = false;
    PostOrderVisit(stmts[i], [&](const ffi::ObjectRef &node) {
      control_flow |= node.as<IfThenElseNode>() || node.as<ForNode>() ||
                      node.as<WhileNode>();
      if (const auto *block = node.as<SBlockRealizeNode>()) {
        control_flow |= !is_one(block->predicate);
      }
    });
    auto record_event = [&]() {
      nested_event |= control_flow;
      if (events && (events->empty() || events->back() != i)) {
        events->push_back(i);
      }
    };
    auto wait_wgmma_groups = [&](int64_t retain) {
      // Even an initially empty wait must not move into a later MMA group.
      record_event();
      while (wgmma.size() > static_cast<size_t>(retain)) {
        append(&reads[i], wgmma.front());
        wgmma.pop_front();
      }
    };
    PostOrderVisit(stmts[i], [&](const ffi::ObjectRef &node) {
      const auto *call = node.as<CallNode>();
      if (!call || !call->op.as<OpNode>()) {
        return;
      }
      if (call->op.same_as(mbarrier_wait_parity())) {
        auto barrier = barrier_load(call->args[0]);
        if (commit_barrier && barrier &&
            commit_barrier.value()->buffer.same_as(barrier.value()->buffer)) {
          const auto &a = commit_barrier.value()->indices;
          const auto &b = barrier.value()->indices;
          bool matches = a.size() == b.size();
          for (size_t j = 0; matches && j < a.size(); ++j) {
            for (const PrimExpr &index : {a[j], b[j]}) {
              PostOrderVisit(index, [&](const ffi::ObjectRef &n) {
                matches &= n.as<BufferLoadNode>() == nullptr;
              });
            }
            matches &= analyzer.CanProveEqual(a[j], b[j]);
          }
          if (matches) {
            record_event();
            append(&reads[i], committed);
            committed = {};
            commit_barrier = std::nullopt;
          }
        }
        return;
      }
      if (call->op.as<OpNode>()->name == "tl.tcgen05_mma_arrive") {
        if (!uncommitted.empty()) {
          record_event();
          auto barrier = barrier_load(call->args[0]);
          ICHECK(barrier) << "Async GEMM commit requires a known barrier";
          commit(barrier.value());
        }
        return;
      }
      if (call->op.same_as(wait_wgmma()) ||
          call->op.as<OpNode>()->name == "tl.warpgroup_wait") {
        const int64_t *retain = as_const_int(call->args[0]);
        ICHECK(retain && *retain >= 0 && *retain <= 7)
            << "WGMMA wait count must be a constant in [0, 7]";
        wait_wgmma_groups(*retain);
        return;
      }
      TileOperator op = ParseOperator(ffi::GetRef<Call>(call));
      if (!op.defined()) {
        return;
      }
      bool is_tcgen05 = false, is_wgmma = false;
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
      if (is_wgmma && wait == 0) {
        wait_wgmma_groups(0);
        return;
      }
      has_async = true;
      record_event();
      ffi::Array<BufferRegion> operands;
      for (const BufferRegion &region : op->GetAccessRegions().reads) {
        if (IsSharedBuffer(region->buffer)) {
          operands.push_back(region);
        }
      }
      if (is_tcgen05) {
        append(&uncommitted, operands);
        if (barrier) {
          commit(barrier.value());
        }
      } else {
        wgmma.push_back(operands);
        if (wait > 0) {
          wait_wgmma_groups(wait);
        }
      }
    });
  }
  if (has_async && nested_event) {
    LOG(FATAL) << "Async GEMM pipeline requires straight-line MMA and "
                  "completion waits; nested control flow is not supported";
  }
  if (!has_async && events) {
    events->clear();
  }
  if (!wgmma.empty() || !uncommitted.empty() || commit_barrier) {
    LOG(FATAL) << "Async GEMM pipeline requires every MMA to complete within "
                  "the iteration; no matching completion wait was found";
  }
  return reads;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_

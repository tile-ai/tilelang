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
#include <algorithm>
#include <deque>
#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <utility>
#include <vector>

#include "../../op/builtin.h"
#include "../../op/gemm.h"
#include "../../op/gemm_sp.h"
#include "../../op/operator.h"
#include "../../op/utils.h"

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

/*!
 * \brief Collect shared-memory operand regions whose asynchronous GEMM reads
 *        complete at each statement in a pipeline body.
 *
 * Asynchronous GEMM tile-ops (`tcgen05_gemm`, `tcgen05_gemm_blockscaled`,
 * `tcgen05_gemm_sp`, `wgmma_gemm`, `wgmma_gemm_sp`, and `gemm`/`gemm_sp` with
 * `wg_wait != 0`) keep reading their shared-memory operands after issue until a
 * subsequent completion wait (`mbarrier_wait_parity` for TCGEN05, or
 * `wait_wgmma` / `warpgroup_wait` for WGMMA) retires them.
 *
 * Synchronous WGMMA tile-ops also complete preceding groups via their implicit
 * wait_group(0). Their own reads are already covered by the issue statement.
 * Returns false if any asynchronous GEMM in `stmts` lacks a provable
 * unconditional completion at a later statement in `stmts`.
 */
inline bool CollectAsyncGemmCompletionReads(
    const ffi::Array<Stmt> &stmts,
    std::vector<ffi::Array<BufferRegion>> *completed_reads) {
  ICHECK(completed_reads != nullptr);
  completed_reads->assign(stmts.size(), ffi::Array<BufferRegion>{});

  arith::Analyzer analyzer;
  auto has_buffer_load = [](const PrimExpr &expr) {
    bool found = false;
    PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
      found |= node.as<BufferLoadNode>() != nullptr;
    });
    return found;
  };
  auto is_conditional_or_loop = [](const Stmt &stmt) {
    bool found = false;
    PostOrderVisit(stmt, [&](const ffi::ObjectRef &node) {
      if (found) {
        return;
      }
      if (node.as<IfThenElseNode>() || node.as<ForNode>() ||
          node.as<WhileNode>()) {
        found = true;
      } else if (const auto *realize = node.as<SBlockRealizeNode>()) {
        if (!is_one(realize->predicate)) {
          found = true;
        }
      }
    });
    return found;
  };
  auto extract_barrier_load =
      [](const PrimExpr &expr) -> ffi::Optional<BufferLoad> {
    if (const auto *load = expr.as<BufferLoadNode>()) {
      return ffi::GetRef<BufferLoad>(load);
    }
    if (const auto *call = expr.as<CallNode>()) {
      if (call->op.same_as(access_ptr()) && !call->args.empty()) {
        if (const auto *load = call->args[0].as<BufferLoadNode>()) {
          return ffi::GetRef<BufferLoad>(load);
        }
      }
    }
    return ffi::Optional<BufferLoad>();
  };

  struct PendingTcgen05Commit {
    size_t commit_stmt_idx;
    BufferLoad barrier;
    ffi::Array<BufferRegion> operands;
    int wait_stmt_idx = -1;
  };
  struct InflightWgmma {
    size_t issue_stmt_idx;
    ffi::Array<BufferRegion> operands;
    bool conditional;
  };

  ffi::Array<BufferRegion> uncommitted_tcgen05;
  std::vector<PendingTcgen05Commit> pending_tcgen05;
  std::deque<InflightWgmma> inflight_wgmma;
  bool valid = true;

  auto check_barrier_indices = [&](const BufferLoad &barrier) {
    for (const PrimExpr &index : barrier->indices) {
      if (has_buffer_load(index)) {
        valid = false;
        return false;
      }
    }
    return true;
  };
  auto same_barrier = [&](const BufferLoad &lhs, const BufferLoad &rhs) {
    if (!lhs->buffer.same_as(rhs->buffer) ||
        lhs->indices.size() != rhs->indices.size()) {
      return false;
    }
    if (!check_barrier_indices(rhs)) {
      return false;
    }
    for (size_t k = 0; k < lhs->indices.size(); ++k) {
      if (!analyzer.CanProveEqual(lhs->indices[k], rhs->indices[k])) {
        return false;
      }
    }
    return true;
  };
  auto commit_tcgen05 = [&](size_t stmt_idx, const BufferLoad &barrier,
                            const ffi::Array<BufferRegion> &extra_operands) {
    if (!check_barrier_indices(barrier)) {
      return;
    }
    ffi::Array<BufferRegion> operands = std::move(uncommitted_tcgen05);
    uncommitted_tcgen05 = {};
    operands.insert(operands.end(), extra_operands.begin(),
                    extra_operands.end());
    if (operands.empty()) {
      return;
    }
    for (auto it = pending_tcgen05.begin(); it != pending_tcgen05.end();) {
      if (same_barrier(it->barrier, barrier)) {
        if (it->wait_stmt_idx == -1) {
          valid = false;
          return;
        }
        auto &dst = (*completed_reads)[it->wait_stmt_idx];
        dst.insert(dst.end(), it->operands.begin(), it->operands.end());
        it = pending_tcgen05.erase(it);
      } else {
        ++it;
      }
    }
    pending_tcgen05.push_back({stmt_idx, barrier, std::move(operands), -1});
  };
  auto retire_wgmma = [&](size_t stmt_idx, int64_t retain) {
    if (retain < 0 ||
        (retain > 0 &&
         std::any_of(
             inflight_wgmma.begin(), inflight_wgmma.end(),
             [](const InflightWgmma &e) { return e.conditional; }))) {
      valid = false;
      return;
    }
    while (inflight_wgmma.size() > static_cast<size_t>(retain)) {
      InflightWgmma entry = std::move(inflight_wgmma.front());
      inflight_wgmma.pop_front();
      if (entry.issue_stmt_idx >= stmt_idx) {
        valid = false;
        return;
      }
      auto &dst = (*completed_reads)[stmt_idx];
      dst.insert(dst.end(), entry.operands.begin(), entry.operands.end());
    }
  };

  for (size_t i = 0; i < stmts.size() && valid; ++i) {
    bool conditional = is_conditional_or_loop(stmts[i]);
    PostOrderVisit(stmts[i], [&](const ffi::ObjectRef &node) {
      if (!valid) {
        return;
      }
      const auto *call = node.as<CallNode>();
      if (!call) {
        return;
      }
      const auto *op_node = call->op.as<OpNode>();
      if (call->op.same_as(mbarrier_wait_parity()) && !call->args.empty()) {
        if (conditional) {
          return;
        }
        auto wait_opt = extract_barrier_load(call->args[0]);
        if (!wait_opt.defined()) {
          return;
        }
        for (auto &commit : pending_tcgen05) {
          if (commit.commit_stmt_idx < i &&
              same_barrier(commit.barrier, wait_opt.value())) {
            if (!valid || commit.wait_stmt_idx != -1) {
              valid = false;
              return;
            }
            commit.wait_stmt_idx = static_cast<int>(i);
          }
        }
        return;
      }
      if (op_node && op_node->name == "tl.tcgen05_mma_arrive" &&
          !call->args.empty()) {
        if (uncommitted_tcgen05.empty()) {
          return;
        }
        auto mbar_opt = extract_barrier_load(call->args[0]);
        if (conditional || !mbar_opt.defined()) {
          valid = false;
          return;
        }
        commit_tcgen05(i, mbar_opt.value(), {});
        return;
      }
      if (call->op.same_as(wait_wgmma()) ||
          (op_node && op_node->name == "tl.warpgroup_wait")) {
        if (!inflight_wgmma.empty() && !conditional) {
          const int64_t *retain =
              call->args.empty() ? nullptr : as_const_int(call->args[0]);
          retire_wgmma(i, retain ? *retain : -1);
        }
        return;
      }
      if (!op_node) {
        return;
      }
      TileOperator tile_op = ParseOperator(ffi::GetRef<Call>(call));
      if (!tile_op.defined()) {
        return;
      }
      bool is_tcgen05 = false;
      bool is_async_wgmma = false;
      int wg_wait = 0;
      ffi::Optional<BufferLoad> gemm_mbar;
      if (const auto *gemm = tile_op.as<GemmNode>()) {
        is_tcgen05 = gemm->isTcgen05_;
        is_async_wgmma =
            !gemm->isTcgen05_ && (gemm->isWgmma_ || gemm->wgWait_ != 0);
        wg_wait = gemm->wgWait_;
        if (gemm->mbar_.defined()) {
          gemm_mbar = gemm->mbar_;
        }
      } else if (const auto *gemm_sp = tile_op.as<GemmSPNode>()) {
        is_tcgen05 = gemm_sp->isTcgen05_;
        is_async_wgmma = !gemm_sp->isTcgen05_ &&
                         (gemm_sp->isWgmma_ || gemm_sp->wg_wait != 0);
        wg_wait = gemm_sp->wg_wait;
      }
      if (!is_tcgen05 && !is_async_wgmma) {
        return;
      }
      if (is_async_wgmma && wg_wait == 0) {
        // Retire only preceding issues: this operation completes its own reads
        // before returning, so it must not enter the asynchronous queue.
        if (!conditional) {
          retire_wgmma(i, 0);
        }
        return;
      }
      ffi::Array<BufferRegion> operands;
      for (const BufferRegion &region : tile_op->GetAccessRegions().reads) {
        if (IsSharedBuffer(region->buffer)) {
          operands.push_back(region);
        }
      }
      if (operands.empty()) {
        return;
      }
      if (is_tcgen05) {
        if (gemm_mbar.defined()) {
          if (conditional) {
            valid = false;
            return;
          }
          commit_tcgen05(i, gemm_mbar.value(), operands);
        } else {
          if (conditional) {
            valid = false;
            return;
          }
          uncommitted_tcgen05.insert(uncommitted_tcgen05.end(),
                                     operands.begin(), operands.end());
        }
      } else {
        inflight_wgmma.push_back({i, std::move(operands), conditional});
        if (wg_wait > 0 && !conditional) {
          retire_wgmma(i, wg_wait);
        }
      }
    });
  }

  if (!valid || !uncommitted_tcgen05.empty() || !inflight_wgmma.empty()) {
    completed_reads->assign(stmts.size(), ffi::Array<BufferRegion>{});
    return false;
  }
  for (const auto &commit : pending_tcgen05) {
    if (commit.wait_stmt_idx == -1) {
      completed_reads->assign(stmts.size(), ffi::Array<BufferRegion>{});
      return false;
    }
    auto &dst = (*completed_reads)[commit.wait_stmt_idx];
    dst.insert(dst.end(), commit.operands.begin(), commit.operands.end());
  }
  return true;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_PIPELINE_UTILS_H_

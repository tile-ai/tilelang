/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file loop_partition.cc
 * \brief Partition parallel loops onto threads
 */

#include "loop_partition.h"
#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>

#include <tvm/tirx/stmt_functor.h>

#include <utility>

#include "../op/utils.h"
#include "loop_vectorize.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class BufferIndiceSimplify : public StmtExprMutator {
public:
  BufferIndiceSimplify(arith::Analyzer *analyzer) : analyzer_(analyzer) {}

private:
  PrimExpr VisitExpr_(const BufferLoadNode *node) final {
    auto visited = StmtExprMutator::VisitExpr_(node);
    auto n = Downcast<BufferLoad>(visited);
    auto nptr = n.CopyOnWrite();
    nptr->indices = nptr->indices.Map(
        [&](const auto &e) { return analyzer_->Simplify(e); });
    return n;
  }
  Stmt VisitStmt_(const BufferStoreNode *node) final {
    auto visited = StmtExprMutator::VisitStmt_(node);
    auto n = Downcast<BufferStore>(visited);
    auto nptr = n.CopyOnWrite();
    nptr->indices = nptr->indices.Map(
        [&](const auto &e) { return analyzer_->Simplify(e); });
    return n;
  }
  arith::Analyzer *analyzer_;
};

// Rewrite the parallel loop into a common loop, which is mapped to threads
For PartitionLoop(For op, Var thread_var, arith::Analyzer *analyzer,
                  const Fragment &loop_layout) {
  ICHECK(loop_layout.defined());
  ICHECK(thread_var.defined());
  int old_loop_depth = loop_layout->InputDim();
  int new_loop_depth = loop_layout->OutputDim();
  // Create the new loop iter var
  Array<Var> vars;
  for (int i = 0; i < new_loop_depth; i++) {
    Var var = Var(std::string{char('i' + i)});
    analyzer->Bind(var, Range::FromMinExtent(make_zero(var->dtype),
                                             loop_layout->OutputShape()[i]));
    vars.push_back(var);
  }
  vars.push_back(thread_var);
  // create the substitute map, and the loop body
  Map<Var, PrimExpr> vmap;
  Stmt body = std::move(op);
  Array<PrimExpr> loop_mins;
  Array<PrimExpr> loop_extents;
  auto inverse_info = loop_layout->InverseWithLevel();
  auto inv_loop = inverse_info.first;
  auto indices = inv_loop->Forward(Array<PrimExpr>(vars.begin(), vars.end()));
  // Normalize thread var once so we can reuse the same substitution later.
  Map<Var, PrimExpr> thread_offset_map;
  bool has_thread_offset = false;
  if (loop_layout->ThreadRange().defined()) {
    auto range = loop_layout->ThreadRange();
    thread_offset_map.Set(thread_var, thread_var - range->min);
    has_thread_offset = true;
  }
  for (int i = 0; i < old_loop_depth; i++) {
    const ForNode *loop = body.as<ForNode>();
    ICHECK(loop != nullptr)
        << "No extra statements are allowed between nested parallel loops.";
    vmap.Set(loop->loop_var, indices[i]);
    loop_mins.push_back(loop->min);
    loop_extents.push_back(loop->extent);
    body = loop->body;
  }
  // substitute and re-construct the serial loop
  body = Substitute(body, vmap);
  // Guard executes the recovered loop body only if each inverse-mapped iterator
  // falls back into the original For ranges. We first check every axis from the
  // old loop nest (old_loop_depth) and then the extra index produced by inverse
  // layouts that carry a replicate/thread component (`inv_output_shape`). Both
  // must stay within bounds to ensure correctness. Example: layout([i, j]) =
  // floor((i * 16 + j) / 32) may generate extra points when the new loop
  // enumerates 0..31; the guard drops iterations whose inverse-mapped (i, j)
  // or replicate index fall outside their original extents.
  // Example: layout([i, j]) = floor((i * 16 + j) / 32) may produce extra points
  // when the new loop enumerates 0..31; this guard skips iterations where the
  // inverse i, j land outside the original extents. This protects
  // non-surjective loop_layout mappings that otherwise over-cover the parallel
  // space.
  // Always build guard and let analyzer decide if it can be proved true.
  // This handles both non-bijective layouts and cases where loop extent
  // differs from layout input shape (e.g., loop extent=4 with
  // Fragment([8]->[1]) produces inverse index `tx % 8` ranging 0-7, requiring
  // guard `tx % 8 < 4`).
  PrimExpr guard = const_true();
  for (int i = 0; i < old_loop_depth; i++) {
    PrimExpr index = indices[i];
    if (has_thread_offset) {
      index = Substitute(index, thread_offset_map);
    }
    PrimExpr lower_bound = analyzer->Simplify(index >= loop_mins[i]);
    PrimExpr upper_bound =
        analyzer->Simplify(index < loop_mins[i] + loop_extents[i]);
    guard = And(guard, And(lower_bound, upper_bound));
  }
  auto inv_output_shape = inv_loop->OutputShape();
  if (inv_output_shape.size() > static_cast<size_t>(old_loop_depth)) {
    PrimExpr replicate_index = indices[old_loop_depth];
    if (has_thread_offset) {
      replicate_index = Substitute(replicate_index, thread_offset_map);
    }
    PrimExpr replicate_extent = inv_output_shape[old_loop_depth];
    PrimExpr lower_bound = analyzer->Simplify(
        replicate_index >= make_zero(replicate_index.dtype()));
    PrimExpr upper_bound =
        analyzer->Simplify(replicate_index < replicate_extent);
    guard = And(guard, And(lower_bound, upper_bound));
  }
  PrimExpr simplified_guard = analyzer->Simplify(guard);
  if (!analyzer->CanProve(simplified_guard)) {
    body = IfThenElse(simplified_guard, body, Stmt());
  }

  for (int i = new_loop_depth - 1; i >= 0; i--) {
    body = For(vars[i], make_zero(vars[i]->dtype), inv_loop->InputShape()[i],
               ForKind::kSerial, body);
    analyzer->Bind(vars[i], Range(0, inv_loop->InputShape()[i]));
  }

  body = BufferIndiceSimplify(analyzer)(body);

  if (has_thread_offset) {
    body = Substitute(body, thread_offset_map);
  }
  return Downcast<For>(body);
}

class LoopPramaUnroller : public StmtExprMutator {
public:
  LoopPramaUnroller() = default;

private:
  Stmt VisitStmt_(const ForNode *node) final {
    if (node->kind == ForKind::kSerial) {
      auto analyzer = std::make_shared<arith::Analyzer>();
      if (as_const_int(analyzer->Simplify(node->extent)) == nullptr) {
        return StmtExprMutator::VisitStmt_(node);
      }
      For new_for = GetRef<For>(node);
      auto for_ptr = new_for.CopyOnWrite();
      for_ptr->annotations.Set(tirx::attr::pragma_unroll_explicit, Bool(false));
      for_ptr->kind = ForKind::kUnrolled;
      return new_for;
    }
    return StmtExprMutator::VisitStmt_(node);
  }
};

class LoopPartitioner : public StmtExprVisitor {
public:
  LoopPartitioner() = default;

  Fragment Partition(const For &op, int num_thread, int vectorize_size,
                     int replicate_num_thread = -1) {
    this->VisitStmt(op);
    if (replicate_num_thread < 0) {
      replicate_num_thread = num_thread;
    }
    DataType dtype = DataType::Int(32);
    if (!loop_vars_.empty()) {
      dtype = loop_vars_.back()->var.dtype();
    }
    PrimExpr flattened = make_const(dtype, 0);
    PrimExpr vector_extent = make_const(dtype, vectorize_size);
    PrimExpr thread_extent_const = make_const(dtype, num_thread);
    for (size_t i = 0; i < loop_vars_.size(); i++) {
      PrimExpr extent = loop_vars_[i]->dom->extent;
      flattened = flattened * extent + loop_vars_[i]->var;
    }
    PrimExpr access_idx = FloorDiv(flattened, vector_extent);
    PrimExpr thd = FloorMod(access_idx, thread_extent_const);
    PrimExpr idx = FloorDiv(access_idx, thread_extent_const) * vector_extent +
                   FloorMod(flattened, vector_extent);

    auto fragment = Fragment(loop_vars_, {idx}, {thd}, {});
    if (has_fragment_) {
      // Fragment loops need a layout for the whole block. Build the base
      // partition with active threads, then replicate it across the full block
      // when the active width is a factor of the block size.
      auto thread_extent = *as_const_int(fragment->ThreadExtent());
      ICHECK_EQ(replicate_num_thread % thread_extent, 0)
          << "Cannot replicate fragment loop layout with thread extent "
          << thread_extent << " across " << replicate_num_thread << " threads";
      auto num_thread_fragment = replicate_num_thread / thread_extent;
      fragment = fragment->Replicate(num_thread_fragment);
    }
    return fragment;
  }

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    if (IsFragmentBuffer(op->buffer)) {
      has_fragment_ = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (IsFragmentBuffer(op->buffer)) {
      has_fragment_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const ForNode *node) final {
    if (node->kind == ForKind::kParallel) {
      body_ = node->body;
      loop_vars_.push_back(
          IterVar(Range::FromMinExtent(node->min, node->extent), node->loop_var,
                  IterVarType::kDataPar));
    }
    StmtExprVisitor::VisitStmt_(node);
  }

  Stmt body_;
  PrimExpr flattened = 0;
  bool has_fragment_ = false;
  Array<IterVar> loop_vars_;
};

class LoopPartitionFragmentAccessDetector : public StmtExprVisitor {
public:
  bool HasFragmentAccess(const For &op) {
    VisitStmt(op);
    return has_fragment_;
  }

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    if (IsFragmentBuffer(op->buffer)) {
      has_fragment_ = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (IsFragmentBuffer(op->buffer)) {
      has_fragment_ = true;
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  bool has_fragment_ = false;
};

static PrimExpr ComputeLoopTotalSize(const For &op) {
  PrimExpr loop_total_size = 1;
  for (Stmt l = op; l.as<For>().has_value(); l = l.as<For>().value()->body) {
    loop_total_size = loop_total_size * l.as<For>().value()->extent;
  }
  return loop_total_size;
}

int64_t SelectActiveThreadExtent(const For &op, int64_t max_num_thread,
                                 int vectorize_size, arith::Analyzer *analyzer,
                                 bool require_full_thread_replication) {
  ICHECK_GT(max_num_thread, 0);
  ICHECK_GT(vectorize_size, 0);

  arith::Analyzer local_analyzer;
  arith::Analyzer *az = analyzer != nullptr ? analyzer : &local_analyzer;
  PrimExpr loop_total_size = ComputeLoopTotalSize(op);

  for (int64_t active_threads = max_num_thread; active_threads >= 1;
       --active_threads) {
    if (require_full_thread_replication &&
        max_num_thread % active_threads != 0) {
      continue;
    }
    PrimExpr partition_width = Integer(active_threads * vectorize_size);
    if (az->CanProve(floormod(loop_total_size, partition_width) == 0)) {
      return active_threads;
    }
  }
  return 0;
}

Fragment PlanLoopPartition(const For &op, size_t num_thread,
                           int vectorize_size) {
  LoopPartitioner partitioner;
  return partitioner.Partition(op, num_thread, vectorize_size);
}

Fragment PlanLoopPartition(const For &op, int vectorize_size,
                           const Range &thread_range, arith::Analyzer *analyzer,
                           bool require_full_thread_replication) {
  const int64_t *num_thread_ptr = as_const_int(thread_range->extent);
  ICHECK(num_thread_ptr != nullptr)
      << "PlanLoopPartition requires constant thread extent, got "
      << thread_range;
  int64_t num_thread = *num_thread_ptr;
  bool needs_full_thread_replication =
      require_full_thread_replication ||
      LoopPartitionFragmentAccessDetector().HasFragmentAccess(op);
  int64_t active_threads = SelectActiveThreadExtent(
      op, num_thread, vectorize_size, analyzer, needs_full_thread_replication);
  ICHECK_NE(active_threads, 0)
      << "Cannot find an active thread extent <= " << num_thread
      << " that evenly partitions loop_total_size=" << ComputeLoopTotalSize(op)
      << " with vector_size=" << vectorize_size;
  if (active_threads != num_thread) {
    if (needs_full_thread_replication) {
      DLOG(INFO) << "[PlanLoopPartition] using " << active_threads
                 << "-thread fragment partition replicated across "
                 << num_thread << " block threads.";
    } else {
      DLOG(INFO) << "[PlanLoopPartition] using " << active_threads
                 << " active threads out of " << num_thread
                 << " to avoid a ragged loop layout.";
    }
  }
  LoopPartitioner partitioner;
  Fragment fragment = partitioner.Partition(
      op, static_cast<int>(active_threads), vectorize_size,
      needs_full_thread_replication ? static_cast<int>(num_thread)
                                    : static_cast<int>(active_threads));
  return fragment->BindThreadRange(thread_range);
}

For PragmaUnrollLoop(For stmt) {
  LoopPramaUnroller unroller;
  For unrolled = Downcast<For>(unroller(std::move(stmt)));
  return unrolled;
}

Stmt LowerParallelLoop(For loop, const Fragment &loop_layout, Var thread_var,
                       arith::Analyzer *analyzer, const LayoutMap &layout_map,
                       Optional<PrimExpr> predicate, bool parallel_loop,
                       bool should_vectorize) {
  // Save analyzer state to prevent conflicted bindings during vectorization
  auto saved_analyzer = analyzer->Clone();

  For result_loop = loop;
  // Strip parallel-loop layout/predicate annotations on the original loop.
  // After partitioning/vectorization, keeping them can confuse later passes.
  // Also, annotations may contain complex expressions; mutators do not visit
  // inside annotation payloads, so explicit removal here prevents stale state
  // from leaking into subsequent transforms.
  // Note: Map::erase(key) is a no-op if key doesn't exist.
  result_loop.CopyOnWrite()->annotations.erase(attr::kParallelLoopLayout);
  result_loop.CopyOnWrite()->annotations.erase(attr::kParallelLoopPredicate);

  // Step 1: Partition the loop based on the layout (if this is a parallel loop)
  if (parallel_loop) {
    result_loop = PartitionLoop(result_loop, thread_var, analyzer, loop_layout);
  }

  // Step 2: Vectorize the loop (if requested)
  if (should_vectorize) {
    result_loop = VectorizeLoop(result_loop, saved_analyzer.get(), layout_map);
  }

  result_loop = PragmaUnrollLoop(result_loop);

  // Step 3: Wrap with predicate if provided and this is a parallel loop
  if (predicate.defined() && parallel_loop) {
    return IfThenElse(predicate.value(), result_loop);
  }

  return result_loop;
}

} // namespace tl
} // namespace tvm

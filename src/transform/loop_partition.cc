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

#include <tvm/tir/stmt_functor.h>

#include <utility>

#include "../op/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

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
  // Must check the guard if the layout can not be proved as bijective
  bool need_guard = inverse_info.second != arith::IterMapLevel::Bijective;
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
  PrimExpr guard = const_true();

  if (need_guard) {
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

  auto for_node = LoopPragmaUnroll(Downcast<For>(body));
  return for_node;
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
      For new_for = tvm::ffi::GetRef<For>(node);
      auto for_ptr = new_for.CopyOnWrite();
      for_ptr->annotations.Set(tir::attr::pragma_unroll_explicit, Bool(false));
      for_ptr->kind = ForKind::kUnrolled;
      return new_for;
    }
    return StmtExprMutator::VisitStmt_(node);
  }
};

class LoopPartitioner : public StmtExprVisitor {
public:
  LoopPartitioner() = default;

  Fragment Partition(const For &op, int num_thread, int vectorize_size) {
    this->VisitStmt(op);
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
      // for fragment buffer, we don't need to replicate the loop layout
      auto thread_extent = *as_const_int(fragment->ThreadExtent());
      auto num_thread_fragment = num_thread / thread_extent;
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

Fragment PlanLoopPartition(const For &op, size_t num_thread,
                           int vectorize_size) {
  LoopPartitioner partitioner;
  return partitioner.Partition(op, num_thread, vectorize_size);
}

Fragment PlanLoopPartition(const For &op, int vectorize_size,
                           const Range &thread_range) {
  size_t num_thread = *as_const_int(thread_range->extent);
  LoopPartitioner partitioner;
  Fragment fragment = partitioner.Partition(op, num_thread, vectorize_size);
  return fragment->BindThreadRange(thread_range);
}

For LoopPragmaUnroll(For stmt) {
  LoopPramaUnroller unroller;
  For unrolled = Downcast<For>(unroller(std::move(stmt)));
  return unrolled;
}

} // namespace tl
} // namespace tvm

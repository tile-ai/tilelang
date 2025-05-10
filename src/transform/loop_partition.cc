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
                  Fragment loop_layout) {
  ICHECK(loop_layout.defined());
  ICHECK(thread_var.defined());
  int old_loop_depth = loop_layout->InputDim();
  int new_loop_depth = loop_layout->OutputDim();

  // Create the new loop iter var
  Array<Var> vars;
  for (int i = 0; i < new_loop_depth; i++) {
    Var var = Var(std::string{char('i' + i)});
    vars.push_back(var);
  }
  vars.push_back(thread_var);
  // create the substitute map, and the loop body
  Map<Var, PrimExpr> vmap;
  Stmt body = op;
  auto inv_loop = loop_layout->Inverse();
  auto indices = inv_loop->Forward(Array<PrimExpr>(vars.begin(), vars.end()));
  for (int i = 0; i < old_loop_depth; i++) {
    const ForNode *loop = body.as<ForNode>();
    ICHECK(loop != nullptr);
    vmap.Set(loop->loop_var, indices[i]);
    body = loop->body;
  }

  // substitute and re-construct the serial loop
  body = Substitute(body, vmap);
  for (int i = new_loop_depth - 1; i >= 0; i--) {
    body = For(vars[i], make_zero(vars[i]->dtype), inv_loop->InputShape()[i],
               ForKind::kSerial, body);
    analyzer->Bind(vars[i], Range(0, inv_loop->InputShape()[i]));
  }

  body = BufferIndiceSimplify(analyzer)(body);

  auto for_node = LoopPragmaUnroll(Downcast<For>(body));
  if (loop_layout->ThreadRange().defined()) {
    auto range = loop_layout->ThreadRange();
    auto thread_var_with_offset = thread_var - range->min;
    for_node.CopyOnWrite()->body =
        Substitute(for_node->body, {{thread_var, thread_var_with_offset}});
  }
  return for_node;
}

class LoopPramaUnroller : public StmtExprMutator {
public:
  LoopPramaUnroller() = default;

private:
  Stmt VisitStmt_(const ForNode *node) final {
    if (node->kind == ForKind::kSerial) {
      For new_for = GetRef<For>(node);
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

  Fragment Partition(For op, int num_thread, int vectorize_size) {
    this->VisitStmt(op);
    int loop_size_full = 1;
    PrimExpr flattened = 0;
    for (size_t i = 0; i < loop_vars_.size(); i++) {
      auto ext_ptr = as_const_int(loop_vars_[i]->dom->extent);
      ICHECK(ext_ptr)
          << "Loop partitioner only works with constant loop sizes, but got "
          << loop_vars_[i]->dom->extent;
      int extent = *ext_ptr;
      loop_size_full *= extent;
      flattened = flattened * extent + loop_vars_[i]->var;
    }
    ICHECK(loop_size_full % vectorize_size == 0);
    PrimExpr access_idx = FloorDiv(flattened, vectorize_size);
    PrimExpr thd = FloorMod(access_idx, num_thread);
    PrimExpr idx = FloorDiv(access_idx, num_thread) * vectorize_size +
                   FloorMod(flattened, vectorize_size);
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
    if (op->buffer.scope() == "local.fragment") {
      has_fragment_ = true;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (op->buffer.scope() == "local.fragment") {
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

Fragment PlanLoopPartition(For op, size_t num_thread, int vectorize_size) {
  LoopPartitioner partitioner;
  return partitioner.Partition(op, num_thread, vectorize_size);
}

Fragment PlanLoopPartition(For op, int vectorize_size, Range thread_range) {
  size_t num_thread = *as_const_int(thread_range->extent);
  LoopPartitioner partitioner;
  Fragment fragment = partitioner.Partition(op, num_thread, vectorize_size);
  return fragment->BindThreadRange(thread_range);
}

For LoopPragmaUnroll(For stmt) {
  LoopPramaUnroller unroller;
  For unrolled = Downcast<For>(unroller(stmt));
  return unrolled;
}

} // namespace tl
} // namespace tvm

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
 * \brief Native thread partitioning, guarded replicas and loop intent.
 */
#include "loop_partition.h"
#include "../op/utils.h"
#include "ir/analyzer.h"
#include "ir/annotations.h"
#include "ir/loop_math.h"
#include "ir/stmt_functor.h"
#include "layout/expr_utils.h"
#include "loop_vectorize.h"
#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl {
using namespace tirx;
using namespace prim;
using namespace ffi;

class BufferIndiceSimplify : public LayoutMapMutator {
public:
  explicit BufferIndiceSimplify(sym::AnalyzerObj *analyzer)
      : analyzer_(analyzer) {}
  using LayoutMapMutator::Mutate_;
  UnchangedOr<Stmt> Mutate_(const BindObj *node, InplaceMode mode) final {
    auto result = LayoutMapMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(GetRef<Stmt>(node))
                      .as_or_throw<Bind>();
    if (auto var = result->var.as<PrimVar>()) {
      if (auto expr = result->value.as<PrimExpr>();
          expr && SideEffect(*expr) <= CallEffectKind::kPure)
        BindAnalyzer(analyzer_, *var, *expr, true);
    }
    return result;
  }
  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *node,
                                InplaceMode mode) final {
    auto result = LayoutMapMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(PrimExpr(GetRef<TensorLoad>(node)))
                      .as_or_throw<TensorLoad>();
    auto indices = result->indices.Map(
        [&](const PrimExpr &index) { return analyzer_->Simplify(index); });
    return TensorLoad(result->source, indices, result->loc);
  }
  UnchangedOr<Stmt> Mutate_(const TensorStoreObj *node,
                            InplaceMode mode) final {
    auto result = LayoutMapMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(GetRef<Stmt>(node))
                      .as_or_throw<TensorStore>();
    auto indices = result->indices.Map(
        [&](const PrimExpr &index) { return analyzer_->Simplify(index); });
    return TensorStore(result->dest, indices, result->value, result->loc);
  }

private:
  sym::AnalyzerObj *analyzer_;
};

class MultiplicityMarkerLowerer : public LayoutMapMutator {
public:
  explicit MultiplicityMarkerLowerer(Optional<PrimExpr> guard)
      : guard_(std::move(guard)) {}
  using LayoutMapMutator::Mutate_;
  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    if (op->op->name == "tl.attr") {
      auto metadata = ReadAnnotations(op);
      if (auto key = metadata.Get("key");
          key && key->cast<String>() == "tl.parallel_multiplicity") {
        ICHECK(op->body_params.empty() && op->result_vars.empty());
        SeqStmt body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
        if (guard_)
          return If(*guard_, body, std::nullopt, op->loc);
        return body;
      }
    }
    return LayoutMapMutator::Mutate_(op, mode);
  }

private:
  Optional<PrimExpr> guard_;
};

namespace {

// A loop slice can have a lexical thread origin, e.g. i*32 + k/8 with
// serial i in [0,4). Only the relative k/8 part belongs to the inverse's
// rectangular thread domain. Normalize this query locally after proving the
// origin and domain fit the real participant range; buffer layouts and the
// caller's absolute-owner guards keep their original coordinates.
std::pair<Fragment, PrimExpr>
PartitionInverseView(const Fragment &layout, PrimExpr thread,
                     const sym::AnalyzerObj *context) {
  auto range = layout->ThreadRange();
  if (!range)
    return {layout, thread};
  auto proof = context ? context->Clone() : sym::Analyzer();
  Map<PrimVar, PrimExpr> zero;
  for (int i = 0; i < layout->InputDim(); ++i) {
    PrimVar input = InputPlaceholder(i);
    zero.Set(input, IntImm(input.ty(), 0));
  }
  PrimVar replica = ReplicationPlaceholder();
  zero.Set(replica, IntImm(replica.ty(), 0));
  PrimExpr origin =
      proof->Simplify(SubstituteLayoutVars(layout->GetForwardThread(), zero));
  if (IsZero(origin))
    return {layout, thread};
  PrimExpr relative = proof->Simplify(layout->GetForwardThread() - origin);
  Fragment normalized(layout->InputShape(), layout->GetForwardIndex(), relative,
                      layout->ReplicateExtent(), std::nullopt);
  auto extent = TryConstInt(normalized->ThreadExtent());
  if (!extent || *extent <= 0)
    return {layout, thread};
  for (int i = 0; i < layout->InputDim(); ++i) {
    PrimVar input = InputPlaceholder(i);
    BindAnalyzer(
        proof, input,
        Range::FromMinExtent(IntImm(input.ty(), 0), layout->InputShape()[i]));
  }
  BindAnalyzer(
      proof, replica,
      Range::FromMinExtent(IntImm(replica.ty(), 0), layout->ReplicateExtent()));
  PrimExpr bound = IntImm(relative.ty(), *extent);
  if (!proof->CanProve(origin >= IntImm(origin.ty(), 0)) ||
      !proof->CanProve(range->extent >= bound) ||
      !proof->CanProve(origin <= range->extent - bound) ||
      !proof->CanProve(relative >= IntImm(relative.ty(), 0)) ||
      !proof->CanProve(relative < bound))
    return {layout, thread};
  return {normalized, proof->Simplify(thread - origin)};
}

// NoCheck supplies an inverse candidate for a padded physical domain. Prove
// that it recovers every logical element and replica before using it; a bounds
// guard alone cannot repair an inverse that loses valid logical coordinates.
void ValidateGuardedInverse(const Fragment &layout, const Layout &inverse,
                            const sym::AnalyzerObj *context) {
  auto analyzer = context ? context->Clone() : sym::Analyzer();
  Array<PrimExpr> logical;
  PrimExpr nonempty = IntImm::Bool(true);
  for (const PrimExpr &extent : layout->InputShape()) {
    PrimVar index("guarded_logical_" + std::to_string(logical.size()),
                  extent.ty());
    BindAnalyzer(analyzer, index,
                 Range::FromMinExtent(IntImm(index.ty(), 0), extent));
    logical.push_back(index);
    nonempty = nonempty && (extent > IntImm(extent.ty(), 0));
  }
  PrimExpr replicas = layout->ReplicateExtent();
  PrimVar replica("guarded_replica", replicas.ty());
  BindAnalyzer(analyzer, replica,
               Range::FromMinExtent(IntImm(replica.ty(), 0), replicas));
  nonempty = nonempty && (replicas > IntImm(replicas.ty(), 0));
  ScopedAnalyzerConstraints constraint(analyzer, nonempty);
  auto physical = layout->Forward(logical);
  physical.push_back(layout->ForwardThread(logical, replica));
  auto recovered = inverse->Forward(physical);
  logical.push_back(replica);
  TVM_FFI_CHECK_EQ(recovered.size(), logical.size(), ValueError)
      << "Guarded loop inverse must preserve every logical axis and replica";
  for (size_t i = 0; i < logical.size(); ++i) {
    TVM_FFI_CHECK(analyzer->CanProveEqual(recovered[i], logical[i]), ValueError)
        << "Guarded loop inverse does not recover logical coordinate " << i
        << " for " << layout->DebugOutput();
  }
}

} // namespace

For PartitionLoop(For op, PrimExpr thread_index, sym::AnalyzerObj *analyzer,
                  const Fragment &layout, bool require_padding_guard) {
  ICHECK(layout.defined());
  const Location loc = op->loc;
  const int old_depth = layout->InputDim(), new_depth = layout->OutputDim();
  Array<PrimVar> vars;
  for (int i = 0; i < new_depth; ++i) {
    PrimExpr extent = layout->OutputShape()[i];
    PrimVar var("p" + std::to_string(i), extent.ty());
    BindAnalyzer(analyzer, var,
                 Range::FromMinExtent(IntImm(var.ty(), 0), extent));
    vars.push_back(var);
  }
  PrimExpr thread = thread_index;
  if (auto range = layout->ThreadRange())
    thread = thread - range->begin;
  auto inverse_view = PartitionInverseView(layout, thread, analyzer);
  Array<PrimExpr> input(vars.begin(), vars.end());
  input.push_back(inverse_view.second);
  Layout inverse =
      inverse_view.first->InverseWithLevel(require_padding_guard).first;
  if (require_padding_guard)
    ValidateGuardedInverse(inverse_view.first, inverse, analyzer);
  auto indices = inverse->Forward(input);
  ICHECK_GE(indices.size(), static_cast<size_t>(old_depth));
  Map<Var, PrimExpr> replacements;
  Stmt body = op;
  Array<PrimExpr> begins, ends;
  for (int i = 0; i < old_depth; ++i) {
    const auto *loop = SingleStatement(body).as<ForObj>();
    ICHECK(loop) << "No statements are allowed between the parallel loops";
    (void)LoopExtent(loop); // validates the explicit unit step
    replacements.Set(loop->var, indices[i]);
    begins.push_back(loop->begin);
    ends.push_back(loop->end);
    body = loop->body;
  }
  body = tl::SubstituteWithDataTypeLegalization(
      body, [&](const Var &var) -> ffi::Optional<PrimExpr> {
        return replacements.Get(var);
      });
  PrimExpr guard = IntImm::Bool(true);
  if (!inverse_view.first.same_as(layout)) {
    // The normalized inverse has a smaller physical thread domain. Guard it
    // explicitly: an inverse may use modulo and otherwise wrap a participant
    // outside this slice back onto a valid logical coordinate.
    const PrimExpr &relative_thread = inverse_view.second;
    guard = (relative_thread >= IntImm(relative_thread.ty(), 0)) &&
            (relative_thread < inverse_view.first->ThreadExtent());
  }
  for (int i = 0; i < old_depth; ++i)
    guard = guard && (indices[i] >= begins[i]) && (indices[i] < ends[i]);
  if (indices.size() > static_cast<size_t>(old_depth)) {
    PrimExpr replica = indices[old_depth];
    guard = guard && (replica >= IntImm(replica.ty(), 0)) &&
            (replica < layout->ReplicateExtent());
  }
  if (require_padding_guard) {
    // A rectangular physical domain may contain holes as well as a final
    // tail. Admit only points in the original forward image, so padding never
    // duplicates a logical store or assigns an element to another thread.
    Array<PrimExpr> logical(indices.begin(), indices.begin() + old_depth);
    auto physical = layout->Forward(logical);
    ICHECK_EQ(physical.size(), vars.size());
    for (size_t i = 0; i < physical.size(); ++i)
      guard = guard && (physical[i] == vars[i]);
    guard =
        guard && (layout->ForwardThread(logical, indices[old_depth]) == thread);
  }
  Optional<PrimExpr> replica_guard;
  if (indices.size() > static_cast<size_t>(old_depth)) {
    PrimExpr replica = indices[old_depth];
    PrimExpr zero_replica =
        analyzer->Simplify(replica == IntImm(replica.ty(), 0));
    if (!analyzer->CanProve(zero_replica))
      replica_guard = zero_replica;
  }
  auto multiplicity = make_object<MultiplicityMarkerLowerer>(replica_guard);
  body = multiplicity->Mutate(body).ValueOrUnchanged(body);
  guard = analyzer->Simplify(guard);
  if (!analyzer->CanProve(guard))
    body = If(guard, SeqStmt(body), std::nullopt, loc);
  for (int i = new_depth - 1; i >= 0; --i) {
    PrimExpr extent = inverse->InputShape()[i];
    body =
        For(vars[i], IntImm(vars[i].ty(), 0), extent, IntImm(vars[i].ty(), 1),
            ForKind::kDefault, SeqStmt(body), {}, loc);
    BindAnalyzer(analyzer, vars[i],
                 Range::FromMinExtent(IntImm(vars[i].ty(), 0), extent), true);
  }
  auto simplify = make_object<BufferIndiceSimplify>(analyzer);
  body = simplify->Mutate(body).ValueOrUnchanged(body);
  return body.as_or_throw<For>();
}

class LoopPragmaUnroller : public LayoutMapMutator {
public:
  using LayoutMapMutator::Mutate_;
  UnchangedOr<Stmt> Mutate_(const ForObj *node, InplaceMode mode) final {
    if (node->kind == ForKind::kDefault && IsOne(node->step)) {
      sym::Analyzer analyzer;
      if (TryConstInt(analyzer->Simplify(LoopExtent(node)))) {
        auto copy = make_object<ForObj>(*node);
        copy->kind = ForKind::kUnrolled;
        // Preserve the original convention: mark the first eligible loop,
        // without recursively unrolling a serial reduction inside its body.
        return For(copy);
      }
    }
    return LayoutMapMutator::Mutate_(node, mode);
  }
};

class LoopPartitioner : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;
  Optional<VisitInterrupt> Visit_(const ForObj *node) final {
    if (node->kind == ForKind::kParallel) {
      (void)LoopExtent(node);
      loop_vars_.push_back({Range(node->begin, node->end), node->var});
    }
    return StmtExprVisitor::Visit_(node);
  }
  Optional<VisitInterrupt> Visit_(const TensorLoadObj *node) final {
    if (auto tensor = node->source.as<TensorVar>())
      has_fragment_ |= IsFragmentBuffer(*tensor);
    return StmtExprVisitor::Visit_(node);
  }
  Optional<VisitInterrupt> Visit_(const TensorStoreObj *node) final {
    if (auto tensor = node->dest.as<TensorVar>())
      has_fragment_ |= IsFragmentBuffer(*tensor);
    return StmtExprVisitor::Visit_(node);
  }
  Fragment Partition(const For &loop, size_t num_thread, int vector_size) {
    ICHECK_GT(num_thread, 0U);
    ICHECK_GT(vector_size, 0);
    Visit(loop);
    bool has_dynamic_wide_extent = false;
    for (const auto &iv : loop_vars_) {
      has_dynamic_wide_extent |=
          iv->var.ty().bits() >= 64 && !TryConstInt(iv->dom->extent);
    }
    if (num_thread == 1 && !has_fragment_ && has_dynamic_wide_extent) {
      // Without distributed registers, one thread can traverse these axes
      // directly. Flattening a symbolic int64 nest would introduce an
      // extra extent product that need not fit int64 even when every source
      // axis does. Keep the identity map and all PartitionLoop guards.
      Array<PrimExpr> indices;
      for (const auto &iv : loop_vars_)
        indices.push_back(iv->var);
      return Fragment(loop_vars_, indices, {IntImm::Int32(0)}, {});
    }
    PrimType dtype =
        loop_vars_.empty() ? PrimType::Int(32) : loop_vars_.back()->var.ty();
    // This flat coordinate is generated by the partitioner, not supplied by
    // the user. A dynamic int32 dimension can produce more than INT32_MAX
    // points when multiplied by another axis, even though each axis fits.
    // Widen before that multiplication; user-provided layouts keep their
    // original arithmetic and narrowing semantics.
    bool widen_dynamic_index = false;
    if (dtype.bits() < 64) {
      for (const auto &iv : loop_vars_) {
        if (!TryConstInt(iv->dom->extent)) {
          dtype = PrimType::Int(64);
          widen_dynamic_index = true;
          break;
        }
      }
    }
    PrimExpr flat = IntImm(dtype, 0), vec = IntImm(dtype, vector_size),
             threads = IntImm(dtype, num_thread);
    for (const auto &iv : loop_vars_) {
      flat = widen_dynamic_index
                 ? flat * cast(iv->dom->extent, dtype) + cast(iv->var, dtype)
                 : flat * iv->dom->extent + iv->var;
    }
    PrimExpr access = floordiv(flat, vec);
    PrimExpr thread = floormod(access, threads);
    PrimExpr index = floordiv(access, threads) * vec + floormod(flat, vec);
    Fragment fragment(loop_vars_, {index}, {thread}, {});
    if (has_fragment_) {
      auto extent = TryConstInt(fragment->ThreadExtent());
      ICHECK(extent && *extent > 0);
      // Replication can occupy all threads only when there is an integral
      // number of complete copies. Otherwise keep the existing owners;
      // ParallelOp's thread-extent predicate excludes the inactive tail.
      if (num_thread % *extent == 0)
        fragment = fragment->Replicate(num_thread / *extent);
    }
    return fragment;
  }

private:
  bool has_fragment_ = false;
  Array<LayoutIterVar> loop_vars_;
};

Fragment PlanLoopPartition(const For &op, size_t threads, int vector_size) {
  return make_object<LoopPartitioner>()->Partition(op, threads, vector_size);
}

Fragment PlanLoopPartition(const For &op, int vector_size, const Range &range) {
  auto threads = TryConstInt(range->extent);
  ICHECK(threads && *threads > 0);
  return PlanLoopPartition(op, static_cast<size_t>(*threads), vector_size)
      ->BindThreadRange(range);
}

For PragmaUnrollLoop(For stmt) {
  return make_object<LoopPragmaUnroller>()
      ->Mutate(stmt)
      .ValueOrUnchanged(stmt)
      .as_or_throw<For>();
}

Stmt LowerParallelLoop(For loop, const Fragment &layout, PrimExpr thread_index,
                       const sym::AnalyzerObj *analyzer,
                       const LayoutMap &layout_map,
                       Optional<PrimExpr> predicate, bool parallel_loop,
                       bool require_padding_guard, Optional<Target> target) {
  For result = loop;
  auto writer = result.CopyOnWrite();
  writer->annotations.erase(attr::kParallelLoopLayout);
  writer->annotations.erase(attr::kParallelLoopPredicate);
  writer->annotations.erase(attr::kParallelLoopRequiresPaddingGuard);
  if (parallel_loop) {
    auto fork = analyzer ? analyzer->Clone() : sym::Analyzer();
    result = PartitionLoop(result, thread_index, fork.get(), layout,
                           require_padding_guard);
  }
  result = VectorizeLoop(result, analyzer, layout_map, -1, std::move(target));
  result = PragmaUnrollLoop(result);
  if (predicate && parallel_loop)
    return If(*predicate, SeqStmt(result), std::nullopt, loop->loc);
  return result;
}
} // namespace tvm::tl

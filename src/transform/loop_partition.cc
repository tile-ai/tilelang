#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "op/prim.h"
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

#include "compat/region.h"
#include "loop_partition.h"

#include "op/utils.h"
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>

#include <limits>
#include <utility>

namespace tvm::tl {
namespace {

class BufferIndexSimplifier : public tvm::tl::StmtExprMutator {
public:
  explicit BufferIndexSimplifier(sym::Analyzer analyzer)
      : analyzer_(std::move(analyzer)) {}

private:
  ffi::UnchangedOr<Stmt> Mutate_(const BindNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt original = ffi::GetRef<Stmt>(node);
    Bind binding = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                       .ValueOrUnchanged(original)
                       .as_or_throw<Bind>();
    if (auto value = binding->value.as<PrimExpr>();
        value && binding->var.as<PrimVar>() &&
        SideEffect(value.value()) <= CallEffectKind::kPure) {
      tl::arith::Bind(analyzer_, binding->var, value.value(), true);
    }
    return binding;
  }

  ffi::UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *node,
                                     ffi::InplaceMode mode) final {
    PrimExpr original = ffi::GetRef<PrimExpr>(node);
    TensorLoad load = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                          .ValueOrUnchanged(original)
                          .as_or_throw<TensorLoad>();
    load.CopyOnWrite()->indices = load->indices.Map([&](const PrimExpr &index) {
      return tl::arith::Simplify(analyzer_, index);
    });
    return load;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt original = ffi::GetRef<Stmt>(node);
    TensorStore store = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                            .ValueOrUnchanged(original)
                            .as_or_throw<TensorStore>();
    store.CopyOnWrite()->indices =
        store->indices.Map([&](const PrimExpr &index) {
          return tl::arith::Simplify(analyzer_, index);
        });
    return store;
  }

  sym::Analyzer analyzer_;
};

class MultiplicityMarkerLowerer : public tvm::tl::StmtExprMutator {
public:
  explicit MultiplicityMarkerLowerer(ffi::Optional<PrimExpr> guard)
      : guard_(std::move(guard)) {}

private:
  ffi::UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt statement = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                         .ValueOrUnchanged(ffi::GetRef<Stmt>(node));
    auto tensor = node->dest.as<TensorVar>();
    if (guard_ && tensor && tensor.value().scope().name() == "global")
      return If(*guard_, SeqStmt({statement}), std::nullopt, node->loc);
    return statement;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    Stmt original = ffi::GetRef<Stmt>(node);
    RegionStmt region = tvm::tl::StmtExprMutator::Mutate_(node, mode)
                            .ValueOrUnchanged(original)
                            .as_or_throw<RegionStmt>();
    if (region->op->name != "tl.parallel_multiplicity") {
      return region;
    }
    if (guard_.has_value()) {
      return If(guard_.value(), region->body, std::nullopt, region->loc);
    }
    return RegionStmt(::tvm::tl::scope_op(), {}, {}, DictAttrs(), region->body,
                      {}, region->loc);
  }

  ffi::Optional<PrimExpr> guard_;
};

const ForNode *UnwrapLoop(const Stmt &statement) {
  Stmt current = statement;
  while (const auto *sequence = current.as<SeqStmtNode>()) {
    if (sequence->seq.size() != 1)
      return nullptr;
    current = sequence->seq[0];
  }
  return current.as<ForNode>();
}

class FragmentAccessFinder : public tvm::tl::StmtExprVisitor {
public:
  bool has_fragment{false};

private:
  ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorLoadNode *load) final {
    if (auto tensor = load->source.as<tirx::TensorVar>()) {
      has_fragment |= IsFragmentBuffer(tensor.value());
    }
    return tvm::tl::StmtExprVisitor::Visit_(load);
  }
  ffi::Optional<ffi::VisitInterrupt>
  Visit_(const TensorStoreNode *store) final {
    if (auto tensor = store->dest.as<tirx::TensorVar>()) {
      has_fragment |= IsFragmentBuffer(tensor.value());
    }
    return tvm::tl::StmtExprVisitor::Visit_(store);
  }
};

class LoopPragmaUnroller : public tvm::tl::StmtExprMutator {
private:
  ffi::UnchangedOr<Stmt> Mutate_(const ForNode *node,
                                 ffi::InplaceMode mode) final {
    if (node->kind == ForKind::kDefault) {
      sym::Analyzer analyzer;
      if (GetConstInt(tl::arith::Simplify(analyzer, node->end - node->begin))
              .has_value()) {
        For loop = ffi::GetRef<For>(node);
        loop.CopyOnWrite()->kind = ForKind::kUnrolled;
        return loop;
      }
    }
    return tvm::tl::StmtExprMutator::Mutate_(node, mode);
  }
};

} // namespace

For PartitionLoop(For loop, PrimExpr thread_index, sym::Analyzer analyzer,
                  const Fragment &loop_layout, bool require_padding_guard) {
  TVM_FFI_CHECK(loop_layout.defined(), ValueError)
      << "A loop fragment layout is required";
  Location location = loop->loc;
  size_t input_rank = loop_layout->InputDim();
  size_t output_rank = loop_layout->OutputDim();
  ffi::Array<PrimVar> variables;
  ffi::Array<PrimExpr> output_shape = loop_layout->OutputShape();
  for (size_t axis = 0; axis < output_rank; ++axis) {
    PrimVar variable("partition_" + std::to_string(axis),
                     output_shape[axis].ty());
    tl::arith::Bind(analyzer, variable,
                    Range::FromMinExtent(tl::MakeConst(variable.ty(), 0),
                                         output_shape[axis]));
    variables.push_back(variable);
  }
  PrimExpr normalized_thread = thread_index;
  if (auto range = loop_layout->ThreadRange()) {
    normalized_thread = thread_index - range.value()->begin;
  }
  ffi::Array<PrimExpr> coordinates(variables.begin(), variables.end());
  coordinates.push_back(normalized_thread);
  Layout inverse = loop_layout->InverseWithLevel(require_padding_guard).first;
  ffi::Array<PrimExpr> indices = inverse->Forward(coordinates);
  ffi::Map<PrimVar, PrimExpr> replacements;
  ffi::Array<PrimExpr> begins;
  ffi::Array<PrimExpr> ends;
  Stmt body = loop;
  for (size_t axis = 0; axis < input_rank; ++axis) {
    const ForNode *nested = UnwrapLoop(body);
    TVM_FFI_CHECK(nested && nested->kind == ForKind::kParallel, ValueError)
        << "A layout requires a perfectly nested parallel loop";
    TVM_FFI_CHECK(nested->HasTrivialStep(), ValueError)
        << "PartitionLoop requires unit-step parallel loops";
    replacements.Set(nested->var, indices[axis]);
    begins.push_back(nested->begin);
    ends.push_back(nested->end);
    body = nested->body;
  }
  body = Substitute(body, replacements);
  PrimExpr guard = (normalized_thread >= 0) &&
                   (normalized_thread < loop_layout->ThreadExtent());
  for (size_t axis = 0; axis < input_rank; ++axis) {
    guard = guard && (indices[axis] >= begins[axis]) &&
            (indices[axis] < ends[axis]);
  }
  ffi::Array<PrimExpr> inverse_shape = inverse->OutputShape();
  if (inverse_shape.size() > input_rank) {
    guard = guard && (indices[input_rank] >= 0) &&
            (indices[input_rank] < inverse_shape[input_rank]);
  }
  ffi::Optional<PrimExpr> replica_guard;
  if (indices.size() > input_rank) {
    PrimExpr primary_replica =
        tl::arith::Simplify(analyzer, indices[input_rank] == 0);
    if (!tl::arith::CanProve(analyzer, primary_replica))
      replica_guard = primary_replica;
  }
  auto multiplicity =
      ffi::make_object<MultiplicityMarkerLowerer>(replica_guard);
  body = multiplicity->Mutate(body).ValueOrUnchanged(body);
  guard = tl::arith::Simplify(analyzer, guard);
  if (!tl::arith::CanProve(analyzer, guard)) {
    body = If(guard, SeqStmt(body), std::nullopt, location);
  }
  if (output_rank == 0) {
    PrimVar single_iteration("partition_once", PrimType::Int(32));
    body = For(single_iteration, 0, 1, 1, ForKind::kDefault, SeqStmt(body), {},
               location);
  } else {
    for (size_t remaining = output_rank; remaining > 0; --remaining) {
      size_t axis = remaining - 1;
      PrimExpr extent = inverse->InputShape()[axis];
      PrimType type = variables[axis].ty();
      body = For(variables[axis], tl::MakeConst(type, 0), extent,
                 tl::MakeConst(type, 1), ForKind::kDefault, SeqStmt(body), {},
                 location);
      tl::arith::Bind(analyzer, variables[axis],
                      Range::FromMinExtent(tl::MakeConst(type, 0), extent),
                      true);
    }
  }
  auto simplifier = ffi::make_object<BufferIndexSimplifier>(analyzer);
  return simplifier->Mutate(body).ValueOrUnchanged(body).as_or_throw<For>();
}

Fragment PlanLoopPartition(const For &loop, size_t num_thread,
                           int vectorize_size) {
  TVM_FFI_CHECK(num_thread > 0 &&
                    num_thread <= std::numeric_limits<int32_t>::max(),
                ValueError)
      << "Thread count must be a positive int32";
  TVM_FFI_CHECK_GT(vectorize_size, 0, ValueError);
  ffi::Array<LayoutAxis> axes;
  Stmt body = loop;
  const ForNode *current = UnwrapLoop(body);
  while (current && current->kind == ForKind::kParallel) {
    TVM_FFI_CHECK(current->HasTrivialStep() && IsZero(current->begin),
                  ValueError)
        << "Loop partition planning requires zero-based, unit-step parallel "
           "loops";
    Range domain(current->begin, current->end);
    axes.push_back(LayoutAxis(domain, current->var));
    body = current->body;
    current = UnwrapLoop(body);
  }
  TVM_FFI_CHECK(!axes.empty(), ValueError)
      << "Loop partition planning requires a parallel loop";
  PrimType type = axes.back()->var.ty();
  PrimExpr flattened = tl::MakeConst(type, 0);
  for (const LayoutAxis &axis : axes) {
    flattened = flattened * axis->dom->extent + axis->var;
  }
  PrimExpr width = tl::MakeConst(type, vectorize_size);
  PrimExpr threads = tl::MakeConst(type, static_cast<int32_t>(num_thread));
  PrimExpr access = floordiv(flattened, width);
  PrimExpr thread = floormod(access, threads);
  PrimExpr offset =
      floordiv(access, threads) * width + floormod(flattened, width);
  Fragment fragment(axes, {offset}, thread, std::nullopt);
  auto finder = ffi::make_object<FragmentAccessFinder>();
  finder->Visit(body);
  if (finder->has_fragment) {
    auto used_threads = GetConstInt(fragment->ThreadExtent());
    TVM_FFI_CHECK(used_threads.has_value() && *used_threads > 0 &&
                      static_cast<size_t>(*used_threads) <= num_thread,
                  ValueError)
        << "Fragment thread count must fit within the participating threads";
    fragment = fragment->Replicate(num_thread / *used_threads);
  }
  return fragment;
}

Fragment PlanLoopPartition(const For &loop, int vectorize_size,
                           const Range &thread_range) {
  auto threads = GetConstInt(thread_range->extent);
  TVM_FFI_CHECK(threads.has_value() && *threads > 0, ValueError)
      << "Thread range extent must be a positive constant";
  return PlanLoopPartition(loop, static_cast<size_t>(*threads), vectorize_size)
      ->BindThreadRange(thread_range);
}

For PragmaUnrollLoop(For loop) {
  auto unroller = ffi::make_object<LoopPragmaUnroller>();
  return unroller->Mutate(loop).ValueOrUnchanged(loop).as_or_throw<For>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.parallel_multiplicity")
      .signature()
      .set_attr(tvm::op_attr::kRegionGetBodyParams,
                FRegionGetBodyParams::FromNative<
                    +[](const CallNode *) -> ffi::Array<Var> { return {}; }>())
      .set_attr(tvm::op_attr::kRegionValidate,
                FRegionValidate::FromNative<+[](const RegionStmtNode *region) {
                  TVM_FFI_CHECK(region->attrs->dict.empty() &&
                                    region->result_vars.empty(),
                                ValueError)
                      << "Parallel multiplicity regions have no attributes or "
                         "results";
                }>());
  ffi::reflection::GlobalDef()
      .def("tl.PartitionLoop", PartitionLoop)
      .def("tl.PlanLoopPartition",
           [](For loop, Range thread_range, int width) {
             return PlanLoopPartition(loop, width, thread_range);
           })
      .def("tl.PragmaUnrollLoop", PragmaUnrollLoop);
}

} // namespace tvm::tl

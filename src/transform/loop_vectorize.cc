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
 * \file loop_vectorize.cc
 * \brief A tool to automatically vectorize a for loop
 */

#include "loop_vectorize.h"

#include "../config.h"
#include "../op/builtin.h"
#include "../op/utils.h"
#include "backend/common/target_utils.h"
#include "common/int64_promoter.h"
#include "compat/visit.h"
#include "support/check.h"

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <optional>
#include <vector>

namespace tvm::tl {

using namespace ffi;
using namespace prim;

namespace {

class AnalysisLetInliner : public tirx::StmtExprMutator {
private:
  UnchangedOr<PrimExpr> Mutate_(const LetNode *node, InplaceMode mode) final {
    PrimExpr result = StmtExprMutator::Mutate_(node, mode)
                          .ValueOrUnchanged(GetRef<PrimExpr>(node));
    const auto *binding = result.as<LetNode>();
    if (SideEffect(binding->value) != CallEffectKind::kPure)
      return result;
    return Substitute(binding->body,
                      {{binding->var.as_or_throw<PrimVar>(), binding->value}});
  }
};

PrimExpr PrepareExprForAnalyzer(const PrimExpr &expression, int scale,
                                sym::Analyzer analyzer) {
  ICHECK_GE(scale, 1);
  bool promote = false;
  PostOrderVisit(expression, [&](const ObjectRef &object) {
    auto scalar = object.as<PrimExpr>();
    if (!scalar || promote)
      return;
    PrimType type = scalar.value().ty();
    if (!type.MatchesCode(kDLInt) || type.bits() >= 64 || !type.IsScalar())
      return;
    auto bounds = analyzer->const_int_bound(scalar.value());
    int64_t maximum = (int64_t{1} << (type.bits() - 1)) - 1;
    int64_t minimum = -(int64_t{1} << (type.bits() - 1));
    promote = bounds->max_value > maximum / scale ||
              bounds->min_value < minimum / scale;
  });
  PrimExpr prepared = promote ? Int64Promoter()(expression) : expression;
  return ffi::make_object<AnalysisLetInliner>()
      ->Mutate(prepared)
      .ValueOrUnchanged(prepared);
}

PrimExpr FlatOffset(const TensorVar &tensor, const Array<PrimExpr> &indices) {
  auto offsets = tensor.ty()->ElemOffset(indices);
  ICHECK_EQ(offsets.size(), 1U)
      << "Vectorization requires a memory-only tensor layout";
  return offsets[0];
}

class MemoryAccessFinder : public tirx::StmtExprVisitor {
public:
  bool global{false};
  bool shared{false};

private:
  void Record(const Expr &expression) {
    if (auto tensor = expression.as<TensorVar>()) {
      global |= IsGlobalBuffer(tensor.value());
      shared |= IsSharedBuffer(tensor.value());
    }
  }

  Optional<VisitInterrupt> Visit_(const TensorLoadNode *node) final {
    Record(node->source);
    return StmtExprVisitor::Visit_(node);
  }

  Optional<VisitInterrupt> Visit_(const TensorStoreNode *node) final {
    Record(node->dest);
    return StmtExprVisitor::Visit_(node);
  }
};

bool ForBodyContainsSeqStmt(const For &loop) {
  bool multiple = false;
  PostOrderVisit(loop->body, [&](const ObjectRef &object) {
    if (auto sequence = object.as<SeqStmtNode>()) {
      int statements = 0;
      for (const Stmt &statement : sequence->seq) {
        if (!statement.as<BindNode>())
          ++statements;
      }
      multiple |= statements > 1;
    }
  });
  return multiple;
}

} // namespace

Optional<TensorLoad> ExtractBufferLoadForAtomic(const Expr &expression) {
  if (auto load = expression.as<TensorLoad>())
    return load;
  if (auto call = expression.as<CallNode>();
      call && (call->op.same_as(tirx::address_of_op()) ||
               call->op.same_as(tl::access_ptr()))) {
    ICHECK_EQ(call->args.size(),
              call->op.same_as(tirx::address_of_op()) ? 1U : 3U);
    if (auto region = call->args[0].as<TensorRegion>()) {
      Array<PrimExpr> indices;
      for (const Range &range : region.value()->region)
        indices.push_back(range->begin);
      return TensorLoad(region.value()->source, indices, region.value()->loc);
    }
    return call->args[0].as<TensorLoad>();
  }
  return std::nullopt;
}

int GetMaxAtomicVectorSize(PrimType type, const String &scope,
                           const Target &target) {
  if (!type.IsScalar())
    return 1;
  if (type.MatchesElementType(kDLFloat, 16) ||
      type.MatchesElementType(kDLBfloat, 16))
    return 2;
  if (type.MatchesElementType(kDLFloat, 32) &&
      TargetHasSMVersionGE(target, 90) && (scope.empty() || scope == "global"))
    return 4;
  return 1;
}

enum class VectorConstraintKind {
  kMustScalarize,
  kCall,
  kCast,
  kLocal,
  kMemory,
  kBroadcastLoad,
};

struct BufferVectorInfo {
  Optional<TensorVar> buffer;
  int vector_size;
  bool is_store;
  Array<PrimExpr> indices;
  bool is_cast = false;
  bool requires_scalarization = false;
};

struct VectorConstraintSummary {
  explicit VectorConstraintSummary(int width)
      : local_min(width), memory_min(width), broadcast_load_min(width),
        call_min(width), non_cast_call_min(width) {}
  int local_min;
  int memory_min;
  int broadcast_load_min;
  int call_min;
  int non_cast_call_min;
  bool requires_scalarization = false;
  bool has_memory_access = false;
  std::vector<BufferVectorInfo> deferred_accesses;
};

class VectorizePlanner : public tirx::StmtExprMutatorWithAnalyzer {
public:
  VectorizePlanner(sym::Analyzer analyzer, LayoutMap layouts)
      : StmtExprMutatorWithAnalyzer(analyzer),
        analyzer_context_(std::move(analyzer)),
        layout_map_(std::move(layouts)) {}

  int Plan(const For &loop) {
    Target target = Target::Current(false).value();
    auto memory = ffi::make_object<MemoryAccessFinder>();
    memory->Visit(loop);
    initial_vector_size_ = vector_load_bits_max_ = loop_extent_vector_size_ =
        MaxVectorLoadBits(target, memory->global && !memory->shared);
    ICHECK_GT(initial_vector_size_, 0);
    this->Mutate(loop, InplaceMode::kDisallow);
    if (innermost_loop_count_ != 1)
      return 1;
    bool verbose = tl_config::VectorizePlannerVerboseEnabled();
    auto summary = SummarizeConstraints(verbose);
    int selected =
        SelectVectorSize(summary, ForBodyContainsSeqStmt(loop), verbose);
    selected = std::gcd(loop_extent_vector_size_, selected);
    if (target->kind->name == "metal") {
      int element_lanes = 1;
      PostOrderVisit(loop->body, [&](const ObjectRef &object) {
        if (auto expression = object.as<Expr>())
          if (auto type = expression.value()->ty.as<PrimType>())
            element_lanes = std::max(element_lanes, type.value().lanes());
      });
      selected = std::gcd(selected, std::max(1, 4 / element_lanes));
    }
    if (verbose)
      std::cerr << "=== Final vector_size: " << selected << " ===\n";
    return selected;
  }

private:
  PrimExpr MemoryOffset(const TensorVar &tensor,
                        const Array<PrimExpr> &indices) const {
    auto wide_indices = indices.Map(
        [](const PrimExpr &index) { return Int64Promoter()(index); });
    if (auto layout = layout_map_.Get(tensor)) {
      ICHECK(tensor.ty()->HasDefaultLayout(analyzer_context_))
          << tensor << " already owns a non-default native layout";
      auto coordinates = layout.value()->Forward(wide_indices);
      auto shape = layout.value()->OutputShape();
      ICHECK_EQ(coordinates.size(), shape.size());
      PrimExpr offset = 0;
      for (size_t axis = 0; axis < coordinates.size(); ++axis) {
        offset = offset * shape[axis] + coordinates[axis];
      }
      return offset;
    }
    return FlatOffset(tensor, wide_indices);
  }

  static const char *ConstraintKindName(VectorConstraintKind kind) {
    switch (kind) {
    case VectorConstraintKind::kMustScalarize:
      return "must-scalarize";
    case VectorConstraintKind::kCall:
      return "call";
    case VectorConstraintKind::kCast:
      return "cast";
    case VectorConstraintKind::kLocal:
      return "local";
    case VectorConstraintKind::kMemory:
      return "memory";
    case VectorConstraintKind::kBroadcastLoad:
      return "broadcast-load";
    }
    LOG(FATAL) << "Unknown vector constraint kind";
    return "unknown";
  }

  bool IsBroadcastLoad(const BufferVectorInfo &info) const {
    // A loop-invariant non-local load becomes one scalar load broadcast to all
    // lanes, not a vector memory access. DecoupleTypeCast also does not create
    // a cast buffer for it, so classify it separately from memory accesses.
    // Stores are never broadcasts: same-address lane stores carry ordering
    // semantics and are either marked must-scalarize or kept as memory access.
    if (info.is_store || info.indices.empty()) {
      return false;
    }
    PrimExpr elem_offset = MemoryOffset(info.buffer.value(), info.indices);
    return IsExprInvariantInVectorBoundary(
        elem_offset, inner_for_.value()->var,
        (inner_for_.value()->end - inner_for_.value()->begin),
        initial_vector_size_, analyzer_context_, inner_for_.value()->begin);
  }

  VectorConstraintKind ClassifyConstraint(const BufferVectorInfo &info) const {
    if (info.requires_scalarization) {
      return VectorConstraintKind::kMustScalarize;
    }
    if (!info.buffer.has_value()) {
      return info.is_cast ? VectorConstraintKind::kCast
                          : VectorConstraintKind::kCall;
    }
    if (IsLocalBuffer(info.buffer.value(), /*allow_var=*/true) ||
        IsFragmentBuffer(info.buffer.value())) {
      return VectorConstraintKind::kLocal;
    }
    if (IsBroadcastLoad(info)) {
      return VectorConstraintKind::kBroadcastLoad;
    }
    return VectorConstraintKind::kMemory;
  }

  void LogConstraint(const BufferVectorInfo &info,
                     VectorConstraintKind kind) const {
    if (info.buffer.has_value()) {
      std::cerr << "  Buffer: " << info.buffer.value()->name
                << " (scope=" << info.buffer.value().scope().name() << ")"
                << " -> vector_size=" << info.vector_size
                << (info.is_store ? " [store]" : " [load]");
    } else {
      std::cerr << "  [" << (info.is_cast ? "cast" : "call")
                << "] -> vector_size=" << info.vector_size;
    }
    if (kind != VectorConstraintKind::kCall &&
        kind != VectorConstraintKind::kCast) {
      std::cerr << " [constraint=" << ConstraintKindName(kind) << "]";
    }
    std::cerr << "\n";
  }

  VectorConstraintSummary SummarizeConstraints(bool verbose) const {
    VectorConstraintSummary summary(initial_vector_size_);
    for (const BufferVectorInfo &info : buffer_vector_infos_) {
      VectorConstraintKind kind = ClassifyConstraint(info);
      if (verbose) {
        LogConstraint(info, kind);
      }
      switch (kind) {
      case VectorConstraintKind::kMustScalarize:
        summary.requires_scalarization = true;
        break;
      case VectorConstraintKind::kCall:
        summary.call_min = std::gcd(summary.call_min, info.vector_size);
        summary.non_cast_call_min =
            std::gcd(summary.non_cast_call_min, info.vector_size);
        break;
      case VectorConstraintKind::kCast:
        summary.call_min = std::gcd(summary.call_min, info.vector_size);
        break;
      case VectorConstraintKind::kLocal:
        summary.local_min = std::gcd(summary.local_min, info.vector_size);
        summary.deferred_accesses.push_back(info);
        break;
      case VectorConstraintKind::kMemory:
        summary.memory_min = std::gcd(summary.memory_min, info.vector_size);
        summary.has_memory_access = true;
        break;
      case VectorConstraintKind::kBroadcastLoad:
        summary.broadcast_load_min =
            std::gcd(summary.broadcast_load_min, info.vector_size);
        summary.deferred_accesses.push_back(info);
        break;
      }
    }
    if (verbose) {
      int legacy_local_fragment_min =
          std::gcd(summary.local_min, summary.broadcast_load_min);
      // Keep the established summary for log consumers. Historically,
      // local_fragment_min also contained non-local broadcast loads.
      std::cerr << "  Computed mins: local_fragment_min="
                << legacy_local_fragment_min
                << ", memory_min=" << summary.memory_min
                << ", call_node_min=" << summary.call_min << "\n";
      std::cerr << "  Classified constraints: must_scalarize="
                << (summary.requires_scalarization ? "true" : "false")
                << ", local=" << summary.local_min
                << ", broadcast_load=" << summary.broadcast_load_min << "\n";
    }
    return summary;
  }

  int RevalidateDeferredAccesses(int vector_size,
                                 const std::vector<BufferVectorInfo> &accesses,
                                 bool verbose) const {
    // The simple memory strategy may select a width larger than a deferred
    // local/fragment or broadcast-load constraint. Re-check its indices at the
    // selected boundary and fold the original width back in when needed.
    for (const BufferVectorInfo &info : accesses) {
      if (vector_size <= info.vector_size || info.indices.empty()) {
        continue;
      }
      PrimExpr elem_offset = MemoryOffset(info.buffer.value(), info.indices);
      if (!IndicesCanVectorize(
              elem_offset, inner_for_.value()->var,
              (inner_for_.value()->end - inner_for_.value()->begin),
              vector_size, analyzer_context_,
              /*allow_broadcast=*/true, inner_for_.value()->begin)) {
        int old_vector_size = vector_size;
        vector_size = std::gcd(vector_size, info.vector_size);
        if (verbose) {
          std::cerr << "  [Re-validate] Local buffer '"
                    << info.buffer.value()->name
                    << "' not invariant at vector_size=" << old_vector_size
                    << ", GCD with " << info.vector_size
                    << " -> vector_size=" << vector_size << "\n";
        }
      }
    }
    return vector_size;
  }

  int SelectVectorSize(const VectorConstraintSummary &summary,
                       bool has_seq_stmt, bool verbose) const {
    if (summary.requires_scalarization) {
      if (verbose) {
        std::cerr << "  [Strategy] Semantic constraint requires scalarization"
                  << " -> vector_size=1\n";
      }
      return 1;
    }
    int deferred_min = std::gcd(summary.local_min, summary.broadcast_load_min);
    int selected;
    if (has_seq_stmt) {
      // Multiple statements may carry interactions not represented by the
      // simple single-store optimization, so include every constraint.
      selected = std::gcd(std::gcd(deferred_min, summary.memory_min),
                          summary.call_min);
      if (verbose) {
        std::cerr << "  [Strategy] Has SeqStmt, using conservative GCD of all"
                  << " -> vector_size=" << selected << "\n";
      }
    } else if (summary.has_memory_access) {
      // For a simple memory loop, local/fragment accesses do not impose memory
      // alignment constraints. Cast constraints are also deferred because
      // DecoupleTypeCast later splits mixed-type operations into separate
      // loops, allowing memory copies to retain their wider vector width.
      selected = std::gcd(summary.memory_min, summary.non_cast_call_min);
      if (verbose) {
        std::cerr << "  [Strategy] Has memory buffers (simple case), using "
                  << "memory_min=" << summary.memory_min
                  << ", non_cast_call_node_min=" << summary.non_cast_call_min
                  << " (ignoring local/fragment_min=" << deferred_min << ")"
                  << "\n";
      }
      selected = RevalidateDeferredAccesses(selected, summary.deferred_accesses,
                                            verbose);
    } else {
      // With no vector memory access, local/fragment, broadcast-load, and call
      // constraints jointly determine the available vector width.
      selected = std::gcd(deferred_min, summary.call_min);
      if (verbose) {
        std::cerr << "  [Strategy] Only local/fragment buffers, using "
                     "GCD(local_fragment_min, call_node_min)="
                  << selected << "\n";
      }
    }
    return selected;
  }

  UnchangedOr<Stmt> Mutate_(const ForNode *node, InplaceMode mode) final {
    inner_for_ = GetRef<For>(node);
    bool nested = false;
    PostOrderVisit(node->body, [&](const ObjectRef &object) {
      nested |= object.as<ForNode>() != nullptr;
    });
    if (!nested) {
      ++innermost_loop_count_;
      auto extent = GetConstInt(analyzer_->Simplify(node->end - node->begin));
      if (!IsOne(node->step) || !extent || extent.value() <= 0) {
        loop_extent_vector_size_ = 1;
        return Unchanged();
      }
      loop_extent_vector_size_ = static_cast<int>(
          std::gcd<int64_t>(initial_vector_size_, extent.value()));
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(node, mode);
  }

  void RecordAccess(const TensorVar &tensor, const Array<PrimExpr> &indices,
                    bool store) {
    if (!inner_for_)
      return;
    PrimExpr offset = MemoryOffset(tensor, indices);
    const For &loop = inner_for_.value();
    PrimExpr extent = loop->end - loop->begin;
    int width = loop_extent_vector_size_;
    int bits = tensor.dtype().bits() * tensor.dtype().lanes();
    int minimum = std::gcd(width, std::max(1, vector_load_bits_max_ / bits));
    bool invariant = false;
    for (int candidate = width; candidate >= minimum; candidate /= 2) {
      invariant = IsExprInvariantInVectorBoundary(
          offset, loop->var, extent, candidate, analyzer_context_, loop->begin);
      if (invariant) {
        width = candidate;
        break;
      }
      if (candidate == 1)
        break;
    }
    if (!invariant)
      width = minimum;
    bool independent =
        CanProveIndependent(offset, loop->var, analyzer_context_);
    bool scalarize = store && (invariant || independent);
    if (scalarize)
      width = 1;
    if (!independent && !scalarize) {
      while (width > 1 &&
             !IndicesCanVectorize(offset, loop->var, extent, width,
                                  analyzer_context_, true, loop->begin))
        width /= 2;
    }
    buffer_vector_infos_.push_back(
        {tensor, width, store, indices, false, scalarize});
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *node,
                                InplaceMode mode) final {
    RecordAccess(node->source.as_or_throw<TensorVar>(), node->indices, false);
    return StmtExprMutatorWithAnalyzer::Mutate_(node, mode);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *node,
                            InplaceMode mode) final {
    RecordAccess(node->dest.as_or_throw<TensorVar>(), node->indices, true);
    return StmtExprMutatorWithAnalyzer::Mutate_(node, mode);
  }

  int InvariantWidth(const PrimExpr &expression, int width) const {
    if (!inner_for_)
      return width;
    const For &loop = inner_for_.value();
    while (width > 1 && !IsExprInvariantInVectorBoundary(
                            expression, loop->var, loop->end - loop->begin,
                            width, analyzer_context_, loop->begin))
      width /= 2;
    return width;
  }

  void RecordCall(int width, bool cast = false) {
    buffer_vector_infos_.push_back({std::nullopt, width, false, {}, cast});
  }

  UnchangedOr<Stmt> Mutate_(const IfNode *node, InplaceMode mode) final {
    RecordCall(InvariantWidth(node->condition, loop_extent_vector_size_));
    If analysis = GetRef<If>(node);
    analysis.CopyOnWrite()->condition = PrepareExprForAnalyzer(
        node->condition, initial_vector_size_, analyzer_context_);
    return StmtExprMutatorWithAnalyzer::Mutate_(analysis.get(), mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const LetNode *node, InplaceMode mode) final {
    PrimExpr analysis = PrepareExprForAnalyzer(
        GetRef<PrimExpr>(node), initial_vector_size_, analyzer_context_);
    if (auto binding = analysis.as<LetNode>())
      return StmtExprMutatorWithAnalyzer::Mutate_(binding, mode);
    return Mutate(analysis, mode).ValueOrUnchanged(analysis);
  }

  UnchangedOr<PrimExpr> Mutate_(const SelectNode *node,
                                InplaceMode mode) final {
    auto target = Target::Current(false).value();
    if (!target->GetAttr<bool>("supports_vector_predicate").value_or(false)) {
      RecordCall(InvariantWidth(node->condition, loop_extent_vector_size_));
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(node, mode);
  }

  int AtomicWidth(const Expr &destination) const {
    auto load = ExtractBufferLoadForAtomic(destination);
    if (!load || !inner_for_)
      return 1;
    TensorVar tensor = load.value()->source.as_or_throw<TensorVar>();
    int width =
        std::gcd(loop_extent_vector_size_,
                 GetMaxAtomicVectorSize(tensor.dtype(), tensor.scope().name(),
                                        Target::Current(false).value()));
    PrimExpr offset = MemoryOffset(tensor, load.value()->indices);
    const For &loop = inner_for_.value();
    while (width > 1 &&
           !IndicesCanVectorize(offset, loop->var, loop->end - loop->begin,
                                width, analyzer_context_, false, loop->begin))
      width /= 2;
    return width;
  }

  int AsyncCopyWidth(const CallNode *node) const {
    ICHECK_GE(node->args.size(), 3U);
    auto count = GetConstInt(node->args[2].as_or_throw<PrimExpr>());
    if (!count || count.value() <= 0)
      return 1;
    auto destination = node->args[0]->ty.as<PtrType>();
    auto source = node->args[1]->ty.as<PtrType>();
    if (!destination || !source)
      return 1;
    auto destination_type = destination.value()->element_type.as<PrimType>();
    auto source_type = source.value()->element_type.as<PrimType>();
    if (!destination_type || !source_type)
      return 1;
    int destination_bits =
        destination_type.value().bits() * destination_type.value().lanes();
    int source_bits = source_type.value().bits() * source_type.value().lanes();
    ICHECK_EQ(destination_bits, source_bits);
    int64_t transfer_bits = count.value() * destination_bits;
    int width = 1;
    for (int bits : {32, 64, 128}) {
      if (bits % transfer_bits == 0)
        width = std::max(width, static_cast<int>(bits / transfer_bits));
    }
    return width;
  }

  UnchangedOr<Expr> Mutate_(const CallNode *node, InplaceMode mode) final {
    auto operation = node->op.as<Op>();
    if (node->op.same_as(prim::if_then_else_op())) {
      RecordCall(InvariantWidth(node->args[0].as_or_throw<PrimExpr>(),
                                loop_extent_vector_size_));
    } else if (node->op.same_as(tl::rng_rand_float()) ||
               node->op.same_as(tl::rng_rand())) {
      RecordCall(1);
    } else if (node->op.same_as(tl::atomic_add_elem_op())) {
      ICHECK_GE(node->args.size(), 2U);
      RecordCall(AtomicWidth(node->args[0]));
    } else if (node->op.same_as(tl::ptx_cp_async())) {
      RecordCall(AsyncCopyWidth(node));
    } else if (operation &&
               operation.value()->name == "tl.cast_with_rounding") {
      auto result_type = node->ty.as_or_throw<PrimType>();
      int bits = std::max(result_type.bits(),
                          node->args[0].as_or_throw<PrimExpr>().ty().bits());
      int width = std::gcd(initial_vector_size_,
                           std::max(1, vector_load_bits_max_ / bits));
      if (result_type.bits() == 4)
        width = std::gcd(width, 4);
      if (node->args.size() == 2) {
        width = InvariantWidth(node->args[1].as_or_throw<PrimExpr>(), width);
      }
      RecordCall(width, true);
    } else if (node->op.same_as(prim::assume_op()) &&
               SideEffect(node->args[0]) <= CallEffectKind::kPure) {
    } else if (node->op.same_as(tirx::address_of_op()) ||
               node->op.same_as(tl::access_ptr()) ||
               (operation && operation.value()->name == "tl.reducer_update")) {
    } else if (!operation || !Op::GetAttrMap(tirx::op_attr::kVectorizable)
                                  .get(operation.value(), false)) {
      int width = loop_extent_vector_size_;
      PostOrderVisit(GetRef<Expr>(node), [&](const ObjectRef &object) {
        if (auto load = object.as<TensorLoadNode>()) {
          width =
              InvariantWidth(MemoryOffset(load->source.as_or_throw<TensorVar>(),
                                          load->indices),
                             width);
        } else if (auto scalar = object.as<PrimExpr>();
                   scalar && !scalar.value().ty().IsVoid() &&
                   !object.as<TensorLoadNode>()) {
          width = InvariantWidth(scalar.value(), width);
        }
      });
      RecordCall(width);
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(node, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const CastNode *node, InplaceMode mode) final {
    int bits = std::max(node->ty.as_or_throw<PrimType>().bits(),
                        node->value.ty().bits());
    RecordCall(std::gcd(initial_vector_size_,
                        std::max(1, vector_load_bits_max_ / bits)),
               true);
    return StmtExprMutatorWithAnalyzer::Mutate_(node, mode);
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *node, InplaceMode mode) final {
    Stmt result = StmtExprMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(GetRef<Stmt>(node));
    Bind binding = result.as_or_throw<Bind>();
    if (auto value = binding->value.as<PrimExpr>();
        value && binding->var.as<PrimVar>() &&
        SideEffect(value.value()) <= CallEffectKind::kPure) {
      analyzer_->Bind(binding->var, value.value(), true);
    }
    return result;
  }

  sym::Analyzer analyzer_context_;
  LayoutMap layout_map_;
  Optional<For> inner_for_;
  int vector_load_bits_max_{128};
  int initial_vector_size_{128};
  int loop_extent_vector_size_{128};
  int innermost_loop_count_{0};
  std::vector<BufferVectorInfo> buffer_vector_infos_;
};

class VectorizeRewriter : public tirx::StmtExprMutator {
public:
  VectorizeRewriter(int width, sym::Analyzer analyzer)
      : width_(width), analyzer_(std::move(analyzer)) {}

private:
  UnchangedOr<Stmt> Mutate_(const ForNode *node, InplaceMode mode) final {
    bool nested = false;
    PostOrderVisit(node->body, [&](const ObjectRef &object) {
      nested |= object.as<ForNode>() != nullptr;
    });
    For loop = StmtExprMutator::Mutate_(node, mode)
                   .ValueOrUnchanged(GetRef<Stmt>(node))
                   .as_or_throw<For>();
    if (nested || width_ == 1) {
      if (loop->kind == ForKind::kParallel)
        loop.CopyOnWrite()->kind = ForKind::kDefault;
      return loop;
    }
    ICHECK(IsOne(loop->step)) << "Vectorization requires a unit loop step";
    auto extent = GetConstInt(analyzer_->Simplify(loop->end - loop->begin));
    ICHECK(extent && extent.value() > 0 && extent.value() % width_ == 0)
        << "The vector width must divide a positive static loop extent";
    PrimType type = loop->var.ty();
    PrimExpr zero = MakeConst(type, 0);
    PrimExpr width = MakeConst(type, width_);
    PrimVar lane("vec", type);
    if (extent.value() == width_) {
      Stmt body = Substitute(loop->body, {{loop->var, loop->begin + lane}});
      return For(lane, zero, width, MakeConst(type, 1), ForKind::kVectorized,
                 body, loop->annotations, loop->loc);
    }
    PrimVar group(loop->var->name, type);
    Stmt body = Substitute(loop->body,
                           {{loop->var, loop->begin + group * width + lane}});
    body = For(lane, zero, width, MakeConst(type, 1), ForKind::kVectorized,
               body, {}, loop->loc);
    ForKind kind =
        loop->kind == ForKind::kParallel ? ForKind::kDefault : loop->kind;
    return For(group, zero, MakeConst(type, extent.value() / width_),
               MakeConst(type, 1), kind, body, loop->annotations, loop->loc);
  }

  int width_;
  sym::Analyzer analyzer_;
};

bool CanProveIndependent(const PrimExpr &expression, PrimVar variable,
                         sym::Analyzer analyzer) {
  PrimVar other(variable->name + "_independent", variable.ty());
  PrimExpr prepared = PrepareExprForAnalyzer(expression, 1, analyzer);
  return analyzer->CanProveEqual(prepared,
                                 Substitute(prepared, {{variable, other}}));
}

bool IsExprInvariantInVectorBoundary(const PrimExpr &expression,
                                     const PrimVar &variable,
                                     const PrimExpr &extent, int width,
                                     sym::Analyzer analyzer,
                                     const PrimExpr &minimum) {
  ICHECK_GE(width, 1);
  PrimVar ordinal("ordinal", variable.ty());
  PrimExpr prepared = PrepareExprForAnalyzer(expression, width, analyzer);
  prepared = Substitute(
      prepared, {{variable, prim::cast(minimum, variable.ty()) + ordinal}});
  analyzer->Bind(ordinal, Range(MakeConst(ordinal.ty(), 0), extent));
  PrimExpr aligned = floordiv(ordinal, width) * width;
  if (analyzer->CanProveEqual(prepared,
                              Substitute(prepared, {{ordinal, aligned}})))
    return true;
  PrimVar lane("lane", variable.ty());
  PrimVar group("group", variable.ty());
  PrimExpr zero = MakeConst(variable.ty(), 0);
  analyzer->Bind(lane, Range(zero, MakeConst(variable.ty(), width)));
  analyzer->Bind(group,
                 Range(zero, analyzer->Simplify(ceildiv(extent, width))));
  PrimExpr grouped = analyzer->Simplify(
      Substitute(prepared, {{ordinal, group * width + lane}}));
  PrimExpr base = analyzer->Simplify(Substitute(grouped, {{lane, zero}}));
  return analyzer->CanProveEqual(grouped, base);
}

bool IndicesCanVectorize(const PrimExpr &expression, const PrimVar &variable,
                         const PrimExpr &extent, int width,
                         sym::Analyzer analyzer, bool allow_broadcast,
                         const PrimExpr &minimum) {
  ICHECK_GE(width, 1);
  if (width == 1)
    return true;
  if (!analyzer->CanProveEqual(floormod(extent, width), 0))
    return false;
  if (allow_broadcast &&
      IsExprInvariantInVectorBoundary(expression, variable, extent, width,
                                      analyzer, minimum))
    return true;
  PrimVar ordinal("ordinal", variable.ty());
  PrimExpr prepared = PrepareExprForAnalyzer(expression, width, analyzer);
  prepared = Substitute(
      prepared, {{variable, prim::cast(minimum, variable.ty()) + ordinal}});
  PrimExpr zero = MakeConst(ordinal.ty(), 0);
  PrimExpr first = analyzer->Simplify(Substitute(prepared, {{ordinal, zero}}));
  if (!analyzer->CanProveEqual(floormod(first, width), 0))
    return false;
  PrimVar lane("lane", ordinal.ty());
  PrimVar group("group", ordinal.ty());
  analyzer->Bind(lane, Range(zero, MakeConst(ordinal.ty(), width)));
  analyzer->Bind(group,
                 Range(zero, analyzer->Simplify(floordiv(extent, width))));
  PrimExpr grouped = analyzer->Simplify(
      Substitute(prepared, {{ordinal, group * width + lane}}));
  PrimExpr base = analyzer->Simplify(Substitute(grouped, {{lane, zero}}));
  if (!analyzer->CanProveEqual(grouped, base + prim::cast(lane, base.ty())))
    return false;
  return allow_broadcast || analyzer->CanProveEqual(floormod(base, width), 0);
}

int MaxVectorLoadBits(const Target &target, bool global_only_access) {
  if (auto bits = target->GetAttr<int64_t>("max_vector_bits")) {
    ICHECK_GT(bits.value(), 0);
    ICHECK_LE(bits.value(), 4096);
    return static_cast<int>(bits.value());
  }
  return TargetSupportVectorize256(target) &&
                 !tl_config::Vectorize256Disabled() && global_only_access
             ? 256
             : 128;
}

int GetVectorizeSize(const For &loop, const LayoutMap &layouts) {
  return GetVectorizeSize(loop, sym::Analyzer(), layouts);
}

int GetVectorizeSize(const For &loop, sym::Analyzer analyzer,
                     const LayoutMap &layouts) {
  return ffi::make_object<VectorizePlanner>(std::move(analyzer), layouts)
      ->Plan(loop);
}

For VectorizeLoop(const For &loop, const LayoutMap &layouts, int hint) {
  return VectorizeLoop(loop, sym::Analyzer(), layouts, hint);
}

For VectorizeLoop(const For &loop, sym::Analyzer analyzer,
                  const LayoutMap &layouts, int hint) {
  if (hint <= 0)
    hint = GetVectorizeSize(loop, analyzer, layouts);
  return ffi::make_object<VectorizeRewriter>(hint, std::move(analyzer))
      ->Mutate(loop, InplaceMode::kDisallow)
      .ValueOrUnchanged(loop)
      .as_or_throw<For>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.GetVectorizeSize",
           [](For loop, sym::Analyzer analyzer, LayoutMap layouts) {
             return GetVectorizeSize(loop, analyzer, layouts);
           })
      .def("tl.VectorizeLoop",
           [](For loop, sym::Analyzer analyzer, LayoutMap layouts, int hint) {
             return VectorizeLoop(loop, analyzer, layouts, hint);
           })
      .def("tl.IndicesCanVectorize", IndicesCanVectorize)
      .def("tl.IsExprInvariantInVectorBoundary",
           IsExprInvariantInVectorBoundary)
      .def("tl.GetMaxAtomicVectorSize", GetMaxAtomicVectorSize)
      .def("tl.MaxVectorLoadBits", MaxVectorLoadBits);
}

} // namespace tvm::tl

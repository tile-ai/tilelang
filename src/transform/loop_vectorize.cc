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
#include "../op/reducer.h"
#include "../op/utils.h"
#include "arith/int_operator.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "backend/common/target_utils.h"
#include "common/int64_promoter.h"
#include "common/loop_vectorization_utils.h"
#include "support/check.h"
#include "tir/transforms/ir_utils.h"
#include <iostream>
#include <optional>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/var.h>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

PrimExpr SimplifyExprForAnalyzer(const PrimExpr &expr, int scale,
                                 arith::Analyzer *analyzer);

} // namespace

/*!
 * \brief Check if buffer strides represent a contiguous (row-major) layout.
 * \param buffer The buffer to check.
 * \param analyzer The analyzer for symbolic comparison.
 * \return True if strides are empty (implicitly contiguous) or match row-major
 * layout.
 */
bool IsBufferContiguous(const Buffer &buffer, arith::Analyzer *analyzer) {
  if (buffer->strides.empty()) {
    return true;
  }
  if (buffer->strides.size() != buffer->shape.size()) {
    return false;
  }
  // For row-major layout:
  // strides[n-1] = 1
  // strides[i] = strides[i+1] * shape[i+1]
  int n = buffer->shape.size();
  PrimExpr expected_stride = make_const(buffer->shape[0].dtype(), 1);
  for (int i = n - 1; i >= 0; --i) {
    if (!analyzer->CanProveEqual(buffer->strides[i], expected_stride)) {
      return false;
    }
    if (i > 0) {
      expected_stride = expected_stride * buffer->shape[i];
    }
  }
  return true;
}

struct VectorizePlanResult {
  int vector_size;
  bool dynamic;
  PrimExpr condition;
};

enum class VectorConstraintKind {
  kMustScalarize,
  kCall,
  kCast,
  kLocal,
  kMemory,
  kBroadcastLoad,
};

struct VectorSizeConstraint {
  int vector_size;
  bool requires_scalarization;
};

struct BufferVectorInfo {
  Buffer buffer;
  int vector_size;
  bool is_store;
  Array<PrimExpr> indices;
  bool is_cast = false; // true for CastNode constraints (vs CallNode)
  bool requires_scalarization = false;
};

struct VectorConstraintSummary {
  explicit VectorConstraintSummary(int initial_vector_size)
      : local_min(initial_vector_size), memory_min(initial_vector_size),
        broadcast_load_min(initial_vector_size), call_min(initial_vector_size),
        non_cast_call_min(initial_vector_size) {}

  int local_min;
  int memory_min;
  int broadcast_load_min;
  int call_min;
  int non_cast_call_min;
  bool requires_scalarization = false;
  bool has_memory_access = false;
  std::vector<BufferVectorInfo> deferred_accesses;
};

Array<PrimExpr> GetBufferStrides(const Buffer &buffer) {
  if (!buffer->strides.empty()) {
    return buffer->strides;
  }
  Array<PrimExpr> strides;
  PrimExpr stride = 1;
  for (int i = buffer->shape.size() - 1; i >= 0; --i) {
    strides.push_back(stride);
    stride = stride * buffer->shape[i];
  }
  return Array<PrimExpr>{strides.rbegin(), strides.rend()};
}

Optional<BufferLoad> ExtractBufferLoadForAtomic(const PrimExpr &expr) {
  if (const auto *call = expr.as<CallNode>();
      call != nullptr && (call->op.same_as(builtin::address_of()) ||
                          call->op.same_as(tl::access_ptr()))) {
    ICHECK_EQ(call->args.size(),
              call->op.same_as(builtin::address_of()) ? 1U : 3U);
    return Downcast<BufferLoad>(call->args[0]);
  }
  return expr.as<BufferLoad>();
}

int GetMaxAtomicVectorSize(DataType dtype, const String &storage_scope,
                           const Target &target) {
  if (dtype.lanes() != 1) {
    return 1;
  }
  if (dtype.is_float16() || dtype.is_bfloat16()) {
    return 2;
  }
  if (dtype.is_float() && dtype.bits() == 32 &&
      TargetHasSMVersionGE(target, 90) &&
      (storage_scope.empty() || storage_scope == "global")) {
    // CUDA's float2/float4 atomicAdd overloads support global memory only.
    // An empty pointer storage scope denotes global memory, as in
    // Buffer::scope.
    return 4;
  }
  return 1;
}

class VectorizeFindMemoryAccess : public StmtExprVisitor {
public:
  VectorizeFindMemoryAccess() = default;

  bool HasGlobalAccess(const Stmt &stmt) {
    this->operator()(stmt);
    return has_global_access_;
  }

  bool HasSharedAccess(const Stmt &stmt) {
    this->operator()(stmt);
    return has_shared_access_;
  }

  static bool MaySupportVectorize256(const Stmt &stmt) {
    VectorizeFindMemoryAccess visitor;
    visitor(stmt);
    return visitor.has_global_access_ && !visitor.has_shared_access_;
  }

private:
  bool has_global_access_ = false;
  bool has_shared_access_ = false;

  void VisitStmt_(const BufferStoreNode *node) final {
    if (IsGlobalBuffer(node->buffer))
      has_global_access_ = true;
    if (IsSharedBuffer(node->buffer))
      has_shared_access_ = true;
    return StmtExprVisitor::VisitStmt_(node);
  }

  void VisitExpr_(const BufferLoadNode *node) final {
    if (IsGlobalBuffer(node->buffer))
      has_global_access_ = true;
    if (IsSharedBuffer(node->buffer))
      has_shared_access_ = true;
    return StmtExprVisitor::VisitExpr_(node);
  }
};

/*!
 * \brief Check if a For loop body contains SeqStmt (multiple statements).
 *
 * When the For body has SeqStmt, the vectorization analysis is more complex
 * and we should be conservative - treating local buffers the same as memory
 * buffers instead of ignoring their constraints.
 *
 * Currently we only handle simple single BufferStore cases specially for
 * local buffer optimization.
 */
bool ForBodyContainsSeqStmt(const For &loop) {
  // Ignore flat Bind nodes (SSA value defs): a single store wrapped in leading
  // Binds is not multi-stmt. Keeps this in sync with DecoupleTypeCast.
  bool has_seq_stmt = false;
  PostOrderVisit(loop->body, [&](const ObjectRef &obj) {
    if (auto seq = obj.as<SeqStmtNode>()) {
      int num_real_stmts = 0;
      for (const Stmt &s : seq->seq) {
        if (!s.as<BindNode>()) {
          ++num_real_stmts;
        }
      }
      if (num_real_stmts >= 2) {
        has_seq_stmt = true;
      }
    }
  });
  return has_seq_stmt;
}

class VectorizePlanner : public arith::IRMutatorWithAnalyzer {
public:
  explicit VectorizePlanner(arith::Analyzer *analyzer,
                            const LayoutMap &layout_map = {})
      : arith::IRMutatorWithAnalyzer(analyzer), layout_map_(layout_map) {}

  int Plan(const For &node) {
    Target target = Target::Current(false);
    bool verbose = tl_config::VectorizePlannerVerboseEnabled();

    vector_load_bits_max_ = initial_vector_size_ = loop_extent_vector_size_ =
        MaxVectorLoadBits(
            target, VectorizeFindMemoryAccess::MaySupportVectorize256(node));

    // Check if For body contains SeqStmt (multiple statements).
    // When there's SeqStmt, we use conservative strategy - treating local
    // buffers the same as memory buffers. The special local buffer optimization
    // (ignoring local buffer constraints) only applies to simple single
    // BufferStore cases.
    bool has_seq_stmt = ForBodyContainsSeqStmt(node);

    // Clear previous buffer info and collect new ones
    buffer_vector_infos_.clear();
    this->operator()(node);

    if (verbose) {
      std::cerr << "=== VectorizePlanner: Collected buffer vector sizes ==="
                << "\n";
      std::cerr << "  initial_vector_size=" << initial_vector_size_
                << ", loop_extent_vector_size=" << loop_extent_vector_size_
                << ", has_seq_stmt=" << (has_seq_stmt ? "true" : "false")
                << "\n";
    }

    // Compute the final vector size from the classified constraints.
    // Strategy:
    // - A must-scalarize constraint always selects vector_size=1.
    // - If the body contains SeqStmt, combine all constraints conservatively.
    // - If there is no vector memory access, combine local, broadcast-load,
    //   and call constraints.
    // - Otherwise, select from memory and non-cast call constraints, then
    //   revalidate deferred local and broadcast-load accesses.
    // Rationale: local/fragment accesses are register-level and impose no
    // memory alignment constraints. Invariant non-local loads become scalar
    // broadcasts, while invariant stores carry ordering semantics and must
    // remain scalar. Explicit classification keeps these semantic properties
    // separate from physical buffer scope.
    VectorConstraintSummary constraints = SummarizeConstraints(verbose);
    vector_size_ = SelectVectorSize(constraints, has_seq_stmt, verbose);

    // The vector width must divide the innermost loop's trip count.
    vector_size_ = arith::ZeroAwareGCD(loop_extent_vector_size_, vector_size_);

    if (verbose) {
      std::cerr << "=== Final vector_size: " << vector_size_ << " ===" << "\n";
    }
    return vector_size_;
  }

private:
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
    Array<PrimExpr> strides = GetBufferStrides(info.buffer);
    PrimExpr elem_offset = 0;
    for (size_t i = 0; i < info.indices.size(); ++i) {
      elem_offset += info.indices[i] * strides[i];
    }
    return IsExprInvariantInVectorBoundary(
        elem_offset, inner_for_->loop_var, inner_for_->extent,
        initial_vector_size_, analyzer_, inner_for_->min);
  }

  VectorConstraintKind ClassifyConstraint(const BufferVectorInfo &info) const {
    if (info.requires_scalarization) {
      return VectorConstraintKind::kMustScalarize;
    }
    if (!info.buffer.defined()) {
      return info.is_cast ? VectorConstraintKind::kCast
                          : VectorConstraintKind::kCall;
    }
    if (IsLocalBuffer(info.buffer, /*allow_var=*/true) ||
        IsFragmentBuffer(info.buffer)) {
      return VectorConstraintKind::kLocal;
    }
    if (IsBroadcastLoad(info)) {
      return VectorConstraintKind::kBroadcastLoad;
    }
    return VectorConstraintKind::kMemory;
  }

  void LogConstraint(const BufferVectorInfo &info,
                     VectorConstraintKind kind) const {
    if (info.buffer.defined()) {
      std::cerr << "  Buffer: " << info.buffer->name
                << " (scope=" << info.buffer.scope() << ")"
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
        summary.call_min =
            arith::ZeroAwareGCD(summary.call_min, info.vector_size);
        summary.non_cast_call_min =
            arith::ZeroAwareGCD(summary.non_cast_call_min, info.vector_size);
        break;
      case VectorConstraintKind::kCast:
        summary.call_min =
            arith::ZeroAwareGCD(summary.call_min, info.vector_size);
        break;
      case VectorConstraintKind::kLocal:
        summary.local_min =
            arith::ZeroAwareGCD(summary.local_min, info.vector_size);
        summary.deferred_accesses.push_back(info);
        break;
      case VectorConstraintKind::kMemory:
        summary.memory_min =
            arith::ZeroAwareGCD(summary.memory_min, info.vector_size);
        summary.has_memory_access = true;
        break;
      case VectorConstraintKind::kBroadcastLoad:
        summary.broadcast_load_min =
            arith::ZeroAwareGCD(summary.broadcast_load_min, info.vector_size);
        summary.deferred_accesses.push_back(info);
        break;
      }
    }
    if (verbose) {
      int legacy_local_fragment_min =
          arith::ZeroAwareGCD(summary.local_min, summary.broadcast_load_min);
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
      Array<PrimExpr> strides = GetBufferStrides(info.buffer);
      PrimExpr elem_offset = 0;
      for (size_t i = 0; i < info.indices.size(); ++i) {
        elem_offset += info.indices[i] * strides[i];
      }
      if (!IndicesCanVectorize(elem_offset, inner_for_->loop_var,
                               inner_for_->extent, vector_size, analyzer_,
                               /*allow_broadcast=*/true, inner_for_->min)) {
        int old_vector_size = vector_size;
        vector_size = arith::ZeroAwareGCD(vector_size, info.vector_size);
        if (verbose) {
          std::cerr << "  [Re-validate] Local buffer '" << info.buffer->name
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
    int deferred_min =
        arith::ZeroAwareGCD(summary.local_min, summary.broadcast_load_min);
    int selected;
    if (has_seq_stmt) {
      // Multiple statements may carry interactions not represented by the
      // simple single-store optimization, so include every constraint.
      selected = arith::ZeroAwareGCD(
          arith::ZeroAwareGCD(deferred_min, summary.memory_min),
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
      selected =
          arith::ZeroAwareGCD(summary.memory_min, summary.non_cast_call_min);
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
      selected = arith::ZeroAwareGCD(deferred_min, summary.call_min);
      if (verbose) {
        std::cerr << "  [Strategy] Only local/fragment buffers, using "
                     "GCD(local_fragment_min, call_node_min)="
                  << selected << "\n";
      }
    }
    return selected;
  }

  Stmt VisitStmt_(const ForNode *node) final {
    inner_for_ = GetRef<For>(node);
    bool contains_nested_for = false;
    // Only the innermost domain determines lane grouping.
    PostOrderVisit(Downcast<Stmt>(node->body), [&](const ObjectRef &obj) {
      if (obj.as<ForNode>()) {
        contains_nested_for = true;
      }
    });

    if (!contains_nested_for) {
      PrimExpr extent = analyzer_->Simplify(node->extent);
      const int64_t *extent_ptr = as_const_int(extent);
      // Splitting currently requires a unit step and a static, positive extent.
      if (!node->HasTrivialStep() || !extent_ptr || *extent_ptr <= 0) {
        loop_extent_vector_size_ = 1;
        return GetRef<Stmt>(node);
      }
      loop_extent_vector_size_ =
          arith::ZeroAwareGCD(initial_vector_size_, *extent_ptr);
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(node);
  }

  PrimExpr VisitExpr_(const BufferLoadNode *node) final {
    if (IsSharedBuffer(node->buffer) || IsGlobalBuffer(node->buffer))
      has_nonlocal_memory_access_ = true;
    if (node->buffer->shape.size() == 1) {
      // TODO(lei): This should be improved as
      // constant buffer that tl hack to use as local register.
      auto boundary_check = node->buffer->shape[0].as<IntImmNode>();
      if (boundary_check && boundary_check->value == 1) {
        return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
      }
    }
    UpdateVectorSize(node->indices, node->buffer, false);
    return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
  }

  Stmt VisitStmt_(const BufferStoreNode *node) final {
    if (IsSharedBuffer(node->buffer) || IsGlobalBuffer(node->buffer))
      has_nonlocal_memory_access_ = true;
    UpdateVectorSize(node->indices, node->buffer, true);
    return arith::IRMutatorWithAnalyzer::VisitStmt_(node);
  }

  Stmt VisitStmt_(const IfThenElseNode *node) final {
    CheckConditionVectorized(node->condition);
    return arith::IRMutatorWithAnalyzer::VisitStmt_(node);
  }

  PrimExpr VisitExpr_(const SelectNode *node) final {
    // Select stays an expression-level ternary. Constrain its vector width
    // using the same condition-uniformity rule as IfThenElse, unless the
    // target codegen supports lane-wise vector predicates and therefore
    // does not need this control-flow restriction.
    Target target = Target::Current(false);
    bool vector_predicate =
        target.defined() &&
        target->GetAttr<Bool>("supports_vector_predicate", Bool(false)).value();
    if (!vector_predicate) {
      CheckConditionVectorized(node->condition);
    }
    return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
  }

  static std::optional<int> GetAccessPtrElementBits(const PrimExpr &expr) {
    const auto *ptr_call = expr.as<CallNode>();
    if (ptr_call == nullptr) {
      return std::nullopt;
    }
    if (ptr_call->op.same_as(builtin::tvm_access_ptr())) {
      ICHECK(!ptr_call->args.empty());
      DataType dtype = ptr_call->args[0].dtype();
      return dtype.bits() * dtype.lanes();
    }
    if (ptr_call->op.same_as(tl::access_ptr())) {
      ICHECK_EQ(ptr_call->args.size(), 3U)
          << "tl.access_ptr expects 3 args: (BufferLoad, extent, rw_mask)";
      const auto *buffer_load = ptr_call->args[0].as<BufferLoadNode>();
      ICHECK(buffer_load) << "tl.access_ptr arg0 must be BufferLoad";
      DataType dtype = buffer_load->buffer->dtype;
      return dtype.bits() * dtype.lanes();
    }
    return std::nullopt;
  }

  static std::optional<int> GetCPAsyncBitsPerCall(const CallNode *node) {
    ICHECK_GE(node->args.size(), 3U)
        << "cp.async expects at least 3 arguments, but got " << node->args;
    const auto *count_imm = node->args[2].as<IntImmNode>();
    ICHECK(count_imm) << "cp.async transfer count must be IntImm, but got "
                      << node->args[2];
    int count = static_cast<int>(count_imm->value);
    if (count <= 0) {
      return std::nullopt;
    }
    if (node->op.same_as(builtin::ptx_cp_async())) {
      return count * 8;
    }
    ICHECK(node->op.same_as(tl::ptx_cp_async()));
    auto dst_elem_bits = GetAccessPtrElementBits(node->args[0]);
    auto src_elem_bits = GetAccessPtrElementBits(node->args[1]);
    if (!dst_elem_bits.has_value() || !src_elem_bits.has_value()) {
      return std::nullopt;
    }
    int dst_total_bits = count * dst_elem_bits.value();
    int src_total_bits = count * src_elem_bits.value();
    ICHECK_EQ(dst_total_bits, src_total_bits)
        << "tl.ptx_cp_async requires src/dst transfer widths to match, but got "
        << dst_total_bits << " vs " << src_total_bits << " bits";
    return dst_total_bits;
  }

  static int GetMaxCPAsyncVectorizeLength(int per_call_bits) {
    if (per_call_bits <= 0) {
      return 1;
    }
    int vectorize_length = 1;
    for (int target_bytes : {16, 8, 4}) {
      int target_bits = target_bytes * 8;
      if (target_bits % per_call_bits == 0) {
        vectorize_length =
            std::max(vectorize_length, target_bits / per_call_bits);
      }
    }
    return vectorize_length;
  }

  PrimExpr VisitExpr_(const CallNode *node) final {
    if (node->op == builtin::if_then_else()) {
      CheckConditionVectorized(node->args[0]);
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    } else if (node->op.same_as(builtin::tvm_access_ptr())) {
      HandleTvmAccessPtr(node);
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    } else if (node->op.same_as(tl::rng_rand_float())) {
      // Stateful RNG calls produce scalar values before any destination cast.
      // Keep that source width as a non-cast constraint: the simple-store
      // strategy intentionally ignores cast constraints, while later passes
      // handle the packed destination conversion independently.
      int value_bits = node->dtype.bits() * node->dtype.lanes();
      ICHECK_GT(value_bits, 0)
          << "tl.rng_rand_float requires a fixed-width result dtype, got "
          << node->dtype;
      int vectorize_length =
          std::max(1, std::min(4, vector_load_bits_max_ / value_bits));
      buffer_vector_infos_.push_back(
          {Buffer(), vectorize_length, false, {}, /*is_cast=*/false});
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    } else if (node->op == tl::atomic_add_elem_op()) {
      ICHECK_GE(node->args.size(), 2U)
          << "atomic_add_elem_op requires at least 2 args (dst and src)";

      // Keep this as a call constraint: simple-memory planning must not defer
      // the atomic's legality together with local or broadcast-load accesses.
      buffer_vector_infos_.push_back(
          {Buffer(), ComputeAtomicVectorSize(node->args[0]), false, {}});
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    } else if (node->op.same_as(builtin::ptx_cp_async()) ||
               node->op.same_as(tl::ptx_cp_async())) {
      // builtin::ptx_cp_async stores bytes, while tl::ptx_cp_async stores
      // logical element counts. In both cases we pick the largest vector width
      // whose eventual PTX payload is one of {4, 8, 16} bytes.
      int vectorize_length =
          GetMaxCPAsyncVectorizeLength(GetCPAsyncBitsPerCall(node).value_or(0));
      buffer_vector_infos_.push_back({Buffer(), vectorize_length, false, {}});
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    } else if (node->op == builtin::address_of() ||
               node->op == tl::access_ptr()) {
      // address_of and tl.access_ptr have buffer load value so we should
      // analysis the buffer load node to update vector_size_.
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    } else if (node->op.same_as(tl::reducer_update())) {
      // The accumulate itself never loop-vectorizes: by the time execution
      // vectorization runs (post ReducerPlanAndMaterialize) this call has
      // been rewritten into an ordinary read-modify-write store, which the
      // reduction-axis scalarization guard below keeps correct. During
      // layout inference the call therefore only influences the LAYOUT
      // SHAPE this nest plans. Shape it by the contribution loads (visit
      // the args as ordinary expressions; the accumulator access is
      // loop-invariant and adds no constraint) instead of the generic
      // opaque-call rule, whose all-lanes-invariant test degrades every
      // update nest to a scalar-shaped (elementwise mod-threads) plan and
      // forces that shape onto the fragments feeding it — scalarizing
      // their shared-memory copies (observed as a 1.1-1.3x latency
      // regression on production kernels when such a plan wins).
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    }

    // vectorizable property
    OpAttrMap<TVectorizable> op_vectorizable_ =
        Op::GetAttrMap<TVectorizable>("TVectorizable");

    auto optional_op = node->op.as<Op>();
    bool vectorizable = op_vectorizable_.get(optional_op.value(), false) &&
                        !node->dtype.is_scalable_vector();
    if (vectorizable) {
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    }

    // For other call nodes, use PostOrderVisit to check buffer accesses
    // and determine if the given vector size is invariant
    auto check_buffer_access_invariant = [&](int target_vec_size) -> bool {
      if (!inner_for_.defined())
        return true;
      bool all_invariant = true;
      PostOrderVisit(GetRef<PrimExpr>(node), [&](const ObjectRef &obj) {
        if (!all_invariant)
          return;
        if (auto *load = obj.as<BufferLoadNode>()) {
          auto transformed_indices =
              TransformIndices(load->indices, load->buffer);
          Array<PrimExpr> strides = GetBufferStrides(load->buffer);
          PrimExpr elem_offset = 0;
          for (size_t i = 0; i < transformed_indices.size(); ++i) {
            elem_offset += transformed_indices[i] * strides[i];
          }
          if (!IsExprInvariantInVectorBoundary(
                  elem_offset, inner_for_->loop_var, inner_for_->extent,
                  target_vec_size, analyzer_, inner_for_->min)) {
            all_invariant = false;
          }
        } else if (auto *store = obj.as<BufferStoreNode>()) {
          auto transformed_indices =
              TransformIndices(store->indices, store->buffer);
          Array<PrimExpr> strides = GetBufferStrides(store->buffer);
          PrimExpr elem_offset = 0;
          for (size_t i = 0; i < transformed_indices.size(); ++i) {
            elem_offset += transformed_indices[i] * strides[i];
          }
          if (!IsExprInvariantInVectorBoundary(
                  elem_offset, inner_for_->loop_var, inner_for_->extent,
                  target_vec_size, analyzer_, inner_for_->min)) {
            all_invariant = false;
          }
        } else if (auto *call = obj.as<CallNode>()) {
          // tvm_access_ptr(dtype_annotation, data, offset, extent, rw_mask)
          // The offset (args[2]) is the element offset into the buffer.
          if (call->op.same_as(builtin::tvm_access_ptr()) &&
              call->args.size() >= 3) {
            PrimExpr offset = call->args[2];
            if (!IsExprInvariantInVectorBoundary(
                    offset, inner_for_->loop_var, inner_for_->extent,
                    target_vec_size, analyzer_, inner_for_->min)) {
              all_invariant = false;
            }
          }
        }
      });
      return all_invariant;
    };
    // Find the largest vector size where all buffer accesses are invariant
    int call_node_vector_size = loop_extent_vector_size_;
    while (call_node_vector_size > 1) {
      if (check_buffer_access_invariant(call_node_vector_size)) {
        break;
      }
      call_node_vector_size /= 2;
    }
    buffer_vector_infos_.push_back(
        {Buffer(), call_node_vector_size, false, {}});
    return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
  }

  int ComputeAtomicVectorSize(const PrimExpr &destination) {
    ICHECK(inner_for_.defined())
        << "Atomic vectorization analysis requires a loop context";

    DataType dtype;
    String storage_scope;
    PrimExpr offset;
    if (auto load = ExtractBufferLoadForAtomic(destination); load.defined()) {
      const Buffer &buffer = load.value()->buffer;
      dtype = buffer->dtype;
      storage_scope = buffer.scope();
      // Keep the original Buffer identity for layout lookup.
      Array<PrimExpr> indices = TransformIndices(load.value()->indices, buffer);
      Array<PrimExpr> strides = GetBufferStrides(buffer);
      offset = buffer->elem_offset;
      for (size_t i = 0; i < indices.size(); ++i) {
        offset = offset + indices[i] * strides[i];
      }
    } else {
      Call ptr = Downcast<Call>(destination);
      ICHECK(ptr->op.same_as(builtin::tvm_access_ptr()))
          << "Unsupported atomic destination: " << destination;
      ICHECK_EQ(ptr->args.size(), 5U)
          << "tvm_access_ptr expects (dtype, data, offset, extent, rw_mask)";
      dtype = ptr->args[0].dtype();
      storage_scope = GetPtrStorageScope(Downcast<Var>(ptr->args[1]));
      // tvm_access_ptr already carries the physical element offset.
      offset = ptr->args[2];
    }

    int vector_size = arith::ZeroAwareGCD(
        loop_extent_vector_size_,
        GetMaxAtomicVectorSize(dtype, storage_scope, Target::Current(false)));
    // A wide atomic updates distinct consecutive elements. Unlike a load,
    // its destination cannot broadcast. Search smaller widths here, before
    // partitioning/layout choices and before rewriting the loop body.
    while (vector_size > 1 &&
           !IndicesCanVectorize(offset, inner_for_->loop_var,
                                inner_for_->extent, vector_size, analyzer_,
                                /*allow_broadcast=*/false, inner_for_->min)) {
      vector_size /= 2;
    }
    return vector_size;
  }

  void CheckConditionVectorized(const PrimExpr &cond) {
    if (!inner_for_.defined()) {
      return;
    }
    int condition_vector_size = loop_extent_vector_size_;
    PrimExpr condition = cond;
    if (condition_vector_size > 1) {
      condition =
          SimplifyExprForAnalyzer(cond, condition_vector_size, analyzer_);
    }
    while (condition_vector_size > 1 &&
           !IsExprInvariantInVectorBoundary(
               condition, inner_for_->loop_var, inner_for_->extent,
               condition_vector_size, analyzer_, inner_for_->min)) {
      condition_vector_size /= 2;
    }
    buffer_vector_infos_.push_back(
        {Buffer(), condition_vector_size, false, {}});
  }

  void HandleTvmAccessPtr(const CallNode *node) {
    // tvm_access_ptr format: (ptype, data, offset, extent, rw_mask)
    if (!inner_for_.defined()) {
      return;
    }
    ICHECK(node->args.size() >= 3U)
        << "tvm_access_ptr requires at least 3 args";

    // args[0] is TypeAnnotation(dtype[/lanes]); dtype() encodes the element
    // type. See tvm::tirx::Buffer::access_ptr implementation.
    DataType dtype = node->args[0].dtype();
    Var data_var;
    if (auto data_var_node = node->args[1].as<VarNode>()) {
      data_var = Downcast<Var>(node->args[1]);
    }
    ICHECK(data_var.defined()) << "tvm_access_ptr second arg must be a var";
    PrimExpr offset = node->args[2];

    Optional<Buffer> buffer_opt;
    Optional<Layout> layout_opt;

    // Find the Buffer whose data pointer matches data_var by searching
    // layout_map_. The layout_map_ maps Buffer -> Layout, so we iterate
    // to find the buffer whose ->data field is the same Var.
    if (layout_map_.defined()) {
      for (auto [buf, layout] : layout_map_) {
        if (buf->data.same_as(data_var)) {
          buffer_opt = buf;
          layout_opt = layout;
          break;
        }
      }
    }

    // Base vector size from loop extent.
    int access_vec_size = loop_extent_vector_size_;
    // Constrain by dtype lane capacity (128/256-bit vector load/store width).
    // This mirrors ComputeBufferVectorSize's dtype-based lower bound.
    int dtype_bits = dtype.bits() * dtype.lanes();
    if (dtype_bits > 0) {
      int dtype_lane_bound = vector_load_bits_max_ / dtype_bits;
      if (dtype_lane_bound <= 0) {
        dtype_lane_bound = 1;
      }
      access_vec_size = arith::ZeroAwareGCD(access_vec_size, dtype_lane_bound);
    }

    // If the buffer has a layout, use the last output dimension as a proxy for
    // the maximum contiguous vector length implied by the layout.
    if (layout_opt.defined()) {
      Array<PrimExpr> out_shape = layout_opt.value()->OutputShape();
      if (!out_shape.empty()) {
        PrimExpr contig = analyzer_->Simplify(out_shape.back());
        if (auto contig_int = as_const_int(contig);
            contig_int && *contig_int > 1) {
          access_vec_size = arith::ZeroAwareGCD(access_vec_size, *contig_int);
        }
      }
    }
    // tvm_access_ptr itself is not vectorizable in TLVectorizer. If its offset
    // depends on the vectorized loop var, TLVectorizer will force scalarization
    // of the whole loop body. To avoid planning a vector size that will be
    // immediately scalarized (and to keep semantics sane for side-effectful
    // calls), require the offset to be invariant within the vector boundary.
    PrimExpr offset_s = offset;
    if (access_vec_size > 1) {
      offset_s = SimplifyExprForAnalyzer(offset, access_vec_size, analyzer_);
    }
    while (access_vec_size > 1 &&
           !IndicesCanVectorize(offset_s, inner_for_->loop_var,
                                inner_for_->extent, access_vec_size, analyzer_,
                                /*allow_broadcast=*/true, inner_for_->min)) {
      access_vec_size /= 2;
    }
    // Record as a memory-like constraint if we can resolve the buffer.
    buffer_vector_infos_.push_back(
        {buffer_opt.value_or(Buffer()), access_vec_size, false, {}});
  }

  Array<PrimExpr> TransformIndices(const Array<PrimExpr> &indices,
                                   const Buffer &buffer) {
    auto transformed_indices = indices;
    if (layout_map_.defined() && layout_map_.count(buffer)) {
      ICHECK(IsBufferContiguous(buffer, analyzer_))
          << buffer
          << " has non-contiguous strides, but layout map is provided.";
      // forward indices
      auto layout = layout_map_[buffer];
      transformed_indices = layout->Forward(indices);
      // Reshape transformed_indices to match buffer->shape dimensions if needed
      if (transformed_indices.size() != buffer->shape.size()) {
        // Step 1: Compute linear offset using layout->OutputShape()
        auto output_shape = layout->OutputShape();
        ICHECK_EQ(transformed_indices.size(), output_shape.size())
            << "Forward indices size " << transformed_indices.size()
            << " != OutputShape size " << output_shape.size();
        PrimExpr linear_offset = 0;
        PrimExpr stride = 1;
        for (int i = output_shape.size() - 1; i >= 0; --i) {
          linear_offset = linear_offset + transformed_indices[i] * stride;
          stride = stride * output_shape[i];
        }
        // Step 2: Decompose linear_offset into buffer->shape dimensions
        Array<PrimExpr> new_indices;
        for (int i = buffer->shape.size() - 1; i >= 0; --i) {
          new_indices.push_back(FloorMod(linear_offset, buffer->shape[i]));
          linear_offset = FloorDiv(linear_offset, buffer->shape[i]);
        }
        transformed_indices =
            Array<PrimExpr>{new_indices.rbegin(), new_indices.rend()};
      }
    }
    return transformed_indices;
  }

  PrimExpr VisitExpr_(const CastNode *node) final {
    // Consider both source and target types to ensure all intermediate
    // vector types can be represented. For example, casting int32 to
    // float8_e4m3fn: target allows 128/8=16 lanes but int32 only supports
    // up to 128/32=4 lanes in CUDA vector types.
    int target_lanes = vector_load_bits_max_ / node->dtype.bits();
    int source_bits = node->value.dtype().bits();
    int max_lanes = target_lanes;
    if (source_bits > 0) {
      int source_lanes = vector_load_bits_max_ / source_bits;
      max_lanes = std::min(target_lanes, source_lanes);
    }
    int cast_vector_size = arith::ZeroAwareGCD(max_lanes, initial_vector_size_);
    // For casts that carry a "rbits" annotation (stochastic rounding), the
    // rbits operand must be invariant within the vectorized boundary so the
    // same value can be shared across the lanes of each PTX cvt.rs.*
    // instruction. Halve the vector size until invariance holds, mirroring
    // the call-node fallback above.
    if (inner_for_.defined()) {
      auto it = node->annotations.find("rbits");
      if (it != node->annotations.end()) {
        PrimExpr rbits = Downcast<PrimExpr>((*it).second);
        while (cast_vector_size > 1 &&
               !IsExprInvariantInVectorBoundary(
                   rbits, inner_for_->loop_var, inner_for_->extent,
                   cast_vector_size, analyzer_, inner_for_->min)) {
          cast_vector_size /= 2;
        }
      }
    }
    // Record cast constraint (use empty buffer to indicate cast)
    // Mark is_cast=true so Plan() can distinguish cast from other call nodes
    buffer_vector_infos_.push_back(
        {Buffer(), cast_vector_size, false, {}, /*is_cast=*/true});
    return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
  }

  VectorSizeConstraint ComputeBufferVectorSize(const Array<PrimExpr> &indices,
                                               const Buffer &buffer,
                                               bool is_store) {
    if (!inner_for_.defined())
      return {initial_vector_size_, /*requires_scalarization=*/false};

    int buffer_vec_size = loop_extent_vector_size_;

    // Transform indices using layout_map if present
    auto transformed_indices = TransformIndices(indices, buffer);

    // 1. Compute raw element offset
    Array<PrimExpr> strides = GetBufferStrides(buffer);

    PrimExpr elem_offset = 0;
    for (size_t i = 0; i < transformed_indices.size(); ++i) {
      elem_offset += transformed_indices[i] * strides[i];
    }

    // 2. Check if current buffer_vec_size works with invariant boundary check
    // In some cases, buffer_vec_size is max (e.g. 128), but
    // IsExprInvariantInVectorBoundary may only be true at a smaller size (e.g.
    // 64). Recursively halve buffer_vec_size until we find a size where
    // is_invariant is true. Fallback: minimum vector size based on buffer dtype
    int min_vec_size = arith::ZeroAwareGCD(
        buffer_vec_size,
        vector_load_bits_max_ / (buffer->dtype.bits() * buffer->dtype.lanes()));
    bool is_invariant = false;
    int try_vec_size = buffer_vec_size;
    while (try_vec_size >= min_vec_size) {
      is_invariant = IsExprInvariantInVectorBoundary(
          elem_offset, inner_for_->loop_var, inner_for_->extent, try_vec_size,
          analyzer_, inner_for_->min);
      if (is_invariant) {
        buffer_vec_size = try_vec_size;
        break;
      }
      try_vec_size /= 2;
    }
    // If is_invariant is still false, use the fallback min_vec_size
    if (!is_invariant) {
      buffer_vec_size = min_vec_size;
    }

    // 3. If element offset is independent with loop_var, ignore it.
    bool is_independent =
        CanProveIndependent(elem_offset, inner_for_->loop_var, analyzer_);
    // For ordinary BufferStore, if indices are invariant or independent with
    // loop_var, vectorization would turn scalar lane stores into a broadcast
    // store. Keep those scalar. This is also the guard that keeps reducer
    // combine stores (read-modify-write chains) correct: a store whose index
    // does not advance with the loop var is a reduction along the vectorized
    // axis, and vectorizing it would collapse the dependent chain into
    // last-lane-wins. Output-axis combines, whose target does advance with
    // the loop var, vectorize like any other store.
    if (is_store && (is_invariant || is_independent)) {
      return {1, /*requires_scalarization=*/true};
    }
    if (is_independent) {
      return {buffer_vec_size, /*requires_scalarization=*/false};
    }
    // 4. Try to find max vectorize size for this buffer
    while (buffer_vec_size > 1 &&
           !IndicesCanVectorize(elem_offset, inner_for_->loop_var,
                                inner_for_->extent, buffer_vec_size, analyzer_,
                                /*allow_broadcast=*/true, inner_for_->min)) {
      buffer_vec_size /= 2;
    }
    return {buffer_vec_size, /*requires_scalarization=*/false};
  }

  void UpdateVectorSize(const Array<PrimExpr> &indices, const Buffer &buffer,
                        bool is_store) {
    VectorSizeConstraint constraint =
        ComputeBufferVectorSize(indices, buffer, is_store);
    buffer_vector_infos_.push_back({buffer, constraint.vector_size, is_store,
                                    indices, /*is_cast=*/false,
                                    constraint.requires_scalarization});
  }

  // NOTE(wt): The base class IRMutatorWithAnalyzer::VisitStmt_(BindNode*)
  // binds let variables, but this causes issues when the same variable name
  // appears multiple times with different values (e.g., in pipelined loops
  // where the body is duplicated). For this case, we allow the analyzer to
  // override the binding. Check the impl of
  // IRMutatorWithAnalyzer::VisitStmt_(BindNode*) in:
  // tvm/src/arith/ir_mutator_with_analyzer.cc
  Stmt VisitStmt_(const BindNode *op) final {
    PrimExpr value = this->VisitExpr(op->value);
    if (SideEffect(value) <= CallEffectKind::kPure) {
      // Allow override to handle duplicated loop bodies in pipelined loops
      analyzer_->Bind(op->var, value, /*allow_override=*/true);
    }
    if (value.same_as(op->value)) {
      return GetRef<Stmt>(op);
    } else {
      auto n = this->CopyOnWrite(op);
      n->value = std::move(value);
      return Stmt(n);
    }
  }

  int vector_load_bits_max_;
  int initial_vector_size_ = 128;
  int loop_extent_vector_size_ = 128;

  For inner_for_;
  bool has_nonlocal_memory_access_ = false;
  int vector_size_ = 128;
  std::vector<BufferVectorInfo> buffer_vector_infos_;
  LayoutMap layout_map_;
};

class VectorizeRewriter : public StmtExprMutator {
public:
  VectorizeRewriter(int vector_size, arith::Analyzer *analyzer)
      : vector_size_(vector_size), analyzer_(analyzer) {}

private:
  Stmt VisitStmt_(const ForNode *node) final {
    inner_for_ = GetRef<For>(node);
    auto ret = StmtExprMutator::VisitStmt_(node);
    if (inner_for_.same_as(GetRef<For>(node))) { // rewrite the innermost loop
      For fnode = ret.as<For>().value();
      auto old_var = fnode->loop_var;
      ICHECK(fnode->HasTrivialStep())
          << "Vectorization currently requires a unit loop step";
      PrimExpr simplified_extent = analyzer_->Simplify(fnode->extent);
      auto extent_ptr = as_const_int(simplified_extent);
      ICHECK(extent_ptr) << simplified_extent;
      int64_t extent = *extent_ptr;
      ICHECK(extent % vector_size_ == 0)
          << "extent: " << extent << " vector_size_: " << vector_size_
          << " for loop: " << fnode;
      DataType dtype = old_var.dtype();
      PrimExpr zero = make_zero(dtype);
      PrimExpr width = make_const(dtype, vector_size_);
      Var inner_var("vec", dtype);
      if (extent == vector_size_) {
        Stmt body =
            Substitute(fnode->body, {{old_var, fnode->min + inner_var}});
        return For(inner_var, zero, width, ForKind::kVectorized, body,
                   fnode->thread_binding, fnode->annotations, std::nullopt,
                   fnode->span);
      } else {
        Var outer_var(old_var->name_hint, dtype);
        Map<Var, PrimExpr> vmap;
        vmap.Set(old_var, fnode->min + (outer_var * width + inner_var));
        Stmt body = Substitute(fnode->body, vmap);
        body = For(inner_var, zero, width, ForKind::kVectorized, body,
                   std::nullopt, {}, std::nullopt, fnode->span);
        // TileLang uses ForKind::kParallel in frontend SIMT loops. After
        // vectorization, keep semantics equivalent but downgrade to serial so
        // subsequent passes (e.g. pragma-unroll) can run.
        ForKind outer_kind = fnode->kind;
        if (outer_kind == ForKind::kParallel) {
          outer_kind = ForKind::kSerial;
        }
        body = For(outer_var, zero, make_const(dtype, extent / vector_size_),
                   outer_kind, body, fnode->thread_binding, fnode->annotations,
                   std::nullopt, fnode->span);
        return body;
      }
    } else {
      // Keep other loops intact, except for TileLang frontend "parallel" loops
      // which should behave as serial loops after lowering.
      For loop = ret.as<For>().value();
      if (loop->kind == ForKind::kParallel) {
        loop.CopyOnWrite()->kind = ForKind::kSerial;
      }
      return loop;
    }
  }

  For inner_for_;
  const int vector_size_;
  arith::Analyzer *analyzer_;
};

int GetVectorizeSize(const For &loop, const LayoutMap &layout_map) {
  arith::Analyzer analyzer;
  return VectorizePlanner(&analyzer, layout_map).Plan(loop);
}

int GetVectorizeSize(const For &loop, arith::Analyzer *analyzer,
                     const LayoutMap &layout_map) {
  return VectorizePlanner(analyzer, layout_map).Plan(loop);
}

namespace {

// Canonical simplification may multiply an integer subexpression's coefficient
// by the candidate vector width before it constructs an IntImm. Widen the
// analysis copy if any such scaled bound can exceed the source dtype. Inspect
// subexpressions as well as the root so boolean conditions and cancelling
// affine terms cannot hide a risky integer coefficient.
bool ScaledIntSubexprMayOverflow(const PrimExpr &expr, int scale,
                                 arith::Analyzer *analyzer) {
  ICHECK_GE(scale, 1);
  bool may_overflow = false;
  PostOrderVisit(expr, [&](const ObjectRef &obj) {
    if (may_overflow) {
      return;
    }
    Optional<PrimExpr> opt_subexpr = obj.as<PrimExpr>();
    if (!opt_subexpr.defined()) {
      return;
    }
    PrimExpr subexpr = opt_subexpr.value();
    DataType dtype = subexpr.dtype();
    if (!dtype.is_int() || dtype.bits() >= 64 || dtype.lanes() != 1) {
      return;
    }

    arith::ConstIntBound bound = analyzer->const_int_bound(subexpr);
    const int64_t type_max = (1LL << (dtype.bits() - 1)) - 1;
    const int64_t type_min = -(1LL << (dtype.bits() - 1));
    may_overflow = bound->max_value > type_max / scale ||
                   bound->min_value < type_min / scale;
  });
  return may_overflow;
}

PrimExpr PrepareExprForAnalyzer(const PrimExpr &expr, int scale,
                                arith::Analyzer *analyzer) {
  if (!ScaledIntSubexprMayOverflow(expr, scale, analyzer)) {
    return expr;
  }
  Int64Promoter promoter;
  return promoter(expr);
}

PrimExpr SimplifyExprForAnalyzer(const PrimExpr &expr, int scale,
                                 arith::Analyzer *analyzer) {
  return analyzer->Simplify(PrepareExprForAnalyzer(expr, scale, analyzer));
}

} // namespace

bool CanProveIndependent(const PrimExpr &expr, Var var,
                         arith::Analyzer *analyzer) {
  // 1. if var doesn't exist, it is independent
  bool used_var = UsesVar(
      expr, [&](const VarNode *v) { return GetRef<Var>(v).same_as(var); });
  if (!used_var) {
    return true;
  }
  // 2. if \forall v_1, v_2, f(v_1) == f(v_2), f is independent with v
  Var var_1("_t", var.dtype());
  PrimExpr expr_1 = Substitute(expr, {{var, var_1}});
  PrimExpr equality = PrepareExprForAnalyzer(expr == expr_1, 1, analyzer);
  const auto *equality_node = equality.as<EQNode>();
  ICHECK(equality_node);
  if (analyzer->CanProveEqual(equality_node->a, equality_node->b)) {
    return true;
  }
  return false;
}

bool IsExprInvariantInVectorBoundary(const PrimExpr &expr, const Var &var,
                                     const PrimExpr &extent,
                                     int target_vectorized_size,
                                     arith::Analyzer *analyzer,
                                     const PrimExpr &min) {
  // Groups begin at iteration ordinals 0, width, ... rather than logical
  // indices divisible by width. For i in [2, 6), i // 4 is not a broadcast
  // at width 4: its lane values are [0, 0, 1, 1].
  Var ordinal("ordinal", var.dtype());
  PrimExpr analysis_expr =
      PrepareExprForAnalyzer(expr, target_vectorized_size, analyzer);
  analysis_expr =
      Substitute(analysis_expr, {{var, cast(var.dtype(), min) + ordinal}});
  analyzer->Bind(ordinal,
                 Range::FromMinExtent(make_zero(ordinal.dtype()), extent));
  PrimExpr aligned =
      floordiv(ordinal, target_vectorized_size) * target_vectorized_size;
  PrimExpr expr_aligned = Substitute(analysis_expr, {{ordinal, aligned}});
  return analyzer->CanProveEqual(analysis_expr, expr_aligned);
}

bool IndicesCanVectorize(const PrimExpr &expr, const Var &var,
                         const PrimExpr &extent, int target_vectorized_size,
                         arith::Analyzer *analyzer, bool allow_broadcast,
                         const PrimExpr &min) {
  ICHECK(target_vectorized_size >= 1);
  if (target_vectorized_size == 1) {
    return true;
  }
  // The extent must be divisible by the selected vector width.
  PrimExpr target_size_for_iter =
      make_const(extent.dtype(), target_vectorized_size);
  if (!analyzer->CanProveEqual(FloorMod(extent, target_size_for_iter), 0))
    return false;

  if (allow_broadcast &&
      IsExprInvariantInVectorBoundary(expr, var, extent, target_vectorized_size,
                                      analyzer, min)) {
    return true;
  }

  Var ordinal("ordinal", var.dtype());
  PrimExpr analysis_expr =
      PrepareExprForAnalyzer(expr, target_vectorized_size, analyzer);
  analysis_expr =
      Substitute(analysis_expr, {{var, cast(var.dtype(), min) + ordinal}});
  PrimExpr target_size_for_expr =
      make_const(analysis_expr.dtype(), target_vectorized_size);
  PrimExpr target_size_for_ordinal =
      make_const(ordinal.dtype(), target_vectorized_size);
  PrimExpr zero_ordinal = make_const(ordinal.dtype(), 0);
  PrimExpr zero_expr = make_const(analysis_expr.dtype(), 0);

  PrimExpr simplified_expr =
      analyzer->Simplify(Substitute(analysis_expr, {{ordinal, zero_ordinal}}));
  // The base offset must be divisible
  if (!analyzer->CanProveEqual(FloorMod(simplified_expr, target_size_for_expr),
                               zero_expr)) {
    return false;
  }

  // Bind thread range
  Var v0("v0", ordinal.dtype()), v1("v1", ordinal.dtype());
  analyzer->Bind(v0, Range(zero_ordinal, target_size_for_ordinal));
  analyzer->Bind(v1, Range(zero_ordinal, analyzer->Simplify(FloorDiv(
                                             extent, target_size_for_iter))));
  PrimExpr expr_transformed = analyzer->Simplify(Substitute(
      analysis_expr, {{ordinal, v1 * target_size_for_ordinal + v0}}));
  Vectorizer vectorizer(v0, target_size_for_ordinal);
  PrimExpr expr_vectorized = vectorizer.VisitExpr(expr_transformed);

  // This simplify is necessary for thread region specified
  // optimizations.
  expr_vectorized = analyzer->Simplify(expr_vectorized);
  auto ramp_node = expr_vectorized.as<RampNode>();
  if (!ramp_node) {
    return allow_broadcast && expr_vectorized.dtype().lanes() == 1;
  }
  if (!is_one(ramp_node->stride)) {
    return false;
  }
  // The first-iteration check above only covers the first group. For example,
  // (ordinal // 4) * 6 + ordinal % 4 has group bases 0, 6, ... at width 4: the
  // first group is aligned, but the second is not. Atomic destinations must
  // align every group's base; the planner can retry this example at width 2.
  return allow_broadcast ||
         analyzer->CanProveEqual(
             FloorMod(ramp_node->base, target_size_for_expr), zero_expr);
}

int MaxVectorLoadBits(const Target &target, bool global_only_access) {
  if (auto max_bits = target->GetAttr<Integer>("max_vector_bits")) {
    return static_cast<int>(max_bits.value()->value);
  }
  if (TargetSupportVectorize256(target) && !tl_config::Vectorize256Disabled() &&
      global_only_access) {
    return 256;
  }
  return 128;
}

namespace {

/*!
 * \brief Convert TIR parallel loops into serial loops.
 *
 * TileLang uses ForKind::kParallel in a few places as a frontend "SIMT loop"
 * marker. When vectorize size resolves to 1 (i.e. no vectorization is applied),
 * keeping these loops as kParallel can block later loop transforms that only
 * apply to serial loops (e.g. pragma-unroll rewriting).
 *
 * This rewriter is intentionally conservative: it only downgrades kParallel to
 * kSerial and leaves all other loop kinds untouched.
 */
class ParallelToSerialRewriter : public StmtExprMutator {
private:
  Stmt VisitStmt_(const ForNode *node) final {
    Stmt visited = StmtExprMutator::VisitStmt_(node);
    For loop = Downcast<For>(visited);
    if (loop->kind == ForKind::kParallel) {
      loop.CopyOnWrite()->kind = ForKind::kSerial;
    }
    return loop;
  }
};

For ParallelToSerial(const For &loop) {
  ParallelToSerialRewriter rewriter;
  return Downcast<For>(rewriter(loop));
}

} // namespace

For VectorizeLoop(const For &loop, const LayoutMap &layout_map,
                  int vectorize_hint) {
  arith::Analyzer analyzer;
  return VectorizeLoop(loop, &analyzer, layout_map, vectorize_hint);
}

For VectorizeLoop(const For &loop, arith::Analyzer *analyzer,
                  const LayoutMap &layout_map, int vectorize_hint) {
  if (vectorize_hint <= 0) {
    VectorizePlanner planner(analyzer, layout_map);
    vectorize_hint = planner.Plan(loop);
  }
  if (vectorize_hint == 1)
    return ParallelToSerial(loop);
  auto rewriter = VectorizeRewriter(vectorize_hint, analyzer);
  return Downcast<For>(rewriter(loop));
}

} // namespace tl
} // namespace tvm

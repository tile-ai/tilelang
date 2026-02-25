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
#include "../target/utils.h"
#include "arith/int_operator.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "common/loop_vectorization_utils.h"
#include "tvm/tir/analysis.h"
#include "tvm/tir/var.h"
#include <iostream>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

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

struct BufferVectorInfo {
  Buffer buffer;
  int vector_size;
  bool is_store;
  Array<PrimExpr> indices;
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
  bool has_seq_stmt = false;
  PostOrderVisit(loop->body, [&](const ObjectRef &obj) {
    if (obj.as<SeqStmtNode>()) {
      has_seq_stmt = true;
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
    bool disable_vectorize_256 = tl_config::Vectorize256Disabled();
    bool verbose = tl_config::VectorizePlannerVerboseEnabled();

    if (TargetSupportVectorize256(Target::Current(false)) &&
        !disable_vectorize_256 &&
        VectorizeFindMemoryAccess::MaySupportVectorize256(node)) {
      vector_load_bits_max_ = initial_vector_size_ = loop_extent_vector_size_ =
          256;
    } else {
      vector_load_bits_max_ = initial_vector_size_ = loop_extent_vector_size_ =
          128;
    }

    // Check if For body contains SeqStmt (multiple statements).
    // When there's SeqStmt, we use conservative strategy - treating local
    // buffers the same as memory buffers. The special local buffer optimization
    // (ignoring local buffer constraints) only applies to simple single
    // BufferStore cases.
    bool has_seq_stmt = ForBodyContainsSeqStmt(node);

    // Clear previous buffer info and collect new ones
    buffer_vector_infos_.clear();
    this->operator()(node);

    // Compute final vector size from collected buffer infos
    // Strategy:
    // - If For body contains SeqStmt: take min of all buffers (conservative)
    // - Else if all buffers are local/fragment: take min of all
    // - Else if there are global/shared buffers: ignore local/fragment
    //   constraints and only take min of global/shared buffers
    // Rationale: local/fragment are register-level, no memory alignment
    // constraints. But for complex cases (SeqStmt), we stay conservative.
    vector_size_ = initial_vector_size_;

    if (verbose) {
      std::cerr << "=== VectorizePlanner: Collected buffer vector sizes ==="
                << "\n";
      std::cerr << "  initial_vector_size=" << initial_vector_size_
                << ", loop_extent_vector_size=" << loop_extent_vector_size_
                << ", has_seq_stmt=" << (has_seq_stmt ? "true" : "false")
                << "\n";
    }

    // Separate buffers into local/fragment vs memory (global/shared) vs
    // call/cast
    int local_fragment_min = initial_vector_size_;
    int memory_min = initial_vector_size_;
    int call_node_min = initial_vector_size_;
    bool has_global_or_shared_buffer = false;

    auto is_local_or_fragment = [](const Buffer &buf) {
      return IsLocalBuffer(buf, /*allow_var=*/true) || IsFragmentBuffer(buf);
    };

    std::vector<BufferVectorInfo> local_fragment_buffers;

    for (const auto &info : buffer_vector_infos_) {
      auto buffer = info.buffer;
      if (verbose) {
        if (buffer.defined()) {
          std::cerr << "  Buffer: " << buffer->name
                    << " (scope=" << buffer.scope() << ")"
                    << " -> vector_size=" << info.vector_size
                    << (info.is_store ? " [store]" : " [load]") << "\n";
        } else {
          std::cerr << "  [cast/call] -> vector_size=" << info.vector_size
                    << "\n";
        }
      }
      if (!buffer.defined()) {
        // CastNode, CallNode do not have buffer defined.
        call_node_min = arith::ZeroAwareGCD(call_node_min, info.vector_size);
      } else if (is_local_or_fragment(buffer)) {
        local_fragment_min =
            arith::ZeroAwareGCD(local_fragment_min, info.vector_size);
        local_fragment_buffers.push_back(info);
      } else {
        // global, shared, shared.dyn
        memory_min = arith::ZeroAwareGCD(memory_min, info.vector_size);
        has_global_or_shared_buffer = true;
      }
    }

    if (verbose) {
      std::cerr << "  Computed mins: local_fragment_min=" << local_fragment_min
                << ", memory_min=" << memory_min
                << ", call_node_min=" << call_node_min << "\n";
    }

    if (has_seq_stmt) {
      // For body contains SeqStmt (multiple statements).
      // Use conservative strategy: take GCD of all buffers including local.
      // The special local buffer optimization only applies to simple single
      // BufferStore cases where we can be confident about the access pattern.
      vector_size_ = arith::ZeroAwareGCD(
          arith::ZeroAwareGCD(local_fragment_min, memory_min), call_node_min);
      if (verbose) {
        std::cerr << "  [Strategy] Has SeqStmt, using conservative GCD of all"
                  << " -> vector_size=" << vector_size_ << "\n";
      }
    } else if (has_global_or_shared_buffer) {
      // Has memory buffers and simple case (no SeqStmt):
      // ignore local/fragment constraints
      vector_size_ = arith::ZeroAwareGCD(memory_min, call_node_min);
      if (verbose) {
        std::cerr << "  [Strategy] Has memory buffers (simple case), using "
                     "memory_min="
                  << memory_min
                  << " (ignoring local/fragment_min=" << local_fragment_min
                  << ")" << "\n";
      }
      // vector_size may be greater than local/fragment buffers' vector_size.
      // In such case, we need to re-validate if the indices are vectorizable
      // at the new vector_size boundary. If not, take GCD.
      for (const auto &info : local_fragment_buffers) {
        if (vector_size_ > info.vector_size && !info.indices.empty()) {
          // Compute elem_offset from indices and strides
          Array<PrimExpr> strides = GetBufferStrides(info.buffer);
          PrimExpr elem_offset = 0;
          for (size_t i = 0; i < info.indices.size(); ++i) {
            elem_offset += info.indices[i] * strides[i];
          }
          if (!IndicesCanVectorize(elem_offset, inner_for_->loop_var,
                                   inner_for_->extent, vector_size_,
                                   analyzer_)) {
            // Not invariant at this vector_size, need to take GCD
            int old_vector_size = vector_size_;
            vector_size_ = arith::ZeroAwareGCD(vector_size_, info.vector_size);
            if (verbose) {
              std::cerr << "  [Re-validate] Local buffer '" << info.buffer->name
                        << "' not invariant at vector_size=" << old_vector_size
                        << ", GCD with " << info.vector_size
                        << " -> vector_size=" << vector_size_ << "\n";
            }
          }
        }
      }
    } else {
      // Only local/fragment buffers: use GCD of local_fragment_min and
      // call_node_min
      vector_size_ = arith::ZeroAwareGCD(local_fragment_min, call_node_min);
      if (verbose) {
        std::cerr << "  [Strategy] Only local/fragment buffers, using "
                     "GCD(local_fragment_min, call_node_min)="
                  << vector_size_ << "\n";
      }
    }

    // GCD with loop extent to ensure vector_size divides the loop extent
    vector_size_ = arith::ZeroAwareGCD(loop_extent_vector_size_, vector_size_);

    if (verbose) {
      std::cerr << "=== Final vector_size: " << vector_size_ << " ===" << "\n";
    }
    return vector_size_;
  }

private:
  Stmt VisitStmt_(const ForNode *node) final {
    inner_for_ = node;
    bool contains_nested_for = false;
    // Must analysis vectorization on the innermost loop
    PostOrderVisit(Downcast<Stmt>(node->body), [&](const ObjectRef &obj) {
      if (obj.as<ForNode>()) {
        contains_nested_for = true;
      }
    });

    if (!contains_nested_for) {
      auto extent_ptr = as_const_int(analyzer_->Simplify(node->extent));
      // Here I disable dynamic shape completely,
      //   In order to do it, the Planner should accept an analyzer with
      //   arithmetic info outside to prove the dividiblity of vector size
      // Note(lei): This is somehow make sense because we should assume the
      // tiling size is always static.
      if (!extent_ptr) {
        loop_extent_vector_size_ = 1;
        return ffi::GetRef<Stmt>(node);
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

  PrimExpr VisitExpr_(const CallNode *node) final {
    if (node->op == builtin::if_then_else()) {
      CheckConditionVectorized(node->args[0]);
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    } else if (node->op.same_as(builtin::tvm_access_ptr())) {
      HandleTvmAccessPtr(node);
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    } else if (node->op == tl::atomic_add_elem_op()) {
      // Assert at least 2 args (dst_ptr and src)
      ICHECK(node->args.size() >= 2)
          << "atomic_add_elem_op requires at least 2 args (dst and src)";

      // Get dst dtype from args[0] (tvm_access_ptr or address_of(BufferLoad))
      const CallNode *dst_ptr_call = node->args[0].as<CallNode>();
      ICHECK(dst_ptr_call) << "atomic_add_elem_op first arg must be a call";

      DataType dtype;
      if (dst_ptr_call->op.same_as(builtin::address_of())) {
        auto buffer_load = dst_ptr_call->args[0].as<BufferLoadNode>();
        ICHECK(buffer_load) << "address_of arg must be BufferLoad";
        dtype = buffer_load->buffer->dtype;
      } else if (dst_ptr_call->op.same_as(builtin::tvm_access_ptr())) {
        ICHECK(!dst_ptr_call->args.empty());
        dtype = dst_ptr_call->args[0].dtype();
      } else if (dst_ptr_call->op.same_as(tl::access_ptr())) {
        ICHECK_EQ(dst_ptr_call->args.size(), 3U)
            << "tl.access_ptr expects 3 args: (BufferLoad, extent, rw_mask)";
        auto buffer_load = dst_ptr_call->args[0].as<BufferLoadNode>();
        ICHECK(buffer_load) << "tl.access_ptr arg0 must be BufferLoad";
        dtype = buffer_load->buffer->dtype;
      } else {
        LOG(FATAL) << "atomic_add_elem_op first arg must be tvm_access_ptr, "
                      "tl.access_ptr, or address_of call, but got "
                   << node->args[0];
      }
      int vectorize_length = 1;
      if (dtype.is_float16() || dtype.is_bfloat16()) {
        vectorize_length = 2;
      } else if (dtype.is_float() && dtype.bits() == 32 &&
                 TargetHasSMVersionGE(Target::Current(false), 90)) {
        vectorize_length = 4;
      }

      buffer_vector_infos_.push_back({Buffer(), vectorize_length, false, {}});
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    } else if (node->op == builtin::address_of() ||
               node->op == tl::access_ptr()) {
      // address_of and tl.access_ptr have buffer load value so we should
      // analysis the buffer load node to update vector_size_.
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
      if (!inner_for_)
        return true;
      bool all_invariant = true;
      PostOrderVisit(ffi::GetRef<PrimExpr>(node), [&](const ObjectRef &obj) {
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
          if (!IsExprInvariantInVectorBoundary(elem_offset,
                                               inner_for_->loop_var,
                                               target_vec_size, analyzer_)) {
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
          if (!IsExprInvariantInVectorBoundary(elem_offset,
                                               inner_for_->loop_var,
                                               target_vec_size, analyzer_)) {
            all_invariant = false;
          }
        } else if (auto *call = obj.as<CallNode>()) {
          // tvm_access_ptr(dtype_annotation, data, offset, extent, rw_mask)
          // The offset (args[2]) is the element offset into the buffer.
          if (call->op.same_as(builtin::tvm_access_ptr()) &&
              call->args.size() >= 3) {
            PrimExpr offset = call->args[2];
            if (!IsExprInvariantInVectorBoundary(offset, inner_for_->loop_var,
                                                 target_vec_size, analyzer_)) {
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

  void CheckConditionVectorized(const PrimExpr &cond) {
    // TODO: perform some checks here
  }

  void HandleTvmAccessPtr(const CallNode *node) {
    // tvm_access_ptr format: (ptype, data, offset, extent, rw_mask)
    if (!inner_for_) {
      return;
    }
    ICHECK(node->args.size() >= 3U)
        << "tvm_access_ptr requires at least 3 args";

    // args[0] is TypeAnnotation(dtype[/lanes]); dtype() encodes the element
    // type. See tvm::tir::Buffer::access_ptr implementation.
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
    PrimExpr offset_s = analyzer_->Simplify(offset);
    while (access_vec_size > 1 &&
           !IndicesCanVectorize(offset_s, inner_for_->loop_var,
                                inner_for_->extent, access_vec_size,
                                analyzer_)) {
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
    int cast_vector_size = arith::ZeroAwareGCD(
        vector_load_bits_max_ / node->dtype.bits(), initial_vector_size_);
    // Record cast constraint (use empty buffer to indicate cast)
    buffer_vector_infos_.push_back({Buffer(), cast_vector_size, false, {}});
    return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
  }

  int ComputeBufferVectorSize(const Array<PrimExpr> &indices,
                              const Buffer &buffer, bool is_store) {
    if (!inner_for_)
      return initial_vector_size_;

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
          elem_offset, inner_for_->loop_var, try_vec_size, analyzer_);
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
    // For BufferStore, if indices is invariant or independent with loop_var,
    // we should not vectorize it (broadcasting store is not supported).
    if (is_store && (is_invariant || is_independent)) {
      return 1;
    }
    if (is_independent) {
      return buffer_vec_size; // only limited constraint from this buffer
    }
    // 4. Try to find max vectorize size for this buffer
    while (buffer_vec_size > 1 &&
           !IndicesCanVectorize(elem_offset, inner_for_->loop_var,
                                inner_for_->extent, buffer_vec_size,
                                analyzer_)) {
      buffer_vec_size /= 2;
    }
    return buffer_vec_size;
  }

  void UpdateVectorSize(const Array<PrimExpr> &indices, const Buffer &buffer,
                        bool is_store) {
    int buffer_vec_size = ComputeBufferVectorSize(indices, buffer, is_store);
    buffer_vector_infos_.push_back(
        {buffer, buffer_vec_size, is_store, indices});
  }

  // NOTE(wt): The base class IRMutatorWithAnalyzer::VisitStmt_(LetStmtNode*)
  // binds let variables, but this causes issues when the same variable name
  // appears multiple times with different values (e.g., in pipelined loops
  // where the body is duplicated). For this case, we allow the analyzer to
  // override the binding. Check the impl of
  // IRMutatorWithAnalyzer::VisitStmt_(LetStmtNode*) in:
  // tvm/src/arith/ir_mutator_with_analyzer.cc
  Stmt VisitStmt_(const LetStmtNode *op) final {
    PrimExpr value = this->VisitExpr(op->value);
    if (SideEffect(value) <= CallEffectKind::kPure) {
      // Allow override to handle duplicated loop bodies in pipelined loops
      analyzer_->Bind(op->var, value, /*allow_override=*/true);
    }
    // Continue visiting the body to collect vectorization info
    Stmt body = this->VisitStmt(op->body);
    if (value.same_as(op->value) && body.same_as(op->body)) {
      return ffi::GetRef<Stmt>(op);
    } else {
      auto n = this->CopyOnWrite(op);
      n->value = std::move(value);
      n->body = std::move(body);
      return Stmt(n);
    }
  }

  int vector_load_bits_max_;
  int initial_vector_size_ = 128;
  int loop_extent_vector_size_ = 128;

  const ForNode *inner_for_{};
  bool has_nonlocal_memory_access_ = false;
  int vector_size_ = 128;
  std::vector<BufferVectorInfo> buffer_vector_infos_;
  LayoutMap layout_map_;
};

class VectorizeRewriter : public StmtExprMutator {
public:
  VectorizeRewriter(int vector_size) : vector_size_(vector_size) {}

private:
  Stmt VisitStmt_(const ForNode *node) final {
    inner_for_ = node;
    auto ret = StmtExprMutator::VisitStmt_(node);
    if (inner_for_ == node) { // rewrite the innermost loop
      For fnode = ret.as<For>().value();
      auto old_var = fnode->loop_var;
      auto extent_ptr = as_const_int(fnode->extent);
      ICHECK(extent_ptr) << fnode->extent;
      int extent = *extent_ptr;
      ICHECK(extent % vector_size_ == 0)
          << "extent: " << extent << " vector_size_: " << vector_size_
          << " for loop: " << fnode;
      ICHECK(is_zero(fnode->min));
      if (extent == vector_size_) {
        fnode.CopyOnWrite()->kind = ForKind::kVectorized;
        return fnode;
      } else {
        Var inner_var = Var("vec");
        Var outer_var = Var(old_var->name_hint);
        Map<Var, PrimExpr> vmap;
        vmap.Set(fnode->loop_var, outer_var * vector_size_ + inner_var);
        Stmt body = Substitute(fnode->body, vmap);
        body = For(inner_var, 0, vector_size_, ForKind::kVectorized, body);
        // TileLang uses ForKind::kParallel in frontend SIMT loops. After
        // vectorization, keep semantics equivalent but downgrade to serial so
        // subsequent passes (e.g. pragma-unroll) can run.
        ForKind outer_kind = fnode->kind;
        if (outer_kind == ForKind::kParallel) {
          outer_kind = ForKind::kSerial;
        }
        body = For(outer_var, 0, extent / vector_size_, outer_kind, body,
                   fnode->thread_binding, fnode->annotations, fnode->step,
                   fnode->span);
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

  const ForNode *inner_for_{};
  const int vector_size_;
};

int GetVectorizeSize(const For &loop, const LayoutMap &layout_map) {
  arith::Analyzer analyzer;
  return VectorizePlanner(&analyzer, layout_map).Plan(loop);
}

int GetVectorizeSize(const For &loop, arith::Analyzer *analyzer,
                     const LayoutMap &layout_map) {
  return VectorizePlanner(analyzer, layout_map).Plan(loop);
}

bool CanProveIndependent(const PrimExpr &expr, Var var,
                         arith::Analyzer *analyzer) {
  // 1. if var doesn't exist, it is independent
  bool used_var = UsesVar(expr, [&](const VarNode *v) {
    return tvm::ffi::GetRef<Var>(v).same_as(var);
  });
  if (!used_var) {
    return true;
  }
  // 2. if \forall v_1, v_2, f(v_1) == f(v_2), f is independent with v
  Var var_1("_t", var.dtype());
  auto expr_1 = Substitute(expr, {{var, var_1}});
  if (analyzer->CanProveEqual(expr, expr_1)) {
    return true;
  }
  return false;
}

bool IsExprInvariantInVectorBoundary(const PrimExpr &expr, Var var,
                                     int target_vectorized_size,
                                     arith::Analyzer *analyzer) {
  // Check if expr is invariant within vector boundaries
  // We're trying to prove the access expression A[f(var)] depends only on
  // floor(var/vecsize), not on var%vecsize
  // Mathematically:
  // \forall var, f(floor(var/vecsize)*vecsize + var%vecsize) ==
  // f(floor(var/vecsize)*vecsize + 0)
  // Example: for i in T.vectorized(8):
  //     A[i] = B[i] * C[i//4]
  // if vecsize=4, f(i)=i//4 depends only on i//4
  // Therefore A[i] = B[i] * C[i//4] can be vectorized with vecsize=4
  PrimExpr var_aligned =
      floordiv(var, target_vectorized_size) * target_vectorized_size;
  PrimExpr expr_aligned = Substitute(expr, {{var, var_aligned}});
  if (analyzer->CanProveEqual(expr, expr_aligned)) {
    return true;
  }
  return false;
}

bool IndicesCanVectorize(const PrimExpr &expr, Var var,
                         const PrimExpr &iter_var_size,
                         int target_vectorized_size,
                         arith::Analyzer *analyzer) {
  ICHECK(target_vectorized_size >= 1);
  if (target_vectorized_size == 1)
    return true;

  // Extent must be divisible
  PrimExpr target_size_for_iter =
      make_const(iter_var_size.dtype(), target_vectorized_size);
  PrimExpr target_size_for_expr =
      make_const(expr.dtype(), target_vectorized_size);
  PrimExpr target_size_for_var =
      make_const(var.dtype(), target_vectorized_size);
  PrimExpr zero = make_const(var.dtype(), 0);

  if (!analyzer->CanProveEqual(FloorMod(iter_var_size, target_size_for_iter),
                               0))
    return false;

  if (IsExprInvariantInVectorBoundary(expr, var, target_vectorized_size,
                                      analyzer)) {
    return true;
  }

  auto simplified_expr = analyzer->Simplify(Substitute(expr, {{var, zero}}));
  // The base offset must be divisible
  if (!analyzer->CanProveEqual(FloorMod(simplified_expr, target_size_for_expr),
                               zero)) {
    return false;
  }

  // Bind thread range
  Var v0("v0", var.dtype()), v1("v1", var.dtype());
  analyzer->Bind(v0, Range(zero, target_size_for_var));
  analyzer->Bind(v1, Range(zero, analyzer->Simplify(FloorDiv(
                                     iter_var_size, target_size_for_iter))));
  PrimExpr expr_transformed = analyzer->Simplify(
      Substitute(expr, {{var, v0 + v1 * target_size_for_var}}));
  Vectorizer vectorizer(v0, target_size_for_var);
  PrimExpr expr_vectorized = vectorizer.VisitExpr(expr_transformed);

  // This simplify is necessary for thread region specified
  // optimizations.
  expr_vectorized = analyzer->Simplify(expr_vectorized);
  auto ramp_node = expr_vectorized.as<RampNode>();
  if (!ramp_node) {
    // Broadcast value
    if (expr_vectorized.dtype().lanes() == 1)
      return true;
    else
      return false;
  } else {
    return is_one(ramp_node->stride);
  }
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
  if (vectorize_hint <= 0) {
    arith::Analyzer analyzer;
    VectorizePlanner planner(&analyzer, layout_map);
    vectorize_hint = planner.Plan(loop);
  }
  if (vectorize_hint == 1)
    return ParallelToSerial(loop);
  auto rewriter = VectorizeRewriter(vectorize_hint);
  return Downcast<For>(rewriter(loop));
}

For VectorizeLoop(const For &loop, arith::Analyzer *analyzer,
                  const LayoutMap &layout_map, int vectorize_hint) {
  if (vectorize_hint <= 0) {
    VectorizePlanner planner(analyzer, layout_map);
    vectorize_hint = planner.Plan(loop);
  }
  if (vectorize_hint == 1)
    return ParallelToSerial(loop);
  auto rewriter = VectorizeRewriter(vectorize_hint);
  return Downcast<For>(rewriter(loop));
}

} // namespace tl
} // namespace tvm

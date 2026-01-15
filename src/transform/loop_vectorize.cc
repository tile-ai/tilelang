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
#include "../op/builtin.h"
#include "../op/utils.h"
#include "../target/utils.h"
#include "arith/int_operator.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "common/loop_vectorization_utils.h"
#include "tvm/tir/analysis.h"
#include "tvm/tir/var.h"
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

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

class VectorizeFindGlobalAccess : public StmtExprVisitor {
public:
  VectorizeFindGlobalAccess() = default;

  bool HasGlobalAccess(const Stmt &stmt) {
    this->operator()(stmt);
    return has_global_access_;
  }

private:
  bool has_global_access_ = false;

  void VisitStmt_(const BufferStoreNode *node) final {
    if (node->buffer.scope() == "global")
      has_global_access_ = true;
    return StmtExprVisitor::VisitStmt_(node);
  }

  void VisitExpr_(const BufferLoadNode *node) final {
    if (node->buffer.scope() == "global")
      has_global_access_ = true;
    return StmtExprVisitor::VisitExpr_(node);
  }
};

class VectorizePlanner : public arith::IRMutatorWithAnalyzer {
public:
  explicit VectorizePlanner(arith::Analyzer *analyzer,
                            const LayoutMap &layout_map = {})
      : arith::IRMutatorWithAnalyzer(analyzer), layout_map_(layout_map) {}

  int Plan(const For &node) {
    tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
    Optional<Bool> opt_disable_vectorize_256 =
        ctxt->GetConfig(kDisableVectorize256, Optional<Bool>());
    bool disable_vectorize_256 =
        opt_disable_vectorize_256.value_or(Bool(false));
    if (tvm::tl::TargetIsSm100(Target::Current(false)) &&
        !disable_vectorize_256 &&
        VectorizeFindGlobalAccess().HasGlobalAccess(node)) {
      vector_load_bits_max_ = vector_size_ = 256;
    } else {
      vector_load_bits_max_ = vector_size_ = 128;
    }
    this->operator()(node);
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
      if (!extent_ptr) {
        vector_size_ = 1;
        return ffi::GetRef<Stmt>(node);
      }
      vector_size_ = arith::ZeroAwareGCD(vector_size_, *extent_ptr);
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
    } else if (node->op == tl::atomic_add_elem_op()) {
      // Assert at least 2 args (dst_ptr and src)
      ICHECK(node->args.size() >= 2)
          << "atomic_add_elem_op requires at least 2 args (dst and src)";

      // Get dst dtype from args[0] (address_of call containing BufferLoad)
      auto address_of_call = node->args[0].as<CallNode>();
      ICHECK(address_of_call && address_of_call->op == builtin::address_of())
          << "atomic_add_elem_op first arg must be address_of call";

      auto buffer_load = address_of_call->args[0].as<BufferLoadNode>();
      ICHECK(buffer_load) << "address_of arg must be BufferLoad";

      DataType dtype = buffer_load->buffer->dtype;
      int vectorize_length = 1;
      if (dtype.is_float16() || dtype.is_bfloat16()) {
        vectorize_length = 2;
      } else if (dtype.is_float() && dtype.bits() == 32 &&
                 TargetHasSMVersionGE(Target::Current(false), 90)) {
        vectorize_length = 4;
      }

      vector_size_ = arith::ZeroAwareGCD(vector_size_, vectorize_length);
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    } else if (node->op == builtin::address_of()) {
      // address_of have buffer load value so we should analysis the buffer load
      // node to update vector_size_.
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    } else if (node->op.same_as(tir::builtin::bitwise_and()) ||
               node->op.same_as(tir::builtin::bitwise_or()) ||
               node->op.same_as(tir::builtin::bitwise_xor()) ||
               node->op.same_as(tir::builtin::bitwise_not()) ||
               node->op.same_as(tir::builtin::shift_left()) ||
               node->op.same_as(tir::builtin::shift_right())) {
      // Bitwise operations can be vectorized
      return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
    } else {
      // Other calls should not be vectorized
      vector_size_ = 1;
      return ffi::GetRef<PrimExpr>(node);
    }
    return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
  }

  void CheckConditionVectorized(const PrimExpr &cond) {
    // TODO: perform some checks here
  }

  PrimExpr VisitExpr_(const CastNode *node) final {
    vector_size_ = arith::ZeroAwareGCD(
        vector_load_bits_max_ / node->dtype.bits(), vector_size_);
    return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
  }

  void UpdateVectorSize(const Array<PrimExpr> indices, const Buffer &buffer,
                        bool is_store) {
    if (!inner_for_)
      return;
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

    // 1. Compute raw element offset
    auto strides = buffer->strides;
    if (buffer->strides.empty()) {
      PrimExpr stride = 1;
      for (int i = transformed_indices.size() - 1; i >= 0; --i) {
        strides.push_back(stride);
        stride = stride * buffer->shape[i];
      }
      strides = Array<PrimExpr>{strides.rbegin(), strides.rend()};
    }
    PrimExpr elem_offset = 0;
    for (int i = 0; i < transformed_indices.size(); ++i) {
      elem_offset += transformed_indices[i] * strides[i];
    }
    // 2. If element offset is independent with loop_var, ignore it.
    if (CanProveIndependent(elem_offset, inner_for_->loop_var, analyzer_)) {
      // Specially, if it's a BufferStore, we should not vectorize it.
      if (is_store) {
        vector_size_ = 1;
      }
      return;
    }
    // 3. Check if current vector_size_ works with invariant boundary check
    if (!IsExprInvariantInVectorBoundary(elem_offset, inner_for_->loop_var,
                                         vector_size_, analyzer_)) {
      // If not, tight vectorize bound with buffer dtype constraint
      vector_size_ = arith::ZeroAwareGCD(
          vector_size_, vector_load_bits_max_ /
                            (buffer->dtype.bits() * buffer->dtype.lanes()));
    } else if (is_store) {
      // If the indices is invariant for BufferStore, we should also not
      // vectorize it.
      vector_size_ = 1;
    }
    // 4. Try to vectorize buffer load
    while (!IndiceCanVectorize(elem_offset, inner_for_->loop_var,
                               inner_for_->extent, vector_size_, analyzer_)) {
      vector_size_ /= 2;
    }
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

  const ForNode *inner_for_{};
  bool has_nonlocal_memory_access_ = false;
  int vector_size_ = 128;
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
          << "extent: " << extent << " vector_size_: " << vector_size_;
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
        body = For(outer_var, 0, extent / vector_size_, fnode->kind, body,
                   fnode->thread_binding, fnode->annotations, fnode->step,
                   fnode->span);
        return body;
      }
    } else {
      return ret;
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

bool IndiceCanVectorize(const PrimExpr &expr, Var var,
                        const PrimExpr &iter_var_size,
                        int target_vectorized_size, arith::Analyzer *analyzer) {
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

For VectorizeLoop(const For &loop, const LayoutMap &layout_map,
                  int vectorize_hint) {
  if (vectorize_hint <= 0) {
    arith::Analyzer analyzer;
    VectorizePlanner planner(&analyzer, layout_map);
    vectorize_hint = planner.Plan(loop);
  }
  if (vectorize_hint == 1)
    return loop;
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
    return loop;
  auto rewriter = VectorizeRewriter(vectorize_hint);
  return Downcast<For>(rewriter(loop));
}

} // namespace tl
} // namespace tvm

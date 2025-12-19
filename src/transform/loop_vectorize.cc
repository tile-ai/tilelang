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
  explicit VectorizePlanner(arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer) {}

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
    if (node->buffer.scope() == "shared" || node->buffer.scope() == "global" ||
        node->buffer.scope() == "shared.dyn")
      has_nonlocal_memory_access_ = true;
    if (node->buffer->shape.size() == 1) {
      // TODO(lei): This should be improved as
      // constant buffer that tl hack to use as local register.
      auto boundary_check = node->buffer->shape[0].as<IntImmNode>();
      if (boundary_check && boundary_check->value == 1) {
        return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
      }
    }
    UpdateVectorSize(node->indices, node->buffer);
    return arith::IRMutatorWithAnalyzer::VisitExpr_(node);
  }

  Stmt VisitStmt_(const BufferStoreNode *node) final {
    if (node->buffer.scope() == "shared" || node->buffer.scope() == "global" ||
        node->buffer.scope() == "shared.dyn")
      has_nonlocal_memory_access_ = true;
    UpdateVectorSize(node->indices, node->buffer);
    return arith::IRMutatorWithAnalyzer::VisitStmt_(node);
  }

  Stmt VisitStmt_(const IfThenElseNode *node) final {
    CheckConditionVectorized(node->condition);
    return arith::IRMutatorWithAnalyzer::VisitStmt_(node);
  }

  PrimExpr VisitExpr_(const CallNode *node) final {
    if (node->op == builtin::if_then_else()) {
      CheckConditionVectorized(node->args[0]);
    } else if (node->op == builtin::call_extern()) {
      // do not vectorize extern calls
      vector_size_ = 1;
    } else if (node->op.same_as(tl::rng_rand()) ||
               node->op.same_as(tl::rng_init())) {
      // do not vectorize random operation
      vector_size_ = 1;
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

  void UpdateVectorSize(const Array<PrimExpr> indices, const Buffer &buffer) {
    if (!inner_for_)
      return;
    // 1. Compute raw element offset
    auto strides = buffer->strides;
    if (buffer->strides.empty()) {
      PrimExpr stride = 1;
      for (int i = indices.size() - 1; i >= 0; --i) {
        strides.push_back(stride);
        stride = stride * buffer->shape[i];
      }
      strides = Array<PrimExpr>{strides.rbegin(), strides.rend()};
    }
    PrimExpr elem_offset = 0;
    for (int i = 0; i < indices.size(); ++i) {
      elem_offset += indices[i] * strides[i];
    }
    // 2. If element offset is independent with loop_var, ignore it
    if (CanProveIndependent(elem_offset, inner_for_->loop_var, analyzer_)) {
      return;
    }
    // 3. Check if current vector_size_ works with invariant boundary check
    if (!IsExprInvariantInVectorBoundary(elem_offset, inner_for_->loop_var,
                                         vector_size_, analyzer_)) {
      // If not, tight vectorize bound with buffer dtype constraint
      vector_size_ = arith::ZeroAwareGCD(
          vector_size_, vector_load_bits_max_ /
                            (buffer->dtype.bits() * buffer->dtype.lanes()));
    }
    // 4. Try to vectorize buffer load
    while (!IndiceCanVectorize(elem_offset, inner_for_->loop_var,
                               inner_for_->extent, vector_size_, analyzer_)) {
      vector_size_ /= 2;
    }
  }

  int vector_load_bits_max_;

  const ForNode *inner_for_{};
  bool has_nonlocal_memory_access_ = false;
  int vector_size_ = 128;
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

int GetVectorizeSize(const For &loop) {
  arith::Analyzer analyzer;
  return VectorizePlanner(&analyzer).Plan(loop);
}

int GetVectorizeSize(const For &loop, arith::Analyzer *analyzer) {
  return VectorizePlanner(analyzer).Plan(loop);
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

For VectorizeLoop(const For &loop, int vectorize_hint) {
  if (vectorize_hint <= 0) {
    arith::Analyzer analyzer;
    VectorizePlanner planner(&analyzer);
    vectorize_hint = planner.Plan(loop);
  }
  if (vectorize_hint == 1)
    return loop;
  auto rewriter = VectorizeRewriter(vectorize_hint);
  return Downcast<For>(rewriter(loop));
}

For VectorizeLoop(const For &loop, arith::Analyzer *analyzer,
                  int vectorize_hint) {
  if (vectorize_hint <= 0) {
    VectorizePlanner planner(analyzer);
    vectorize_hint = planner.Plan(loop);
  }
  if (vectorize_hint == 1)
    return loop;
  auto rewriter = VectorizeRewriter(vectorize_hint);
  return Downcast<For>(rewriter(loop));
}

} // namespace tl
} // namespace tvm

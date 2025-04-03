// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file loop_vectorize_dynamic.cc
 * \brief A tool to automatically vectorize a for loop with dynamic shape
 * \brief Reference to loop_vectorize.cc and vectorize_loop.cc
 */

#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>

#include <numeric>

#include "../layout/layout.h"
#include "../layout/utils.h"
#include "arith/int_operator.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "common/loop_vectorization_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRMutatorWithAnalyzer;

struct VectorizePlanResult {
  int vector_size;
  bool dynamic;
  PrimExpr condition;
};

bool IndiceCanVectorizeDynamic(PrimExpr expr, Var var, PrimExpr iter_var_size,
                               int target_vectorized_size,
                               arith::Analyzer *analyzer) {
  ICHECK(target_vectorized_size >= 1);
  if (target_vectorized_size == 1)
    return true;
  if (!analyzer->CanProveEqual(FloorMod(iter_var_size, target_vectorized_size),
                               0))
    return false;
  Var v0("v0"), v1("v1");
  analyzer->Bind(v0, Range(0, target_vectorized_size));
  analyzer->Bind(v1, Range(0, FloorDiv(iter_var_size, target_vectorized_size)));
  PrimExpr expr_transformed = analyzer->Simplify(
      Substitute(expr, {{var, v0 + v1 * target_vectorized_size}}));

  Vectorizer vectorizer(v0, IntImm(v0->dtype, target_vectorized_size));
  PrimExpr expr_vectorized = vectorizer.VisitExpr(expr_transformed);
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

class VectorizePlannerDynamic : public arith::IRVisitorWithAnalyzer {
public:
  VectorizePlannerDynamic() = default;

  int Plan(const For &node) {
    this->operator()(node);
    // Always Enable vectorization
    // if (!has_nonlocal_memory_access_) return 1;
    return vector_size_;
  }

  bool GetDynamic() { return dynamic_; }

  PrimExpr GetCondition() { return condition_; }

private:
  void VisitStmt_(const ForNode *node) final {
    inner_for_ = node;
    iter_map_.Set(node->loop_var, Range(node->min, node->extent));
    arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitExpr_(const BufferLoadNode *node) final {
    if (node->buffer.scope() == "shared" || node->buffer.scope() == "global" ||
        node->buffer.scope() == "shared.dyn")
      has_nonlocal_memory_access_ = true;
    if (node->buffer->shape.size() == 1) {
      // TODO(lei): This should be improved as
      // constant buffer that tl hack to use as local register.
      auto boundary_check = node->buffer->shape[0].as<IntImmNode>();
      if (boundary_check && boundary_check->value == 1) {
        return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
      }
    }
    UpdateVectorSize(node->indices, node->buffer);
    return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
  }

  void VisitStmt_(const BufferStoreNode *node) final {
    if (node->buffer.scope() == "shared" || node->buffer.scope() == "global" ||
        node->buffer.scope() == "shared.dyn")
      has_nonlocal_memory_access_ = true;
    UpdateVectorSize(node->indices, node->buffer);
    return arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitStmt_(const IfThenElseNode *node) final {
    CheckConditionVectorized(node->condition);
    return arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitExpr_(const CallNode *node) final {
    if (node->op == builtin::if_then_else()) {
      CheckConditionVectorized(node->args[0]);
    } else if (node->op == builtin::call_extern()) {
      // do not vectorize extern calls
      vector_size_ = 1;
    }
    return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
  }

  void CheckConditionVectorized(const PrimExpr &cond) {
    // TODO: may perform some checks here
  }

  void UpdateVectorSize(const Array<PrimExpr> indices, const Buffer &buffer) {
    if (!inner_for_)
      return;
    auto extent_ptr = inner_for_->extent.as<IntImmNode>();
    if (!extent_ptr)
      return;

    const DataType &access_type = buffer->dtype;
    // i // 2, i % 8 can also be vectorized as factor 16
    int max_vector_size = vector_load_bits_max_ / access_type.bits();
    if (access_type.is_e4m3_float8() or access_type.is_e5m2_float8()) {
      max_vector_size = 1; // [temporarily] do not vectorize float8
    }
    // so we should disable this GCD optimization
    max_vector_size = arith::ZeroAwareGCD(max_vector_size, extent_ptr->value);

    auto last_dim = buffer->shape.back();
    auto mod_set = analyzer_.modular_set(last_dim);
    // when dynamic shape like [m, k]: coeff=1, base=0, GCD will block
    // conditionally tail vectorize
    if (buffer->shape.back().as<IntImmNode>()) {
      max_vector_size = arith::ZeroAwareGCD(max_vector_size, mod_set->coeff);

      auto gcd_base = arith::ZeroAwareGCD(max_vector_size, mod_set->base);
      // If gcd_base is equal to the last dimension,
      // we should analyze the second-to-last dimension
      // in relation to the last dimension.
      if (gcd_base < Downcast<IntImm>(last_dim)->value) {
        max_vector_size = gcd_base;
      }

      vector_size_ = arith::ZeroAwareGCD(max_vector_size, vector_size_);

      PrimExpr elem_offset = 0;
      PrimExpr stride = 1;
      for (int i = indices.size() - 1; i >= 0; --i) {
        elem_offset = elem_offset + indices[i] * stride;
        stride = stride * buffer->shape[i];
      }
      while (!IndiceCanVectorizeDynamic(elem_offset, inner_for_->loop_var,
                                        inner_for_->extent, vector_size_,
                                        &analyzer_)) {
        vector_size_ /= 2;
      }
    } else if (vector_size_ <= vector_load_bits_max_ / buffer->dtype.bits()) {
      // dynamic shape load: get the vectorization condition
      dynamic_ = true;
      PrimExpr offset = buffer.OffsetOf(indices).back();
      // condition for alignment, maybe useless
      condition_ = (FloorMod(offset, vector_size_) == 0);
    }
  }

  const int vector_load_bits_max_ = 128;

  const ForNode *inner_for_;
  Map<Var, Range> iter_map_;
  bool has_nonlocal_memory_access_ = false;
  int vector_size_ = 128;
  // conditionally vectorize
  bool dynamic_ = false;
  PrimExpr condition_;
};

class VectorizedBodyMutator : public StmtExprMutator {
public:
  VectorizedBodyMutator(Var inner_var, int vector_size,
                        std::vector<PrimExpr> conditions)
      : inner_var_(inner_var), vector_size_(vector_size),
        conditions_(conditions) {}

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::if_then_else())) {
      // TODO: Currently not ramp, but only reserve the "then" part (because
      // conditions are move outside this vectorized loop)
      PrimExpr ifexpr = op->args[0];
      PrimExpr thenexpr = op->args[1];
      bool flag = false;
      for (auto &cond : conditions_) {
        if (ifexpr.get() == cond.get()) {
          flag = true;
        }
      }
      if (flag) {
        return thenexpr;
      } else {
        return GetRef<PrimExpr>(op);
      }
    } else {
      return GetRef<PrimExpr>(op);
    }
  }

  Var inner_var_;
  int vector_size_;
  std::vector<PrimExpr> conditions_;
};

class VectorizedConditionExtracter : public StmtExprVisitor {
public:
  VectorizedConditionExtracter() = default;
  std::vector<PrimExpr> GetConditions(Stmt body) {
    this->VisitStmt(body);
    return conditions_;
  }

private:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::if_then_else())) {
      PrimExpr cond = op->args[0];
      conditions_.emplace_back(cond);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const IfThenElseNode *node) final {
    conditions_.emplace_back(node->condition);
    StmtExprVisitor::VisitStmt_(node);
  }

  std::vector<PrimExpr> conditions_;
};

class NestedLoopChecker : public StmtExprVisitor {
public:
  NestedLoopChecker() : loop_num_(0) {}
  int GetNestLoopNum(Stmt body) {
    this->VisitStmt(body);
    return loop_num_;
  }

private:
  void VisitStmt_(const ForNode *node) final {
    loop_num_++;
    StmtExprVisitor::VisitStmt_(node);
  }
  int loop_num_;
};

class VectorizeRewriterDynamic : public StmtExprMutator {
public:
  VectorizeRewriterDynamic(VectorizePlanResult plan)
      : vector_size_(plan.vector_size), condition_(plan.condition),
        dynamic_(plan.dynamic) {}

private:
  Stmt VisitStmt_(const ForNode *node) final {
    inner_for_ = node;
    auto ret = StmtExprMutator::VisitStmt_(node);
    if (inner_for_ != node) {
      return ret;
    }
    For fnode = ret.as<For>().value();
    auto old_var = fnode->loop_var;
    if (!fnode->extent.as<IntImmNode>()) {
      return ret;
    }
    int extent = Downcast<IntImm>(fnode->extent)->value;

    if (!dynamic_) {
      return fnode;
    }
    ICHECK(extent % vector_size_ == 0)
        << "extent: " << extent << " vector_size_: " << vector_size_;
    ICHECK(is_zero(fnode->min));
    Var inner_var = Var("vec");
    Var outer_var = Var(old_var->name_hint);
    Map<Var, PrimExpr> vmap;
    vmap.Set(fnode->loop_var, outer_var * vector_size_ + inner_var);
    Stmt body = Substitute(fnode->body, vmap);

    VectorizedConditionExtracter extracter;
    std::vector<PrimExpr> conditions = extracter.GetConditions(body);

    // Set vectorize variable to the max value of the extent (i.e.
    // vector_size_ - 1)
    PrimExpr condition = conditions[0];
    for (int i = 1; i < conditions.size(); ++i) {
      condition = condition && conditions[i];
    }

    // add condition ifthenelse here
    Map<Var, PrimExpr> vmap_condition;
    vmap_condition.Set(inner_var, vector_size_ - 1);
    PrimExpr condition_bound = Substitute(condition, vmap_condition);

    // modify body in the vectorized loop
    VectorizedBodyMutator mutator(inner_var, vector_size_, conditions);
    Stmt vectorize_body = mutator(body);

    For vectorize_for =
        For(inner_var, 0, vector_size_, ForKind::kVectorized, vectorize_body);
    For serial_for = For(inner_var, 0, vector_size_, ForKind::kSerial, body);
    body = IfThenElse(condition_bound, vectorize_for, serial_for);
    body = For(outer_var, 0, extent / vector_size_, fnode->kind, body,
               fnode->thread_binding, fnode->annotations, fnode->span);
    return body;
  }

  const ForNode *inner_for_;
  const int vector_size_;
  const PrimExpr condition_;
  const bool dynamic_;
};

VectorizePlanResult GetVectorizePlanResultDynamic(const For &loop) {
  VectorizePlannerDynamic planner;
  int vector_size = planner.Plan(loop);
  bool dynamic = planner.GetDynamic();
  PrimExpr condition = planner.GetCondition();
  return {vector_size, dynamic, condition};
}

class LoopVectorizerDynamic : public IRMutatorWithAnalyzer {
public:
  static Stmt Substitute(Stmt stmt) {
    arith::Analyzer analyzer;
    LoopVectorizerDynamic substituter(&analyzer);
    stmt = substituter.VisitStmt(stmt);
    return stmt;
  }

private:
  LoopVectorizerDynamic(arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer) {}

  Stmt VisitStmt_(const ForNode *op) final {
    For for_node = Downcast<For>(IRMutatorWithAnalyzer::VisitStmt_(op));
    VectorizePlanResult res{128, false, 0};
    res = GetVectorizePlanResultDynamic(for_node);
    NestedLoopChecker checker;
    int nest_num = checker.GetNestLoopNum(for_node);
    if (nest_num > 1) { // only rewrite the innermost loop
      return for_node;
    }
    int vectorize_hint = res.vector_size;
    auto rewriter = VectorizeRewriterDynamic(res);
    return Downcast<For>(rewriter(for_node));
  }
};

class VectorizeSkipperDynamic : public StmtMutator {
public:
  Stmt VisitStmt_(const ForNode *op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    op = stmt.as<ForNode>();
    if (op->kind == ForKind::kVectorized) {
      return For(op->loop_var, op->min, op->extent, ForKind::kSerial, op->body);
    } else {
      return stmt;
    }
  }
};

tvm::transform::Pass LoopVectorizeDynamic() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto *n = f.CopyOnWrite();
    n->body = tvm::tl::LoopVectorizerDynamic::Substitute(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LoopVectorizeDynamic", {});
}

// Register the pass globally so it can be used in the compilation pipeline
TVM_REGISTER_GLOBAL("tl.transform.LoopVectorizeDynamic")
    .set_body_typed(LoopVectorizeDynamic);

} // namespace tl
} // namespace tvm

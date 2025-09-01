/*!
 * \file atomicadd_vectorize.cc
 * \brief A tool to automatically vectorize atomic add
 */

#include "../layout/layout.h"
#include "../layout/utils.h"
#include "arith/int_operator.h"
#include "arith/ir_visitor_with_analyzer.h"
#include "common/loop_vectorization_utils.h"
#include <numeric>
#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <utility>

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRMutatorWithAnalyzer;
using arith::IRVisitorWithAnalyzer;

struct AtomicAddVectorizePlanResult {
  int vector_size;
  bool dynamic;
  PrimExpr condition;
};

class AtomicAddVectorizePlanner : public arith::IRVisitorWithAnalyzer {
public:
  AtomicAddVectorizePlanner() = default;
  int max_vector_size = 1;
  AtomicAddVectorizePlanResult Plan(const For &node, Var thread_var,
                                    Range thread_bounds, int vectorize_hint) {
    this->max_vector_size = vectorize_hint;
    this->thread_var = std::move(thread_var);
    this->thread_bounds = std::move(thread_bounds);
    this->operator()(node);
    return {vector_size_, dynamic_, condition_};
  }

private:
  void VisitStmt_(const ForNode *node) final {
    inner_for_ = node;
    iter_map_.Set(node->loop_var, Range(node->min, node->extent));

    arith::IRVisitorWithAnalyzer::VisitStmt_(node);
  }

  void VisitExpr_(const CallNode *node) final {
    if (node->op == builtin::call_extern() && node->args.size() >= 2) {
      if (const auto *func_name = node->args[0].as<StringImmNode>()) {
        if (func_name->value == "AtomicAdd") {

          const CallNode *addr_call = node->args[1].as<CallNode>();
          if (addr_call && addr_call->op == builtin::address_of() &&
              addr_call->args.size() == 1) {

            const BufferLoadNode *buffer_load_dst =
                addr_call->args[0].as<BufferLoadNode>();
            const BufferLoadNode *buffer_load_src =
                node->args[2].as<BufferLoadNode>();
            if (buffer_load_src && buffer_load_src->buffer.defined() &&
                buffer_load_dst && buffer_load_dst->buffer.defined()) {

              Buffer dst_buffer = buffer_load_dst->buffer;
              Array<PrimExpr> indices_dst = buffer_load_dst->indices;
              UpdateVectorSize(indices_dst, dst_buffer);
              Buffer src_buffer = buffer_load_src->buffer;
              Array<PrimExpr> indices_src = buffer_load_src->indices;
              UpdateVectorSize(indices_src, src_buffer);
            }
          }
        }
      }
    }
    return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
  }

  void UpdateVectorSize(const Array<PrimExpr> &indices, const Buffer &buffer) {
    if (!inner_for_)
      return;
    auto extent_ptr = inner_for_->extent.as<IntImmNode>();
    if (!extent_ptr)
      return;

    const DataType &access_type = buffer->dtype;
    // i // 2, i % 8 can also be vectorized as factor 16
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
      PrimExpr thread_extent = thread_bounds->extent;
      while (!IndiceCanVectorize(elem_offset, thread_var, thread_extent,
                                 vector_size_, &analyzer_)) {
        vector_size_ /= 2;
      }
    } else if (vector_size_ <= 4) {
      // dynamic shape load: get the vectorization condition
      dynamic_ = true;
      PrimExpr offset = buffer.OffsetOf(indices).back();
      condition_ = (truncmod(offset, vector_size_) == 0);
    }
  }

  const ForNode *inner_for_;
  Map<Var, Range> iter_map_;
  bool has_nonlocal_memory_access_ = false;
  int vector_size_ = 4;
  Var thread_var;
  Range thread_bounds;
  bool dynamic_ = false;
  PrimExpr condition_;
};

class AtomicAddVectorizeRewriter : public StmtExprMutator {
public:
  AtomicAddVectorizeRewriter(const AtomicAddVectorizePlanResult &plan,
                             Var thread_var, PrimExpr by_var, PrimExpr bx_var,
                             const Range &thread_bounds, int stride_y,
                             int stride_x)
      : vector_size_(plan.vector_size), condition_(plan.condition),
        dynamic_(plan.dynamic), tx_var_(std::move(thread_var)),
        by_var_(std::move(by_var)), bx_var_(std::move(bx_var)),
        stride_y_(stride_y), stride_x_(stride_x) {
    const int64_t *tx_ext = as_const_int(thread_bounds->extent);
    ICHECK(tx_ext)
        << "thread_bounds->extent must be a constant for vectorization.";
    extent_tx_ = static_cast<int>(*tx_ext);
  }

private:
  /**
   * @brief Visits a For node and rewrites the innermost loop for atomic-add
   * vectorization.
   *
   * If the visited For node is the recorded innermost loop, this method
   * validates that the loop extent is a constant, divisible by the planned
   * vector size, and has a zero minimum. When vectorization is enabled
   * (dynamic_ == false) it:
   *  - locates the thread index variable named "tx" inside the loop body,
   *  - creates a new outer loop variable named "<old_loop_var>_outer",
   *  - substitutes occurrences of `tx` with `tx * vector_size_` and the old
   * loop var with `outer_var * vector_size_` so each outer iteration maps to a
   * contiguous vector-sized chunk,
   *  - returns a new For with extent divided by vector_size_ and the
   * transformed body.
   *
   * If dynamic_ is true, the method returns the (possibly mutated) inner For
   * unchanged.
   *
   * Side effects:
   *  - updates inner_for_ to point to the current For node during visitation.
   *  - performs runtime checks (ICHECK) to enforce: constant extent, extent %
   * vector_size_ == 0, and zero loop minimum; violations terminate execution.
   *
   * @return The original or transformed For statement as a Stmt.
   */
  Stmt VisitStmt_(const ForNode *node) final {
    inner_for_ = node;
    iter_var_ = Var(node->loop_var->name_hint + "_outer");
    auto ret = StmtExprMutator::VisitStmt_(node);
    if (inner_for_ == node) { // rewrite the innermost loop
      For fnode = ret.as<For>().value();
      auto extent_ptr = as_const_int(fnode->extent);
      ICHECK(extent_ptr) << fnode->extent;
      int extent = *extent_ptr;
      ICHECK(extent % vector_size_ == 0)
          << "extent: " << extent << " vector_size_: " << vector_size_;
      ICHECK(is_zero(fnode->min));
      if (!dynamic_) {
        Map<Var, PrimExpr> vmap;
        vmap.Set(fnode->loop_var, iter_var_);
        Stmt body = Substitute(fnode->body, vmap);
        return For(iter_var_, 0, extent / vector_size_, fnode->kind, body,
                   fnode->thread_binding, fnode->annotations, fnode->span);
      }
    }
    return ret;
  }

  PrimExpr VisitExpr_(const CallNode *node) final {
    if (dynamic_) {
      return StmtExprMutator::VisitExpr_(node);
    }
    if (vector_size_ == 2 || vector_size_ == 4) {
      if (node->op == builtin::call_extern() && node->args.size() >= 2) {
        if (const auto *func_name = node->args[0].as<StringImmNode>()) {
          if (func_name->value == "AtomicAdd") {
            // Matrix[by * stride_y + i / (stride_x / (tx_txtent *
            // vector_size_)) + tx_var_ / (stride_x / vector_size_),
            //        bx * stride_x + (i % (stride_x / (tx_extent *
            //        vector_size_)) * (tx_extent * vector_size_) + (tx_var_ %
            //        (stride / vector_size_)) * vector_size_]
            const CallNode *addr_call = node->args[1].as<CallNode>();
            if (!addr_call || addr_call->op != builtin::address_of() ||
                addr_call->args.size() != 1) {
              return StmtExprMutator::VisitExpr_(node);
            }
            const BufferLoadNode *old_dst_node =
                addr_call->args[0].as<BufferLoadNode>();
            const BufferLoadNode *old_value_node =
                node->args[2].as<BufferLoadNode>();
            if (!old_dst_node || !old_value_node) {
              return StmtExprMutator::VisitExpr_(node);
            }
            Array<PrimExpr> dst_indices, value_indices;
            if ((extent_tx_ * vector_size_) > stride_x_) {
              dst_indices.push_back(
                  by_var_ * stride_y_ +
                  iter_var_ * (extent_tx_ * vector_size_ / stride_x_) +
                  truncdiv(tx_var_, stride_x_ / vector_size_));
              dst_indices.push_back(
                  bx_var_ * stride_x_ +
                  truncmod(tx_var_, stride_x_ / vector_size_) * vector_size_);
              value_indices.push_back(
                  iter_var_ * (extent_tx_ * vector_size_ / stride_x_) +
                  truncdiv(tx_var_ * vector_size_, stride_x_));
              value_indices.push_back(
                  truncmod(tx_var_, stride_x_ / vector_size_) * vector_size_);
            } else {
              dst_indices.push_back(
                  by_var_ * stride_y_ +
                  truncdiv(iter_var_, stride_x_ / (extent_tx_ * vector_size_)) +
                  truncdiv(tx_var_, stride_x_ / vector_size_));
              dst_indices.push_back(
                  bx_var_ * stride_x_ +
                  truncmod(iter_var_, stride_x_ / (extent_tx_ * vector_size_)) *
                      (extent_tx_ * vector_size_) +
                  truncmod(tx_var_, stride_x_ / vector_size_) * vector_size_);
              value_indices.push_back(
                  truncdiv(iter_var_, stride_x_ / (extent_tx_ * vector_size_)) +
                  truncdiv(tx_var_, stride_x_ / vector_size_));
              value_indices.push_back(
                  truncmod(iter_var_, stride_x_ / (extent_tx_ * vector_size_)) *
                      (extent_tx_ * vector_size_) +
                  truncmod(tx_var_, stride_x_ / vector_size_) * vector_size_);
            }

            BufferLoad dst_node =
                BufferLoad(old_dst_node->buffer, dst_indices,
                           old_dst_node->predicate, old_dst_node->span);
            BufferLoad value_node =
                BufferLoad(old_value_node->buffer, value_indices,
                           old_value_node->predicate, old_value_node->span);
            Call address_of_dst =
                Call(DataType::Handle(), builtin::address_of(), {dst_node});
            Call address_of_value =
                Call(DataType::Handle(), builtin::address_of(), {value_node});
            Array<PrimExpr> new_args;
            if (vector_size_ == 2) {
              new_args.push_back(StringImm("AtomicAddx2"));
            } else {
              new_args.push_back(StringImm("AtomicAddx4"));
            }
            new_args.push_back(address_of_dst);
            new_args.push_back(address_of_value);

            Call new_call =
                tvm::tir::Call(node->dtype, builtin::call_extern(), new_args);

            return new_call;
          }
        }
      }
    }
    return StmtExprMutator::VisitExpr_(node);
  }

  const ForNode *inner_for_;
  const int vector_size_;
  const PrimExpr condition_;
  const bool dynamic_;
  const PrimExpr by_var_, bx_var_;
  int stride_y_, stride_x_;
  const Var tx_var_;
  Var iter_var_;
  int extent_tx_;
};

static int GetVectorizeSizeMax(int compute_capability, DataType dtype) {

  if (dtype == DataType::Float(16)) {
    return 2;
  }
  if (dtype == DataType::BFloat(16)) {
    if (compute_capability > 75) {
      return 2;
    } else {
      return 1;
    }
  }
  if (dtype == DataType::Float(32)) {
    if (compute_capability >= 90) {
      return 4;
    } else {
      return 1;
    }
  }
  return 1;
}

For VectorizeAtomicAdd(const For &for_node, const Var &thread_var,
                       const Range &thread_bounds, int compute_capability) {

  int vectorize_size_max = 1;
  int stride_x = -1, stride_y = -1;
  PrimExpr bx_var, by_var;

  PostOrderVisit(for_node->body, [&](const ObjectRef &obj) {
    if (const auto *call = obj.as<CallNode>()) {
      if (call->op == builtin::call_extern() && call->args.size() >= 2) {
        const auto *func_name = call->args[0].as<StringImmNode>();
        if (func_name->value == "AtomicAdd") {
          DataType dtype =
              call->args[1].as<CallNode>()->args[0].as<BufferLoadNode>()->dtype;
          vectorize_size_max = GetVectorizeSizeMax(compute_capability, dtype);
        }
      }
    }
    if (const MulNode *mul = obj.as<MulNode>()) {
      const VarNode *var = nullptr;
      const IntImmNode *imm = nullptr;
      PrimExpr var_expr;
      if ((var = mul->a.as<VarNode>()) && (imm = mul->b.as<IntImmNode>())) {
        var_expr = mul->a;
      } else if ((var = mul->b.as<VarNode>()) &&
                 (imm = mul->a.as<IntImmNode>())) {
        var_expr = mul->b;
      }
      if (var && imm) {
        if (var->name_hint == "bx") {
          stride_x = imm->value;
          bx_var = var_expr;
        } else if (var->name_hint == "by") {
          stride_y = imm->value;
          by_var = var_expr;
        }
      }
    }
  });
  if (vectorize_size_max != 1) {
    int vectorize_hint = vectorize_size_max;
    AtomicAddVectorizePlanResult res = {1, false, 0};
    AtomicAddVectorizePlanner planner;
    res = planner.Plan(for_node, thread_var, thread_bounds, vectorize_hint);
    vectorize_hint = res.vector_size;

    if (vectorize_hint == 1 || stride_x == -1 || stride_y == -1 ||
        !bx_var.defined() || !by_var.defined())
      return for_node;
    auto rewriter = AtomicAddVectorizeRewriter(
        res, thread_var, by_var, bx_var, thread_bounds, stride_y, stride_x);
    return Downcast<For>(rewriter(for_node));
  } else {
    return for_node;
  }
}

} // namespace tl
} // namespace tvm
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
    this->thread_var = thread_var;
    this->thread_bounds = thread_bounds;
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

  void UpdateVectorSize(const Array<PrimExpr> indices, const Buffer &buffer) {
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
      condition_ = (FloorMod(offset, vector_size_) == 0);
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
  AtomicAddVectorizeRewriter(AtomicAddVectorizePlanResult plan)
      : vector_size_(plan.vector_size), condition_(plan.condition),
        dynamic_(plan.dynamic) {}

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
      if (!dynamic_) {
        Var tx_var;
        PostOrderVisit(fnode->body, [&tx_var](const ObjectRef &node) {
          if (const VarNode *var = node.as<VarNode>()) {
            if (var->name_hint == "tx") {
              tx_var = GetRef<Var>(var);
            }
          }
        });
        ICHECK(tx_var.defined()) << "Failed to find tx var";
        Var outer_var = Var(old_var->name_hint + "_outer");
        Map<Var, PrimExpr> vmap;
        vmap.Set(tx_var,
                 truncmod(tx_var, extent / vector_size_) * vector_size_);
        vmap.Set(fnode->loop_var, outer_var * vector_size_ +
                                      truncdiv(tx_var, extent / vector_size_));
        Stmt body = Substitute(fnode->body, vmap);
        return For(outer_var, 0, extent / vector_size_, fnode->kind, body,
                   fnode->thread_binding, fnode->annotations, fnode->span);
      } else {
        return fnode;
      }
    } else {
      return ret;
    }
  }

  PrimExpr VisitExpr_(const CallNode *node) final {

    if (vector_size_ == 2 || vector_size_ == 4) {
      if (node->op == builtin::call_extern() && node->args.size() >= 2) {
        if (const auto *func_name = node->args[0].as<StringImmNode>()) {
          if (func_name->value == "AtomicAdd") {
            PrimExpr value_node = node->args[2];

            Call address_of_value = tvm::tir::Call(
                DataType::Handle(), builtin::address_of(), {value_node});

            Array<PrimExpr> new_args;
            if (vector_size_ == 2) {
              new_args.push_back(StringImm("AtomicAddx2"));
            } else {
              new_args.push_back(StringImm("AtomicAddx4"));
            }

            new_args.push_back(node->args[1]);
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

For VectorizeAtomicAdd(const For &for_node, Var thread_var, Range thread_bounds,
                       int compute_capability) {

  int vectorize_size_max = 1;

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
  });

  if (vectorize_size_max != 1) {
    int vectorize_hint = vectorize_size_max;
    AtomicAddVectorizePlanResult res = {1, false, 0};
    AtomicAddVectorizePlanner planner;
    res = planner.Plan(for_node, thread_var, thread_bounds, vectorize_hint);
    vectorize_hint = res.vector_size;

    if (vectorize_hint == 1)
      return for_node;
    auto rewriter = AtomicAddVectorizeRewriter(res);
    return Downcast<For>(rewriter(for_node));
  } else {
    return for_node;
  }
}

} // namespace tl
} // namespace tvm

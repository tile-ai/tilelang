/*!
 * \file atomicadd_vectorize.cc
 * \brief A tool to automatically vectorize atomic add
 */

#include "atomicadd_vectorize.h"

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRMutatorWithAnalyzer;
using arith::IRVisitorWithAnalyzer;

AtomicAddVectorizePlanner::AtomicAddVectorizePlanner() = default;

AtomicAddVectorizePlanResult
AtomicAddVectorizePlanner::Plan(const For &node, int compute_capability) {
  int vectorize_size_max = 1;
  this->vector_size_ = 4;
  this->dynamic_ = false;
  this->condition_ = PrimExpr();

  PostOrderVisit(node, [&](const ObjectRef &obj) {
    if (const auto *call = obj.as<CallNode>()) {
      if (call->op == atomicadd_elem_op()) {
        if (call->args.size() < 2) {
          // Fallback: unexpected arity
          vectorize_size_max = 1;
          DLOG(WARNING) << "[AtomicAddVectorizePlanner] atomicadd_elem_op "
                           "expects 2 args, got "
                        << call->args.size() << "; Fallback to no vectorize";
          return;
        }
        DataType dtype;
        if (const auto *load = call->args[0].as<BufferLoadNode>()) {
          dtype = load->dtype;
          vectorize_size_max = GetVectorizeSizeMax(compute_capability, dtype);
        } else if (const auto *ite = call->args[0].as<IfThenElseNode>()) {
          if (const auto *then_load = ite->then_case.as<BufferLoadNode>()) {
            dtype = then_load->dtype;
            vectorize_size_max = GetVectorizeSizeMax(compute_capability, dtype);
          } else if (const auto *else_load =
                         ite->else_case.as<BufferLoadNode>()) {
            dtype = else_load->dtype;
            vectorize_size_max = GetVectorizeSizeMax(compute_capability, dtype);
          } else {
            // fallback
            vectorize_size_max = 1;
            DLOG(WARNING) << "[AtomicAddVectorizePlanner] IfThenElse case "
                             "has no BufferLoad; Fallback to no vectorize";
          }
        } else {
          // fallback
          vectorize_size_max = 1;
          DLOG(WARNING) << "[AtomicAddVectorizePlanner] Unexpected arg1 type "
                        << call->args[1]->GetTypeKey()
                        << "; Fallback to no vectorize";
        }
      }
    }
  });

  if (vectorize_size_max <= 1) {
    return {1, dynamic_, condition_};
  }

  this->max_vector_size = vectorize_size_max;
  this->operator()(node);
  return {vector_size_, dynamic_, condition_};
}

void AtomicAddVectorizePlanner::VisitStmt_(const ForNode *node) {
  inner_for_ = node;
  arith::IRVisitorWithAnalyzer::VisitStmt_(node);
}

void AtomicAddVectorizePlanner::VisitExpr_(const CallNode *node) {
  if (node->op == atomicadd_elem_op() && !node->args.empty()) {
    if (node->args.size() < 2) {
      return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
    }
    const BufferLoadNode *buffer_load_dst = node->args[0].as<BufferLoadNode>();
    const BufferLoadNode *buffer_load_src = node->args[1].as<BufferLoadNode>();
    if (buffer_load_src && buffer_load_src->buffer.defined() &&
        buffer_load_dst && buffer_load_dst->buffer.defined()) {
      Buffer dst_buffer = buffer_load_dst->buffer;
      UpdateVectorSize(buffer_load_dst->indices, dst_buffer);

      Buffer src_buffer = buffer_load_src->buffer;
      UpdateVectorSize(buffer_load_src->indices, src_buffer);
    }
  }
  return arith::IRVisitorWithAnalyzer::VisitExpr_(node);
}

int AtomicAddVectorizePlanner::GetVectorizeSizeMax(int compute_capability,
                                                   DataType dtype) {
  if (dtype == DataType::Float(16)) {
    return 2;
  }
  if (dtype == DataType::BFloat(16)) {
    return compute_capability > 75 ? 2 : 1;
  }
  if (dtype == DataType::Float(32)) {
    return compute_capability >= 90 ? 4 : 1;
  }
  return 1;
}

void AtomicAddVectorizePlanner::UpdateVectorSize(const Array<PrimExpr> &indices,
                                                 const Buffer &buffer) {
  if (!inner_for_)
    return;
  auto extent_ptr = inner_for_->extent.as<IntImmNode>();
  if (!extent_ptr)
    return;

  const DataType &access_type = buffer->dtype;
  max_vector_size = arith::ZeroAwareGCD(max_vector_size, extent_ptr->value);

  auto last_dim = buffer->shape.back();
  auto mod_set = analyzer_.modular_set(last_dim);

  if (buffer->shape.back().as<IntImmNode>()) {
    max_vector_size = arith::ZeroAwareGCD(max_vector_size, mod_set->coeff);
    auto gcd_base = arith::ZeroAwareGCD(max_vector_size, mod_set->base);

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

    while (!IndiceCanVectorize(elem_offset, inner_for_->loop_var,
                               inner_for_->extent, vector_size_, &analyzer_)) {
      vector_size_ /= 2;
    }
  } else if (vector_size_ <= 4) {
    dynamic_ = true;
    PrimExpr offset = buffer.OffsetOf(indices).back();
    condition_ = (truncmod(offset, vector_size_) == 0);
  }
}

class AtomicAddVectorizeRewriter : public StmtExprMutator {
public:
  AtomicAddVectorizeRewriter(const AtomicAddVectorizePlanResult &plan)
      : vector_size_(plan.vector_size), dynamic_(plan.dynamic),
        condition_(plan.condition) {}

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
    auto ret = StmtExprMutator::VisitStmt_(node);
    if (vector_size_ == 1)
      return ret;
    if (inner_for_ == node) {
      For fnode = ret.as<For>().value();
      auto old_var = fnode->loop_var;
      auto new_var = Var(old_var->name_hint);
      auto extent_ptr = as_const_int(fnode->extent);
      ICHECK(extent_ptr) << fnode->extent;
      int extent = *extent_ptr;
      ICHECK(extent % vector_size_ == 0)
          << "extent: " << extent << " vector_size_: " << vector_size_;
      ICHECK(is_zero(fnode->min));
      if (!dynamic_) {
        Map<Var, PrimExpr> vmap;
        vmap.Set(old_var, new_var * vector_size_);
        Stmt body = Substitute(fnode->body, vmap);
        return For(new_var, 0, extent / vector_size_, fnode->kind, body,
                   fnode->thread_binding, fnode->annotations, fnode->step,
                   fnode->span);
      }
    }
    return ret;
  }

  PrimExpr VisitExpr_(const CallNode *node) final {
    bool legal_vectorize = true;
    if (dynamic_)
      legal_vectorize = false;
    if (!(node->op == atomicadd_elem_op()))
      legal_vectorize = false;
    if (node->args.size() < 2)
      legal_vectorize = false;
    if (legal_vectorize) {
      const BufferLoadNode *temp_dst_node = node->args[0].as<BufferLoadNode>();
      const BufferLoadNode *temp_value_node =
          node->args[1].as<BufferLoadNode>();
      if (!temp_dst_node || !temp_value_node)
        legal_vectorize = false;
    }
    if (legal_vectorize) {
      const BufferLoad dst_node = Downcast<BufferLoad>(node->args[0]);
      const BufferLoad value_node = Downcast<BufferLoad>(node->args[1]);
      // The default memory order is relaxed
      // Ref: src/tl_templates/cuda/atomic.h::AtomicAdd
      const IntImm memory_order =
          node->args.size() >= 3 ? Downcast<IntImm>(node->args[2]) : IntImm(0);
      Array<PrimExpr> new_args;
      Call address_of_dst =
          Call(DataType::Handle(), builtin::address_of(), {dst_node});
      Call address_of_value =
          Call(DataType::Handle(), builtin::address_of(), {value_node});
      if (vector_size_ == 4) {
        new_args.push_back(StringImm("AtomicAddx4"));
        new_args.push_back(address_of_dst);
        new_args.push_back(address_of_value);
      } else if (vector_size_ == 2) {
        new_args.push_back(StringImm("AtomicAddx2"));
        new_args.push_back(address_of_dst);
        new_args.push_back(address_of_value);
      } else {
        // Scalar case: AtomicAdd now expects a pointer to destination.
        new_args.push_back(StringImm("AtomicAdd"));
        new_args.push_back(address_of_dst);
        new_args.push_back(value_node);
      }
      new_args.push_back(memory_order);

      Call new_call =
          tvm::tir::Call(node->dtype, builtin::call_extern(), new_args);

      return new_call;
    } else {
      Array<PrimExpr> new_args;
      new_args.push_back(StringImm("AtomicAdd"));
      // Ensure first argument is an address; keep value as-is.
      if (!node->args.empty()) {
        if (const auto *bl = node->args[0].as<BufferLoadNode>()) {
          Call address_of_dst = Call(DataType::Handle(), builtin::address_of(),
                                     {Downcast<BufferLoad>(node->args[0])});
          new_args.push_back(address_of_dst);
        } else if (const auto *call = node->args[0].as<CallNode>()) {
          // If it's already an address_of, forward it; otherwise, keep
          // original.
          if (call->op.same_as(builtin::address_of())) {
            new_args.push_back(node->args[0]);
          } else {
            new_args.push_back(node->args[0]);
          }
        } else {
          new_args.push_back(node->args[0]);
        }
        // Push remaining args unchanged (value, optional memory_order, ...)
        for (size_t i = 1; i < node->args.size(); ++i) {
          new_args.push_back(node->args[i]);
        }
      }

      Call new_call =
          tvm::tir::Call(node->dtype, builtin::call_extern(), new_args);

      return new_call;
    }
  }

  const ForNode *inner_for_;
  const int vector_size_;
  const PrimExpr condition_;
  const bool dynamic_;
};

For VectorizeAtomicAdd(const For &for_node, int compute_capability) {
  AtomicAddVectorizePlanResult res = {1, false, 0};
  AtomicAddVectorizePlanner planner;
  res = planner.Plan(for_node, compute_capability);
  auto rewriter = AtomicAddVectorizeRewriter(res);
  return Downcast<For>(rewriter(for_node));
}

} // namespace tl
} // namespace tvm

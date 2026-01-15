/*!
 * \file atomicadd_vectorize.cc
 * \brief Automatic vectorization pass for atomic add operations.
 *
 * This pass detects atomic_add_elem_op inside vectorized loops and converts
 * them to vectorized versions (atomic_addx2_elem_op or atomic_addx4_elem_op).
 */

#include "atomicadd_vectorize.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

/*!
 * \brief Extract BufferLoad from an expression that may be wrapped in
 * address_of.
 */
Optional<BufferLoad> ExtractBufferLoad(const PrimExpr &expr) {
  if (const auto *load = expr.as<BufferLoadNode>()) {
    return tvm::ffi::GetRef<BufferLoad>(load);
  }
  if (const auto *call = expr.as<CallNode>()) {
    if (call->op.same_as(builtin::address_of()) && !call->args.empty()) {
      if (const auto *load = call->args[0].as<BufferLoadNode>()) {
        return tvm::ffi::GetRef<BufferLoad>(load);
      }
    }
  }
  return Optional<BufferLoad>();
}

/*!
 * \brief Get the vectorized atomic add op based on vector size.
 */
Op GetVectorizedAtomicOp(int vector_size) {
  switch (vector_size) {
  case 4:
    return atomic_addx4_elem_op();
  case 2:
    return atomic_addx2_elem_op();
  default:
    return atomic_add_elem_op();
  }
}

/*!
 * \brief Rewriter that transforms atomic_add_elem_op inside vectorized loops.
 *
 * Strategy: Detect ForKind::kVectorized loops, use their extent as vector size,
 * and convert atomic_add_elem_op to the corresponding vectorized version.
 */
class AtomicAddVectorizeRewriter : public StmtExprMutator {
public:
  explicit AtomicAddVectorizeRewriter(Target target) : target_(target) {}

private:
  /*!
   * \brief Get the max vector size supported by the given dtype.
   */
  int GetMaxVectorSize(DataType dtype) const {
    if (dtype.is_float16() || dtype.is_bfloat16()) {
      return 2;
    }
    if (dtype.is_float() && dtype.bits() == 32 &&
        TargetHasSMVersionGE(target_, 90)) {
      return 4;
    }
    return 1;
  }

  Stmt VisitStmt_(const ForNode *node) final {
    // Check if this is a vectorized loop
    if (node->kind == ForKind::kVectorized) {
      auto extent_ptr = as_const_int(node->extent);
      if (!extent_ptr) {
        return StmtExprMutator::VisitStmt_(node);
      }

      int vec_size = static_cast<int>(*extent_ptr);
      // Push vectorized context
      vectorized_loop_ = node;
      vector_size_ = vec_size;
      Stmt body = VisitStmt(node->body);
      // If we successfully vectorized atomic ops, transform the loop
      if (has_vectorized_atomic_) {
        has_vectorized_atomic_ = false;
        vectorized_loop_ = nullptr;
        vector_size_ = 1;
        // Change loop extent to 1 since atomic op now handles all elements
        return For(node->loop_var, node->min, Integer(1), node->kind, body,
                   node->thread_binding, node->annotations, node->step,
                   node->span);
      }

      vectorized_loop_ = nullptr;
      vector_size_ = 1;

      if (body.same_as(node->body)) {
        return tvm::ffi::GetRef<Stmt>(node);
      }
      return For(node->loop_var, node->min, node->extent, node->kind, body,
                 node->thread_binding, node->annotations, node->step,
                 node->span);
    }
    return StmtExprMutator::VisitStmt_(node);
  }

  PrimExpr VisitExpr_(const CallNode *node) final {
    if (node->op != atomic_add_elem_op() || node->args.size() < 2) {
      return StmtExprMutator::VisitExpr_(node);
    }

    // Must be inside a vectorized loop
    if (!vectorized_loop_ || vector_size_ <= 1) {
      return StmtExprMutator::VisitExpr_(node);
    }

    auto dst_load = ExtractBufferLoad(node->args[0]);
    auto src_load = ExtractBufferLoad(node->args[1]);

    if (!dst_load.defined() || !src_load.defined()) {
      return StmtExprMutator::VisitExpr_(node);
    }

    // Check if dtype supports this vector size
    DataType dtype = dst_load.value()->buffer->dtype;
    if (vector_size_ > GetMaxVectorSize(dtype)) {
      return StmtExprMutator::VisitExpr_(node);
    }

    // Mark that we have vectorized an atomic op
    has_vectorized_atomic_ = true;

    // Create vectorized atomic op
    Call addr_dst(DataType::Handle(), builtin::address_of(),
                  {dst_load.value()});
    Call addr_src(DataType::Handle(), builtin::address_of(),
                  {src_load.value()});

    return Call(node->dtype, GetVectorizedAtomicOp(vector_size_),
                {addr_dst, addr_src});
  }

  Target target_;
  const ForNode *vectorized_loop_ = nullptr;
  int vector_size_ = 1;
  bool has_vectorized_atomic_ = false;
};

} // namespace

For VectorizeAtomicAdd(const For &for_node) {
  Target target = Target::Current(false);
  return Downcast<For>(AtomicAddVectorizeRewriter(target)(for_node));
}

} // namespace tl
} // namespace tvm

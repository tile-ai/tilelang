/*!
 * \file tl/op/multimem_rewriter.h
 * \brief Post-process IR to replace vectorized BufferLoad/Store on mcast
 * buffers with multimem call_extern instructions.
 */

#ifndef TVM_TL_OP_MULTIMEM_REWRITER_H_
#define TVM_TL_OP_MULTIMEM_REWRITER_H_

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <sstream>

#include "multimem.h"

namespace tvm {
namespace tl {

using namespace tir;

static inline std::string GetReduceOpStr(int reduce_op) {
  switch (reduce_op) {
  case 0:
    return "tl::multimem::ReduceOp::ADD";
  case 1:
    return "tl::multimem::ReduceOp::MIN";
  case 2:
    return "tl::multimem::ReduceOp::MAX";
  default:
    LOG(FATAL) << "Invalid reduce_op: " << reduce_op;
    return "";
  }
}

/*!
 * \brief Rewrites BufferLoad/BufferStore involving a multicast buffer
 *        into multimem call_extern instructions.
 *
 * After ParallelOp + VectorizeLoop, the IR contains ForKind::kVectorized loops
 * with scalar loop variables (Ramp is not materialized until codegen).
 * This rewriter detects two patterns:
 *
 * 1. ForKind::kVectorized loop containing mcast buffer access:
 *    for (vec: kVectorized, extent=N) { dst[base+vec] = mcast[base+vec] }
 *    → call_extern("LdReduceVN<...>::run", &dst[base], &mcast[base])
 *
 * 2. Scalar BufferStore with Ramp indices (if vectorization produced Ramp):
 *    dst[Ramp(base,1,N)] = mcast[Ramp(base,1,N)]
 *    → call_extern("LdReduceVN<...>::run", &dst[base], &mcast[base])
 */
class MultimemRewriter : public StmtExprMutator {
public:
  MultimemRewriter(Buffer mcast_buf, MultimemMode mode, int reduce_op)
      : mcast_buf_(std::move(mcast_buf)), mode_(mode), reduce_op_(reduce_op) {}

  Stmt Rewrite(Stmt stmt) { return VisitStmt(std::move(stmt)); }

protected:
  /*!
   * \brief Handle ForKind::kVectorized loops.
   * If the loop body is a single BufferStore involving the mcast buffer,
   * replace the entire loop with a single vectorized multimem call.
   */
  Stmt VisitStmt_(const ForNode *op) override {
    if (op->kind == ForKind::kVectorized) {
      auto extent_ptr = op->extent.as<IntImmNode>();
      if (extent_ptr) {
        int lanes = static_cast<int>(extent_ptr->value);
        // Try to match the loop body as a single BufferStore from mcast buffer
        auto result = TryRewriteVectorizedLoop(op, lanes);
        if (result.defined()) {
          return result;
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  /*!
   * \brief Handle scalar BufferStore with Ramp indices (fallback).
   */
  Stmt VisitStmt_(const BufferStoreNode *op) override {
    if (mode_ == MultimemMode::kLdReduce) {
      if (auto *load = op->value.as<BufferLoadNode>()) {
        if (load->buffer.same_as(mcast_buf_)) {
          int lanes = GetLanes(load->indices);
          if (lanes > 1) {
            return MakeMultimemCall(op->buffer, op->indices, load->buffer,
                                    load->indices, lanes);
          }
        }
      }
    } else {
      if (op->buffer.same_as(mcast_buf_)) {
        if (auto *load = op->value.as<BufferLoadNode>()) {
          int lanes = GetLanes(op->indices);
          if (lanes > 1) {
            return MakeMultimemCall(load->buffer, load->indices, op->buffer,
                                    op->indices, lanes);
          }
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

private:
  Buffer mcast_buf_;
  MultimemMode mode_;
  int reduce_op_;

  /*!
   * \brief Try to rewrite a kVectorized for-loop containing a mcast
   * BufferStore. Returns the replacement Stmt, or undefined if the pattern
   * doesn't match.
   */
  Stmt TryRewriteVectorizedLoop(const ForNode *op, int lanes) {
    // The body should be a single BufferStore (possibly wrapped in IfThenElse)
    const BufferStoreNode *store = nullptr;
    Stmt body = op->body;

    // Unwrap IfThenElse if present
    if (auto *ite = body.as<IfThenElseNode>()) {
      store = ite->then_case.as<BufferStoreNode>();
    } else {
      store = body.as<BufferStoreNode>();
    }

    if (!store)
      return Stmt();

    const BufferLoadNode *load = store->value.as<BufferLoadNode>();
    if (!load)
      return Stmt();

    // Check if this involves the mcast buffer
    bool matches = false;
    const Buffer *local_buf_ptr = nullptr;
    const Array<PrimExpr> *local_indices_ptr = nullptr;
    const Buffer *mc_buf_ptr = nullptr;
    const Array<PrimExpr> *mc_indices_ptr = nullptr;

    if (mode_ == MultimemMode::kLdReduce) {
      if (load->buffer.same_as(mcast_buf_)) {
        matches = true;
        local_buf_ptr = &store->buffer;
        local_indices_ptr = &store->indices;
        mc_buf_ptr = &load->buffer;
        mc_indices_ptr = &load->indices;
      }
    } else {
      if (store->buffer.same_as(mcast_buf_)) {
        matches = true;
        local_buf_ptr = &load->buffer;
        local_indices_ptr = &load->indices;
        mc_buf_ptr = &store->buffer;
        mc_indices_ptr = &store->indices;
      }
    }

    if (!matches)
      return Stmt();

    // Substitute vec_var = 0 to get the base address
    Var vec_var = op->loop_var;
    Map<Var, PrimExpr> vmap;
    vmap.Set(vec_var, IntImm(vec_var.dtype(), 0));

    Array<PrimExpr> local_base_indices;
    for (const auto &idx : *local_indices_ptr) {
      local_base_indices.push_back(Substitute(idx, vmap));
    }
    Array<PrimExpr> mc_base_indices;
    for (const auto &idx : *mc_indices_ptr) {
      mc_base_indices.push_back(Substitute(idx, vmap));
    }

    return MakeMultimemCall(*local_buf_ptr, local_base_indices, *mc_buf_ptr,
                            mc_base_indices, lanes);
  }

  /*!
   * \brief Get the vector lanes from Ramp indices.
   */
  int GetLanes(const Array<PrimExpr> &indices) const {
    if (indices.size() == 1) {
      if (auto *ramp = indices[0].as<RampNode>()) {
        return static_cast<int>(ramp->lanes.as<IntImmNode>()->value);
      }
    }
    return 1;
  }

  /*!
   * \brief Create the call_extern for a multimem instruction.
   */
  Stmt MakeMultimemCall(const Buffer &local_buf,
                        const Array<PrimExpr> &local_indices,
                        const Buffer &mc_buf, const Array<PrimExpr> &mc_indices,
                        int lanes) const {
    std::string func_name = MakeFuncName(lanes, local_buf->dtype);

    Array<PrimExpr> args;
    args.push_back(StringImm(func_name));

    if (mode_ == MultimemMode::kLdReduce) {
      args.push_back(MakeAddressOf(local_buf, local_indices));
      args.push_back(MakeAddressOf(mc_buf, mc_indices));
    } else {
      args.push_back(MakeAddressOf(mc_buf, mc_indices));
      args.push_back(MakeAddressOf(local_buf, local_indices));
    }

    auto call = Call(DataType::Handle(), builtin::call_extern(), args);
    return Evaluate(call);
  }

  /*!
   * \brief Construct the template function name.
   */
  std::string MakeFuncName(int lanes, DataType dtype) const {
    std::string dtype_tag = DTypeToTag(dtype);
    std::string reduce_op_str = GetReduceOpStr(reduce_op_);

    std::stringstream ss;
    switch (mode_) {
    case MultimemMode::kLdReduce:
      ss << "tl::multimem::LdReduceV" << lanes;
      break;
    case MultimemMode::kSt:
      ss << "tl::multimem::StV" << lanes;
      break;
    case MultimemMode::kRed:
      ss << "tl::multimem::RedV" << lanes;
      break;
    }
    ss << "<";
    if (mode_ != MultimemMode::kSt) {
      ss << reduce_op_str << ", ";
    }
    ss << dtype_tag;
    ss << ">::run";
    return ss.str();
  }

  std::string DTypeToTag(DataType dtype) const {
    if (dtype.is_float() && dtype.bits() == 32)
      return "float";
    if (dtype.is_float16())
      return "half_t";
    if (dtype.is_bfloat16())
      return "bfloat16_t";
    LOG(FATAL) << "Unsupported dtype for multimem: " << dtype;
    return "";
  }

  /*!
   * \brief Create address_of expression. Handles Ramp by extracting base.
   */
  PrimExpr MakeAddressOf(const Buffer &buffer,
                         const Array<PrimExpr> &indices) const {
    Array<PrimExpr> scalar_indices;
    for (const auto &idx : indices) {
      if (auto *ramp = idx.as<RampNode>()) {
        scalar_indices.push_back(ramp->base);
      } else {
        scalar_indices.push_back(idx);
      }
    }
    return Call(DataType::Handle(), builtin::address_of(),
                {BufferLoad(buffer, scalar_indices)});
  }
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_MULTIMEM_REWRITER_H_

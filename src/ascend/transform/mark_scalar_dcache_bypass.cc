/*!
 * \file mark_scalar_dcache_bypass.cc
 * \brief Rewrite scalar BufferLoad/BufferStore on global buffers that have
 * writes (either scalar BufferStore or MTE copy) into explicit
 * ascend_read/write_gm_bypass_dcache Call nodes.
 *
 * Pure-read global buffers keep their normal BufferLoad (dcache path).
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../../../3rdparty/tvm/src/tirx/transform/ir_utils.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

/*!
 * \brief Check whether a Var points to a global-memory buffer.
 */
bool IsGlobalBufferVar(const Var &var) {
  std::string scope = GetPtrStorageScope(var);
  return scope == "global";
}

/*!
 * \brief Collect the set of global-buffer data Vars that have writes.
 *
 * Walks the PrimFunc body and records:
 *  - BufferStore to a global buffer
 *  - MTE copy ops (ascend_copy_ubuf_to_gm / ascend_copy_matrix_cc_to_gm)
 *    whose destination is a global buffer.
 */
std::unordered_set<const VarNode *>
CollectGlobalWriteBufferVars(const PrimFunc &f) {
  std::unordered_set<const VarNode *> write_vars;

  PostOrderVisit(f->body, [&](const ObjectRef &n) {
    if (const auto *store = n.as<BufferStoreNode>()) {
      if (IsGlobalBufferVar(store->buffer->data)) {
        write_vars.insert(store->buffer->data.get());
      }
    }
    if (const auto *call = n.as<CallNode>()) {
      if (!call->op.same_as(ascend_copy_ubuf_to_gm()) &&
          !call->op.same_as(ascend_copy_matrix_cc_to_gm())) {
        return;
      }
      // First arg is tvm_access_ptr(TypeAnnotation, data_var, offset, ...)
      if (call->args.empty())
        return;
      const auto *access_ptr = call->args[0].as<CallNode>();
      if (!access_ptr || !access_ptr->op.same_as(builtin::tvm_access_ptr())) {
        return;
      }
      if (access_ptr->args.size() < 2)
        return;
      const auto *data_var = access_ptr->args[1].as<VarNode>();
      if (!data_var)
        return;
      if (IsGlobalBufferVar(GetRef<Var>(data_var))) {
        write_vars.insert(data_var);
      }
    }
  });

  return write_vars;
}

} // namespace

class ScalarDcacheBypassRewriter : public StmtExprMutator {
public:
  explicit ScalarDcacheBypassRewriter(
      std::unordered_set<const VarNode *> bypass_vars)
      : bypass_vars_(std::move(bypass_vars)) {}

  static PrimFunc Rewrite(PrimFunc f) {
    auto write_vars = CollectGlobalWriteBufferVars(f);
    if (write_vars.empty()) {
      // No global buffers have writes → nothing to do.
      return f;
    }
    ScalarDcacheBypassRewriter rewriter(std::move(write_vars));
    f.CopyOnWrite()->body = rewriter(f->body);
    return f;
  }

private:
  std::unordered_set<const VarNode *> bypass_vars_;
  bool inside_vf_{false};

  bool ShouldBypass(const Buffer &buffer) const {
    if (inside_vf_)
      return false;
    if (!IsGlobalBufferVar(buffer->data))
      return false;
    return bypass_vars_.count(buffer->data.get()) > 0;
  }

  PrimExpr MakeAddressOf(PrimExpr load) const {
    return Call(DataType::Handle(), tirx::builtin::address_of(), {load});
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    bool is_vf = (op->name_hint == "SIMT_VF" || op->name_hint == "SIMD_VF");
    if (is_vf)
      inside_vf_ = true;
    Stmt ret = StmtExprMutator::VisitStmt_(op);
    if (is_vf)
      inside_vf_ = false;
    return ret;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    if (op->indices.size() == 1 && op->dtype.is_scalar() &&
        ShouldBypass(op->buffer)) {
      ICHECK(!op->predicate.defined())
          << "ScalarDcacheBypass: predicated buffer load not implemented";
      // Replace with: ascend_read_gm_bypass_dcache(address_of(BufferLoad))
      return Call(op->dtype, ascend_read_gm_bypass_dcache(),
                  {MakeAddressOf(GetRef<PrimExpr>(op))});
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    if (op->indices.size() == 1 && op->buffer->dtype.is_scalar() &&
        ShouldBypass(op->buffer)) {
      ICHECK(!op->predicate.defined())
          << "ScalarDcacheBypass: predicated buffer store not implemented";
      ICHECK(op->value.dtype().is_scalar())
          << "ScalarDcacheBypass: vector-valued buffer store not implemented";
      // Replace with:
      //   Evaluate(ascend_write_gm_bypass_dcache(address_of(BufferLoad),
      //                                           value))
      auto addr =
          MakeAddressOf(BufferLoad(op->buffer, op->indices, op->predicate));
      auto call = Call(DataType::Void(), ascend_write_gm_bypass_dcache(),
                       {addr, VisitExpr(op->value)});
      return Evaluate(call);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

using namespace tirx::transform;

tvm::transform::Pass MarkScalarDcacheBypass() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return ScalarDcacheBypassRewriter::Rewrite(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MarkScalarDcacheBypass", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MarkScalarDcacheBypass",
                        MarkScalarDcacheBypass);
}

} // namespace tl
} // namespace tvm

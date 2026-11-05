/*!
 * \file lower_pdl.cc
 * \brief Mark native device functions that use CUDA PDL operations.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "cuda/op/builtin.h"

namespace tvm::tl {
using namespace tirx;

class CudaSyncCallCollector : public StmtExprVisitor {
public:
  using StmtExprVisitor::Visit_;

  ffi::Optional<ffi::VisitInterrupt> Visit_(const CallObj *call) final {
    has_trigger_launch |= call->op.same_as(pdl_trigger());
    has_grid_sync |= call->op.same_as(pdl_sync());
    if (auto op = call->op.as<OpObj>())
      has_ldg |= op->name == "tl.__ldg";
    return StmtExprVisitor::Visit_(call);
  }

  bool has_trigger_launch{false};
  bool has_grid_sync{false};
  bool has_ldg{false};
};

tvm::transform::Pass MarkCudaSyncCallsPass(bool support_pdl) {
  auto pass = [support_pdl](tirx::Function function, IRModule,
                            tvm::transform::PassContext) {
    auto collector = ffi::make_object<CudaSyncCallCollector>();
    collector->Visit(function->body);
    TVM_FFI_CHECK(support_pdl || (!collector->has_trigger_launch &&
                                  !collector->has_grid_sync),
                  ValueError)
        << "PDL is not supported";
    // NVCC's read-only __ldg cache is incompatible with PDL's visibility
    // boundary. This concerns the explicit __ldg intrinsic, not all global
    // loads or the independent ldg32/64/128/256 instruction family.
    TVM_FFI_CHECK(!collector->has_grid_sync || !collector->has_ldg, ValueError)
        << "Cannot invoke __ldg function with pdl_sync";
    if (collector->has_trigger_launch)
      function =
          WithAttr(std::move(function), "has_cuda_pdl_trigger", int64_t{1});
    if (collector->has_grid_sync)
      function = WithAttr(std::move(function), "has_cuda_pdl_sync", int64_t{1});
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.MarkCudaSyncCalls");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.MarkCudaSyncCalls",
                                   MarkCudaSyncCallsPass);
}
} // namespace tvm::tl

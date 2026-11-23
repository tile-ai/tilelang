/*!
 * \file legalize_parallel_to_pto.cc
 * \brief Placeholder for the PTO Parallel legalization stage.
 *
 * Only the registration exists for now, so that the formal PTO pipeline
 * order (VerifyParallelToPTO -> LegalizeParallelToPTO ->
 * VectorizeParallelToPTO) is routing-testable. The placeholder preserves
 * the input IR and claims no semantic support: it neither rewrites nor
 * rejects any construct. The safe scalar If/Select/Min/Max rewrites are
 * left to a later change.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

tvm::transform::Pass LegalizeParallelToPTO() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) { return func; };
  return CreatePrimFuncPass(pass_func, 0, "tl.LegalizeParallelToPTO", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeParallelToPTO",
                        LegalizeParallelToPTO);
}

} // namespace tl
} // namespace tvm

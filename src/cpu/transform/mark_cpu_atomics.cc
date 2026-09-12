/*!
 * \file mark_cpu_atomics.cc
 * \brief Mark PrimFuncs that call atomic ops, for the CPU parallel-grid pass.
 *
 * Both atomic forms are lowered to plain read-modify-write long before the
 * tail of the CPU pipeline (region forms `tl.tileop.atomic*` inside
 * LowerTileOp, scalar `tl.atomic_*_elem_op` intrinsics by LowerCPUAtomics),
 * so the parallel-grid pass cannot see them anymore. This pass runs before
 * LowerTileOp and tags the function with the ``tl.cpu_had_atomics``
 * attribute when any atomic op call is present; MaterializeCPUParallelGrid
 * refuses to parallelize such kernels — a parallel grid would turn the
 * serial RMW into a data race.
 */

#include "support/check.h"
#include "transform/common/attr.h"
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tvm::ffi;

namespace transform {

using namespace tirx::transform;

namespace {

bool HasAtomicCall(const Stmt &body) {
  bool found = false;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    if (const auto *call = node.as<CallNode>()) {
      if (const auto *op = call->op.as<OpNode>();
          op && (op->name.find("tl.atomic") == 0 ||
                 op->name.find("tl.tileop.atomic") == 0)) {
        found = true;
      }
    }
  });
  return found;
}

} // namespace

tvm::transform::Pass MarkCPUAtomics() {
  auto pass_func = [](PrimFunc func, const IRModule &mod,
                      const tvm::transform::PassContext &ctx) -> PrimFunc {
    if (HasAtomicCall(func->body)) {
      return WithAttr(std::move(func), attr::kCPUHadAtomics, Bool(true));
    }
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.MarkCPUAtomics", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.cpu.transform.MarkCPUAtomics", MarkCPUAtomics);
}

} // namespace transform

} // namespace tl
} // namespace tvm

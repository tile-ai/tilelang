/*!
 * \file tl/backend/rocm/op/atomic_add.cc
 * \brief ROCm implementation for tl.atomic_add lowering.
 */

#include "op/atomic_add.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace rocm {

struct AtomicAdd {
  static LayoutMap InferLayout(const AtomicAddNode &op,
                               const LayoutInferArgs &T, InferLevel level) {
    ICHECK(!op.GetUseTMA())
        << "TMA atomic_add is only supported by the CUDA backend";
    return op.InferSIMTLayout(T, level);
  }

  static Stmt Lower(const AtomicAddNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    ICHECK(!op.GetUseTMA())
        << "TMA atomic_add is only supported by the CUDA backend";
    return op.LowerSIMT(T, analyzer);
  }
};

} // namespace rocm

namespace {

bool MatchROCmAtomicAddTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmAtomicAdd() {
  RegisterAtomicAddImpl(AtomicAddImpl{
      "rocm.AtomicAdd",
      MatchROCmAtomicAddTarget,
      rocm::AtomicAdd::InferLayout,
      rocm::AtomicAdd::Lower,
  });
  return true;
}

const bool rocm_atomic_add_registered = RegisterROCmAtomicAdd();

} // namespace

} // namespace tl
} // namespace tvm

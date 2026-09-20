/*!
 * \file tl/cpu/op/atomic_add.cc
 * \brief CPU implementation for tl.atomicadd lowering (serial RMW).
 *
 * CPU execution is serial, so atomic_add degenerates to a plain
 * read-modify-write (`dst = dst + value`) inside a kSerial loop nest built
 * by cpu::LowerAtomicRMW. See atomic_rmw.h for the full contract
 * (memory_order ignored, use_tma rejected).
 */

#include "op/atomic_add.h"

#include "atomic_rmw.h"
#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace cpu {

struct AtomicAdd {
  static LayoutMap InferLayout(const AtomicAddNode &, const LayoutInferArgs &,
                               InferLevel) {
    // CPU has no fragment/shared layouts to infer.
    return LayoutMap{};
  }

  static Stmt Lower(const AtomicAddNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    (void)analyzer;
    return LowerAtomicRMW(op, AtomicCombine::kAdd, lower_args, "atomic_add");
  }
};

} // namespace cpu

namespace {

bool MatchCPUAtomicAddTarget(Target target) { return TargetIsCPU(target); }

bool RegisterCPUAtomicAdd() {
  RegisterAtomicAddImpl(AtomicAddImpl{
      "cpu.AtomicAdd",
      MatchCPUAtomicAddTarget,
      cpu::AtomicAdd::InferLayout,
      cpu::AtomicAdd::Lower,
  });
  return true;
}

const bool cpu_atomic_add_registered = RegisterCPUAtomicAdd();

} // namespace

} // namespace tl
} // namespace tvm

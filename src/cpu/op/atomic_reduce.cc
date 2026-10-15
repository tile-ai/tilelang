/*!
 * \file tl/cpu/op/atomic_reduce.cc
 * \brief CPU implementation for tl.atomicmax/tl.atomicmin lowering
 *        (serial RMW).
 *
 * CPU execution is serial, so atomic_max/atomic_min degenerate to a plain
 * read-modify-write (`dst = max/min(dst, value)`) inside a kSerial loop nest
 * built by cpu::LowerAtomicRMW. Both ops share the AtomicReduceImpl registry
 * and are told apart via GetElemOp(). See atomic_rmw.h for the full contract
 * (memory_order ignored, use_tma rejected).
 */

#include "op/atomic_reduce.h"

#include "atomic_rmw.h"
#include "backend/common/target_utils.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace cpu {

struct AtomicReduce {
  static LayoutMap InferLayout(const AtomicOpBaseNode &,
                               const LayoutInferArgs &, InferLevel) {
    // CPU has no fragment/shared layouts to infer.
    return LayoutMap{};
  }

  static Stmt Lower(const AtomicOpBaseNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    (void)analyzer;
    AtomicCombine combine;
    if (op.GetElemOp().same_as(atomic_max_elem_op())) {
      combine = AtomicCombine::kMax;
    } else {
      ICHECK(op.GetElemOp().same_as(atomic_min_elem_op()))
          << "CPU atomic_reduce: unexpected elem op " << op.GetElemOp()->name
          << " (only atomic_max/atomic_min are registered).";
      combine = AtomicCombine::kMin;
    }
    return LowerAtomicRMW(op, combine, lower_args, "atomic_reduce");
  }
};

} // namespace cpu

namespace {

bool MatchCPUAtomicReduceTarget(Target target) { return TargetIsCPU(target); }

bool RegisterCPUAtomicReduce() {
  RegisterAtomicReduceImpl(AtomicReduceImpl{
      "cpu.AtomicReduce",
      MatchCPUAtomicReduceTarget,
      cpu::AtomicReduce::InferLayout,
      cpu::AtomicReduce::Lower,
  });
  return true;
}

const bool cpu_atomic_reduce_registered = RegisterCPUAtomicReduce();

} // namespace

} // namespace tl
} // namespace tvm

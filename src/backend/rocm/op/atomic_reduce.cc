/*!
 * \file tl/backend/rocm/op/atomic_reduce.cc
 * \brief ROCm implementation for tl.atomicmax/tl.atomicmin lowering.
 */

#include "backend/common/op/atomic_reduce.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchROCmAtomicReduceTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmAtomicReduce() {
  RegisterAtomicReduceImpl(AtomicReduceImpl{
      "rocm.AtomicReduce",
      MatchROCmAtomicReduceTarget,
      backend::AtomicReduce::InferLayout,
      backend::AtomicReduce::Lower,
  });
  return true;
}

const bool rocm_atomic_reduce_registered = RegisterROCmAtomicReduce();

} // namespace

} // namespace tl
} // namespace tvm

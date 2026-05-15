/*!
 * \file tl/backend/cuda/op/atomic_reduce.cc
 * \brief CUDA implementation for tl.atomicmax/tl.atomicmin lowering.
 */

#include "backend/common/op/atomic_reduce.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchCudaAtomicReduceTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaAtomicReduce() {
  RegisterAtomicReduceImpl(AtomicReduceImpl{
      "cuda.AtomicReduce",
      MatchCudaAtomicReduceTarget,
      backend::AtomicReduce::InferLayout,
      backend::AtomicReduce::Lower,
  });
  return true;
}

const bool cuda_atomic_reduce_registered = RegisterCudaAtomicReduce();

} // namespace

} // namespace tl
} // namespace tvm

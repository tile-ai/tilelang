/*!
 * \file tl/rocm/op/atomic_add.cc
 * \brief ROCm implementation for tl.atomic_add lowering.
 */

#include "op/atomic_add.h"
#include "backend/common/op/atomic_reduce.h"
#include "backend/common/target_utils.h"

namespace tvm::tl {
namespace {

void VerifyROCmAtomicAdd(const AtomicAddNode &operation) {
  auto use_tma = operation.annotations.Get("use_tma");
  TVM_FFI_CHECK(!use_tma || GetIntAnnotation(*use_tma).value_or(0) == 0,
                ValueError)
      << "TMA atomic_add is only supported by the CUDA backend";
}

LayoutMap InferROCmAtomicAdd(const AtomicAddNode &operation,
                             const LayoutInferArgs &arguments,
                             InferLevel level) {
  VerifyROCmAtomicAdd(operation);
  if (operation.src && IsFragmentBuffer(*operation.src) &&
      IsFragmentBuffer(operation.dst)) {
    auto source_layout = arguments.layout_map.Get(*operation.src);
    auto destination_layout = arguments.layout_map.Get(operation.dst);
    TVM_FFI_CHECK(
        !source_layout || !destination_layout ||
            ffi::StructuralEqual()(*source_layout, *destination_layout),
        ValueError)
        << "AtomicAdd requires source and destination fragments to have the "
           "same layout";
  }
  return backend::AtomicReduce::InferLayout(operation, arguments, level);
}

Stmt LowerROCmAtomicAdd(const AtomicAddNode &operation,
                        const LowerArgs &arguments, sym::Analyzer analyzer) {
  VerifyROCmAtomicAdd(operation);
  return backend::AtomicReduce::Lower(operation, arguments, analyzer);
}

bool RegisterROCmAtomicAdd() {
  RegisterAtomicAddImpl(AtomicAddImpl{"rocm.AtomicAdd", TargetIsRocm,
                                      InferROCmAtomicAdd, LowerROCmAtomicAdd});
  return true;
}

const bool rocm_atomic_add_registered = RegisterROCmAtomicAdd();

} // namespace
} // namespace tvm::tl

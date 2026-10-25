#include <tvm/ir/transform.h>

#include "op/builtin.h"

namespace tvm::tl {

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kDebugMergeSharedMemoryAllocations);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kDisableSafeMemoryLegalize);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kDisableThreadStorageSync);
  tvm::transform::PassContext::RegisterConfigOption<int64_t>(
      kConfigIndexBitwidth);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kEnableAggressiveSharedMemoryMerge);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kDisableSharedMemoryReuse);
  tvm::transform::PassContext::RegisterConfigOption<bool>(kForceLetInline);
  tvm::transform::PassContext::RegisterConfigOption<bool>(kEnableFastMath);
  tvm::transform::PassContext::RegisterConfigOption<bool>(kEnableAsyncCopy);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kReducerForceBaseline);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kEnableReducerPlanVerbose);
  tvm::transform::PassContext::RegisterConfigOption<ffi::String>(
      kLayoutCostModel);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kEnableVectorizePlannerVerbose);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kStorageRewriteDetectInplace);
  tvm::transform::PassContext::RegisterConfigOption<bool>(kASTPrintEnable);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kLayoutVisualizationEnable);
  tvm::transform::PassContext::RegisterConfigOption<ffi::String>(
      kLayoutVisualizationFormats);
  tvm::transform::PassContext::RegisterConfigOption<ffi::Array<ffi::String>>(
      kDeviceCompileFlags);
  tvm::transform::PassContext::RegisterConfigOption<bool>(kEmitLineDirectives);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kDisableDataRaceCheck);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kDisableBufferInitCheck);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kDisableLoopUnswitching);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kLoopUnswitchingAllowNonTrivialElse);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kIfStmtBindingInlineReplayableBinds);
  tvm::transform::PassContext::RegisterConfigOption<bool>(
      kDisableOutOfBoundWarning);
  tvm::transform::PassContext::RegisterConfigOption<bool>(kEnableDumpIR);
  tvm::transform::PassContext::RegisterConfigOption<ffi::String>(kDumpIRDir);
  tvm::transform::PassContext::RegisterConfigOption<bool>(kPassProfile);
  tvm::transform::PassContext::RegisterConfigOption<double>(
      kPassProfileThresholdMs);
}

} // namespace tvm::tl

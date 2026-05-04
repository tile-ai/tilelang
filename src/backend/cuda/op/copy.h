/*!
 * \file tl/backend/cuda/op/copy.h
 * \brief CUDA copy instruction classification helpers.
 */

#ifndef TVM_TL_BACKEND_CUDA_OP_COPY_H_
#define TVM_TL_BACKEND_CUDA_OP_COPY_H_

#include "op/copy.h"

#include <cstdint>
#include <string>

namespace tvm {
namespace tl {
namespace cuda {

using namespace tir;

enum class CopyInst : uint8_t {
  kNormal = 0,
  kLDSM = 1,
  kSTSM = 2,
  kBulkLoad = 3,
  kBulkStore = 4,
  kCPAsync = 5,
  kBulkLoad1D = 6,
  kBulkStore1D = 7,
  kTMemLoad = 8,
  kTMemStore = 9,
  kInvalid = 255,
};

const char *CopyInstToString(CopyInst inst);
bool CopyInstIsTMA(CopyInst inst);
bool CopyInstIsCPAsync(CopyInst inst);

struct CopyAnalysisContext {
  Target target;
  const LayoutMap *layout_map = nullptr;
  arith::Analyzer *analyzer = nullptr;
  bool buffer_oob = false;
  bool emit_diagnostics = false;
};

struct CopyInstSelection {
  CopyInst inst = CopyInst::kNormal;
  bool supported = true;
  std::string reason;
};

// Final CUDA lowering decision. Explicit T.tma_copy/T.async_copy semantics are
// enforced here and reported through CopyInstSelection::reason.
CopyInstSelection SelectCopyInstForLowering(const CopyNode &op,
                                            const CopyAnalysisContext &ctx);

// Coarse pre-layout classification used by InstructionAnnotation.
std::string ClassifyCopyForInstructionAnnotation(const CopyNode &op,
                                                 Target target,
                                                 bool in_pipeline);

// Pre-layout producer classification used by warp-specialized scheduling.
CopyInstSelection ClassifyWarpSpecializedProducerCopy(const CopyNode &op,
                                                      Target target);

// Semantic queries used by transform passes that need copy shape/capability
// information without knowing the CUDA lowering policy knobs.
bool IsPipelineManagedCPAsyncCopy(const CopyNode &op, Target target);
bool IsPipelinePlanningTmaCopy(const CopyNode &op, Target target);

} // namespace cuda
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_CUDA_OP_COPY_H_

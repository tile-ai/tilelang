/*!
 * \file tl/cuda/op/copy.h
 * \brief CUDA copy instruction classification helpers.
 */

#ifndef TVM_TL_BACKEND_CUDA_OP_COPY_H_
#define TVM_TL_BACKEND_CUDA_OP_COPY_H_

#include "cuda/op/tma_layout.h"
#include "layout/cute_layout.h"
#include "op/copy.h"
#include "op/operator.h"
#include "support/check.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace tvm {
namespace tl {
namespace cuda {

using namespace tirx;
using namespace ffi;

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
  kBulkLoadGather4 = 10,   // tma cp.async.bulk.tensor.tile::gather4 (sm_100a)
  kBulkStoreScatter4 = 11, // tma cp.async.bulk.tensor.tile::scatter4 (sm_100a)
  kInvalid = 255,
};

const char *CopyInstToString(CopyInst inst);
bool CopyInstIsTMALoad(CopyInst inst);
bool CopyInstIsTMAStore(CopyInst inst);
bool CopyInstIsCPAsync(CopyInst inst);

struct TMADesc {
  size_t rank;
  int data_type;
  Array<PrimExpr> global_shape;
  Array<PrimExpr> global_stride;
  Array<PrimExpr> smem_box;
  Array<PrimExpr> smem_stride;
  PrimExpr global_addr;
  int swizzle;
  int interleave;
  int oob_fill;
  int l2_promotion;

  Array<PrimExpr> EncodeCallArgs() const {
    Array<PrimExpr> args;
    args.reserve(rank * 4 + 7);

    args.push_back(data_type);
    args.push_back(static_cast<int>(rank));
    args.push_back(global_addr);
    for (auto e : global_shape)
      args.push_back(e);
    for (auto e : global_stride)
      args.push_back(e);
    for (auto e : smem_box)
      args.push_back(e);
    for (auto e : smem_stride)
      args.push_back(e);
    args.push_back(interleave);
    args.push_back(swizzle);
    args.push_back(l2_promotion);
    args.push_back(oob_fill);

    return args;
  }
};

// Geometry of a descriptor-based bulk tensor copy between a shared-memory
// tile and a global region, derived with CuTe layout algebra. Every TMADesc
// field is filled except data_type, l2_promotion, oob_fill and interleave,
// which are op policy. The copy issues `rest_size` TMA instructions of
// `box_size` elements each; SharedOffset/TmaCoords give the arguments of
// instruction `rest_idx` (std::nullopt when rest_size == 1).
struct TMABulkCopyPlan {
  TMADesc desc;
  Buffer shared_tensor;          // physical (remapped) shared buffer
  int64_t box_size;              // elements per TMA instruction
  int64_t rest_size;             // TMA instructions per copy
  cute::IntTuple shared_offset;  // physical offset of the tile base
  cute::IntTuple tma_coords;     // global coords of the tile base per TMA mode
  cute::Layout rest_to_smem;     // instruction index -> smem offset step
  cute::Layout rest_to_tma_mode; // instruction index -> TMA coord steps

  PrimExpr SharedOffset(std::optional<PrimExpr> rest_idx) const;
  Array<PrimExpr> TmaCoords(std::optional<PrimExpr> rest_idx) const;

  // One statement per TMA instruction: `make_copy` receives the rest index
  // (std::nullopt when the copy is a single box) and the results replay in an
  // unrolled loop.
  Stmt EmitInstructions(
      const std::function<Stmt(std::optional<PrimExpr>)> &make_copy) const;
};

struct TMABulkCopyAnalysis {
  std::optional<TMABulkCopyPlan> plan;
  std::string reason;
};

// Derive the TMA boxes for copying `global_range` <-> `shared_range` given
// the inferred shared layout, or report why the copy cannot be a bulk TMA.
// `box_dim_caps` overrides the per-global-dim box limit of kTmaMaxBoxDim
// (im2col allows 1024 pixels per column).
TMABulkCopyAnalysis
AnalyzeTMABulkCopy(const LowerArgs &lower_args, const Buffer &global_tensor,
                   Buffer shared_tensor, const Array<Range> &global_range,
                   const Array<Range> &shared_range,
                   const std::vector<int64_t> &box_dim_caps = {});

struct CopyAnalysisContext {
  Target target;
  const LayoutMap *layout_map = nullptr;
  arith::Analyzer *analyzer = nullptr;
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

// Pre-layout classification used by warp-specialized scheduling.
CopyInstSelection ClassifyWarpSpecializedCopy(const CopyNode &op,
                                              Target target);

// Semantic queries used by transform passes that need copy shape/capability
// information without knowing the CUDA lowering policy knobs.
bool IsPipelineManagedCPAsyncCopy(const CopyNode &op, Target target);

} // namespace cuda
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_CUDA_OP_COPY_H_

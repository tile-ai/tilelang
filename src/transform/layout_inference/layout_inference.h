#ifndef TILELANG_TRANSFORM_LAYOUT_INFERENCE_H_
#define TILELANG_TRANSFORM_LAYOUT_INFERENCE_H_

#include "op/operator.h"

#include <tvm/tirx/function.h>

#include <functional>

namespace tvm::tl {

struct LayoutInferenceOptions {
  bool fuse_parallel_loops{true};
  std::function<bool(const RegionStmtNode *)> opaque_region;
  std::function<void(const TileOperator &, sym::Analyzer)> validate_operator;
};

TILELANG_DLL tirx::Function InferLayouts(tirx::Function function,
                                         LayoutInferenceOptions options = {});

} // namespace tvm::tl

#endif

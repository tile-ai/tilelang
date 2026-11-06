#pragma once
#include "transform/common/pipeline_access.h"
#include <functional>
#include <tvm/target/target.h>
#include <tvm/tirx/function.h>
#include <vector>
namespace tvm::tl {
using namespace tirx;
using namespace ffi;
// Analysis-only facts. All IR state remains in native Bind/Call/Region
// operands; this is not an alternative Buffer/Block object model or registry.
struct WSKernelAnalysis {
  SeqStmt body{Array<Stmt>{}};
  PrimExpr threads{ffi::UnsafeInit{}};
  Array<TensorVar> allocations;
  Map<TensorVar, Bind> definitions;
  Array<Bind> views;
  Map<String, Any> annotations;
  std::vector<RegionStmt> wrappers;
  const PipelineMemoryAnalysis *memory{nullptr};
  Location loc{UnknownLoc()};
};

struct WSKernelRewriteResult {
  SeqStmt body;
  Optional<int> threads;
};

using WSKernelRewrite = std::function<Optional<WSKernelRewriteResult>(
    WSKernelAnalysis, PrimVar, Target)>;

SeqStmt RebuildWSKernel(const WSKernelAnalysis &kernel, Stmt body,
                        const Map<TensorVar, TensorVar> &replacements = {},
                        const Array<TensorVar> &extra_allocations = {});
tirx::Function RewriteWSKernels(tirx::Function function,
                                const WSKernelRewrite &rewrite);
} // namespace tvm::tl

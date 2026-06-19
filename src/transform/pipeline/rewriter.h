#ifndef TVM_TL_TRANSFORM_PIPELINE_REWRITER_H_
#define TVM_TL_TRANSFORM_PIPELINE_REWRITER_H_

#include "helpers.h"

namespace tvm {
namespace tl {
namespace software_pipeline {

struct PipelineRewriteResult {
  Stmt pipeline;
  Map<Buffer, Buffer> buffer_remap;
};

PipelineRewriteResult RewritePipeline(
    Map<Var, Buffer> buffer_data_to_buffer,
    const Array<Buffer> &pipeline_allocs, const Array<Buffer> &local_allocs,
    const For &pipeline_loop, const PipelineInfo &pipeline_info,
    const Array<SBlock> &scalar_binding_blocks, Optional<Target> target);

} // namespace software_pipeline
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_PIPELINE_REWRITER_H_

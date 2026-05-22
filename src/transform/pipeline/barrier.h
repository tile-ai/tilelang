#ifndef TVM_TL_TRANSFORM_PIPELINE_BARRIER_H_
#define TVM_TL_TRANSFORM_PIPELINE_BARRIER_H_

#include "helpers.h"

namespace tvm {
namespace tl {
namespace software_pipeline {

using BlockDependencyGraph =
    std::unordered_map<SBlock, Array<SBlock>, ObjectPtrHash, ObjectPtrEqual>;

void BuildDependencyGraph(const Array<SBlock> &blocks,
                          BlockDependencyGraph *dep_src2dst,
                          BlockDependencyGraph *dep_dst2src);

Map<Buffer, Buffer> ExpandPipelineBarriers(
    Array<SBlock> &original_order, PipelineInfo &pipeline_info,
    Map<Var, Buffer> &buffer_data_to_buffer, BufferSet &allocated_buffers,
    Array<Buffer> &block_local_allocs, Array<Buffer> &pipeline_allocs,
    Var loop_var, PrimExpr loop_min, int num_stages);

Buffer RewritePipelineTmaBarriers(
    Array<SBlock> &original_order, PipelineInfo &pipeline_info,
    const Array<Integer> &tma_copies, Map<Var, Buffer> &buffer_data_to_buffer,
    BufferSet &allocated_buffers, Array<Buffer> &block_local_allocs,
    Var loop_var, PrimExpr loop_min, int num_stages);

} // namespace software_pipeline
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_PIPELINE_BARRIER_H_

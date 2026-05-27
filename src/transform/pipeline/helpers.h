#ifndef TVM_TL_TRANSFORM_PIPELINE_HELPERS_H_
#define TVM_TL_TRANSFORM_PIPELINE_HELPERS_H_

#include <tvm/target/target.h>
#include <tvm/tirx/stmt.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
namespace software_pipeline {

using namespace tirx;
using namespace ffi;

using BufferSet = std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;
using BufferMap =
    std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>;
using BufferShapeMap =
    std::unordered_map<Buffer, PrimExpr, ObjectPtrHash, ObjectPtrEqual>;
using BufferCommitGroupMap =
    std::unordered_map<Buffer, int, ObjectPtrHash, ObjectPtrEqual>;

/*! Structure that represents the provided annotation per block or loop. */
struct PipelineAnnotation {
  int stage;
  int order;
  bool async{false};
  int async_group_id{-1};
};

using PipelineInfo = std::unordered_map<SBlock, PipelineAnnotation,
                                        ObjectPtrHash, ObjectPtrEqual>;

struct BufferAccessInfo {
  int def = -1; // the defining stage of the buffer
  int use = -1; // the last using stage of the buffer
};

bool IsReplayableScalarBindBlock(const SBlock &block,
                                 const BufferSet &pipeline_write_buffers);

BufferSet CollectPipelineWriteBuffers(const Array<SBlock> &blocks);

bool UpdateExpandedLayoutMapForRemappedAllocs(
    const std::vector<std::pair<Buffer, Buffer>> &remapped_allocs,
    Map<String, Any> *annotations);

Array<Buffer>
CollectUsedPipelineBuffers(const Stmt &stmt,
                           const Map<Var, Buffer> &buffer_data_to_buffer,
                           const BufferSet &allocated_buffers);

SBlock MakeBlock(const Stmt &body,
                 const Map<Var, Buffer> &buffer_data_to_buffer);

bool ContainsPipelineAsyncControlAttrs(const Stmt &stmt);

Stmt AnnotateSimtProducer(const Stmt &stmt,
                          Optional<Target> target = Optional<Target>());

Stmt AnnotateTileOpMbarPhase(const Stmt &stmt, PrimExpr phase_expr);

Stmt LowerAsyncCommitWaitAttrs(const Stmt &stmt);

} // namespace software_pipeline
} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_PIPELINE_HELPERS_H_

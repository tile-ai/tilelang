/*!
 * \file attr.h
 * \brief Check attributes of the IR
 */

#ifndef TVM_TL_TRANSFORM_COMMON_ATTR_H_
#define TVM_TL_TRANSFORM_COMMON_ATTR_H_

#include <string>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

constexpr const char *HostMainBlockName = "root";

constexpr const char *DeviceMainBlockName = "tilelang_root";

inline bool IsHostMainBlock(const tirx::SBlockNode *node) {
  return node->name_hint == HostMainBlockName;
}

inline bool IsDeviceMainBlock(const tirx::SBlockNode *node) {
  return node->name_hint == DeviceMainBlockName;
}

namespace attr {
// Attributes to mark CUDA sync calls
constexpr const char *kHasTriggerLaunch = "has_cuda_pdl_trigger";
constexpr const char *kHasGridSync = "has_cuda_pdl_sync";

// TileLang-only AttrStmt keys.
constexpr const char *volatile_scope = "volatile_scope";
constexpr const char *coproc_scope = "coproc_scope";
constexpr const char *pipeline_exec_scope = "pipeline_exec_scope";
// Marks user-authored assumptions that require a host runtime check. The
// corresponding tl.assume remains in the IR as an optimizer fact.
constexpr const char *kAssumeRequiresRuntimeCheck =
    "tl.assume_requires_runtime_check";

// Attributes to implement SourceCodeBlock
constexpr const char *kCodeBlockSource = "code_block_source";
constexpr const char *kCodeBlockEntryName = "code_block_entry_name";

// Marks a grid (blockIdx) loop lowered by MaterializeKernelLaunch on a CPU
// target, valued by its grid dimension index. Consumed (and removed) by
// MaterializeCPUParallelGrid late in the CPU pipeline to convert the loop
// nest to ForKind::kParallel. Only added when the tl.cpu_parallel pass
// config is enabled.
constexpr const char *kCPUGridDim = "tl.cpu_grid_dim";

// Requested OpenMP thread count for the CPU grid parallel region, stamped
// by T.Kernel(cpu_num_threads=...) on the outermost grid loop and carried
// through the pipeline as a loop annotation. Consumed by the C codegen as
// the num_threads(n) clause; absent means dynamic (the OpenMP runtime picks
// the thread count, e.g. from OMP_NUM_THREADS).
constexpr const char *kCPUNumThreads = "tl.cpu_num_threads";

/*!
 * \brief Check if attr_key is a code block key extension
 * \param attr_key The attr key to be compared
 * \return true if it is a code block key
 */
inline bool IsCodeBlockKey(const std::string &attr_key) {
  return attr_key.compare(0, 11, "code_block_") == 0;
}

} // namespace attr

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_ATTR_H_

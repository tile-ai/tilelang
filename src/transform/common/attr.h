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

constexpr const char *tilelang_is_cpu_kernel_frame =
    "tilelang.is_cpu_kernel_frame";

namespace attr {
// Attributes to mark CUDA sync calls
constexpr const char *kHasTriggerLaunch = "has_cuda_pdl_trigger";
constexpr const char *kHasGridSync = "has_cuda_pdl_sync";

// TileLang-only AttrStmt keys.
constexpr const char *volatile_scope = "volatile_scope";
constexpr const char *coproc_scope = "coproc_scope";
constexpr const char *pipeline_exec_scope = "pipeline_exec_scope";

// Attributes to implement SourceCodeBlock
constexpr const char *kCodeBlockSource = "code_block_source";
constexpr const char *kCodeBlockEntryName = "code_block_entry_name";

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

/*!
 * \file attr.h
 * \brief Check attributes of the IR
 */

#include "tvm/tir/stmt.h"

namespace tvm {
namespace tl {

constexpr const char *HostMainBlockName = "root";

constexpr const char *DeviceMainBlockName = "tilelang_root";

inline bool IsHostMainBlock(const tir::BlockNode *node) {
  return node->name_hint == HostMainBlockName;
}

inline bool IsDeviceMainBlock(const tir::BlockNode *node) {
  return node->name_hint == DeviceMainBlockName;
}

constexpr const char *tilelang_is_cpu_kernel_frame =
    "tilelang.is_cpu_kernel_frame";

namespace attr {
// Attributes to mark CUDA sync calls
constexpr const char *kHasTriggerLaunch = "has_cuda_pdl_trigger";
constexpr const char *kHasGridSync = "has_cuda_pdl_sync";
} // namespace attr

} // namespace tl
} // namespace tvm

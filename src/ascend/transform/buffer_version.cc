/*!
 * \file tl/ascend/transform/buffer_version.cc
 * \brief LowerTileOp remap hook for buffer version metadata.
 *
 * LowerTileOp rewrites buffer data vars (e.g. when remapping fragment
 * buffers), which invalidates the Var keys of `tl.buffer_version` maps
 * produced by the Ascend multi-buffer passes. Registering this hook keeps
 * the metadata consistent without the shared pass depending on Ascend
 * headers.
 *
 * The map deliberately keys by Var identity rather than name: distinct
 * versioned storages may share a name within one function (see
 * test_auto_schedule_distinguishes_same_name_buffer_version_annotations),
 * so names cannot serve as stable keys.
 */

#include "ascend/transform/buffer_version.h"

#include "support/check.h"

#include "transform/common/attr.h"
#include "transform/common/lower_hooks.h"

namespace tvm {
namespace tl {

namespace {

ffi::Any RemapBufferVersionNode(
    const ffi::Any &node,
    const std::function<tirx::Var(const tirx::Var &)> &remap) {
  auto versions = node.try_cast<BufferVersionMap>();
  ICHECK(versions.has_value())
      << "'" << attr::kBufferVersion
      << "' AttrStmt node must be a buffer version map";
  BufferVersionMap remapped_versions;
  for (const auto &[data, version] : versions.value()) {
    tirx::Var remapped_data = remap(data);
    auto existing = remapped_versions.find(remapped_data);
    ICHECK(existing == remapped_versions.end() ||
           (*existing).second == version)
        << "Conflicting buffer version counts after storage remapping for "
        << remapped_data;
    remapped_versions.Set(std::move(remapped_data), version);
  }
  return remapped_versions;
}

bool RegisterBufferVersionRemapHook() {
  RegisterAttrNodeRemapHook(attr::kBufferVersion, RemapBufferVersionNode);
  return true;
}

const bool buffer_version_remap_registered = RegisterBufferVersionRemapHook();

} // namespace

} // namespace tl
} // namespace tvm

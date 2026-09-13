/*!
 * \file tl/op/region_op.cc
 * \brief Registry storage for region operators.
 */

#include "region_op.h"

#include "support/check.h"

#include <vector>

namespace tvm {
namespace tl {

namespace {

std::vector<RegionOpImpl> &RegionOpRegistry() {
  static std::vector<RegionOpImpl> registry;
  return registry;
}

} // namespace

void RegisterRegionOpImpl(RegionOpImpl impl) {
  ICHECK(impl.block_name != nullptr);
  ICHECK(impl.make != nullptr);
  for (const RegionOpImpl &existing : RegionOpRegistry()) {
    ICHECK(std::string(existing.block_name) != impl.block_name)
        << "Region op already registered for block " << impl.block_name;
  }
  RegionOpRegistry().push_back(impl);
}

const RegionOpImpl *LookupRegionOpImpl(const std::string &block_name) {
  for (const RegionOpImpl &impl : RegionOpRegistry()) {
    if (block_name == impl.block_name) {
      return &impl;
    }
  }
  return nullptr;
}

} // namespace tl
} // namespace tvm

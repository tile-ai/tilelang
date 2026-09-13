/*!
 * \file tl/transform/common/lower_hooks.cc
 * \brief Registry storage for LowerTileOp backend hooks.
 */

#include "lower_hooks.h"

#include "support/check.h"

#include <unordered_map>

namespace tvm {
namespace tl {

namespace {

std::vector<LoweredParallelLoopHook> &ParallelLoopHookRegistry() {
  static std::vector<LoweredParallelLoopHook> registry;
  return registry;
}

std::unordered_map<std::string, AttrNodeRemapHook> &AttrRemapHookRegistry() {
  static std::unordered_map<std::string, AttrNodeRemapHook> registry;
  return registry;
}

} // namespace

void RegisterLoweredParallelLoopHook(LoweredParallelLoopHook hook) {
  ICHECK(hook != nullptr);
  ParallelLoopHookRegistry().push_back(hook);
}

const std::vector<LoweredParallelLoopHook> &LoweredParallelLoopHooks() {
  return ParallelLoopHookRegistry();
}

void RegisterAttrNodeRemapHook(const std::string &attr_key,
                               AttrNodeRemapHook hook) {
  ICHECK(hook != nullptr);
  auto [it, inserted] = AttrRemapHookRegistry().emplace(attr_key, hook);
  ICHECK(inserted) << "Attr remap hook already registered for key "
                   << attr_key;
}

AttrNodeRemapHook LookupAttrNodeRemapHook(const std::string &attr_key) {
  const auto &registry = AttrRemapHookRegistry();
  auto it = registry.find(attr_key);
  return it == registry.end() ? nullptr : it->second;
}

} // namespace tl
} // namespace tvm

#include "role_based_scheduler.h"
#include "transform/common/memory_effects.h"
#include "transform/common/read_only_params.h"

#include <tvm/ffi/function.h>

namespace tvm::tl::cuda {
ffi::Array<Var> ScheduleReadOnlyStorage(const tirx::Function &function) {
  auto alias_function =
      WithAttr(function, tirx::attr::kNoAlias,
               function->GetAttr<bool>(tirx::attr::kNoAlias).value_or(true));
  analysis::AliasOrigins aliases(alias_function);
  auto candidates = ReadOnlyParameterIndices(function);
  std::unordered_set<size_t> positions;
  for (const IntImm &index : candidates)
    positions.insert(index->value.as<size_t>().value());
  analysis::Roots writes;
  for (size_t index = 0; index < function->params.size(); ++index)
    if (!positions.count(index) &&
        analysis::IsAddress(function->params[index])) {
      auto roots = aliases.Get(function->params[index]);
      writes.insert(roots.begin(), roots.end());
    }
  ffi::Array<Var> readonly;
  for (const IntImm &index : candidates) {
    Var parameter = function->params[index->value.as<size_t>().value()];
    if (!aliases.MayAlias(aliases.Get(parameter), writes))
      readonly.push_back(parameter);
  }
  return readonly;
}

ffi::Optional<WSSchedule> RoleBasedSchedule(const RegionStmt &block,
                                            const Stmt &body,
                                            int worker_threads,
                                            const Target &target) {
  auto scheduler = ffi::Function::GetGlobal("tl.cuda.native.RoleBasedSchedule");
  TVM_FFI_CHECK(scheduler, RuntimeError)
      << "Import tilelang to register the native warp scheduler";
  return (*scheduler)(block, body, worker_threads, target)
      .cast<ffi::Optional<WSSchedule>>();
}
} // namespace tvm::tl::cuda

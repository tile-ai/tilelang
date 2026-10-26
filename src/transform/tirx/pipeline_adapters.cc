#include "transform/tirx_passes.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl::tirx_passes {

tvm::transform::Pass BindTarget(Target target) {
  auto bind = [target](IRModule module, tvm::transform::PassContext) {
    ffi::Map<GlobalVar, Target> original_targets;
    for (const auto &[global, function] : module->functions) {
      auto original = function->GetAttr(tvm::attr::kTarget);
      if (original && !original->GetHost() &&
          function->GetAttr(tvm::attr::kCallingConv)
                  .value_or(CallingConv::kDefault) != CallingConv::kDefault) {
        original_targets.Set(global, *original);
      }
    }
    module = tirx::transform::BindTarget(target)(std::move(module));
    for (const auto &[global, original] : original_targets) {
      auto function = module->functions.at(global);
      module.CopyOnWrite()->Update(
          global, WithAttr(function, tvm::attr::kTarget.name, original));
    }
    return module;
  };
  return tvm::transform::CreateModulePass(bind, 0, "tl.BindTarget");
}

tvm::transform::Pass AnnotateEntryFunc() {
  auto annotate = [](IRModule module, tvm::transform::PassContext) {
    ffi::Map<GlobalVar, BaseFunc> host_functions;
    for (const auto &[global, function] : module->functions) {
      if (function->GetAttr(tvm::attr::kCallingConv)
              .value_or(CallingConv::kDefault) !=
          CallingConv::kDeviceKernelLaunch) {
        host_functions.Set(global, function);
      }
    }
    if (host_functions.empty())
      return module;
    IRModule host = module;
    host.CopyOnWrite()->functions = std::move(host_functions);
    host = tirx::transform::AnnotateEntryFunc()(std::move(host));
    module.CopyOnWrite()->Update(host);
    return module;
  };
  return tvm::transform::CreateModulePass(annotate, 0, "tl.AnnotateEntryFunc");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl._tirx_passes.BindTarget", BindTarget)
      .def("tl._tirx_passes.AnnotateEntryFunc", AnnotateEntryFunc);
}

} // namespace tvm::tl::tirx_passes

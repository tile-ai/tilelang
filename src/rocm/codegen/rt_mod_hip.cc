#include "transform/tirx_passes.h"

#include "backend/common/codegen/native_codegen.h"
#include "codegen_hip.h"
#include "config.h"
#include "tirx_compat/build_common.h"
#include "tirx_compat/rocm_fallback_module.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/support/env.h>
#include <tvm/tirx/transform.h>

namespace tvm::codegen {
namespace {

ffi::Module BuildHIPSourceModule(IRModule module, ffi::String source,
                                 ffi::Optional<ffi::Bytes> binary) {
  auto functions = ExtractFuncInfo(module);
  ffi::Map<ffi::String, ffi::String> sources{{"hip", source}};
  if (binary) {
    TVM_FFI_CHECK(binary.value().size() != 0, ValueError)
        << "ROCm compilation returned an empty binary";
    if (!support::GetEnv<bool>("TVM_TIRX_COMPILE_FORCE_FALLBACK", false)) {
      if (auto create = ffi::Function::GetGlobal("ffi.Module.create.rocm")) {
        return (*create)(*binary, "hsaco", functions, sources)
            .cast<ffi::Module>();
      }
    }
  }
  return target::ROCmFallbackModuleCreate(binary.value_or(ffi::Bytes()),
                                          "hsaco", std::move(functions),
                                          std::move(sources));
}

ffi::Module BuildTileLangHIP(IRModule module, Target target, bool source_only) {
  module = tl::PrepareNativeCodegen(std::move(module), target);
  module = tvm::tl::tirx_passes::StmtSimplify()(std::move(module));
  CodeGenTileLangHIP generator;
  generator.Init(false);
  generator.ConfigureModule(module);
  generator.SetTarget(target);
  generator.SetEmitLineDirectives(tl::tl_config::EmitLineDirectivesEnabled());
  for (const auto &[global, base_function] : module->functions)
    generator.DeclareFunction(global,
                              base_function.as_or_throw<tirx::Function>());
  for (const auto &[global, base_function] : module->functions)
    generator.AddFunction(global, base_function.as_or_throw<tirx::Function>());
  ffi::String source = generator.Finish();
  if (auto postprocess =
          ffi::Function::GetGlobal("tilelang_callback_hip_postproc"))
    source = (*postprocess)(source, target).cast<ffi::String>();
  ffi::Optional<ffi::Bytes> binary;
  if (!source_only) {
    auto compiled = ffi::Function::GetGlobalRequired(
        "tilelang_callback_hip_compile")(source, target);
    if (auto bytes = compiled.as<ffi::Bytes>()) {
      binary = *bytes;
    } else {
      auto text = compiled.cast<ffi::String>();
      binary = ffi::Bytes(text.data(), text.size());
    }
  }
  return BuildHIPSourceModule(std::move(module), std::move(source),
                              std::move(binary));
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.codegen.rocm_from_source", BuildHIPSourceModule)
      .def("target.build.tilelang_hip",
           [](IRModule module, Target target) {
             return BuildTileLangHIP(std::move(module), std::move(target),
                                     false);
           })
      .def("target.build.tilelang_hip_without_compile", [](IRModule module,
                                                           Target target) {
        return BuildTileLangHIP(std::move(module), std::move(target), true);
      });
}

} // namespace tvm::codegen

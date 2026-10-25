#include "backend/common/codegen/native_codegen.h"
#include "codegen_hip.h"
#include "config.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

namespace tvm::codegen {
namespace {

ffi::Module BuildTileLangHIP(IRModule module, Target target, bool source_only) {
  module = tl::PrepareNativeCodegen(std::move(module), target);
  module = tirx::transform::StmtSimplify()(std::move(module));
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
  return ffi::Function::GetGlobalRequired("target.build.rocm_from_source")(
             module, source, binary)
      .cast<ffi::Module>();
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
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

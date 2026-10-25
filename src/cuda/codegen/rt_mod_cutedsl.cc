#include "codegen_cutedsl.h"
#include <tvm/ffi/reflection/registry.h>

namespace tvm::codegen {

ffi::Module BuildTileLangCuTeDSLWithoutCompile(IRModule module, Target target) {
  TVM_FFI_CHECK(target->kind->name == "cuda", ValueError)
      << "CuTeDSL requires a CUDA target";
  CodeGenTileLangCuTeDSL generator;
  for (const auto &[global, base] : module->functions) {
    auto function = base.as_or_throw<tirx::Function>();
    generator.AddFunction(global, function);
  }
  std::string source = generator.Finish();
  if (auto postprocess =
          ffi::Function::GetGlobal("tilelang_callback_cutedsl_postproc")) {
    source = (*postprocess)(source, target).cast<std::string>();
  }
  return SourceModuleCreate(source, "py");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "target.build.tilelang_cutedsl_without_compile",
      BuildTileLangCuTeDSLWithoutCompile);
}

} // namespace tvm::codegen

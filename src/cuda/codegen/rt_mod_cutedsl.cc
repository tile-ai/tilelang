#include "backend/cuda/codegen/cuda_fallback_module.h"
#include "codegen/build_common.h"
#include "codegen_cutedsl.h"

namespace tvm::codegen {
using namespace ffi;

Module BuildTileLangCuTeDSLWithoutCompile(IRModule mod, Target target) {
  With<Target> target_scope(target);
  CodeGenTileLangCuTeDSL codegen;
  for (const auto &[gvar, base] : mod->functions) {
    const auto function = base.as_or_throw<tirx::Function>();
    TVM_FFI_CHECK(
        function->GetAttr(tvm::attr::kCallingConv, CallingConv::kDefault)
                .value() == CallingConv::kDeviceKernelLaunch,
        ValueError)
        << "CuTeDSL build requires a device Function after host/device split";
    TVM_FFI_CHECK(function->GetAttr(tvm::attr::kGlobalSymbol).has_value(),
                  ValueError)
        << "CuTeDSL device function requires global_symbol";
    codegen.AddFunction(gvar, function);
  }
  std::string code = codegen.Finish();
  if (auto callback =
          ffi::Function::GetGlobal("tilelang_callback_cutedsl_postproc"))
    code = (*callback)(code, target).cast<std::string>();
  // A source-only module must never invoke the CUDA driver with dummy PTX.
  Map<String, String> sources{{"cuda", code}, {"cutedsl", code}};
  if (auto config = mod->GetAttr<String>("backend_config"))
    sources.Set("backend_config", config.value());
  return target::CUDAFallbackModuleCreate(Bytes(code.data(), code.size()),
                                          "cutedsl", ExtractFuncInfo(mod),
                                          sources);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("target.build.tilelang_cutedsl_without_compile",
                              BuildTileLangCuTeDSLWithoutCompile);
}
} // namespace tvm::codegen

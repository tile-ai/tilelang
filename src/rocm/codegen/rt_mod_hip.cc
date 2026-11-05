/*!
 * \file rocm/codegen/rt_mod_hip.cc
 * \brief Native HIP source generation and runtime module construction.
 */
#include "codegen_hip.h"

#include <unordered_set>

#include "backend/rocm/codegen/rocm_fallback_module.h"
#include "codegen/build_common.h"

namespace tvm::codegen {
using namespace ffi;

namespace {
std::string EmitTileLangHIP(const IRModule &mod, const Target &target) {
  With<Target> target_scope(target);
  CodeGenTileLangHIP codegen;
  codegen.Init(false);
  codegen.SetTarget(target);
  codegen.SetEmitLineDirectives(TileLangEmitLineDirectivesEnabled());
  std::unordered_set<std::string> symbols;
  for (const auto &[gvar, base] : mod->functions) {
    auto func = base.as_or_throw<tirx::Function>();
    auto symbol = func->GetAttr(tvm::attr::kGlobalSymbol);
    TVM_FFI_CHECK(symbol.has_value(), ValueError)
        << "HIP kernel requires global_symbol";
    TVM_FFI_CHECK(symbols.insert(symbol.value()).second, ValueError)
        << "Duplicate HIP kernel global_symbol: " << symbol.value();
    TVM_FFI_CHECK(
        func->GetAttr(tvm::attr::kCallingConv, CallingConv::kDefault).value() ==
            CallingConv::kDeviceKernelLaunch,
        ValueError)
        << "TileLang HIP build requires device functions after host/device "
           "split";
    codegen.AddFunction(gvar, func);
  }
  std::string source = codegen.Finish();
  if (auto postproc =
          ffi::Function::GetGlobal("tilelang_callback_hip_postproc"))
    source = (*postproc)(source, target).cast<std::string>();
  return source;
}

Map<String, runtime::FunctionInfo>
TileLangHIPFunctionInfo(const IRModule &mod) {
  auto infos = ExtractFuncInfo(mod);
  for (const auto &[gvar, base] : mod->functions) {
    auto func = base.as_or_throw<tirx::Function>();
    if (!func->HasNonzeroAttr("use_cooperative_groups"))
      continue;
    auto name = func->GetAttr(tvm::attr::kGlobalSymbol).value();
    const auto &info = infos.at(name);
    Array<String> tags{runtime::launch_param::kUseCooperativeLaunch};
    for (const String &tag : info->launch_param_tags)
      if (tag != runtime::launch_param::kUseCooperativeLaunch)
        tags.push_back(tag);
    infos.Set(name, runtime::FunctionInfo(
                        info->name, info->arg_types, tags, info->arg_extra_tags,
                        info->cuda_launch_fields, info->cuda_kernel_attrs));
  }
  return infos;
}
} // namespace

Module BuildTileLangHIP(IRModule mod, Target target) {
  std::string source = EmitTileLangHIP(mod, target);
  auto compile =
      ffi::Function::GetGlobalRequired("tilelang_callback_hip_compile");
  Any compiled = compile(source, target);
  Bytes binary = [&]() {
    if (auto bytes = compiled.as<Bytes>())
      return bytes.value();
    auto string = compiled.cast<String>();
    return Bytes(string.data(), string.size());
  }();
  TVM_FFI_CHECK_GT(binary.size(), 0, ValueError)
      << "HIP compiler returned an empty kernel binary";
  return target::ROCmModuleCreateWithFallback(
      binary, "hsaco", TileLangHIPFunctionInfo(mod), {{"hip", source}});
}

Module BuildTileLangHIPWithoutCompile(IRModule mod, Target target) {
  std::string source = EmitTileLangHIP(mod, target);
  // Source-only JIT consumers must work without a HIP SDK or runtime sidecar,
  // and must not cause an implicit compilation through runtime registration.
  return target::ROCmFallbackModuleCreate(Bytes(source.data(), source.size()),
                                          "hip", TileLangHIPFunctionInfo(mod),
                                          {{"hip", source}});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::transform::PassContext::RegisterConfigOption<String>(
      "hip.kernels_output_dir");
  reflection::GlobalDef()
      .def("target.build.tilelang_hip", BuildTileLangHIP)
      .def("target.build.tilelang_hip_without_compile",
           BuildTileLangHIPWithoutCompile);
}
} // namespace tvm::codegen

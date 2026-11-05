#include "codegen_cuda.h"

#include <tvm/ir/transform.h>

#include <bit>
#include <string>
#include <unordered_map>
#include <utility>

#include "backend/cuda/codegen/cuda_fallback_module.h"
#include "codegen/build_common.h"
#include "support/check.h"

namespace tvm::codegen {

using namespace ffi;

namespace {

void ValidateDeviceFunctions(const IRModule &mod) {
  std::unordered_map<std::string, std::string> symbols;
  for (const auto &[gvar, base] : mod->functions) {
    const auto func = base.as_or_throw<tirx::Function>();
    auto symbol = func->GetAttr(tvm::attr::kGlobalSymbol);
    TVM_FFI_CHECK(symbol.has_value(), ValueError)
        << "TileLang CUDA kernel " << gvar->name_hint
        << " requires global_symbol";
    auto [previous, inserted] =
        symbols.emplace(symbol.value(), gvar->name_hint);
    TVM_FFI_CHECK(inserted, ValueError)
        << "Duplicate CUDA kernel global_symbol `" << symbol.value()
        << "` on functions `" << previous->second << "` and `"
        << gvar->name_hint
        << "`; raw CUDA source entry names cannot be renamed";
    TVM_FFI_CHECK(
        func->GetAttr(tvm::attr::kCallingConv, CallingConv::kDefault).value() ==
            CallingConv::kDeviceKernelLaunch,
        ValueError)
        << "TileLang CUDA build requires device kernels after host/device "
           "splitting";
  }
}

std::string EmitTileLangCUDA(const IRModule &mod, const Target &target) {
  With<Target> target_scope(target);
  ValidateDeviceFunctions(mod);
  if (auto callback =
          ffi::Function::GetGlobal("tilelang_callback_cuda_validate"))
    (*callback)(mod);
  CodeGenTileLangCUDA codegen;
  codegen.Init(false);
  codegen.SetEmitLineDirectives(TileLangEmitLineDirectivesEnabled());
  for (const auto &[gvar, base] : mod->functions)
    codegen.AddFunction(gvar, base.as_or_throw<tirx::Function>());
  std::string code = codegen.Finish();
  if (auto callback =
          ffi::Function::GetGlobal("tilelang_callback_cuda_postproc"))
    code = (*callback)(code, target).cast<std::string>();
  return code;
}

Map<String, runtime::FunctionInfo> TileLangFunctionInfo(const IRModule &mod) {
  // Native extraction carries TensorMap argument tags and the complete launch
  // field schema. Kernel source, packed calls and runtime must share this ABI.
  auto infos = ExtractFuncInfo(mod);
  for (const auto &[gvar, base] : mod->functions) {
    const auto func = base.as_or_throw<tirx::Function>();
    auto symbol = func->GetAttr(tvm::attr::kGlobalSymbol).value();
    auto info = infos.at(symbol);
    if (func->GetAttr<String>("code_block_source")) {
      auto argument_types = info->arg_types;
      bool has_boolean = false;
      for (size_t index = 0; index < argument_types.size(); ++index) {
        if (argument_types[index].code != kDLBool)
          continue;
        TVM_FFI_CHECK_EQ(argument_types[index].lanes, 1, ValueError)
            << "Raw CUDA boolean arguments must be scalar";
        // Raw CUDA text retains its bool8 parameter. CUDA's void** launcher
        // reads that parameter's one byte from a normalized 0/1 int32 holder,
        // matching the established raw-source ABI on little-endian hosts.
        // Generated kernels have real int32 parameters before this stage.
        TVM_FFI_CHECK(std::endian::native == std::endian::little, ValueError)
            << "Raw CUDA bool argument holders require little-endian hosts";
        argument_types.Set(index, DLDataType{kDLInt, 32, 1});
        has_boolean = true;
      }
      if (has_boolean) {
        info = runtime::FunctionInfo(
            info->name, argument_types, info->launch_param_tags,
            info->arg_extra_tags, info->cuda_launch_fields,
            info->cuda_kernel_attrs);
        infos.Set(symbol, info);
      }
    }
    if (!info->cuda_launch_fields.empty())
      continue;
    Array<String> tags;
    auto append_flag = [&](const char *key, const char *tag) {
      if (!func->HasNonzeroAttr(key))
        return;
      for (const auto &existing : info->launch_param_tags)
        if (existing == tag)
          return;
      tags.push_back(tag);
    };
    append_flag("has_cuda_pdl_sync",
                runtime::launch_param::kUseProgramaticDependentLaunch);
    append_flag("use_cooperative_groups",
                runtime::launch_param::kUseCooperativeLaunch);
    for (const auto &tag : info->launch_param_tags)
      tags.push_back(tag);
    infos.Set(symbol, runtime::FunctionInfo(info->name, info->arg_types, tags,
                                            info->arg_extra_tags,
                                            info->cuda_launch_fields,
                                            info->cuda_kernel_attrs));
  }
  return infos;
}

Map<String, String> SourceMap(const IRModule &mod, const std::string &code) {
  Map<String, String> source{{"cuda", code}};
  if (auto config = mod->GetAttr<String>("backend_config"))
    source.Set("backend_config", config.value());
  return source;
}

} // namespace

Module BuildTileLangCUDA(IRModule mod, Target target) {
  std::string code = EmitTileLangCUDA(mod, target);
  auto compile =
      ffi::Function::GetGlobalRequired("tilelang_callback_cuda_compile");
  auto result =
      compile(code, target, tvm::transform::PassContext::Current()->config);
  Bytes binary = [&]() -> Bytes {
    if (auto bytes = result.as<Bytes>())
      return bytes.value();
    const auto text = result.cast<String>();
    return Bytes(text.data(), text.size());
  }();
  TVM_FFI_CHECK_GT(binary.size(), 0, ValueError)
      << "CUDA compiler returned an empty kernel binary";
  // TileLang's callback returns cubin bytes or NVCC's textual PTX.
  String format = binary.data()[0] == '/' ? "ptx" : "cubin";
  return target::CUDAModuleCreateWithFallback(
      binary, format, TileLangFunctionInfo(mod), SourceMap(mod, code));
}

Module BuildTileLangCUDAWithoutCompile(IRModule mod, Target target) {
  std::string code = EmitTileLangCUDA(mod, target);
  // This entry is used by external JIT adapters. Preserve real source in a
  // serializable CUDA module; neither runtime availability nor import order
  // may turn a source-only request into implicit device compilation.
  return target::CUDAFallbackModuleCreate(Bytes(code.data(), code.size()),
                                          "cuda", TileLangFunctionInfo(mod),
                                          SourceMap(mod, code));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::transform::PassContext::RegisterConfigOption<String>(
      "cuda.kernels_output_dir");
  reflection::GlobalDef()
      .def("target.build.tilelang_cuda", BuildTileLangCUDA)
      .def("target.build.tilelang_cuda_without_compile",
           BuildTileLangCUDAWithoutCompile);
}

} // namespace tvm::codegen

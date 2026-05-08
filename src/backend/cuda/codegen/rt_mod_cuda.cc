#include "codegen_cuda.h"
#include "runtime/cuda/cuda_module.h"
#include "runtime/meta_data.h"
#include "runtime/pack_args.h"
#include "transform/common/attr.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

namespace tvm {
namespace codegen {

static std::string GetDeviceGlobalSymbol(const GlobalVar &gvar,
                                         const tir::PrimFunc &f) {
  if (auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol)) {
    return static_cast<std::string>(global_symbol.value());
  }
  return gvar->name_hint;
}

static void ValidateUniqueDeviceGlobalSymbols(const IRModule &mod) {
  std::unordered_map<std::string, std::string> symbol_to_gvar;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tir::PrimFunc>(kv.second);
    std::string global_symbol = GetDeviceGlobalSymbol(gvar, f);

    auto [it, inserted] =
        symbol_to_gvar.emplace(global_symbol, gvar->name_hint);
    ICHECK(inserted)
        << "Duplicate CUDA kernel global_symbol `" << global_symbol
        << "` found on PrimFuncs `" << it->second << "` and `"
        << gvar->name_hint
        << "`. T.CUDASourceCodeKernel emits raw CUDA source without "
           "renaming, so CUDA entry names must be unique within the compiled "
           "module.";
  }
}

static std::unordered_map<std::string, runtime::FunctionInfo>
ExtractFuncInfo(const IRModule &mod) {
  std::unordered_map<std::string, runtime::FunctionInfo> fmap;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tir::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<tir::PrimFunc>(kv.second);

    runtime::FunctionInfo info;
    for (size_t i = 0; i < f->params.size(); ++i) {
      if (f->params[i]->dtype.is_handle()) {
        auto ptr = f->params[i]->type_annotation.as<PointerTypeNode>();
        if (ptr && ptr->storage_scope == "grid_constant") {
          info.arg_types.push_back(DataType(runtime::kDLGridConstant, 64, 1));
          continue;
        }
      }
      DataType dtype = f->params[i].dtype();
      // Device runtime cannot directly take bool arguments, map to int32.
      if (dtype.is_bool())
        dtype = DataType::Int(32);
      info.arg_types.push_back(dtype);
    }
    if (f->HasNonzeroAttr(tl::attr::kHasGridSync)) {
      info.launch_param_tags.push_back(
          runtime::launch_param::kUseProgramaticDependentLaunch);
    }
    if (f->HasNonzeroAttr("use_cooperative_groups")) {
      info.launch_param_tags.push_back(
          runtime::launch_param::kUseCooperativeLaunch);
    }
    if (f->GetAttr<ffi::Array<Integer>>("cluster_dims").defined()) {
      info.launch_param_tags.push_back(runtime::launch_param::kClusterDimX);
      info.launch_param_tags.push_back(runtime::launch_param::kClusterDimY);
      info.launch_param_tags.push_back(runtime::launch_param::kClusterDimZ);
    }
    if (auto opt = f->GetAttr<ffi::Array<ffi::String>>(
            tir::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        if (tag != runtime::launch_param::kClusterDimX &&
            tag != runtime::launch_param::kClusterDimY &&
            tag != runtime::launch_param::kClusterDimZ) {
          info.launch_param_tags.push_back(tag);
        }
      }
    }
    fmap[GetDeviceGlobalSymbol(Downcast<GlobalVar>(kv.first), f)] = info;
  }
  return fmap;
}

ffi::Module BuildTileLangCUDA(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangCUDA cg;
  cg.Init(output_ssa);

  ValidateUniqueDeviceGlobalSymbols(mod);
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cuda_validate")) {
    (*f)(mod);
  }

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangCUDA: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cuda_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  std::string fmt = "ptx";
  std::string ptx;
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cuda_compile")) {
    // Fetch current pass context config and pass into the compile callback
    tvm::transform::PassContext pass_ctx =
        tvm::transform::PassContext::Current();
    ptx = (*f)(code, target, pass_ctx->config).cast<std::string>();
    if (ptx[0] != '/')
      fmt = "cubin";
  } else {
    ICHECK(0);
  }
  return runtime::CUDAModuleCreate(ptx, fmt, ExtractFuncInfo(mod), code);
}

ffi::Module BuildTileLangCUDAWithoutCompile(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangCUDA cg;
  cg.Init(output_ssa);

  ValidateUniqueDeviceGlobalSymbols(mod);
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cuda_validate")) {
    (*f)(mod);
  }

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangCUDA: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_cuda_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  return runtime::CUDAModuleCreate("ptx", "ptx", ExtractFuncInfo(mod), code);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_cuda", BuildTileLangCUDA)
      .def("target.build.tilelang_cuda_without_compile",
           BuildTileLangCUDAWithoutCompile);
}

} // namespace codegen
} // namespace tvm

#if defined(__linux__)
#include <sys/stat.h>
#include <tvm/ffi/reflection/registry.h>
#endif

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include "codegen_hip.h"
#include "runtime/rocm/rocm_module.h"
#include <tvm/ffi/function.h>

#ifndef kTVMGridConstant
#define kTVMGridConstant 130
#endif

namespace tvm {
namespace codegen {

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
          info.arg_types.push_back(DataType(kTVMGridConstant, 64, 1));
          continue;
        }
      }
      DataType dtype = f->params[i].dtype();
      // Device runtime cannot directly take bool arguments, map to int32.
      if (dtype.is_bool())
        dtype = DataType::Int(32);
      info.arg_types.push_back(dtype);
    }
    if (auto opt = f->GetAttr<ffi::Array<ffi::String>>(
            tir::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        info.launch_param_tags.push_back(tag);
      }
    }
    auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    fmap[static_cast<std::string>(global_symbol.value())] = info;
  }
  return fmap;
}

ffi::Module BuildTileLangHIP(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangHIP cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangHIP: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(f);
  }

  std::string code = cg.Finish();

  // Use the new FFI API to get registered functions
  using ffi::Function;
  if (auto f = Function::GetGlobal("tilelang_callback_hip_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }

  std::string fmt = "ptx";
  std::string ptx;

  if (auto f = Function::GetGlobal("tilelang_callback_hip_compile")) {
    ptx = (*f)(code, target).cast<std::string>();
    if (ptx[0] != '/')
      fmt = "hsaco";
  } else {
    ICHECK(false) << "tilelang_callback_hip_compile is not set";
  }

  return ROCMModuleCreate(ptx, fmt, ExtractFuncInfo(mod), code, std::string());
}

ffi::Module BuildTileLangHIPWithoutCompile(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangHIP cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangHIP: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(f);
  }

  std::string code = cg.Finish();

  // Use the new FFI API to get registered functions
  using ffi::Function;
  if (auto f = Function::GetGlobal("tilelang_callback_hip_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }

  return ROCMModuleCreate("ptx", "fmt", ExtractFuncInfo(mod), code,
                          std::string());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_hip", BuildTileLangHIP)
      .def("target.build.tilelang_hip_without_compile",
           BuildTileLangHIPWithoutCompile);
}

} // namespace codegen
} // namespace tvm

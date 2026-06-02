#if defined(__linux__)
#include "support/check.h"
#include <sys/stat.h>
#include <tvm/ir/cast.h>
#endif

#include <hip/hip_runtime.h>

#include "codegen_hip.h"
#include "runtime/pack_args.h"
#include "target/rocm/rocm_fallback_module.h"

namespace tvm {
namespace codegen {

using namespace ffi;

static Map<String, runtime::FunctionInfo> ExtractFuncInfo(const IRModule &mod) {
  Map<String, runtime::FunctionInfo> fmap;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tirx::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<tirx::PrimFunc>(kv.second);

    Array<DLDataType> arg_types;
    Array<String> launch_param_tags;
    for (size_t i = 0; i < f->params.size(); ++i) {
      if (f->params[i]->dtype.is_handle()) {
        auto ptr = f->params[i]->type_annotation.as<PointerTypeNode>();
        if (ptr && ptr->storage_scope == "grid_constant") {
          arg_types.push_back(DataType(runtime::kDLGridConstant, 64, 1));
          continue;
        }
      }
      DataType dtype = f->params[i].dtype();
      // Device runtime cannot directly take bool arguments, map to int32.
      if (dtype.is_bool())
        dtype = DataType::Int(32);
      arg_types.push_back(dtype);
    }
    if (f->HasNonzeroAttr("use_cooperative_groups")) {
      launch_param_tags.push_back(runtime::launch_param::kUseCooperativeLaunch);
    }
    if (auto opt = f->GetAttr<Array<String>>(tirx::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        launch_param_tags.push_back(tag);
      }
    }
    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
    std::string name = static_cast<std::string>(global_symbol.value());
    fmap.Set(String(name), runtime::FunctionInfo(String(name), arg_types,
                                                 launch_param_tags, {}));
  }
  return fmap;
}

Module BuildTileLangHIP(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangHIP cg;
  cg.Init(output_ssa);
  cg.SetTarget(target);

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

  Map<String, String> source_map;
  source_map.Set("hip", code);
  return target::ROCmModuleCreateWithFallback(Bytes(ptx.data(), ptx.size()),
                                              String(fmt), ExtractFuncInfo(mod),
                                              source_map);
}

Module BuildTileLangHIPWithoutCompile(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangHIP cg;
  cg.Init(output_ssa);
  cg.SetTarget(target);

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
  if (auto f = Function::GetGlobal("tilelang_callback_hip_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }

  Map<String, String> source_map;
  source_map.Set("hip", code);
  static constexpr const char kDummyPtx[] = "ptx";
  return target::ROCmModuleCreateWithFallback(
      Bytes(kDummyPtx, sizeof(kDummyPtx) - 1), String("ptx"),
      ExtractFuncInfo(mod), source_map);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_hip", BuildTileLangHIP)
      .def("target.build.tilelang_hip_without_compile",
           BuildTileLangHIPWithoutCompile);
}

} // namespace codegen
} // namespace tvm

#include "codegen_cuda.h"
#include "runtime/pack_args.h"
#include "runtime/thread_storage_scope.h"
#include "support/check.h"
#include "target/cuda/cuda_fallback_module.h"
#include "transform/common/attr.h"
#include <tvm/ir/cast.h>
#include <tvm/ir/transform.h>

namespace tvm {
namespace codegen {

using namespace ffi;

static std::string GetDeviceGlobalSymbol(const GlobalVar &gvar,
                                         const tirx::PrimFunc &f) {
  if (auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
    return static_cast<std::string>(global_symbol.value());
  }
  return gvar->name_hint;
}

static void ValidateUniqueDeviceGlobalSymbols(const IRModule &mod) {
  std::unordered_map<std::string, std::string> symbol_to_gvar;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tirx::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<tirx::PrimFunc>(kv.second);
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
      if (dtype.is_bool())
        dtype = DataType::Int(32);
      arg_types.push_back(dtype);
    }
    if (f->HasNonzeroAttr(tl::attr::kHasGridSync)) {
      launch_param_tags.push_back(
          runtime::launch_param::kUseProgramaticDependentLaunch);
    }
    if (f->HasNonzeroAttr("use_cooperative_groups")) {
      launch_param_tags.push_back(runtime::launch_param::kUseCooperativeLaunch);
    }
    if (f->GetAttr<Array<Integer>>("cluster_dims").defined()) {
      launch_param_tags.push_back(runtime::launch_param::kClusterDimX);
      launch_param_tags.push_back(runtime::launch_param::kClusterDimY);
      launch_param_tags.push_back(runtime::launch_param::kClusterDimZ);
    }
    if (auto opt = f->GetAttr<Array<String>>(tirx::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        if (tag != runtime::launch_param::kClusterDimX &&
            tag != runtime::launch_param::kClusterDimY &&
            tag != runtime::launch_param::kClusterDimZ) {
          launch_param_tags.push_back(tag);
        }
      }
    }
    std::string sym = GetDeviceGlobalSymbol(Downcast<GlobalVar>(kv.first), f);
    fmap.Set(String(sym), runtime::FunctionInfo(String(sym), arg_types,
                                                launch_param_tags, {}));
  }
  return fmap;
}

Module BuildTileLangCUDA(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangCUDA cg;
  cg.Init(output_ssa);

  ValidateUniqueDeviceGlobalSymbols(mod);
  if (const auto f = Function::GetGlobal("tilelang_callback_cuda_validate")) {
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
  if (const auto f = Function::GetGlobal("tilelang_callback_cuda_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  std::string fmt = "ptx";
  std::string ptx;
  if (const auto f = Function::GetGlobal("tilelang_callback_cuda_compile")) {
    // Fetch current pass context config and pass into the compile callback
    tvm::transform::PassContext pass_ctx =
        tvm::transform::PassContext::Current();
    ptx = (*f)(code, target, pass_ctx->config).cast<std::string>();
    if (ptx[0] != '/')
      fmt = "cubin";
  } else {
    ICHECK(0);
  }
  Map<String, String> source_map;
  source_map.Set("cuda", code);
  return target::CUDAModuleCreateWithFallback(Bytes(ptx.data(), ptx.size()),
                                              String(fmt), ExtractFuncInfo(mod),
                                              source_map);
}

Module BuildTileLangCUDAWithoutCompile(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangCUDA cg;
  cg.Init(output_ssa);

  ValidateUniqueDeviceGlobalSymbols(mod);
  if (const auto f = Function::GetGlobal("tilelang_callback_cuda_validate")) {
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
  if (const auto f = Function::GetGlobal("tilelang_callback_cuda_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  Map<String, String> source_map;
  source_map.Set("cuda", code);
  // The no-compile path still needs a code payload and format for the CUDA
  // module container.  Keep a tiny dummy PTX payload; the generated CUDA source
  // is preserved in source_map for InspectSource/get_source.
  static constexpr const char kDummyPtx[] = "ptx";
  return target::CUDAModuleCreateWithFallback(
      Bytes(kDummyPtx, sizeof(kDummyPtx) - 1), String("ptx"),
      ExtractFuncInfo(mod), source_map);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_cuda", BuildTileLangCUDA)
      .def("target.build.tilelang_cuda_without_compile",
           BuildTileLangCUDAWithoutCompile);
}

} // namespace codegen
} // namespace tvm

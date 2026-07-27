#include "codegen_cutedsl.h"
#include "runtime/pack_args.h"
#include "support/check.h"
#include "target/cuda/cuda_fallback_module.h"
#include <tvm/ir/cast.h>

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
    for (size_t i = 0; i < f->params.size(); ++i) {
      if (f->params[i]->dtype.is_handle()) {
        auto ptr = f->params[i]->type_annotation.as<PointerTypeNode>();
        if (ptr && ptr->storage_scope == "grid_constant") {
          arg_types.push_back(DataType(runtime::kDLGridConstant, 64, 1));
          continue;
        }
      }
      arg_types.push_back(f->params[i].dtype());
    }
    Array<String> launch_param_tags;
    if (auto opt = f->GetAttr<Array<String>>(tirx::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        launch_param_tags.push_back(tag);
      }
    }
    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
    std::string name = static_cast<std::string>(global_symbol.value());
    runtime::FunctionInfo info(String(name), arg_types, launch_param_tags,
                               Array<runtime::ArgExtraTags>());
    fmap.Set(String(name), info);
  }
  return fmap;
}

Module BuildTileLangCuTeDSLWithoutCompile(IRModule mod, Target target) {
  CodeGenTileLangCuTeDSL cg;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangCuTeDSL: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f =
          Function::GetGlobal("tilelang_callback_cutedsl_postproc")) {
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
  refl::GlobalDef().def("target.build.tilelang_cutedsl_without_compile",
                        BuildTileLangCuTeDSLWithoutCompile);
}

} // namespace codegen
} // namespace tvm

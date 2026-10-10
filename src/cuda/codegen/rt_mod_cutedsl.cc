#include "codegen_cuda.h"
#include "codegen_cutedsl.h"
#include "runtime/pack_args.h"
#include "support/check.h"
#include "target/cuda/cuda_fallback_module.h"
#include <tvm/ir/cast.h>

namespace tvm {
namespace codegen {

using namespace ffi;

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
      ExtractCudaFuncInfo(mod), source_map);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_cutedsl_without_compile",
           BuildTileLangCuTeDSLWithoutCompile)
      .def("target.build.tilelang_cutedsl", [](IRModule mod, Target target) {
        auto source_mod = BuildTileLangCuTeDSLWithoutCompile(mod, target);
        auto code = source_mod->InspectSource(ffi::String("cuda"));
        auto compile =
            Function::GetGlobalRequired("tilelang_callback_cutedsl_compile");
        auto fmap = ExtractCudaFuncInfo(mod);
        Optional<Module> result;
        for (const auto &kv : mod->functions) {
          auto func = Downcast<tirx::PrimFunc>(kv.second);
          auto name = func->GetAttr<String>(tvm::attr::kGlobalSymbol).value();
          Bytes ptx = compile(code, func, target).cast<Bytes>();
          auto device = target::CUDAModuleCreateWithFallback(
              ptx, String("ptx"), {{name, fmap[name]}}, {{"cuda", code}});
          if (result.defined()) {
            result.value()->ImportModule(device);
          } else {
            result = device;
          }
        }
        return result.value_or(source_mod);
      });
}

} // namespace codegen
} // namespace tvm

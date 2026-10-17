#include "codegen_cuda.h"
#include "codegen_cutedsl.h"
#include "runtime/pack_args.h"
#include "support/bytes_io.h"
#include "support/check.h"
#include "target/cuda/cuda_fallback_module.h"
#include <tvm/ir/cast.h>

namespace tvm {
namespace codegen {

using namespace ffi;

namespace {
// Resolve the logical Host IR name once; launches use the CUDA function
// directly.
class CuTeDSLModuleNode : public ModuleObj {
public:
  Map<String, String> symbols;

  const char *kind() const final { return "tilelang_cutedsl"; }
  int GetPropertyMask() const final {
    return Module::kBinarySerializable | Module::kRunnable;
  }
  Optional<Function> GetFunction(const String &lookup) final {
    if (auto symbol = symbols.Get(lookup)) {
      for (const auto &device : imports()) {
        if (auto func = device.cast<Module>()->GetFunction(*symbol))
          return func;
      }
    }
    return std::nullopt;
  }
  String InspectSource(const String &format) const final {
    return imports()[0].cast<Module>()->InspectSource(format);
  }
  Bytes SaveToBytes() const final {
    std::string buffer;
    support::BytesOutStream stream(&buffer);
    stream.Write(symbols);
    return Bytes(buffer);
  }
};
} // namespace

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
      .def("ffi.Module.load_from_bytes.tilelang_cutedsl",
           [](Bytes bytes) {
             auto node = make_object<CuTeDSLModuleNode>();
             support::BytesInStream stream(bytes);
             ICHECK(stream.Read(&node->symbols));
             return Module(node);
           })
      .def("target.build.tilelang_cutedsl_without_compile",
           BuildTileLangCuTeDSLWithoutCompile)
      .def("target.build.tilelang_cutedsl", [](IRModule mod, Target target) {
        auto source_mod = BuildTileLangCuTeDSLWithoutCompile(mod, target);
        auto code = source_mod->InspectSource(ffi::String("cuda"));
        auto compile =
            Function::GetGlobalRequired("tilelang_callback_cutedsl_compile");
        auto fmap = ExtractCudaFuncInfo(mod);
        auto node = make_object<CuTeDSLModuleNode>();
        for (const auto &kv : mod->functions) {
          auto func = Downcast<tirx::PrimFunc>(kv.second);
          auto name = func->GetAttr<String>(tvm::attr::kGlobalSymbol).value();
          auto compiled = compile(code, func, target).cast<Map<String, Any>>();
          auto symbol = compiled["symbol"].cast<String>();
          node->symbols.Set(name, symbol);
          auto info = fmap[name];
          runtime::FunctionInfo device_info(symbol, info->arg_types,
                                            info->launch_param_tags,
                                            info->arg_extra_tags);
          node->ImportModule(target::CUDAModuleCreateWithFallback(
              compiled["cubin"].cast<Bytes>(), String("cubin"),
              {{symbol, device_info}}, {{"cuda", code}}));
        }
        return node->symbols.empty() ? source_mod : Module(node);
      });
}

} // namespace codegen
} // namespace tvm

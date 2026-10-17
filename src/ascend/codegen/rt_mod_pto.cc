#include "ascend/codegen/codegen_pto.h"
#include "runtime_metadata.h"

#include "support/check.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

namespace tvm {
namespace codegen {

namespace {

std::string PTOCodeGen(IRModule mod) {
  CodeGenTileLangPTO cg;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangPTO: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
  }

  return cg.Finish();
}

std::string ApplyPostproc(std::string code, const Target &target) {
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_ascend_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  return code;
}

} // namespace

ffi::Module BuildTileLangPTO(IRModule mod, Target target) {
  std::string code = ApplyPostproc(PTOCodeGen(mod), target);
  auto compile =
      ffi::Function::GetGlobalRequired("tilelang_callback_pto_compile");
  auto pass_ctx = tvm::transform::PassContext::Current();
  auto fmap = ExtractAscendFuncInfo(mod);
  ffi::Optional<ffi::Module> result;
  // Compile entries independently. Host IR owns launch order and control flow;
  // each device module owns only its binary and standard Ascend launch
  // metadata.
  for (const auto &[name, info] : fmap) {
    std::string binary =
        compile(code, name, target, pass_ctx->config).cast<std::string>();
    ffi::Map<ffi::String, runtime::FunctionInfo> entry;
    entry.Set(name, info);
    ffi::Map<ffi::String, ffi::String> source;
    source.Set("py", code);
    // The Ascend module's default source format remains "asc" for both
    // dialects.
    source.Set("asc", code);
    auto device = ascend::AscendModuleCreate(
        ffi::Bytes(binary.data(), binary.size()), "aibin", entry, source);
    if (!result.defined()) {
      result = device;
    } else {
      result.value()->ImportModule(device);
    }
  }
  ICHECK(result.defined())
      << "PTO device module contains no launchable kernels";
  return result.value();
}

ffi::Module BuildTileLangPTOWithoutCompile(IRModule mod, Target target) {
  std::string code = ApplyPostproc(PTOCodeGen(mod), target);
  return CSourceModuleCreate(code, "py", ffi::Array<ffi::String>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_pto", BuildTileLangPTO)
      .def("target.build.tilelang_pto_without_compile",
           BuildTileLangPTOWithoutCompile);
}

} // namespace codegen
} // namespace tvm

#include "ascend/codegen/codegen_pto.h"
#include "codegen_ascend.h"

#include "runtime/metadata.h"
#include "support/check.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

namespace tvm {

// Defined in src/ascend/runtime/ascend_module.cc.
namespace ascend {
ffi::Module
AscendModuleCreate(ffi::Bytes code, ffi::String fmt,
                   ffi::Map<ffi::String, runtime::FunctionInfo> fmap,
                   ffi::Map<ffi::String, ffi::String> source);
} // namespace ascend

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

  std::string aibin;
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_pto_compile")) {
    tvm::transform::PassContext pass_ctx =
        tvm::transform::PassContext::Current();
    aibin = (*f)(code, target, pass_ctx->config).cast<std::string>();
  } else {
    ICHECK(false) << "tilelang_callback_pto_compile is not registered";
  }

  // The Ascend runtime reads the device source for launch-error reporting
  // and inspect_source under the "asc" key; "pto" exposes the same text
  // under its own name.
  ffi::Map<ffi::String, ffi::String> source_map;
  source_map.Set("asc", ffi::String(code));
  source_map.Set("pto", ffi::String(code));
  return ascend::AscendModuleCreate(
      ffi::Bytes(aibin.data(), aibin.size()), ffi::String("aibin"),
      ExtractAscendFuncInfo(mod), std::move(source_map));
}

ffi::Module BuildTileLangPTOWithoutCompile(IRModule mod, Target target) {
  std::string code = ApplyPostproc(PTOCodeGen(mod), target);
  // Source-only module used by the cython adapter / inspect_source. The "pto"
  // launcher runs the ptodsl -> ptoas -> bisheng pipeline itself, so no
  // runnable module is needed here.
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

#include "ascend/codegen/codegen_pto.h"

#include "support/check.h"

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>

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
  return CSourceModuleCreate(code, "py", ffi::Array<ffi::String>());
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

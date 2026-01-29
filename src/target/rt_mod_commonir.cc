#include "codegen_commonir.h"
#include "runtime/pack_args.h"
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace codegen {

using ffi::String;
using ffi::Array;

ffi::Module BuildTileLangCommonIR(IRModule mod, Target target) {
  // using tvm::runtime::Registry;
  bool output_ssa = false;
  CodeGenTileLangCOMMONIR cg;
  cg.Init(output_ssa);

  Array<String> function_names;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangCOMMONIR: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    cg.AddFunction(gvar, f);
    function_names.push_back(cg.GetFunctionName(gvar));
  }

  std::string code = cg.Finish();

  return CSourceModuleCreate(code, "c", function_names);
}
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.tilelang_commonir", BuildTileLangCommonIR);
}

TVM_REGISTER_TARGET_KIND("commonir", kDLExtDev);

} // namespace codegen
} // namespace tvm
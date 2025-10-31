#include "codegen_cpp.h"
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>

#include "../support/ffi_aliases.h"

namespace tvm {
namespace codegen {

ffi::Module BuildCPPHost(IRModule mod, Target target) {
  bool output_ssa = false;
  bool emit_asserts = false;
  bool emit_fwd_func_decl = true;

  std::unordered_set<std::string> devices;
  if (mod->GetAttr<Map<GlobalVar, String>>("device_contexts") != nullptr) {
    Map<GlobalVar, String> device_contexts =
        mod->GetAttr<Map<GlobalVar, String>>("device_contexts").value();
    for (auto const &context : device_contexts) {
      devices.insert(context.second.data());
    }
  }

  CodeGenTileLangCPP cg;
  cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), devices);
  cg.SetConstantsByteAlignment(
      target->GetAttr<Integer>("constants-byte-alignment").value_or(16));

  auto is_aot_executor_fn = [](const PrimFunc &func) -> bool {
    return func->GetAttr<Bool>("runner_function", Bool(false)).value();
  };

  std::vector<std::pair<GlobalVar, PrimFunc>> funcs;
  for (auto [gvar, base_func] : mod->functions) {
    ICHECK(base_func->IsInstance<PrimFuncNode>())
        << "CodegenCHost: Can only take PrimFunc";
    auto prim_func = Downcast<PrimFunc>(base_func);
    funcs.push_back({gvar, prim_func});
  }

  // Sort functions
  auto sort_key = [&is_aot_executor_fn](const auto &kv) {
    return std::tuple{is_aot_executor_fn(kv.second), kv.first->name_hint};
  };
  std::sort(funcs.begin(), funcs.end(),
            [&sort_key](const auto &kv_a, const auto &kv_b) {
              return sort_key(kv_a) < sort_key(kv_b);
            });

  // Declare all functions first.  This ensures that all functions,
  // including the __tvm_main__ used in AOT, have access to forward
  // declarations of other functions in the IRModule.
  for (const auto &[gvar, prim_func] : funcs) {
    cg.DeclareFunction(gvar, prim_func);
  }

  // Codegen all functions.  Passing emit_fwd_func_decl=true adds a
  // forward declaration for any `builtin::call_extern`, based on the
  // arguments provided to it.
  for (const auto &[gvar, prim_func] : funcs) {
    cg.AddFunction(prim_func);
  }

  if (target->GetAttr<Bool>("system-lib").value_or(Bool(false))) {
    ICHECK_EQ(target->GetAttr<String>("runtime").value_or(""), "c")
        << "c target only supports generating C runtime SystemLibs";
  }

  std::string code = cg.Finish();
  return CSourceModuleCreate(code, "c", cg.GetFunctionNames());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.tilelang_cpp", BuildCPPHost);
}

} // namespace codegen
} // namespace tvm

#include "codegen_c.h"
#include "support/check.h"
#include <tvm/ffi/extra/module.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm {
namespace codegen {

using namespace ffi;

Module BuildTileLangC(IRModule mod, Target target) {
  bool output_ssa = false;
  // Native outlining returns int32 status from CPU compute functions. Pointer
  // and data constraints inside those functions must execute before stores;
  // the host caller propagates the raised FFI error on a nonzero status.
  bool emit_asserts = true;
  bool emit_fwd_func_decl = true;

  std::unordered_set<std::string> devices;
  if (mod->GetAttr<Map<GlobalVar, String>>("device_contexts").has_value()) {
    Map<GlobalVar, String> device_contexts =
        mod->GetAttr<Map<GlobalVar, String>>("device_contexts").value();
    for (auto const &context : device_contexts) {
      devices.insert(context.second.data());
    }
  }

  CodeGenTileLangC cg;
  cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), devices);
  cg.SetConstantsByteAlignment(
      target->GetAttr<int64_t>("constants-byte-alignment").value_or(16));
  cg.SetEmitLineDirectives(TileLangEmitLineDirectivesEnabled());

  auto is_aot_executor_fn = [](const tirx::Function &func) -> bool {
    return func->GetAttr<bool>("runner_function", false).value();
  };

  std::vector<std::pair<GlobalVar, tirx::Function>> funcs;
  for (auto [gvar, base_func] : mod->functions) {
    ICHECK(base_func->IsInstance<tirx::FunctionObj>())
        << "BuildTileLangC: Can only take tirx::Function";
    auto prim_func = base_func.as_or_throw<tirx::Function>();
    if (tirx::ContainsNode<AssertStmtObj>(prim_func->body)) {
      const auto *return_type = prim_func->ret_type.as<PrimTypeObj>();
      TVM_FFI_CHECK(return_type &&
                        ffi::GetRef<PrimType>(return_type) == PrimType::Int(32),
                    TypeError)
          << "TileLang C assertions require an int32 status-returning "
             "function; lower the host/device and packed ABI before codegen";
    }
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
    cg.AddFunction(gvar, prim_func, emit_fwd_func_decl);
  }

  if (target->GetAttr<bool>("system-lib").value_or(false)) {
    ICHECK_EQ(target->GetAttr<String>("runtime").value_or(""), "c")
        << "c target only supports generating C runtime SystemLibs";
  }

  std::string code = cg.Finish();
  return CSourceModuleCreate(code, "c", cg.GetFunctionNames());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("target.build.tilelang_c", BuildTileLangC);
}

} // namespace codegen
} // namespace tvm

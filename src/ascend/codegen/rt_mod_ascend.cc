#include "backend/common/codegen/native_codegen.h"
#include "codegen_ascend.h"
#include "config.h"

#include "support/check.h"
#include <tvm/codegen/build_common.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/transform.h>

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

std::string AscendCodeGen(IRModule mod, const Target &target) {
  for (const auto &[global, base] : mod->functions) {
    auto function = base.as_or_throw<tirx::Function>();
    mod.CopyOnWrite()->functions.Set(
        global, WithAttr(function, tvm::attr::kTarget, target));
  }
  mod = tirx::transform::StmtSimplify()(mod);
  mod = tirx::transform::LowerIntrin()(mod);
  bool output_ssa = false;
  CodeGenTileLangAscend cg;
  cg.Init(output_ssa);
  cg.ConfigureModule(mod);
  cg.SetEmitLineDirectives(tl::tl_config::EmitLineDirectivesEnabled());

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tirx::FunctionNode>())
        << "CodeGenTileLangAscend: Can only take PrimFunc";
    auto f = kv.second.as_or_throw<tirx::Function>();
    auto calling_conv = f->GetAttr(tvm::attr::kCallingConv);
    auto is_global_func =
        f->GetAttr<bool>("tirx.is_global_func").value_or(false);
    // For Ascend, we accept functions that are either:
    // 1. Device kernel launches (calling_conv == kDeviceKernelLaunch), or
    // 2. Global functions called via call_extern (is_global_func == true)
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch || is_global_func)
        << "CodeGenTileLangAscend: Function must have "
           "calling_conv=kDeviceKernelLaunch "
        << "or be marked as a global function (tir.is_global_func=True)";
    cg.AddFunction(kv.first, f);
  }

  return cg.Finish();
}

} // namespace

ffi::Module BuildTileLangAscend(IRModule mod, Target target) {
  mod = tl::PrepareNativeCodegen(std::move(mod), target);
  std::string code = AscendCodeGen(mod, target);

  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_ascend_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }

  ffi::Bytes aibin;
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_ascend_compile")) {
    tvm::transform::PassContext pass_ctx =
        tvm::transform::PassContext::Current();
    auto compiled = (*f)(code, target, pass_ctx->config);
    if (auto binary = compiled.as<ffi::Bytes>()) {
      aibin = binary.value();
    } else {
      auto text_binary = compiled.cast<ffi::String>();
      aibin = ffi::Bytes(text_binary.data(), text_binary.size());
    }
    ICHECK_NE(aibin.size(), 0) << "Ascend compiler returned an empty binary";
  } else {
    ICHECK(false) << "tilelang_callback_ascend_compile is not registered";
  }

  ffi::Map<ffi::String, ffi::String> source_map;
  source_map.Set("asc", ffi::String(code));
  return ascend::AscendModuleCreate(ffi::Bytes(aibin.data(), aibin.size()),
                                    ffi::String("aibin"), ExtractFuncInfo(mod),
                                    std::move(source_map));
}

ffi::Module BuildTileLangAscendWithoutCompile(IRModule mod, Target target) {
  mod = tl::PrepareNativeCodegen(std::move(mod), target);
  std::string code = AscendCodeGen(mod, target);
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_ascend_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  // Source-only module used by the cython adapter / inspect_source. The cython
  // path compiles host+device together with bisheng, so no runnable module is
  // needed here.
  return SourceModuleCreate(code, "asc");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.ascend", BuildTileLangAscend)
      .def("target.build.tilelang_ascend", BuildTileLangAscend)
      .def("target.build.tilelang_ascend_without_compile",
           BuildTileLangAscendWithoutCompile);
}

} // namespace codegen
} // namespace tvm

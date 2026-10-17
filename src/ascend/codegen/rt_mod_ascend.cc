#include "codegen_ascend.h"
#include "config.h"

#include "runtime_metadata.h"
#include "support/check.h"
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>

namespace tvm {

namespace codegen {

namespace {

std::string AscendCodeGen(IRModule mod) {
  bool output_ssa = false;
  CodeGenTileLangAscend cg;
  cg.Init(output_ssa);
  cg.SetEmitLineDirectives(tl::tl_config::EmitLineDirectivesEnabled());

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangAscend: Can only take PrimFunc";
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    auto is_global_func =
        f->GetAttr<Bool>(tvm::tirx::attr::kIsGlobalFunc).value_or(Bool(false));
    // For Ascend, we accept functions that are either:
    // 1. Device kernel launches (calling_conv == kDeviceKernelLaunch), or
    // 2. Global functions called via call_extern (is_global_func == true)
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch || is_global_func)
        << "CodeGenTileLangAscend: Function must have "
           "calling_conv=kDeviceKernelLaunch "
        << "or be marked as a global function (tir.is_global_func=True)";
    cg.AddFunction(f);
  }

  return cg.Finish();
}

} // namespace

ffi::Module BuildTileLangAscend(IRModule mod, Target target) {
  std::string code = AscendCodeGen(mod);

  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_ascend_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }

  std::string aibin;
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_ascend_compile")) {
    tvm::transform::PassContext pass_ctx =
        tvm::transform::PassContext::Current();
    aibin = (*f)(code, target, pass_ctx->config).cast<std::string>();
  } else {
    ICHECK(false) << "tilelang_callback_ascend_compile is not registered";
  }

  ffi::Map<ffi::String, ffi::String> source_map;
  source_map.Set("asc", ffi::String(code));
  return ascend::AscendModuleCreate(
      ffi::Bytes(aibin.data(), aibin.size()), ffi::String("aibin"),
      ExtractAscendFuncInfo(mod), std::move(source_map));
}

ffi::Module BuildTileLangAscendWithoutCompile(IRModule mod, Target target) {
  std::string code = AscendCodeGen(mod);
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_ascend_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  // Source-only module used by the cython adapter / inspect_source. The cython
  // path compiles host+device together with bisheng, so no runnable module is
  // needed here.
  return CSourceModuleCreate(code, "asc", ffi::Array<ffi::String>());
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

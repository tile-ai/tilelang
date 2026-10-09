#include "codegen_ascend.h"
#include "config.h"

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

// Build the per-kernel FunctionInfo table the runtime module needs to recover
// the launch grid (via LaunchParamConfig) and pack arguments. Mirrors the CUDA
// ExtractFuncInfo (src/cuda/codegen/rt_mod_cuda.cc) but only carries what the
// Ascend launch ABI uses: arg types + kernel launch params (thread extents).
ffi::Map<ffi::String, runtime::FunctionInfo>
ExtractFuncInfo(const IRModule &mod) {
  ffi::Map<ffi::String, runtime::FunctionInfo> fmap;
  for (auto kv : mod->functions) {
    auto f = Downcast<tirx::PrimFunc>(kv.second);
    // Only device kernels are launchable; skip call_extern globals.
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    if (!(calling_conv == CallingConv::kDeviceKernelLaunch)) {
      continue;
    }

    ffi::Array<DLDataType> arg_types;
    ffi::Array<ffi::String> launch_param_tags;

    for (size_t i = 0; i < f->params.size(); ++i) {
      DataType dtype = f->params[i].dtype();
      if (dtype.is_bool())
        dtype = DataType::Int(32);
      arg_types.push_back(dtype);
    }
    if (auto opt = f->GetAttr<ffi::Array<ffi::String>>(
            tirx::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        launch_param_tags.push_back(tag);
      }
    }

    auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    std::string sym = global_symbol.has_value()
                          ? static_cast<std::string>(global_symbol.value())
                          : static_cast<std::string>(
                                Downcast<GlobalVar>(kv.first)->name_hint);
    fmap.Set(ffi::String(sym),
             runtime::FunctionInfo(ffi::String(sym), arg_types,
                                   launch_param_tags, {}));
  }
  return fmap;
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
  return ascend::AscendModuleCreate(ffi::Bytes(aibin.data(), aibin.size()),
                                    ffi::String("aibin"), ExtractFuncInfo(mod),
                                    std::move(source_map));
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

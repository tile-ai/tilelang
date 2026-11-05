#include "codegen/c/codegen_source_base.h"
#include "codegen_ascend.h"

#include "runtime/metadata.h"
#include "support/check.h"
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <unordered_set>

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
  With<Target> target_scope(target);
  std::unordered_set<std::string> symbols;
  bool output_ssa = false;
  CodeGenTileLangAscend cg;
  cg.Init(output_ssa);
  cg.SetEmitLineDirectives(TileLangEmitLineDirectivesEnabled());

  for (auto kv : mod->functions) {
    const auto f = kv.second.as_or_throw<tirx::Function>();
    const auto symbol = f->GetAttr(tvm::attr::kGlobalSymbol);
    TVM_FFI_CHECK(symbol.has_value(), ValueError)
        << "Ascend function requires global_symbol";
    TVM_FFI_CHECK(symbols.insert(symbol.value()).second, ValueError)
        << "Duplicate Ascend global_symbol: " << symbol.value();
    const auto calling_conv =
        f->GetAttr(tvm::attr::kCallingConv, CallingConv::kDefault).value();
    const bool is_global_func = f->HasNonzeroAttr("tirx.is_global_func");
    TVM_FFI_CHECK(calling_conv == CallingConv::kDeviceKernelLaunch ||
                      is_global_func,
                  ValueError)
        << "Ascend code generation requires a device kernel or an explicit "
           "tirx.is_global_func helper";
    cg.AddFunction(kv.first, f);
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
    auto f = kv.second.as_or_throw<tirx::Function>();
    // Only device kernels are launchable; skip call_extern globals.
    auto calling_conv =
        f->GetAttr(tvm::attr::kCallingConv, CallingConv::kDefault).value();
    if (!(calling_conv == CallingConv::kDeviceKernelLaunch)) {
      continue;
    }

    ffi::Array<DLDataType> arg_types;
    ffi::Array<ffi::String> launch_param_tags;

    for (size_t i = 0; i < f->params.size(); ++i) {
      const Type type = f->params[i]->ty;
      if (auto scalar = type.as<PrimType>()) {
        TVM_FFI_CHECK(scalar.value().IsScalar() && !scalar.value().IsVoid(),
                      TypeError)
            << "Ascend kernel ABI requires scalar arguments";
        arg_types.push_back(scalar.value().MatchesCode(kDLBool)
                                ? PrimType::Int(32)->dtype
                                : scalar.value()->dtype);
      } else if (type.as<PtrTypeObj>()) {
        arg_types.push_back(DLDataType{kDLOpaqueHandle, 64, 1});
      } else {
        TVM_FFI_THROW(TypeError)
            << "Ascend device ABI requires lowered pointer parameters, got "
            << type;
      }
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
                          : static_cast<std::string>(kv.first->name_hint);
    fmap.Set(ffi::String(sym),
             runtime::FunctionInfo(ffi::String(sym), arg_types,
                                   launch_param_tags, {}));
  }
  return fmap;
}

} // namespace

ffi::Module BuildTileLangAscend(IRModule mod, Target target) {
  std::string code = AscendCodeGen(mod, target);

  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_ascend_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }

  const auto compile =
      ffi::Function::GetGlobalRequired("tilelang_callback_ascend_compile");
  const auto result =
      compile(code, target, tvm::transform::PassContext::Current()->config);
  ffi::Bytes aibin = [&]() {
    if (auto bytes = result.as<ffi::Bytes>())
      return bytes.value();
    auto text = result.cast<ffi::String>();
    return ffi::Bytes(text.data(), text.size());
  }();
  TVM_FFI_CHECK_GT(aibin.size(), 0, ValueError)
      << "Ascend compiler returned an empty binary";

  ffi::Map<ffi::String, ffi::String> source_map;
  source_map.Set("asc", ffi::String(code));
  return ascend::AscendModuleCreate(
      aibin, ffi::String("aibin"), ExtractFuncInfo(mod), std::move(source_map));
}

ffi::Module BuildTileLangAscendWithoutCompile(IRModule mod, Target target) {
  std::string code = AscendCodeGen(mod, target);
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
      .def("tl.codegen.AscendArgumentType",
           [](const Type &type) {
             CodeGenTileLangAscend codegen;
             std::ostringstream result;
             codegen.PrintType(type, result);
             return ffi::String(result.str());
           })
      .def("target.build.ascend", BuildTileLangAscend)
      .def("target.build.tilelang_ascend", BuildTileLangAscend)
      .def("target.build.tilelang_ascend_without_compile",
           BuildTileLangAscendWithoutCompile);
}

} // namespace codegen
} // namespace tvm

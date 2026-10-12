#include "runtime_metadata.h"

#include <tvm/tirx/stmt.h>

namespace tvm {
namespace codegen {

// Build the per-kernel FunctionInfo table the runtime module needs to recover
// the launch grid (via LaunchParamConfig) and pack arguments. Mirrors the CUDA
// ExtractFuncInfo (src/cuda/codegen/rt_mod_cuda.cc) but only carries what the
// Ascend launch ABI uses: arg types + kernel launch params (thread extents).
ffi::Map<ffi::String, runtime::FunctionInfo>
ExtractAscendFuncInfo(const IRModule &mod) {
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

} // namespace codegen
} // namespace tvm

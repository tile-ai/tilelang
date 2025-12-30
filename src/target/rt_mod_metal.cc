#include "codegen_metal.h"


namespace tvm {
namespace codegen {

ffi::Module BuildMetal(IRModule mod, Target target) {
  bool output_ssa = false;
  mod = tir::transform::PointerValueTypeRewrite()(std::move(mod));

  std::ostringstream source_maker;
  std::unordered_map<std::string, std::string> smap;
  const auto fmetal_compile = tvm::ffi::Function::GetGlobal("tvm_callback_metal_compile");
  std::string fmt = fmetal_compile ? "metallib" : "metal";

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>()) << "CodeGenMetal: Can only take PrimFunc";
    auto global_symbol = kv.second->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
    ICHECK(global_symbol.has_value());  
    std::string func_name = global_symbol.value();

    source_maker << "// Function: " << func_name << "\n";
    CodeGenMetal cg(target);
    cg.Init(output_ssa);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch)
        << "CodeGenMetal: expect calling_conv equals CallingConv::kDeviceKernelLaunch";

    cg.AddFunction(kv.first, f);

    std::string fsource = cg.Finish();
    source_maker << fsource << "\n";
    if (fmetal_compile) {
      fsource = (*fmetal_compile)(fsource, target).cast<std::string>();
    }
    smap[func_name] = fsource;
  }

  return codegen::DeviceSourceModuleCreate(source_maker.str(), fmt, ExtractFuncInfo(mod), "metal");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.tilelang_metal", BuildMetal);
}

}  // namespace codegen
}  // namespace tvm

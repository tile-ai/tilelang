/*!
 * \file annotate_device_regions.cc
 * \brief Mark CPU compute regions for native host/device splitting.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {

tvm::transform::Pass AnnotateDeviceRegions() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto target = function->GetAttr<Target>(tvm::attr::kTarget);
    TVM_FFI_CHECK(target.has_value(), ValueError)
        << "AnnotateDeviceRegions requires a target";
    auto kind = target.value()->kind->name;
    // Native SplitHostDevice recognizes launch_thread/device_entry for GPU
    // targets. CPU kernels instead consist of ordinary For loops and still
    // need a compute function with the original TileLang device signature.
    if (kind != "c" && kind != "llvm")
      return function;
    if (function->GetAttr<bool>("tl.cpu_compute_region").value_or(false))
      return function;
    auto region = RegionStmt(
        Op::Get("tirx.device_scope"), {}, {},
        DictAttrs({{tvm::attr::kTarget.name, target.value().WithoutHost()}}),
        function->body);
    function.CopyOnWrite()->body = SeqStmt(region);
    return WithAttr(function, "tl.cpu_compute_region", true);
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.AnnotateDeviceRegions");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.AnnotateDeviceRegions",
                                   AnnotateDeviceRegions);
}
} // namespace tvm::tl

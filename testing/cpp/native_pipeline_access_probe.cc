/*!
 * \file testing/cpp/native_pipeline_access_probe.cc
 * \brief Test-only access to the real pipeline dependence collector.
 */
#include "transform/common/pipeline_access.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl {
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.testing.CollectPipelineAccesses", [](tirx::Function function) {
        PipelineMemoryAnalysis memory(function);
        auto target =
            function->GetAttr<Target>("target").value_or(Target("llvm"));
        auto collector =
            ffi::make_object<PipelineAccessCollector>(&memory, target);
        collector->Visit(function->body);
        return ffi::Map<ffi::String, ffi::Any>{
            {"reads", collector->GetReads()},
            {"writes", collector->GetWrites()},
            {"unknown_address", collector->HasUnknownAddress()},
            {"unknown", collector->HasUnknownAccess()}};
      });
}
} // namespace tvm::tl

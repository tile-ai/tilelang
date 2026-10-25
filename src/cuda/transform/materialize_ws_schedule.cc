#include "compat/visit.h"
#include "transform/common/warp_specialize.h"

#include <tvm/tirx/transform.h>

namespace tvm::tl {
tvm::transform::Pass MaterializeWSSchedule() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    bool scheduled = false;
    if (function->body)
      PostOrderVisit(*function->body, [&](const ffi::ObjectRef &object) {
        if (const auto *region = object.as<RegionStmtNode>())
          scheduled |= region->attrs->dict.count(kWSScheduleKey);
      });
    if (!scheduled)
      return function;
    auto materialize =
        ffi::Function::GetGlobal("tl.cuda.native.MaterializeWSSchedule");
    TVM_FFI_CHECK(materialize, RuntimeError)
        << "Import tilelang to register native WS materialization";
    return (*materialize)(function).cast<tirx::Function>();
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.MaterializeWSSchedule");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.MaterializeWSSchedule",
                                   MaterializeWSSchedule);
}
} // namespace tvm::tl

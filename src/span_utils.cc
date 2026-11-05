/*!
 * \file tl/span_utils.cc
 * \brief Native location metadata used by the TileLang language facade.
 */
#include "span_utils.h"
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/function.h>

namespace tvm::tl {

void StampSubtreeSpans(const Stmt &stmt, const Location &loc) {
  if (loc.as<UnknownLocObj>())
    return;
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      stmt, [&loc](const StmtObj *node) -> ffi::Expected<ffi::WalkResult> {
        if (node->loc.as<UnknownLocObj>())
          node->loc = loc;
        return ffi::WalkResult::Advance();
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.ir.SetStmtSpan",
           [](const Stmt &stmt, Location loc) { stmt->loc = std::move(loc); })
      .def("tl.ir.GetStmtSpan", [](const Stmt &stmt) { return stmt->loc; })
      .def("tl.ir.SetBufferSpan",
           [](const tirx::TensorVar &tensor, Location loc) {
             tensor->loc = std::move(loc);
           })
      .def("tl.ir.GetBufferSpan",
           [](const tirx::TensorVar &tensor) { return tensor->loc; })
      .def("tl.ir.SetPrimFuncSpan",
           [](const tirx::Function &func, Location loc) {
             func->loc = std::move(loc);
           })
      .def("tl.ir.GetPrimFuncSpan",
           [](const tirx::Function &func) { return func->loc; })
      .def("tl.ir.StampSubtreeSpans", &StampSubtreeSpans)
      .def("tl.ir.SpanToString",
           [](const Location &loc) { return ffi::String(FormatSpan(loc)); });
}
} // namespace tvm::tl

/*!
 * \file tl/span_utils.cc
 * \brief FFI entry points to inject and read source spans on tirx IR nodes.
 */

#include "span_utils.h"

#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

void SetStmtSpan(Optional<Stmt> stmt, Optional<Location> span) {
  if (stmt.has_value()) {
    stmt.value()->loc = span.value_or(UnknownLoc());
  }
}

Location GetStmtSpan(const Optional<Stmt> &stmt) {
  return stmt.has_value() ? stmt.value()->loc : UnknownLoc();
}

void SetBufferSpan(Optional<TensorVar> buffer, Optional<Location> span) {
  if (buffer.has_value()) {
    buffer.value()->loc = span.value_or(UnknownLoc());
  }
}

Location GetBufferSpan(const Optional<TensorVar> &buffer) {
  return buffer.has_value() ? buffer.value()->loc : UnknownLoc();
}

void SetPrimFuncSpan(Optional<tirx::Function> func, Optional<Location> span) {
  if (func.has_value()) {
    func.value()->loc = span.value_or(UnknownLoc());
  }
}

Location GetPrimFuncSpan(const Optional<tirx::Function> &func) {
  return func.has_value() ? func.value()->loc : UnknownLoc();
}

String SpanToString(const Location &span) { return FormatSpan(span); }

void StampSubtreeSpans(const Stmt &stmt, const Location &span) {
  if (span.as<UnknownLocNode>()) {
    return;
  }
  ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
      stmt, [&span](const StmtNode *node) -> ffi::Expected<ffi::WalkResult> {
        if (node->loc.as<UnknownLocNode>()) {
          node->loc = span;
        }
        return ffi::WalkResult::Advance();
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.ir.SetStmtSpan", &SetStmtSpan)
      .def("tl.ir.GetStmtSpan", &GetStmtSpan)
      .def("tl.ir.SetBufferSpan", &SetBufferSpan)
      .def("tl.ir.GetBufferSpan", &GetBufferSpan)
      .def("tl.ir.SetPrimFuncSpan", &SetPrimFuncSpan)
      .def("tl.ir.GetPrimFuncSpan", &GetPrimFuncSpan)
      .def("tl.ir.StampSubtreeSpans",
           [](Optional<Stmt> stmt, Optional<Location> span) {
             if (stmt.has_value() && span.has_value()) {
               StampSubtreeSpans(stmt.value(), span.value());
             }
           })
      .def("tl.ir.SpanToString", &SpanToString);
}

} // namespace tl
} // namespace tvm

/*!
 * \file tl/span_utils.cc
 * \brief FFI entry points to inject and read source spans on tirx IR nodes.
 */

#include "span_utils.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

void SetStmtSpan(Stmt stmt, Span span) {
  if (stmt.defined()) {
    stmt.get()->span = std::move(span);
  }
}

Span GetStmtSpan(const Stmt &stmt) {
  return stmt.defined() ? stmt->span : Span();
}

void SetBufferSpan(Buffer buffer, Span span) {
  if (buffer.defined()) {
    buffer.get()->span = std::move(span);
  }
}

Span GetBufferSpan(const Buffer &buffer) {
  return buffer.defined() ? buffer->span : Span();
}

void SetPrimFuncSpan(PrimFunc func, Span span) {
  if (func.defined()) {
    func.get()->span = std::move(span);
  }
}

Span GetPrimFuncSpan(const PrimFunc &func) {
  return func.defined() ? func->span : Span();
}

String SpanToString(const Span &span) { return FormatSpan(span); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.ir.SetStmtSpan", &SetStmtSpan)
      .def("tl.ir.GetStmtSpan", &GetStmtSpan)
      .def("tl.ir.SetBufferSpan", &SetBufferSpan)
      .def("tl.ir.GetBufferSpan", &GetBufferSpan)
      .def("tl.ir.SetPrimFuncSpan", &SetPrimFuncSpan)
      .def("tl.ir.GetPrimFuncSpan", &GetPrimFuncSpan)
      .def("tl.ir.SpanToString", &SpanToString);
}

} // namespace tl
} // namespace tvm

/*!
 * \file layout/printer.h
 * \brief Script representation of layouts with explicit coordinate binders.
 */
#ifndef TILELANG_LAYOUT_PRINTER_H_
#define TILELANG_LAYOUT_PRINTER_H_

#include "layout.h"
#include <tvm/script/printer/doc_translator.h>

namespace tvm::tl {

// Shared by the plain/fragment layouts and backend-owned tagged subtypes.
TVM_TIRX_DLL script::printer::ExprDoc
LayoutMappingDoc(script::printer::DocTranslatorObj *translator,
                 const Layout &layout, script::printer::ExprDoc constructor,
                 ffi::Array<ffi::String> extra_keys = {},
                 ffi::Array<script::printer::ExprDoc> extra_values = {});

} // namespace tvm::tl
#endif // TILELANG_LAYOUT_PRINTER_H_

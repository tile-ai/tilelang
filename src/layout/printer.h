#ifndef TILELANG_LAYOUT_PRINTER_H_
#define TILELANG_LAYOUT_PRINTER_H_

#include <initializer_list>
#include <string_view>

#include <tvm/ffi/reflection/accessor.h>
#include <tvm/script/printer/doc_translator.h>

#include "layout.h"
#include "utils.h"

namespace tvm::tl {

inline ffi::Array<PrimExpr> MapLayoutCoordinates(const tirx::IndexMap &mapping,
                                                 size_t rank, bool replicated) {
  ffi::Array<PrimExpr> coordinates;
  for (size_t axis = 0; axis < rank; ++axis)
    coordinates.push_back(InputPlaceholder(axis));
  if (replicated)
    coordinates.push_back(ReplicationPlaceholder());
  TVM_FFI_CHECK_EQ(mapping->initial_indices.size(), coordinates.size(),
                   ValueError)
      << "Layout index-map arity does not match the logical rank";
  return mapping->MapIndices(coordinates);
}

inline ffi::Optional<script::printer::ExprDoc>
TranslateLayoutFields(script::printer::DocTranslatorObj *translator,
                      ffi::AnyView input, const char *factory,
                      std::initializer_list<const char *> fields) {
  using namespace script::printer;
  auto layout = input.cast<Layout>();
  ffi::Array<ExprDoc> arguments;
  for (const char *field : fields) {
    const TVMFFIFieldInfo *field_info = nullptr;
    ffi::reflection::ForEachFieldInfoWithEarlyStop(
        TVMFFIGetTypeInfo(layout->type_index()),
        [&](const TVMFFIFieldInfo *candidate) {
          if (std::string_view(candidate->name.data, candidate->name.size) !=
              field)
            return false;
          field_info = candidate;
          return true;
        });
    TVM_FFI_CHECK(field_info, RuntimeError)
        << "Cannot find layout field " << field;
    ffi::Any value = ffi::reflection::FieldGetter(field_info)(layout);
    arguments.push_back(translator->Translate(value).value());
  }
  return NamespaceDoc("tilelang")
      ->Attr("_ffi_api")
      ->Attr(factory)
      ->Call(arguments);
}

} // namespace tvm::tl

#endif

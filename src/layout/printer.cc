/*!
 * \file layout/printer.cc
 * \brief Layout script constructors preserve local coordinates and captures.
 */
#include "printer.h"
#include <tvm/ffi/reflection/registry.h>

namespace tvm::tl {
using namespace script::printer;

ExprDoc LayoutMappingDoc(DocTranslatorObj *translator, const Layout &layout,
                         ExprDoc constructor,
                         ffi::Array<ffi::String> extra_keys,
                         ffi::Array<ExprDoc> extra_values) {
  return constructor->Attr("from_mapping")
      ->Call({translator->Translate(layout->InputShape()).value(),
              translator->Translate(layout->CoordinateMapping()).value()},
             extra_keys, extra_values);
}

namespace {
ffi::Optional<ExprDoc> LayoutDocTranslate(DocTranslatorObj *translator,
                                          ffi::AnyView input,
                                          const ffi::Object *) {
  Layout layout = input.cast<Layout>();
  ffi::String name = "Layout";
  ffi::Array<ffi::String> keys;
  ffi::Array<ExprDoc> values;
  if (const auto *fragment = layout.as<FragmentNode>()) {
    name = "Fragment";
    keys.push_back("replicate");
    values.push_back(
        translator->Translate(fragment->ReplicateExtent()).value());
    if (fragment->ThreadRange().has_value()) {
      keys.push_back("thread_range");
      values.push_back(
          translator->Translate(fragment->ThreadRange().value()).value());
    }
  }
  if (const auto *partial = layout.as<PartialFragmentNode>()) {
    name = "PartialFragment";
    keys.push_back("combine");
    values.push_back(translator->Translate(partial->CombineSize()).value());
  }
  return LayoutMappingDoc(translator, layout,
                          NamespaceDoc("tilelang")->Attr("layout")->Attr(name),
                          keys, values);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::TypeAttrDef<LayoutNode>().attr(
      script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&LayoutDocTranslate>());
  ffi::reflection::TypeAttrDef<FragmentNode>().attr(
      script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&LayoutDocTranslate>());
  ffi::reflection::TypeAttrDef<PartialFragmentNode>().attr(
      script::printer::type_attr::kDocTranslate.name,
      FDocTranslate::FromNative<&LayoutDocTranslate>());
}
} // namespace
} // namespace tvm::tl

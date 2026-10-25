#ifndef TILELANG_LAYOUT_ANNOTATIONS_H_
#define TILELANG_LAYOUT_ANNOTATIONS_H_

#include "compat/visit.h"
#include "op/operator.h"

namespace tvm::tl {

inline LayoutMap CollectLayoutHints(const ffi::ObjectRef &root,
                                    LayoutMap layouts = {}) {
  PostOrderVisit(root, [&](const ffi::ObjectRef &object) {
    const auto *call = object.as<CallNode>();
    if (!call || !call->op.same_as(Op::Get("tl.layout_hint")))
      return;
    TVM_FFI_CHECK(call->args.size() == 1 && call->attrs, ValueError)
        << "Malformed layout hint";
    auto tensor = call->args[0].as_or_throw<tirx::TensorVar>();
    auto layout = call->attrs.value()
                      .as_or_throw<DictAttrs>()
                      .GetAttr<Layout>("layout")
                      .value();
    if (auto previous = layouts.Get(tensor)) {
      TVM_FFI_CHECK(previous.value()->IsEqual(layout.get()), ValueError)
          << "Conflicting layout hints for tensor " << tensor->name;
    }
    layouts.Set(tensor, layout);
  });
  return layouts;
}

} // namespace tvm::tl

#endif

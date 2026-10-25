#ifndef TILELANG_LAYOUT_AXIS_H_
#define TILELANG_LAYOUT_AXIS_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>

#include <utility>

namespace tvm::tl {

class LayoutAxisNode : public ffi::Object {
public:
  LayoutAxisNode(Range domain, PrimVar variable)
      : dom(std::move(domain)), var(std::move(variable)) {}
  explicit LayoutAxisNode(ffi::UnsafeInit tag) : dom(tag), var(tag) {}

  Range dom;
  PrimVar var;

  static void RegisterReflection() {
    ffi::reflection::ObjectDef<LayoutAxisNode>()
        .def_ro("dom", &LayoutAxisNode::dom)
        .def_ro("var", &LayoutAxisNode::var);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.LayoutAxis", LayoutAxisNode,
                                    ffi::Object);
};

class LayoutAxis : public ffi::ObjectRef {
public:
  LayoutAxis(Range domain, PrimVar variable)
      : LayoutAxis(ffi::make_object<LayoutAxisNode>(std::move(domain),
                                                    std::move(variable))) {}
  explicit LayoutAxis(ffi::ObjectPtr<LayoutAxisNode> node)
      : ffi::ObjectRef(std::move(node)) {}

  operator PrimExpr() const { return operator->()->var; }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(LayoutAxis, ffi::ObjectRef,
                                                LayoutAxisNode);
};

} // namespace tvm::tl

#endif

/*!
 * \file layout/iter_var.h
 * \brief Native scalar iteration domains used only by TileLang layout algebra.
 */

#ifndef TVM_TL_LAYOUT_ITER_VAR_H_
#define TVM_TL_LAYOUT_ITER_VAR_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>

namespace tvm::tl {

/*!
 * \brief An input coordinate and its domain for Layout/Fragment construction.
 *
 * This is a TileLang inference object, not a loop or a legacy IR IterVar.
 * Its variable is the ordinary shared IR Var identity viewed as a PrimVar.
 */
class LayoutIterVarNode : public ffi::Object {
public:
  Range dom;
  PrimVar var;

  LayoutIterVarNode(Range dom, PrimVar var)
      : dom(std::move(dom)), var(std::move(var)) {}

  static void RegisterReflection() {
    ffi::reflection::ObjectDef<LayoutIterVarNode>()
        .def_ro("dom", &LayoutIterVarNode::dom)
        .def_ro("var", &LayoutIterVarNode::var);
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.LayoutIterVar", LayoutIterVarNode,
                                    ffi::Object);
};

class LayoutIterVar : public ffi::ObjectRef {
public:
  LayoutIterVar(Range dom, PrimVar var)
      : ffi::ObjectRef(ffi::make_object<LayoutIterVarNode>(std::move(dom),
                                                           std::move(var))) {}

  operator PrimExpr() const { return operator->()->var; }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(LayoutIterVar, ffi::ObjectRef,
                                             LayoutIterVarNode);
};

} // namespace tvm::tl

#endif // TVM_TL_LAYOUT_ITER_VAR_H_

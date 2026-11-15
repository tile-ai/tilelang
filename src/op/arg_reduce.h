/*!
 * \file tl/op/arg_reduce.h
 * \brief Index reductions with a joint value/index accumulator.
 */
#ifndef TVM_TL_OP_ARG_REDUCE_H_
#define TVM_TL_OP_ARG_REDUCE_H_

#include "operator.h"

namespace tvm {
namespace tl {

class ArgReduceOpNode : public TileOperatorNode {
public:
  tirx::BufferRegion src_region, dst_region;
  int dim;
  bool is_max;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.ArgReduceOp", ArgReduceOpNode,
                                    TileOperatorNode);
  static void RegisterReflection() {
    namespace refl = ffi::reflection;
    refl::ObjectDef<ArgReduceOpNode>()
        .def_ro("src_region", &ArgReduceOpNode::src_region)
        .def_ro("dst_region", &ArgReduceOpNode::dst_region)
        .def_ro("dim", &ArgReduceOpNode::dim)
        .def_ro("is_max", &ArgReduceOpNode::is_max);
  }

  tirx::Stmt Lower(const LowerArgs &args,
                   arith::Analyzer *analyzer) const final;
  LayoutMap InferLayout(const LayoutInferArgs &args,
                        InferLevel level) const final;
  TileOperator Clone() const final;
};

class ArgReduceOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(ArgReduceOp, TileOperator,
                                             ArgReduceOpNode);
  TVM_DLL ArgReduceOp(ffi::Array<PrimExpr> args,
                      ffi::Map<ffi::String, ffi::ObjectRef> annotations = {});
  static const Op &Get();
};

struct ArgReduceImpl {
  const char *name;
  bool (*match_target)(Target target);
  tirx::Stmt (*lower)(const ArgReduceOpNode &op, const LowerArgs &args,
                      arith::Analyzer *analyzer);
};
void RegisterArgReduceImpl(ArgReduceImpl impl);

} // namespace tl
} // namespace tvm
#endif // TVM_TL_OP_ARG_REDUCE_H_

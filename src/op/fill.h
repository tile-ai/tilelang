/*!
 * \file tl/op/fill.h
 * \brief Fill operations for tensor initialization
 */

#ifndef TVM_TL_OP_FILL_H_
#define TVM_TL_OP_FILL_H_

#include "operator.h"
#include "parallel.h"

namespace tvm {
namespace tl {

using namespace tir;

/// Node class for fill operations
class FillNode : public TileOperatorNode {
public:
  tir::Buffer dst;     ///< Destination buffer to fill
  PrimExpr value;      ///< Value to fill with
  Array<Range> region; ///< Region to fill within the buffer
  static constexpr const char *_type_key = "tl.Fill";
  TVM_DECLARE_FINAL_OBJECT_INFO(FillNode, TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const;
  LayoutMap InferLayout(const LayoutInferArgs &T, InferLevel level) const;
  static const Op &Get();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<FillNode>()
        .def_ro("dst", &FillNode::dst)
        .def_ro("value", &FillNode::value)
        .def_ro("region", &FillNode::region);
  }

  bool SEqualReduce(const FillNode *other, SEqualReducer equal) const {
    return equal(dst, other->dst) && equal(value, other->value) &&
           equal(region, other->region);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(dst);
    hash_reduce(value);
    hash_reduce(region);
  }
  static constexpr bool _type_has_method_sequal_reduce = true;
  static constexpr bool _type_has_method_shash_reduce = true;

  TileOperator Clone() const;

private:
  /// Create SIMT-style parallel loop for filling
  For MakeSIMTLoop(arith::Analyzer *analyzer) const;
};

/// Wrapper class for fill operations
class Fill : public TileOperator {
public:
  TVM_DEFINE_OBJECT_REF_METHODS(Fill, TileOperator, FillNode);
  TVM_DLL Fill(Array<PrimExpr> args, BufferMap vmap);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_FILL_H_
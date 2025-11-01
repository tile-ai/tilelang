/*!
 * \file tl/op/op.h
 * \brief Tile library operations.
 *
 */

#ifndef TVM_TL_OP_REGION_H_
#define TVM_TL_OP_REGION_H_

#include "./operator.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/target/target.h>
#include <tvm/tir/buffer.h>

/**
 * Tile operator representing a memory region (buffer + ranges) used by TL
 * passes.
 *
 * Encapsulates the target tir::Buffer, the region extents as an Array<Range>,
 * and an access mask that indicates permitted or intended accesses for lowering
 * and layout inference.
 */

/**
 * Lower this RegionOp into a TIR statement representing the region access.
 *
 * @param T Lowering-time arguments (e.g., loop/build context and value
 * mappings).
 * @param analyzer Arithmetic analyzer used to simplify and reason about
 * expressions.
 * @return A tir::Stmt that implements the region access/mutation described by
 * this operator.
 */

/**
 * Infer the layout mapping for this region operator.
 *
 * Produces a LayoutMap describing how loop/axis indices map to buffer axes for
 * layout-aware scheduling and subsequent operators.
 *
 * @param T Layout inference arguments (e.g., input layouts and shapes).
 * @param level The inference detail level to use.
 * @return A LayoutMap describing inferred mappings for the operator.
 */

/**
 * Return true when this RegionOp represents the full buffer region (i.e.,
 * ranges cover the entire buffer extent).
 */

/**
 * Create a shallow copy of this operator as a TileOperator handle.
 *
 * @return A TileOperator that references a cloned RegionOpNode.
 */

/**
 * Construct a RegionOp from argument expressions and a buffer map.
 *
 * @param args Positional expressions used to instantiate the operator
 * (semantics depend on how RegionOp is invoked in TL pipelines).
 * @param vmap Mapping from Buffer to replacement Buffer or buffer metadata used
 * during creation.
 */

/**
 * Return the global Op registration for RegionOp.
 *
 * @return Reference to the registered tvm::Op describing the RegionOp.
 */
namespace tvm {
namespace tl {

using namespace tir;

class RegionOpNode : public TileOperatorNode {
public:
  Buffer buffer_;
  Array<Range> ranges_;
  int access_mask_;

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tl.RegionOp", RegionOpNode,
                                    TileOperatorNode);

  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const override;
  LayoutMap InferLayout(const LayoutInferArgs &T,
                        InferLevel level) const override;

  const Buffer &GetBuffer() const { return buffer_; }
  const Array<Range> &GetRanges() const { return ranges_; }
  int GetAccessMask() const { return access_mask_; }
  bool IsFullRegion() const;

  TileOperator Clone() const override;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<RegionOpNode>()
        .def_ro("buffer", &RegionOpNode::buffer_)
        .def_ro("ranges", &RegionOpNode::ranges_)
        .def_ro("access_mask", &RegionOpNode::access_mask_);
  }
};

class RegionOp : public TileOperator {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(RegionOp, TileOperator,
                                             RegionOpNode);
  TVM_DLL RegionOp(Array<PrimExpr> args, BufferMap vmap);

  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_REGION_H_

/*!
 * \file tl/op/region.h
 * \brief Tile memory region descriptor op (bridge to carry BufferRegion via
 * Call args).
 *
 * Why tl.region instead of passing BufferRegion directly?
 *
 * - While TIR can represent a BufferRegion, when a BufferRegion is passed as a
 *   call argument through call_intrin/FFI, the Python->C++ conversion lowers it
 *   to a BufferLoad(indices). To encode an interval inside indices, the FFI
 *   typically uses Ramp(base, stride, lanes) to represent a contiguous slice.
 * - Ramp(lanes) may only be a constant or vscale*k (scalable vector). A general
 *   PrimExpr (e.g., H1 - H0) is not allowed as lanes, so dynamic extents would
 *   make the lowered BufferLoad invalid.
 * - Moreover, BufferLoad only carries indices, not per-axis extents. Downstream
 *   tile operators (e.g., tl.copy, tl.reduce) that require both min and extent
 *   cannot losslessly recover dynamic extents from a BufferLoad alone.
 *
 * tl.region is a small transport-only op that solves this:
 * - The frontend packs buffer + mins (from BufferLoad.indices) + extents into
 *   Call args, allowing dynamic extents to be expressed explicitly.
 * - The backend (NormalizeToBufferRegion) reconstructs a BufferRegion from the
 *   tl.region call without losing information.
 * - The op itself carries no semantics in Lower/InferLayout and is only used as
 *   a bridge for argument passing.
 */

#ifndef TVM_TL_OP_REGION_H_
#define TVM_TL_OP_REGION_H_

#include "./operator.h"
#include <tvm/tir/buffer.h>

namespace tvm {
namespace tl {

using namespace tir;

class RegionOpNode : public TileOperatorNode {
public:
  Buffer buffer_;
  Array<Range> ranges_;
  int access_mask_;

  /*!
   * access_mask_ encodes the intended access type when the region is used as
   * an argument to tile operators: 1=read, 2=write, 3=read-write. The mask is
   * transport metadata only and does not affect lowering.
   */

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
  /*!
   * Build a RegionOp from call arguments:
   * - args[0]: BufferLoad whose indices are per-axis minima.
   * - args[1]: Integer access mask (1=r, 2=w, 3=rw).
   * - args[2 + i]: Extent of axis i (supports dynamic PrimExpr).
   */
  TVM_DLL RegionOp(Array<PrimExpr> args);
  static const Op &Get();
};

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_REGION_H_

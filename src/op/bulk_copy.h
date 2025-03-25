// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/bulk_copy.h
 * \brief Bulk copy operator.
 *
 */

#ifndef TVM_TL_OP_BULK_COPY_H_
#define TVM_TL_OP_BULK_COPY_H_

#include "elem.h"

namespace tvm {
namespace tl {

using namespace tir;

struct TMADesc {
  size_t rank;
  int data_type;
  Array<PrimExpr> global_shape, global_stride;
  Array<PrimExpr> smem_box, smem_stride;
  PrimExpr global_addr;
  int swizzle;
  int interleave;
  int oob_fill;
  int l2_promotion;

  Array<PrimExpr> EncodeCallArgs() const;
};

DataType cuTensorMapType();

struct TMAIm2ColDesc {
  size_t rank;
  int data_type;
  Array<PrimExpr> global_shape, global_stride, elem_stride; // rank
  Array<PrimExpr> lower_corner, upper_corner;               // rank - 2
  PrimExpr global_addr;
  int smem_box_pixel, smem_box_channel;
  int swizzle;
  int interleave;
  int oob_fill;
  int l2_promotion;

  Array<PrimExpr> EncodeCallArgs() const;
};

class Conv2DIm2ColOp : public Operator {
public:
  Conv2DIm2ColOp(Array<PrimExpr> args, BufferMap vmap);
  Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const final;
  static const Op &Get();

private:
  Buffer src, dst;
  int stride, padding, dilation, kernel;
  PrimExpr nhw_step, c_step;
};

} // namespace tl
} // namespace tvm

#endif //  TVM_TL_OP_BULK_COPY_H_
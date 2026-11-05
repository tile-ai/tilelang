/*!
 * \file tl/ir/tensor_utils.h
 * \brief Native tensor identities, regions and typed address construction.
 */

#ifndef TILELANG_IR_TENSOR_UTILS_H_
#define TILELANG_IR_TENSOR_UTILS_H_

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/expr.h>

namespace tvm::tl {

/*! \brief Recover the ordinary Var identity of a tensor region. */
inline tirx::TensorVar RegionTensor(const TensorRegion &region) {
  return region->source.as_or_throw<tirx::TensorVar>();
}

/*! \brief The full logical region, retaining each shape expression's type. */
inline TensorRegion FullTensorRegion(const tirx::TensorVar &tensor) {
  ffi::Array<Range> ranges;
  for (const PrimExpr &extent : tensor.shape()) {
    ranges.push_back(Range::FromMinExtent(IntImm(extent.ty(), 0), extent));
  }
  return TensorRegion(tensor, ranges, TensorRegionType(), tensor->loc);
}

/*!
 * \brief Get one physical element offset from a fully materialized tensor type.
 *
 * The caller must have committed every relevant TileLang Layout/Fragment from
 * its inference map into the tensor definition and all uses. An absent native
 * layout denotes real row-major storage only after that boundary; it must not
 * be interpreted as the physical layout of an unsolved TileLang fragment.
 * Multi-axis distributed layouts are not a flat address and are rejected.
 */
inline PrimExpr TensorElementOffset(const tirx::TensorVar &tensor,
                                    ffi::Array<PrimExpr> indices) {
  ffi::Array<PrimExpr> offsets = tensor.ty()->ElemOffset(std::move(indices));
  TVM_FFI_CHECK_EQ(offsets.size(), 1U, ValueError)
      << "A physical tensor pointer needs one materialized memory-axis offset";
  return offsets[0];
}

/*!
 * \brief Carry a tensor address and its access contract as explicit operands.
 *
 * The base TensorLoad denotes an address, not a memory read. tl.access_ptr is
 * eliminated after tensor layout/alias rewriting; its result is a PtrType
 * matching the tensor's element type and scope. extent is in tensor elements,
 * and rw_mask uses the low read/write bits (1/2). No data-Var map is involved.
 */
inline Expr MakeTensorAccessPtr(const TensorLoad &base, PrimExpr extent,
                                int rw_mask) {
  tirx::TensorVar tensor = base->source.as_or_throw<tirx::TensorVar>();
  TVM_FFI_CHECK_GE(rw_mask, 0, ValueError);
  TVM_FFI_CHECK_LE(rw_mask, 3, ValueError);
  return Call(tensor.ty()->DataPtrType(), Op::Get("tl.access_ptr"),
              {base, std::move(extent), IntImm(PrimType::Int(32), rw_mask)}, {},
              {}, base->loc);
}

inline Expr MakeTensorAccessPtr(const TensorRegion &region, PrimExpr extent,
                                int rw_mask) {
  ffi::Array<PrimExpr> starts;
  for (const Range &range : region->region) {
    starts.push_back(range->begin);
  }
  return MakeTensorAccessPtr(
      TensorLoad(RegionTensor(region), starts, region->loc), std::move(extent),
      rw_mask);
}

/*! \brief Preserve a typed raw pointer's element interval and read/write mask.
 */
inline Expr MakeRawAccessPtr(Expr pointer, IntExpr element_offset,
                             IntExpr extent, int rw_mask) {
  TVM_FFI_CHECK_GE(rw_mask, 0, ValueError);
  TVM_FFI_CHECK_LE(rw_mask, 3, ValueError);
  const PtrType type = pointer->ty.as_or_throw<PtrType>();
  const Location loc = pointer->loc;
  return Call(type, Op::Get("tl.ptr_access"),
              {std::move(pointer), std::move(element_offset), std::move(extent),
               IntImm(PrimType::Int(32), rw_mask)},
              {}, {}, loc);
}

} // namespace tvm::tl

#endif // TILELANG_IR_TENSOR_UTILS_H_

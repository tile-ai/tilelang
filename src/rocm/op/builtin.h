/*!
 * \file tl/rocm/op/builtin.h
 * \brief ROCm-specific TileLang intrinsic Ops.
 */

#ifndef TVM_TL_ROCM_OP_BUILTIN_H_
#define TVM_TL_ROCM_OP_BUILTIN_H_

#include "op/builtin.h"

namespace tvm {
namespace tl {

/*!
 * \brief tilelang intrinsic for gfx950 LDS transpose read, 64-bit, 16-element.
 *
 * Reads 8 bytes from LDS with a 16-element transpose (FP16/BF16 MFMA B-load).
 * Only available on gfx950 (MI350/MI355X).
 *
 * uint32x2 ds_read_tr16_b64(smem_access_ptr)
 */
TVM_DLL const Op &ds_read_tr16_b64();

/*!
 * \brief tilelang intrinsic for gfx950 LDS transpose read, 64-bit, 8-element.
 *
 * Reads 8 bytes from LDS with an 8-element transpose (FP32 MFMA B-load).
 * Only available on gfx950 (MI350/MI355X).
 *
 * uint32x2 ds_read_tr8_b64(smem_access_ptr)
 */
TVM_DLL const Op &ds_read_tr8_b64();

/*!
 * \brief tvm intrinsic for amd matrix core mfma instructions.
 *
 *  void tvm_mfma(StringImm shape, StringImm A_layout, StringImm B_layout,
 *               StringImm A_dtype, StringImm B_dtype, StringImm C_dtype,
 *               Var multiplicand_a, Expr a_index,
 *               Var multiplicand_b, Expr b_index,
 *               Var accumulator, Expr c_index);
 */
TVM_DLL const Op &tvm_mfma();

/*!
 * \brief tvm intrinsic for storing the result of AMD MFMA into a destination
 * pointer.
 *
 *        There is no real instruction that does that, but we want to hide
 * details of complex index manipulation behind this intrinsic to simplify TIR
 * lowering passes (e.g. LowerWarpMemory) like cuda ptx backend does.
 *
 * void tvm_mfma_store(IntImm m, IntImm n, Var dst_ptr, Var src_ptr, Expr
 * src_offset, Var dst_stride);
 */
TVM_DLL const Op &tvm_mfma_store();

/*!
 * \brief tvm intrinsic for amd rdna matrix core instructions.
 *
 *  void tvm_rdna_wmma(StringImm shape, StringImm A_layout, StringImm B_layout,
 *               StringImm A_dtype, StringImm B_dtype, StringImm C_dtype,
 *               Var multiplicand_a, Expr a_index,
 *               Var multiplicand_b, Expr b_index,
 *               Var accumulator, Expr c_index);
 */
TVM_DLL const Op &tvm_rdna_wmma();

/*!
 * \brief tvm intrinsic for storing the result of AMD RDNA WMMA into a
 * destination pointer.
 *
 *        There is no real instruction that does that, but we want to hide
 * details of complex index manipulation behind this intrinsic to simplify TIR
 * lowering passes (e.g. LowerWarpMemory) like cuda ptx backend does.
 *
 * void tvm_rdna_wmma_store(IntImm m, IntImm n, Var dst_ptr, Var src_ptr, Expr
 * src_offset, Var dst_stride);
 */
TVM_DLL const Op &tvm_rdna_wmma_store();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ROCM_OP_BUILTIN_H_

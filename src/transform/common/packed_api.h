#ifndef TL_TRANSFORM_COMMON_PACKED_API_H_
#define TL_TRANSFORM_COMMON_PACKED_API_H_

#include <tvm/ir/prim/expr.h>
#include <tvm/ir/stmt.h>

namespace tvm::tl::tirx_passes {

/*! \brief Decoded packed arguments, with definitions preceding dependent checks
 * and declarations.
 */
struct PackedAPIArguments {
  ffi::Map<Var, Expr> bindings;
  ffi::Array<Stmt> definitions;
  ffi::Array<Stmt> assertions;
  ffi::Array<Stmt> tensor_declarations;
};

/*!
 * \brief Decode and validate packed arguments using the native tensor and
 * symbolic-shape ABI.
 * \param storage_dtypes Optional physical element types; packed elements share
 * the innermost axis.
 * \param nullable_parameters Tensor parameters whose storage is unused by the
 * function.
 * \param required_metadata Live metadata variables that require a provider when
 * shared.
 */
PackedAPIArguments
DecodePackedAPIArguments(const ffi::String &name,
                         const ffi::Array<Var> &parameters,
                         const Var &packed_args, const Var &num_args,
                         const PrimExpr &device_type, const PrimExpr &device_id,
                         const ffi::Map<Var, PrimType> &storage_dtypes = {},
                         const ffi::Array<Var> &nullable_parameters = {},
                         const ffi::Array<Var> &required_metadata = {});

} // namespace tvm::tl::tirx_passes

#endif

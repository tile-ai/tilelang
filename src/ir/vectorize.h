/*!
 * \file tl/ir/vectorize.h
 * \brief TileLang-owned packing rules for explicit vector operations.
 */
#ifndef TILELANG_IR_VECTORIZE_H_
#define TILELANG_IR_VECTORIZE_H_

#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/transform.h>
#include <tvm/target/target.h>

namespace tvm::tl {

// Native TIRx only vectorizes the subset accepted by its scalar-loop analysis.
// TileLang's calls require owning rules for transaction width and side effects.
using FCallVectorize = ffi::TypedFunction<ffi::Optional<Expr>(
    Call, Var, PrimExpr, ffi::Optional<Target>,
    ffi::TypedFunction<Expr(Expr)>)>;
inline constexpr ffi::reflection::AttrKey<FCallVectorize> kCallVectorize{
    "tl.call_vectorize"};

// Preserve TileLang packing contracts before the general upstream vectorizer.
tvm::transform::Pass VectorizeLoop(bool enable_vectorize);

} // namespace tvm::tl

#endif // TILELANG_IR_VECTORIZE_H_

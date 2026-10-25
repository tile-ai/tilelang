#ifndef TILELANG_BACKEND_COMMON_OP_REDUCE_VALUE_H_
#define TILELANG_BACKEND_COMMON_OP_REDUCE_VALUE_H_

#include "cuda/op/builtin.h"
#include "op/builtin.h"
#include "op/reduce.h"
#include "op/reducer.h"

#include <optional>
#include <string>

namespace tvm::tl::backend::reduce {

inline bool UseNanOp(const ReduceOpNode &operation, PrimType type) {
  return operation.nan_propagate && (type.MatchesElementType(kDLFloat, 16) ||
                                     type.MatchesElementType(kDLBfloat, 16));
}

inline PrimExpr MakeInitValue(const ReduceOpNode &operation,
                              int vector_size = 1) {
  TVM_FFI_CHECK_GE(vector_size, 1, ValueError)
      << "Reduction vector size must be positive";
  ReducerV2OpType kind = ReducerV2OpType::kSum;
  if (operation.type->IsMax()) {
    kind = ReducerV2OpType::kMax;
  } else if (operation.type->IsMin()) {
    kind = ReducerV2OpType::kMin;
  } else if (operation.type->IsBitAnd()) {
    kind = ReducerV2OpType::kBitAnd;
  } else if (operation.type->IsBitOr()) {
    kind = ReducerV2OpType::kBitOr;
  } else if (operation.type->IsBitXor()) {
    kind = ReducerV2OpType::kBitXor;
  } else {
    TVM_FFI_CHECK(operation.type->IsSum() || operation.type->IsAbsSum() ||
                      operation.type->IsAbsMax(),
                  ValueError)
        << "Unsupported reduction type";
  }
  PrimExpr identity = ReducerV2Identity(kind, operation.dst.dtype());
  return vector_size == 1 ? identity
                          : PrimExpr(Broadcast(identity, vector_size));
}

inline PrimExpr MakeReduce(const ReduceOpNode &operation, int vector_size,
                           const PrimExpr &accumulator, const PrimExpr &value) {
  TVM_FFI_CHECK(vector_size == 1 || vector_size == 2, ValueError)
      << "Unsupported reduction vector size: " << vector_size;
  PrimExpr contribution = Cast(accumulator.ty(), value);
  auto invoke = [&](const Op &op, Array<Expr> arguments) {
    return Call(accumulator.ty(), op, arguments).as_or_throw<PrimExpr>();
  };
  bool use_nan = UseNanOp(operation, accumulator.ty());
  if (vector_size == 1) {
    if (operation.type->IsAbsSum() || operation.type->IsAbsMax()) {
      if (!contribution.ty().MatchesCode(kDLUInt)) {
        contribution = Max(contribution, -contribution);
      }
    }
    if (operation.type->IsSum() || operation.type->IsAbsSum()) {
      return accumulator + contribution;
    }
    if (operation.type->IsMax() || operation.type->IsAbsMax()) {
      return use_nan ? invoke(tl::max_nan(), {accumulator, contribution})
                     : PrimExpr(Max(accumulator, contribution));
    }
    if (operation.type->IsMin()) {
      return use_nan ? invoke(tl::min_nan(), {accumulator, contribution})
                     : PrimExpr(Min(accumulator, contribution));
    }
    if (operation.type->IsBitAnd()) {
      return accumulator & contribution;
    }
    if (operation.type->IsBitOr()) {
      return accumulator | contribution;
    }
    if (operation.type->IsBitXor()) {
      return accumulator ^ contribution;
    }
  } else {
    if (operation.type->IsAbsSum() || operation.type->IsAbsMax()) {
      contribution = invoke(tl::abs2(), {contribution});
    }
    if (operation.type->IsSum() || operation.type->IsAbsSum()) {
      return invoke(tl::add2(), {accumulator, contribution});
    }
    if (operation.type->IsMax() || operation.type->IsAbsMax()) {
      return invoke(use_nan ? tl::max2_nan() : tl::max2(),
                    {accumulator, contribution});
    }
    if (operation.type->IsMin()) {
      return invoke(use_nan ? tl::min2_nan() : tl::min2(),
                    {accumulator, contribution});
    }
  }
  TVM_FFI_THROW(ValueError)
      << "Unsupported reduction type for vector size " << vector_size;
  TVM_FFI_UNREACHABLE();
}

inline std::optional<std::string>
MakeCodegenReducer(const ReduceOpNode &operation, int vector_size = 1) {
  bool use_nan = UseNanOp(operation, operation.dst.dtype());
  std::string name;
  if (operation.type->IsSum() || operation.type->IsAbsSum()) {
    name = "tl::SumOp";
  } else if (operation.type->IsMax() || operation.type->IsAbsMax()) {
    name = use_nan ? "tl::MaxOpNan" : "tl::MaxOp";
  } else if (operation.type->IsMin()) {
    name = use_nan ? "tl::MinOpNan" : "tl::MinOp";
  } else if (operation.type->IsBitAnd()) {
    name = "tl::BitAndOp";
  } else if (operation.type->IsBitOr()) {
    name = "tl::BitOrOp";
  } else if (operation.type->IsBitXor()) {
    name = "tl::BitXorOp";
  } else {
    TVM_FFI_THROW(ValueError) << "Unsupported reduction type";
  }
  if (vector_size == 1) {
    return name;
  }
  if (vector_size != 2 || operation.type->IsBitAnd() ||
      operation.type->IsBitOr() || operation.type->IsBitXor()) {
    return std::nullopt;
  }
  PrimType type = operation.dst.dtype();
  if (type.MatchesElementType(kDLFloat, 32)) {
    return name + "_f32x2";
  }
  if (type.MatchesElementType(kDLBfloat, 16)) {
    return name + "_bf16x2";
  }
  if (type.MatchesElementType(kDLFloat, 16)) {
    return name + "_fp16x2";
  }
  return std::nullopt;
}

} // namespace tvm::tl::backend::reduce

#endif

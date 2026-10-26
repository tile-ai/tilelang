#include "compat/analyzer.h"
/*!
 * \file vectorize_loop.cc
 */
// Loop vectorizer as in Halide pipeline.
#include "op/prim.h"
#include "transform/tirx_passes.h"

#include "../op/builtin.h"
#include "../op/utils.h"
#include "common/int64_promoter.h"
#include "loop_vectorize.h"
#include "support/check.h"

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

#include <optional>

namespace tvm::tl {

using namespace ffi;
using namespace prim;

namespace {

class VectorCallContext {
public:
  VectorCallContext(PrimVar variable, PrimExpr lanes,
                    TypedFunction<Expr(Expr)> rewrite, Optional<Target> target)
      : variable_(std::move(variable)), lanes_(std::move(lanes)),
        rewrite_(std::move(rewrite)), target_(std::move(target)) {
    if (!target_)
      target_ = Target::Current();
    tl::arith::Bind(analyzer_, variable_,
                    Range(tl::MakeConst(variable_.ty(), 0), lanes_));
  }

  std::optional<int64_t> Width() const { return GetConstInt(lanes_); }

  Optional<Target> target() const { return target_; }

  Optional<PrimExpr> VectorValue(const Expr &expression, int width) const {
    auto value = rewrite_(expression).as<PrimExpr>();
    if (!value)
      return std::nullopt;
    if (value.value().ty().lanes() == width)
      return value;
    if (!value.value().ty().IsScalar())
      return std::nullopt;
    return Broadcast(value.value(), width);
  }

  Optional<PrimExpr> ScalarValue(const Expr &expression) const {
    auto value = expression.as<PrimExpr>();
    if (!value || !CanProveIndependent(value.value(), variable_, analyzer_))
      return std::nullopt;
    auto rewritten = rewrite_(expression).as<PrimExpr>();
    if (!rewritten || !rewritten.value().ty().IsScalar())
      return std::nullopt;
    return rewritten;
  }

  Optional<Expr> Pointer(const Expr &expression, int elements_per_lane,
                         int mask) const {
    auto load = ExtractBufferLoadForAtomic(expression);
    auto width = Width();
    if (!load || !width || width.value() <= 0 || width.value() > 32)
      return std::nullopt;
    TensorVar tensor = load.value()->source.as_or_throw<TensorVar>();
    auto indices = load.value()->indices.Map(
        [](const PrimExpr &index) { return Int64Promoter()(index); });
    auto offsets = tensor.ty()->ElemOffset(indices);
    if (offsets.size() != 1 || !tensor.dtype().IsScalar())
      return std::nullopt;
    PrimExpr offset = offsets[0];
    PrimExpr base_offset =
        Substitute(offset, {{variable_, tl::MakeConst(variable_.ty(), 0)}});
    if (!tl::arith::CanProveEqual(analyzer_,
                                  floormod(base_offset, elements_per_lane), 0))
      return std::nullopt;
    if (!IndicesCanVectorize(floordiv(offset, elements_per_lane), variable_,
                             lanes_, static_cast<int>(width.value()), analyzer_,
                             false, tl::MakeConst(variable_.ty(), 0)))
      return std::nullopt;
    PrimExpr contiguous =
        base_offset + tl::CastValue(variable_, offset.ty()) * elements_per_lane;
    if (!tl::arith::CanProveEqual(analyzer_, offset, contiguous))
      return std::nullopt;
    Array<PrimExpr> base_indices =
        load.value()->indices.Map([&](const PrimExpr &index) {
          return Substitute(index,
                            {{variable_, tl::MakeConst(variable_.ty(), 0)}});
        });
    TensorLoad base(tensor, base_indices, expression->loc);
    Map<String, Any> attributes;
    if (auto call = expression.as<Call>(); call && call.value()->attrs) {
      attributes = call.value()->attrs.value().as_or_throw<DictAttrs>()->dict;
    }
    attributes.Set(attr::kAccessRegion, NormalizeToBufferRegion(tensor));
    attributes.Set(
        attr::kAccessExtent,
        tl::MakeConst(variable_.ty(), width.value() * elements_per_lane));
    attributes.Set(attr::kAccessMask, tl::MakeConst(variable_.ty(), mask));
    return Call(tensor.ty()->DataPtrType(), tirx::address_of_op(), {base},
                DictAttrs(attributes), {}, expression->loc);
  }

private:
  PrimVar variable_;
  PrimExpr lanes_;
  TypedFunction<Expr(Expr)> rewrite_;
  Optional<Target> target_;
  sym::Analyzer analyzer_;
};

Optional<Expr> VectorizeAtomicAdd(Call call, PrimVar variable, PrimExpr lanes,
                                  TypedFunction<Expr(Expr)> rewrite,
                                  Optional<Target> target) {
  VectorCallContext context(variable, lanes, rewrite, target);
  auto width = context.Width();
  auto load = ExtractBufferLoadForAtomic(call->args[0]);
  if (!width || !load || !context.target())
    return std::nullopt;
  int64_t count = width.value();
  if (count != 2 && count != 4)
    return std::nullopt;
  TensorVar tensor = load.value()->source.as_or_throw<TensorVar>();
  if (count > GetMaxAtomicVectorSize(tensor.dtype(), tensor.scope().name(),
                                     context.target().value())) {
    return std::nullopt;
  }
  auto destination = context.Pointer(call->args[0], 1, kAccessReadWrite);
  auto value = context.VectorValue(call->args[1], count);
  if (!destination || !value)
    return std::nullopt;
  Array<Expr> arguments{destination.value(), value.value()};
  for (size_t index = 2; index < call->args.size(); ++index) {
    auto argument = context.ScalarValue(call->args[index]);
    if (!argument)
      return std::nullopt;
    arguments.push_back(argument.value());
  }
  Op operation = count == 2 ? atomic_addx2_elem_op() : atomic_addx4_elem_op();
  return Call(call->ty, operation, arguments, call->attrs, call->ty_args,
              call->loc);
}

Optional<Expr> VectorizeAsyncCopy(Call call, PrimVar variable, PrimExpr lanes,
                                  TypedFunction<Expr(Expr)> rewrite,
                                  Optional<Target> target) {
  if (call->args.size() != 3 && call->args.size() != 4)
    return std::nullopt;
  VectorCallContext context(variable, lanes, rewrite, target);
  auto width = context.Width();
  auto count = GetConstInt(call->args[2].as_or_throw<PrimExpr>());
  if (!width || !count || width.value() < 1 || count.value() < 1)
    return std::nullopt;
  auto destination_load = ExtractBufferLoadForAtomic(call->args[0]);
  auto source_load = ExtractBufferLoadForAtomic(call->args[1]);
  if (!destination_load || !source_load)
    return std::nullopt;
  TensorVar destination_tensor =
      destination_load.value()->source.as_or_throw<TensorVar>();
  TensorVar source_tensor =
      source_load.value()->source.as_or_throw<TensorVar>();
  int bits = source_tensor.dtype().bits();
  if (!source_tensor.dtype().IsScalar() ||
      !destination_tensor.dtype().IsScalar() ||
      bits != destination_tensor.dtype().bits())
    return std::nullopt;
  if (width.value() > 128 / bits ||
      count.value() > 128 / (bits * width.value()))
    return std::nullopt;
  int total_bits = bits * count.value() * width.value();
  if (total_bits % 8 != 0 || !IsValidCPAsyncTransferBytes(total_bits / 8))
    return std::nullopt;
  auto destination =
      context.Pointer(call->args[0], count.value(), kAccessWrite);
  auto source = context.Pointer(call->args[1], count.value(), kAccessRead);
  if (!destination || !source)
    return std::nullopt;
  Array<Expr> arguments{
      destination.value(), source.value(),
      tl::MakeConst(call->args[2].as_or_throw<PrimExpr>().ty(),
                    count.value() * width.value())};
  if (call->args.size() == 4) {
    auto predicate = context.ScalarValue(call->args[3]);
    if (!predicate)
      return std::nullopt;
    arguments.push_back(predicate.value());
  }
  return Call(call->ty, call->op, arguments, call->attrs, call->ty_args,
              call->loc);
}

Optional<Expr> VectorizeRoundedCast(Call call, PrimVar variable, PrimExpr lanes,
                                    TypedFunction<Expr(Expr)> rewrite,
                                    Optional<Target> target) {
  VectorCallContext context(variable, lanes, rewrite, target);
  auto width = context.Width();
  if (!width || *width > 8 ||
      (*width != 1 && *width != 2 && *width != 4 && *width != 8))
    return std::nullopt;
  auto result_type = call->ty.as_or_throw<PrimType>();
  if (result_type.bits() == 4 && *width > 4)
    return std::nullopt;
  auto value = context.VectorValue(call->args[0], *width);
  if (!value)
    return std::nullopt;
  Array<Expr> arguments{value.value()};
  if (call->args.size() == 2) {
    auto random_bits = context.ScalarValue(call->args[1]);
    if (!random_bits)
      return std::nullopt;
    arguments.push_back(random_bits.value());
  }
  return Call(result_type.WithLanes(*width), call->op, arguments, call->attrs,
              call->ty_args, call->loc);
}

} // namespace

tvm::transform::Pass VectorizeLoop(bool enable_vectorize) {
  return tirx_passes::VectorizeLoop(enable_vectorize);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.cast_with_rounding")
      .set_attr(tirx_passes::kVectorizeCall,
                tirx_passes::FVectorizeCall(VectorizeRoundedCast));
  OpDef(atomic_add_elem_op()->name)
      .set_attr(tirx_passes::kVectorizeCall,
                tirx_passes::FVectorizeCall(VectorizeAtomicAdd));
  OpDef(ptx_cp_async()->name)
      .set_attr(tirx_passes::kVectorizeCall,
                tirx_passes::FVectorizeCall(VectorizeAsyncCopy));
  ffi::reflection::GlobalDef().def(
      "tl.transform.VectorizeLoop",
      [](bool enable) { return VectorizeLoop(enable); });
}

} // namespace tvm::tl

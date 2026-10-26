#include "compat/analyzer.h"
#include "op/prim.h"
#include <tvm/ffi/cast.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/prim/pattern_match.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor_with_analyzer.h>
#include <tvm/tirx/transform.h>

#include <limits>
#include <unordered_set>

#include "transform/common/loop_aware_mutator.h"
#include "transform/tirx_passes.h"

namespace tvm::tl::tirx_passes {
using namespace tvm::tirx;
using tvm::tirx::transform::CreateFunctionPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
namespace {

using namespace tvm::prim;

class IntrinInjecter : public LoopAwareMutator {
public:
  using LoopAwareMutator::Mutate;
  using LoopAwareMutator::Mutate_;

  using FLowerGeneral = ffi::TypedFunction<PrimExpr(PrimExpr)>;

  IntrinInjecter(const sym::Analyzer &analyzer, const Target &tgt,
                 bool enable_fast_math)
      : LoopAwareMutator(analyzer) {
    std::string target = tgt->kind->name;
    ffi::String mtriple = tgt->GetAttr<ffi::String>("mtriple").value_or("");

    std::vector<std::string> patterns;

    if (enable_fast_math) {
      patterns.push_back("tl." + target + ".fastmath.FLowerIntrinsic");
      patterns.push_back(target + ".fastmath.FLowerIntrinsic");
      patterns.push_back(target + ".fastmath.FLegalize");
    }
    patterns.push_back("tl." + target + ".FLowerIntrinsic");
    patterns.push_back(target + ".FLowerIntrinsic");
    patterns.push_back(target + ".FLegalize");

    bool is_llvm_aarch64 = (mtriple.find("aarch64") != std::string::npos);
    if (is_llvm_aarch64) {
      patterns.push_back(target + ".aarch64.FLowerIntrinsic");
      patterns.push_back(target + ".aarch64.FLegalize");
    }
    patterns.push_back("tl.default.FLowerIntrinsic");
    patterns.push_back("default.FLowerIntrinsic");
    patterns.push_back("default.FLegalize");

    for (const std::string &pattern : patterns)
      if (Op::HasAttrMap(pattern)) {
        attr_maps_.push_back(Op::GetAttrMap<FLowerGeneral>(pattern));
        if (fma_ == nullptr) {
          static const Op fma_op = Op::Get("prim.fma");
          fma_ = (*attr_maps_.rbegin()).get(fma_op, nullptr);
        }
      }
  }

  UnchangedOr<Expr> Mutate_(const CallNode *op,
                            InplaceMode inplace_mode) final {
    if (auto *ptr_op = op->op.as<OpNode>()) {
      Op op_ref = ffi::GetRef<Op>(ptr_op);
      Expr e = ffi::GetRef<Call>(op);
      if (auto prim_e = e.as<PrimExpr>()) {
        for (const auto &f_attr_map : attr_maps_) {
          FLowerGeneral f = f_attr_map.get(op_ref, nullptr);
          if (f != nullptr) {
            PrimExpr r = f(prim_e.value());
            TVM_FFI_ICHECK(r.defined())
                << "intrinsic rule must always return valid Expr";
            if (!r.same_as(prim_e.value())) {
              r = this->Mutate(r, inplace_mode).ValueOrUnchanged(r);
              if (r.defined()) {
                return r;
              }
            }
          }
        }
      }
    }
    return LoopAwareMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::AddNode *op,
                                InplaceMode inplace_mode) final {
    if (const prim::MulNode *mb = op->b.as<prim::MulNode>()) {
      return MakeFMA(mb->a, mb->b, op->a, op);
    } else if (const prim::MulNode *ma = op->a.as<prim::MulNode>()) {
      return MakeFMA(ma->a, ma->b, op->b, op);
    }
    return LoopAwareMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::FloorDivNode *op,
                                InplaceMode inplace_mode) final {
    auto e = ffi::GetRef<PrimExpr>(op);
    PrimExpr ret = LoopAwareMutator::Mutate_(op, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op));
    op = ret.as<prim::FloorDivNode>();
    if (op == nullptr)
      return ret;
    int shift;
    PrimType dtype = op->ty.as_or_throw<PrimType>();
    TVM_FFI_ICHECK(
        dtype.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt));

    if (support_bitwise_op_ && IsPowerOfTwoInt(op->b, &shift)) {

      return op->a >> IntImm(dtype, shift);
    }

    if (tl::arith::CanProveGreaterEqual(analyzer_, op->b, 0)) {

      if (tl::arith::CanProveGreaterEqual(analyzer_, op->a, 0) ||
          tl::arith::CanProveGreaterEqual(analyzer_, e, 0)) {
        return truncdiv(op->a, op->b);
      }
      const auto *b_as_intimm = op->b.as<IntImmNode>();
      if (auto b_value =
              b_as_intimm ? b_as_intimm->value.as<int64_t>() : std::nullopt;
          b_value.has_value()) {
        if (auto opt_c_value =
                TryFindShiftCoefficientForPositiveRange(op->a, *b_value);
            opt_c_value.has_value()) {
          int64_t c_value = *opt_c_value;

          return truncdiv(op->a + IntImm(dtype, b_as_intimm->value * c_value),
                          op->b) -
                 IntImm(dtype, c_value);
        }
      }
      DLOG(INFO) << "LowerFloorDiv: Cannot decide the sign of divident";
      PrimExpr rdiv = truncdiv(op->a, op->b);
      PrimExpr rmod = truncmod(op->a, op->b);

      if ((dtype == PrimType::Int(32) || dtype == PrimType::Int(64)) &&
          support_bitwise_op_) {

        return rdiv + (rmod >> IntImm(dtype, dtype.bits() - 1));
      } else {
        return prim::Select(rmod >= 0, rdiv, rdiv - tl::MakeConst(dtype, 1));
      }

    } else {
      if (dtype.code() == DLDataTypeCode::kDLFloat) {

        PrimExpr lowered = tvm::prim::floor(op->a / op->b);
        return Mutate(lowered, inplace_mode).ValueOrUnchanged(lowered);
      } else {

        DLOG(INFO) << "LowerFloorDiv: Cannot decide the sign of divisor";
        PrimVar rmod("rmod", dtype);
        PrimVar rdiv("rdiv", dtype);

        PrimExpr let_rdiv = prim::Let(
            rdiv, truncdiv(op->a, op->b),
            prim::Select((op->b >= 0 && rmod >= 0) || (op->b < 0 && rmod <= 0),
                         rdiv, rdiv - tl::MakeConst(dtype, 1)));
        return prim::Let(rmod, truncmod(op->a, op->b), let_rdiv);
      }
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::FloorModNode *op,
                                InplaceMode inplace_mode) final {
    PrimExpr ret = LoopAwareMutator::Mutate_(op, inplace_mode)
                       .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op));
    op = ret.as<prim::FloorModNode>();
    if (op == nullptr)
      return ret;

    int shift;
    PrimType dtype = op->ty.as_or_throw<PrimType>();
    TVM_FFI_ICHECK(
        dtype.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt));

    if (support_bitwise_op_ && IsPowerOfTwoInt(op->b, &shift)) {

      ffi::BigInt mask = (ffi::BigInt(1) << shift) - 1;
      return op->a & IntImm(dtype, mask);
    }

    if (tl::arith::CanProveGreaterEqual(analyzer_, op->b, 0)) {

      if (tl::arith::CanProveGreaterEqual(analyzer_, op->a, 0)) {
        return truncmod(op->a, op->b);
      }
      const auto *b_as_intimm = op->b.as<IntImmNode>();
      if (auto b_value =
              b_as_intimm ? b_as_intimm->value.as<int64_t>() : std::nullopt;
          b_value.has_value()) {
        if (auto opt_c_value =
                TryFindShiftCoefficientForPositiveRange(op->a, *b_value);
            opt_c_value.has_value()) {
          int64_t c_value = *opt_c_value;

          return truncmod(op->a + IntImm(dtype, b_as_intimm->value * c_value),
                          op->b);
        }
      }
      DLOG(INFO) << "LowerFloorMod: Cannot decide the sign of divident";

      PrimExpr rmod = truncmod(op->a, op->b);
      if ((dtype == PrimType::Int(32) || dtype == PrimType::Int(64)) &&
          support_bitwise_op_) {

        return rmod + (op->b & (rmod >> IntImm(dtype, dtype.bits() - 1)));
      } else {
        return prim::Select(rmod >= 0, rmod, rmod + op->b);
      }

    } else {
      if (dtype.code() == DLDataTypeCode::kDLFloat) {

        PrimExpr lowered = tvm::prim::floor(op->a / op->b);
        return op->a -
               (Mutate(lowered, inplace_mode).ValueOrUnchanged(lowered) *
                op->b);
      } else {

        DLOG(INFO)
            << "LowerFloorMod: Cannot decide the sign of divsor and divident";
        PrimVar rmod("rmod", dtype);

        return prim::Let(
            rmod, truncmod(op->a, op->b),
            prim::Select((op->b >= 0 && rmod >= 0) || (op->b < 0 && rmod <= 0),
                         rmod, rmod + op->b));
      }
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::MaxNode *op,
                                InplaceMode inplace_mode) final {
    PVar<PrimExpr> x, y;
    PVar<IntImm> c;
    auto e = ffi::GetRef<PrimExpr>(op);
    if (max(floordiv(x, y), c).Match(e) && c.Eval()->value >= 0 &&
        tl::arith::CanProveGreaterEqual(analyzer_, y.Eval(), 0)) {
      PrimExpr input = truncdiv(x, y).Eval();
      return max(Mutate(input, inplace_mode).ValueOrUnchanged(input), c.Eval());
    }
    return LoopAwareMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::EQNode *op,
                                InplaceMode inplace_mode) final {
    PVar<PrimExpr> x, y;
    auto e = ffi::GetRef<PrimExpr>(op);
    if ((floormod(x, y) == 0).Match(e)) {
      PrimExpr input = (truncmod(x, y) == 0).Eval();
      return Mutate(input, inplace_mode).ValueOrUnchanged(input);
    }
    return LoopAwareMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::NENode *op,
                                InplaceMode inplace_mode) final {
    PVar<PrimExpr> x, y;
    auto e = ffi::GetRef<PrimExpr>(op);
    if ((floormod(x, y) != 0).Match(e)) {
      PrimExpr input = (truncmod(x, y) != 0).Eval();
      return Mutate(input, inplace_mode).ValueOrUnchanged(input);
    }
    return LoopAwareMutator::Mutate_(op, inplace_mode);
  }

private:
  PrimExpr SwapBroadcastCast(const PrimExpr &e) {

    if (const prim::BroadcastNode *bcast = e.as<prim::BroadcastNode>()) {
      if (const prim::CastNode *cast = bcast->value.as<prim::CastNode>()) {
        auto should_swap = [&]() {
          PrimType cast_ty = cast->ty.as_or_throw<PrimType>();
          PrimType value_ty = cast->value.ty();

          if (cast_ty.bits() == value_ty.bits() * 2) {
            return true;
          }

          if (cast_ty.code() != DLDataTypeCode::kDLUInt &&
              cast_ty.code() != DLDataTypeCode::kDLInt) {
            return false;
          }
          if (value_ty.code() != DLDataTypeCode::kDLUInt &&
              value_ty.code() != DLDataTypeCode::kDLInt) {
            return false;
          }

          return cast_ty.bits() > value_ty.bits();
        };

        if (should_swap()) {
          PrimExpr new_bcast = prim::Broadcast(cast->value, bcast->lanes);
          return prim::Cast(bcast->ty.as_or_throw<PrimType>(), new_bcast);
        }
      }
    }
    return e;
  }

  PrimExpr MakeFMA(const PrimExpr &a, const PrimExpr &b, const PrimExpr &c,
                   const prim::AddNode *op) {

    PrimExpr lhs = SwapBroadcastCast(a);
    PrimExpr rhs = SwapBroadcastCast(b);

    if (fma_ != nullptr &&
        op->ty.as_or_throw<PrimType>().code() == DLDataTypeCode::kDLFloat) {
      PrimExpr r =
          fma_(Call(op->ty.as_or_throw<PrimType>(), fma_op(), {lhs, rhs, c})
                   .as_or_throw<PrimExpr>());
      if (r.defined())
        return this->Mutate(r, InplaceMode::kDisallow).ValueOrUnchanged(r);
    } else {
      if (!lhs.same_as(a) || !rhs.same_as(b)) {
        PrimExpr input = prim::Mul(lhs, rhs);
        PrimExpr mul =
            this->Mutate(input, InplaceMode::kDisallow).ValueOrUnchanged(input);
        return prim::Add(
            mul, this->Mutate(c, InplaceMode::kDisallow).ValueOrUnchanged(c));
      }
    }
    return LoopAwareMutator::Mutate_(op, InplaceMode::kDisallow)
        .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op));
  }

  std::optional<int64_t>
  TryFindShiftCoefficientForPositiveRange(const PrimExpr &a, int64_t b_value) {
    if (b_value <= 0) {
      return std::nullopt;
    }

    sym::ConstIntBound const_int_bound_a =
        tl::arith::ConstIntBoundOf(analyzer_, a);
    if (const_int_bound_a->min_value >= 0) {
      return std::nullopt;
    }
    PrimType a_ty = a.ty();

    auto dtype_max = tvm::prim::max_value(PrimType(a_ty.code(), a_ty.bits()))
                         .as_or_throw<IntImm>()
                         ->value.as<int64_t>();
    if (!dtype_max.has_value())
      return std::nullopt;
    const int64_t max_value_of_dtype = *dtype_max;

    if (b_value - 1 > max_value_of_dtype + const_int_bound_a->min_value) {
      return std::nullopt;
    }
    int64_t c_value = ((b_value - 1) - const_int_bound_a->min_value) / b_value;
    TVM_FFI_ICHECK_GT(c_value, 0);

    if (c_value > max_value_of_dtype / b_value)
      return std::nullopt;

    if (const_int_bound_a->max_value > max_value_of_dtype - b_value * c_value) {

      return std::nullopt;
    }
    return c_value;
  }

  std::vector<OpAttrMap<FLowerGeneral>> attr_maps_;
  FLowerGeneral fma_{nullptr};
  bool support_bitwise_op_{true};
};

} // namespace

Pass LowerIntrin() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    auto *n = f.CopyOnWrite();
    auto target = f->GetAttr(tvm::attr::kTarget);
    TVM_FFI_ICHECK(target.has_value())
        << "LowerIntrin: Require the target attribute";
    sym::Analyzer analyzer;
    bool enable_fast_math =
        ctx->GetConfig<bool>("tirx.enable_fast_math").value_or(false);
    n->body = ffi::make_object<IntrinInjecter>(analyzer, target.value(),
                                               enable_fast_math)
                  ->Mutate(n->body, InplaceMode::kAllow)
                  .ValueOrUnchanged(n->body);
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tl.LowerIntrin");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl._tirx_passes.LowerIntrin", LowerIntrin);
}
} // namespace tvm::tl::tirx_passes

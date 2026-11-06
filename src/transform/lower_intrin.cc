/*!
 *  Lower intrinsic calls and ops to device specific ir when possible.
 * \file lower_intrin.cc
 */
#include "ir/stmt_functor.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/prim/pattern_match.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/transform.h>

#include <vector>

#include <tvm/sym/analyzer.h>

namespace tvm {
namespace tl {
using namespace tirx;
using namespace tvm::prim;
using namespace ffi;

class IntrinInjecter : public tl::StmtExprMutatorWithAnalyzer {
public:
  using StmtExprMutatorWithAnalyzer::Mutate;
  using StmtExprMutatorWithAnalyzer::Mutate_;
  using FLowerGeneral = TypedFunction<PrimExpr(PrimExpr)>;

  IntrinInjecter(const sym::Analyzer &analyzer, std::string target,
                 std::string mtriple = "", bool fast_math = false)
      : StmtExprMutatorWithAnalyzer(analyzer) {
    std::vector<std::string> patterns;
    if (fast_math) {
      patterns.push_back("tl." + target + ".fastmath.FLowerIntrinsic");
      patterns.push_back("tl." + target + ".fastmath.FLegalize");
      patterns.push_back(target + ".fastmath.FLowerIntrinsic");
      patterns.push_back(target + ".fastmath.FLegalize");
    }
    patterns.push_back("tl." + target + ".FLowerIntrinsic");
    patterns.push_back("tl." + target + ".FLegalize");
    patterns.push_back(target + ".FLowerIntrinsic");
    patterns.push_back(target + ".FLegalize");
    bool is_llvm_aarch64 = (mtriple.find("aarch64") != std::string::npos);
    if (is_llvm_aarch64) {
      patterns.push_back(target + ".aarch64.FLowerIntrinsic");
      patterns.push_back(target + ".aarch64.FLegalize");
    }
    patterns.push_back("default.FLowerIntrinsic");
    patterns.push_back("default.FLegalize");

    for (const std::string &pattern : patterns)
      if (Op::HasAttrMap(pattern)) {
        attr_maps_.push_back(Op::GetAttrMap<FLowerGeneral>(pattern));
        if (fma_ == nullptr) {
          fma_ = (*attr_maps_.rbegin()).get(Op::Get("prim.fma"), nullptr);
        }
      }
  }

  UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    const auto *ptr_op = op->op.as<OpObj>();
    Expr expression = GetRef<Call>(op);
    if (auto primitive = expression.as<PrimExpr>(); primitive && ptr_op) {
      for (const auto &attributes : attr_maps_) {
        auto lower = attributes.get(GetRef<Op>(ptr_op), nullptr);
        if (lower != nullptr) {
          PrimExpr result = lower(primitive.value());
          TVM_FFI_ICHECK(result.defined())
              << "Intrinsic rule must return a defined primitive expression";
          if (!result.same_as(expression)) {
            return Mutate(result, mode).ValueOrUnchanged(result);
          }
        }
      }
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::AddObj *op,
                                InplaceMode mode) final {
    if (const MulObj *mb = op->b.as<MulObj>()) {
      return MakeFMA(mb->a, mb->b, op->a, op, mode);
    } else if (const MulObj *ma = op->a.as<MulObj>()) {
      return MakeFMA(ma->a, ma->b, op->b, op, mode);
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }

  // We use floordiv for integer analysis,
  // but will need to lower them to native truncdiv instructions
  UnchangedOr<PrimExpr> Mutate_(const prim::FloorDivObj *op,
                                InplaceMode mode) final {
    auto e = GetRef<PrimExpr>(op);
    PrimExpr ret =
        StmtExprMutatorWithAnalyzer::Mutate_(op, mode).ValueOrUnchanged(
            GetRef<PrimExpr>(op));
    op = ret.as<FloorDivObj>();
    if (op == nullptr)
      return ret;
    int shift;
    const PrimType dtype = op->ty.as_or_throw<PrimType>();
    TVM_FFI_ICHECK(dtype.MatchesCode(kDLInt, kDLUInt));

    // Lower division by a positive power of two to an arithmetic shift.
    if (support_bitwise_op_ && IsPowerOfTwoInt(op->b, &shift)) {
      // lower to right shift if possible.
      return op->a >> MakeConst(dtype, shift);
    }

    if (GetSyncedAnalyzer()->CanProveGreaterEqual(op->b, 0)) {
      // Common path, positive divisor
      if (GetSyncedAnalyzer()->CanProveGreaterEqual(op->a, 0) ||
          GetSyncedAnalyzer()->CanProveGreaterEqual(e, 0)) {
        return truncdiv(op->a, op->b);
      }

      // Do not shift the numerator by b*c: TileLang disables that rewrite
      // because its intermediate can overflow even when the final result fits.

      DLOG(INFO) << "LowerFloorDiv: Cannot decide the sign of divident";
      PrimExpr rdiv = truncdiv(op->a, op->b);
      PrimExpr rmod = truncmod(op->a, op->b);
      // condition on b >= 0.
      // truncmod(a, b) < 0 will implies ceildiv,
      // So we need to correct these cases.
      if ((dtype == PrimType::Int(32) || dtype == PrimType::Int(64)) &&
          support_bitwise_op_) {
        // equivalent to rdiv + (rmod >= 0 ? 0: -1);
        return rdiv + (rmod >> MakeConst(dtype, dtype.bits() - 1));
      } else {
        return prim::Select(rmod >= 0, rdiv, rdiv - MakeConst(dtype, 1));
      }

    } else {
      if (dtype.MatchesCode(kDLFloat)) {
        PrimExpr lowered = tvm::prim::floor(op->a / op->b);
        return Mutate(lowered, mode).ValueOrUnchanged(lowered);
      } else {
        // uncommon case
        DLOG(INFO) << "LowerFloorDiv: Cannot decide the sign of divisor";
        auto rmod = PrimVar("rmod", dtype);
        auto rdiv = PrimVar("rdiv", dtype);
        // b >= 0 => (rmod >=0 ? rdiv : rdiv - 1)
        // b < 0  => (rmod <= 0 ? rdiv : rdiv - 1)
        PrimExpr let_rdiv = prim::Let(
            rdiv, truncdiv(op->a, op->b),
            prim::Select((op->b >= 0 && rmod >= 0) || (op->b < 0 && rmod <= 0),
                         rdiv, rdiv - MakeConst(dtype, 1)));
        return Let(rmod, truncmod(op->a, op->b), let_rdiv);
      }
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::FloorModObj *op,
                                InplaceMode mode) final {
    PrimExpr ret =
        StmtExprMutatorWithAnalyzer::Mutate_(op, mode).ValueOrUnchanged(
            GetRef<PrimExpr>(op));
    op = ret.as<FloorModObj>();
    if (op == nullptr)
      return ret;
    // Lower floordiv to native truncdiv.
    int shift;
    const PrimType dtype = op->ty.as_or_throw<PrimType>();
    TVM_FFI_ICHECK(dtype.MatchesCode(kDLInt, kDLUInt));

    if (support_bitwise_op_ && IsPowerOfTwoInt(op->b, &shift)) {
      // lower to masking if possible.
      ffi::BigInt mask = (ffi::BigInt(1) << shift) - 1;
      return op->a & IntImm(dtype, mask);
    }

    if (GetSyncedAnalyzer()->CanProveGreaterEqual(op->b, 0)) {
      // Common pass, positive divisor
      if (GetSyncedAnalyzer()->CanProveGreaterEqual(op->a, 0)) {
        return truncmod(op->a, op->b);
      }

      // Do not shift the numerator by b*c: TileLang disables that rewrite
      // because its intermediate can overflow even when the final result fits.

      DLOG(INFO) << "LowerFloorMod: Cannot decide the sign of divident";
      // NOTE:condition on b >= 0.
      // mod(a, b) < 0 will imply we are doing ceildiv,
      // So we need to correct these cases.
      PrimExpr rmod = truncmod(op->a, op->b);
      if ((dtype == PrimType::Int(32) || dtype == PrimType::Int(64)) &&
          support_bitwise_op_) {
        // (rmod >> shift) & b
        // -> (rmod >= 0 ? 0: -1) & b
        // -> rmod >= 0 ? 0 : b
        return rmod + (op->b & (rmod >> MakeConst(dtype, dtype.bits() - 1)));
      } else {
        return prim::Select(rmod >= 0, rmod, rmod + op->b);
      }

    } else {
      if (dtype.MatchesCode(kDLFloat)) {
        // a - floor(a / b) * b
        PrimExpr lowered = tvm::prim::floor(op->a / op->b);
        return op->a - Mutate(lowered, mode).ValueOrUnchanged(lowered) * op->b;
      } else {
        // uncommon case
        DLOG(INFO)
            << "LowerFloorMod: Cannot decide the sign of divsor and divident";
        auto rmod = PrimVar("rmod", dtype);
        // b > 0 && rmod >= 0 -> rmod
        // b > 0 && rmod < 0  -> rmod + b
        // b < 0 && rmod < 0 -> rmod
        // b < 0 && rmod > 0 -> rmod + b
        return Let(rmod, truncmod(op->a, op->b),
                   Select((op->b >= 0 && rmod >= 0) || (op->b < 0 && rmod <= 0),
                          rmod, rmod + op->b));
      }
    }
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::MaxObj *op,
                                InplaceMode mode) final {
    PVar<PrimExpr> x, y;
    PVar<IntImm> c;
    auto e = GetRef<PrimExpr>(op);
    if (max(floordiv(x, y), c).Match(e) && c.Eval()->value >= 0 &&
        GetSyncedAnalyzer()->CanProveGreaterEqual(y.Eval(), 0)) {
      PrimExpr lowered = truncdiv(x, y).Eval();
      return max(Mutate(lowered, mode).ValueOrUnchanged(lowered), c.Eval());
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::EQObj *op, InplaceMode mode) final {
    PVar<PrimExpr> x, y;
    auto e = GetRef<PrimExpr>(op);
    if ((floormod(x, y) == 0).Match(e)) {
      PrimExpr lowered = (truncmod(x, y) == 0).Eval();
      return Mutate(lowered, mode).ValueOrUnchanged(lowered);
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }

  UnchangedOr<PrimExpr> Mutate_(const prim::NEObj *op, InplaceMode mode) final {
    PVar<PrimExpr> x, y;
    auto e = GetRef<PrimExpr>(op);
    if ((floormod(x, y) != 0).Match(e)) {
      PrimExpr lowered = (truncmod(x, y) != 0).Eval();
      return Mutate(lowered, mode).ValueOrUnchanged(lowered);
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }

private:
  PrimExpr SwapBroadcastCast(const PrimExpr &e) {
    // Try to change broadcast(cast(x)) to cast(broadcast(x))
    // For some targets, LLVM will generate more efficient FMA
    // instruction with the latter. For example, vmla vs. vmlal
    // on ARM.
    if (const BroadcastObj *bcast = e.as<BroadcastObj>()) {
      if (const CastObj *cast = bcast->value.as<CastObj>()) {
        auto should_swap = [&]() {
          // Maintain behaviour (int8 -> int16, fp16 -> fp32).
          if (cast->ty.as_or_throw<PrimType>().bits() ==
              cast->value.ty().bits() * 2) {
            return true;
          }
          // Check both operands are integer-like.
          if (!cast->ty.as_or_throw<PrimType>().MatchesCode(kDLUInt, kDLInt)) {
            return false;
          }
          if (!cast->value.ty().MatchesCode(kDLUInt, kDLInt)) {
            return false;
          }
          // If both are integer-like, swap if we have a widening cast.
          return cast->ty.as_or_throw<PrimType>().bits() >
                 cast->value.ty().bits();
        };

        if (should_swap()) {
          PrimExpr new_bcast = Broadcast(cast->value, bcast->lanes);
          return Cast(bcast->ty.as_or_throw<PrimType>(), new_bcast);
        }
      }
    }
    return e;
  }

  UnchangedOr<PrimExpr> MakeFMA(const PrimExpr &a, const PrimExpr &b,
                                const PrimExpr &c, const prim::AddObj *op,
                                InplaceMode mode) {
    // emit fma instruction: a * b + c
    PrimExpr lhs = SwapBroadcastCast(a);
    PrimExpr rhs = SwapBroadcastCast(b);

    if (fma_ != nullptr &&
        op->ty.as_or_throw<PrimType>().MatchesCode(kDLFloat)) {
      PrimExpr r =
          fma_(Call(op->ty, Op::Get("prim.fma"), {lhs, rhs, c}, {}, {}, op->loc)
                   .as_or_throw<PrimExpr>());
      if (r.defined())
        return Mutate(r, mode).ValueOrUnchanged(r);
    } else {
      if (!lhs.same_as(a) || !rhs.same_as(b)) {
        PrimExpr product = Mul(lhs, rhs);
        PrimExpr mul = Mutate(product, mode).ValueOrUnchanged(product);
        return Add(mul, Mutate(c, mode).ValueOrUnchanged(c));
      }
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }

  // attribute maps, shared only when FLegalize == FLowerIntrinsic
  std::vector<OpAttrMap<FLowerGeneral>> attr_maps_;
  FLowerGeneral fma_{nullptr};
  bool support_bitwise_op_{true};
};

Stmt LowerIntrinStmt(Stmt stmt, const std::string &target) {
  sym::Analyzer analyzer;
  return ffi::make_object<IntrinInjecter>(analyzer, target)
      ->Mutate(stmt, InplaceMode::kAllow)
      .ValueOrUnchanged(stmt);
}

namespace transform {

tvm::transform::Pass LowerIntrin() {
  auto pass = [](tirx::Function func, IRModule,
                 tvm::transform::PassContext ctx) {
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    TVM_FFI_CHECK(target.has_value(), ValueError)
        << "LowerIntrin requires the target attribute";
    sym::Analyzer analyzer;
    ffi::String triple =
        target.value()->GetAttr<ffi::String>("mtriple").value_or("");
    bool fast_math =
        ctx->GetConfig<bool>("tirx.enable_fast_math").value_or(false);
    auto *node = func.CopyOnWrite();
    node->body = ffi::make_object<IntrinInjecter>(
                     analyzer, target.value()->kind->name, triple, fast_math)
                     ->Mutate(node->body, InplaceMode::kAllow)
                     .ValueOrUnchanged(node->body);
    return func;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.LowerIntrin");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerIntrin", LowerIntrin);
}

} // namespace transform
} // namespace tl
} // namespace tvm

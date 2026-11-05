/*!
 * \file rewrite_fp4_to_fp4x2.cc
 * \brief Represent genuine fp4 storage as byte-sized packed pairs.
 *
 * Public tensor parameters retain their logical shape and dtype. Packed body
 * views alias that storage; byte offsets are never divided again. Byte-backed
 * fp4 views have already been handled by AscendCopy and are left unchanged.
 */
#include "ir/alias_analysis.h"
#include "ir/analyzer.h"
#include "ir/stmt_functor.h"
#include "op/builtin.h"
#include "op/native_utils.h"
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/vector_expr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>
#include <unordered_set>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;
using namespace prim;

bool IsFp4Scalar(PrimType type) {
  return type.MatchesElementType(kDLFloat4_e2m1fn, 4) && type.lanes() == 1;
}
PrimType PackedType() { return PrimType(kDLFloat4_e2m1fn, 4, 2); }

bool IsFp4StorageType(const Type &type) {
  if (const auto *tensor = type.as<TensorTypeObj>())
    return IsFp4Scalar(tensor->dtype);
  if (const auto *pointer = type.as<PtrTypeObj>()) {
    auto element = pointer->element_type.as<PrimType>();
    return element && IsFp4Scalar(*element);
  }
  return false;
}

class Fp4StorageRewriter : public StmtExprMutatorWithAnalyzer {
public:
  using StmtExprMutatorWithAnalyzer::Mutate;
  using StmtExprMutatorWithAnalyzer::Mutate_;
  explicit Fp4StorageRewriter(const tirx::Function &function,
                              const sym::Analyzer &analyzer)
      : StmtExprMutatorWithAnalyzer(analyzer), roots_(function) {
    for (const Var &parameter : function->params) {
      parameters_.insert(parameter.get());
      if (IsFp4StorageType(parameter->ty))
        genuine_roots_.insert(parameter.get());
    }
    VisitNativePostOrder(function->body, [&](const ObjectRef &object) {
      if (const auto *bind = object.as<BindObj>()) {
        const auto *call = bind->value.as<CallObj>();
        if (call && call->op.same_as(tirx::alloc_tensor_op()) &&
            IsFp4StorageType(bind->var->ty))
          genuine_roots_.insert(bind->var.get());
      }
    });
  }

  SeqStmt Rewrite(const SeqStmt &body) {
    auto result = Mutate(body).ValueOrUnchanged(body);
    Array<Stmt> statements = parameter_views_;
    statements.push_back(result);
    return SeqStmt(statements, body->loc);
  }

private:
  bool Genuine(const Expr &expression) const {
    auto root = roots_.Root(expression);
    return root && genuine_roots_.count(root.value().get());
  }

  PrimExpr Half(const PrimExpr &value, const char *role) {
    auto expr = Mutate(value).ValueOrUnchanged(value);
    TVM_FFI_CHECK_EQ(expr.ty().lanes(), 1, ValueError)
        << "RewriteFp4ToFp4x2: " << role << " must be scalar";
    auto two = IntImm(expr.ty(), 2);
    TVM_FFI_CHECK(GetSyncedAnalyzer()->CanProveEqual(floormod(expr, two),
                                                     IntImm(expr.ty(), 0)),
                  ValueError)
        << "RewriteFp4ToFp4x2: " << role
        << " must be provably even (fp4 is packed 2 codes/byte), got " << expr;
    return GetSyncedAnalyzer()->Simplify(floordiv(expr, two));
  }

  TensorType PackTensorType(const TensorType &type) {
    auto pattern = tirx::MatchShapeStridePattern(type);
    TVM_FFI_CHECK(pattern, ValueError)
        << "RewriteFp4ToFp4x2 requires a materialized memory-only strided "
           "tensor; lower distributed/composed layouts first";
    auto [shape, strides, origin] = pattern.value();
    if (!shape.empty())
      TVM_FFI_CHECK(GetSyncedAnalyzer()->CanProveEqual(
                        strides.back(), IntImm(strides.back().ty(), 1)),
                    ValueError)
          << "RewriteFp4ToFp4x2 requires unit innermost stride: "
             "two logical fp4 elements must occupy the same physical byte";
    auto node = ffi::make_object<TensorTypeObj>(*type.get());
    node->dtype = PackedType();
    node->shape = shape.Map(
        [this](const PrimExpr &e) { return Mutate(e).ValueOrUnchanged(e); });
    if (!shape.empty()) {
      node->shape.Set(shape.size() - 1, Half(shape.back(), "last dimension"));
      // x_last = 2*y_last, so the innermost stride stays unchanged.
      // Every outer stride and the element origin change to byte units.
      for (size_t i = 0; i + 1 < strides.size(); ++i)
        strides.Set(i, Half(strides[i], "outer stride"));
      strides.Set(strides.size() - 1,
                  Mutate(strides.back()).ValueOrUnchanged(strides.back()));
    }
    origin = Half(origin, "element origin");
    if (type->layout) {
      Array<tirx::Iter> iterators;
      auto memory = tirx::Axis::Get("m");
      for (size_t i = 0; i < node->shape.size(); ++i)
        iterators.push_back(tirx::Iter(node->shape[i], strides[i], memory));
      node->layout = tirx::TileLayout(iterators, {}, {{memory, origin}});
    }
    return TensorType(node);
  }

  Type PackType(const Type &type) {
    if (auto tensor = type.as<TensorType>())
      return PackTensorType(*tensor);
    auto pointer = type.as_or_throw<PtrType>();
    return PtrType(PackedType(), pointer->scope);
  }

  Expr PointerCast(Expr pointer, const PtrType &type) {
    if (ffi::StructuralEqual()(pointer->ty, type))
      return pointer;
    return Call(type, tirx::reinterpret_op(), {pointer}, {}, {}, pointer->loc);
  }

  Var Define(const Var &variable) {
    if (auto mapped = replacements_.Get(variable))
      return *mapped;
    Type type = PackType(variable->ty);
    Var packed(variable->name + "_fp4x2", type, variable->loc);
    replacements_.Set(variable, packed);
    VarRemapSetExpected(variable, packed).value();
    return packed;
  }

  UnchangedOr<Expr> Mutate_(const VarObj *variable, InplaceMode mode) final {
    Var original = GetRef<Var>(variable);
    if (auto mapped = replacements_.Get(original))
      return *mapped;
    if (original->ty.as<TensorTypeObj>() && IsFp4StorageType(original->ty) &&
        Genuine(original)) {
      TVM_FFI_CHECK(parameters_.count(variable), ValueError)
          << "Genuine fp4 storage is used before its native Bind definition";
      Var packed = Define(original);
      Expr value = original;
      if (auto tensor = original.as<TensorVar>()) {
        auto type = packed->ty.as_or_throw<TensorType>();
        value = Call(type, tirx::decl_tensor_op(),
                     {PointerCast(tensor.value().data(), type->DataPtrType())},
                     {}, {type}, original->loc);
      } else {
        value = PointerCast(original, packed->ty.as_or_throw<PtrType>());
      }
      parameter_views_.push_back(Bind(packed, value, original->loc));
      return packed;
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(variable, mode);
  }

  UnchangedOr<Stmt> Mutate_(const BindObj *bind, InplaceMode mode) final {
    const auto *call = bind->value.as<CallObj>();
    if (call &&
        (call->op.same_as(tirx::alloc_tensor_op()) ||
         call->op.same_as(tirx::decl_tensor_op())) &&
        IsFp4StorageType(bind->var->ty) && Genuine(bind->var)) {
      Array<Expr> args;
      for (const Expr &arg : call->args)
        args.push_back(Mutate(arg, mode).ValueOrUnchanged(arg));
      auto packed = Define(bind->var);
      auto type = packed->ty.as_or_throw<TensorType>();
      if (call->op.same_as(tirx::decl_tensor_op()))
        args.Set(0, PointerCast(args[0], type->DataPtrType()));
      return Bind(packed,
                  Call(type, call->op, args, call->attrs, {type}, call->loc),
                  bind->loc);
    }
    auto value = Mutate(bind->value, mode).ValueOrUnchanged(bind->value);
    Var variable = bind->var;
    if (!ffi::StructuralEqual()(variable->ty, value->ty)) {
      // A scalar load used as a raw packed-pair value becomes a two-code
      // primitive; its native SSA definition and all uses must agree.
      variable = Var(variable->name, value->ty, variable->loc);
      replacements_.Set(bind->var, variable);
      VarRemapSetExpected(bind->var, variable).value();
    }
    return Bind(variable, value, bind->loc);
  }

  Array<PrimExpr> PackIndices(const Array<PrimExpr> &indices) {
    auto result = indices.Map(
        [this](const PrimExpr &e) { return Mutate(e).ValueOrUnchanged(e); });
    if (result.empty())
      return result;
    PrimExpr last = result.back();
    if (const auto *ramp = last.as<prim::RampObj>()) {
      const auto *lanes = ramp->lanes.as<IntImmObj>();
      TVM_FFI_CHECK(prim::IsOne(ramp->stride) && lanes && lanes->value > 0 &&
                        lanes->value % 2 == 0,
                    ValueError)
          << "Packed fp4 vector access requires an even unit-stride Ramp";
      auto base = Half(ramp->base, "vector base");
      last = lanes->value == 2
                 ? base
                 : prim::Ramp(base, ramp->stride,
                              IntImm(ramp->lanes.ty(), lanes->value / 2));
    } else {
      last = Half(last, "element index");
    }
    result.Set(result.size() - 1, last);
    return result;
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *load,
                                InplaceMode mode) final {
    if (IsFp4StorageType(load->source->ty) && Genuine(load->source)) {
      auto source = Mutate(load->source, mode).ValueOrUnchanged(load->source);
      return TensorLoad(source, PackIndices(load->indices), load->loc);
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(load, mode);
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreObj *store,
                            InplaceMode mode) final {
    if (IsFp4StorageType(store->dest->ty) && Genuine(store->dest)) {
      auto destination =
          Mutate(store->dest, mode).ValueOrUnchanged(store->dest);
      auto indices = PackIndices(store->indices);
      auto value = Mutate(store->value, mode).ValueOrUnchanged(store->value);
      auto expected = TensorLoad(destination, indices, store->loc).ty();
      TVM_FFI_CHECK(value.ty() == expected, TypeError)
          << "Packed fp4 store requires a whole packed-pair value, got "
          << value.ty() << "; a single fp4 scalar cannot initialize both codes";
      return TensorStore(destination, indices, value, store->loc);
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(store, mode);
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionObj *region,
                            InplaceMode mode) final {
    if (IsFp4StorageType(region->source->ty) && Genuine(region->source)) {
      auto source =
          Mutate(region->source, mode).ValueOrUnchanged(region->source);
      auto ranges = region->region;
      for (size_t i = 0; i < ranges.size(); ++i) {
        const auto &range = region->region[i];
        auto begin = Mutate(range->begin, mode).ValueOrUnchanged(range->begin);
        auto extent =
            Mutate(range->extent, mode).ValueOrUnchanged(range->extent);
        if (i + 1 == ranges.size()) {
          begin = Half(begin, "region begin");
          extent = Half(extent, "region extent");
        }
        ranges.Set(i, Range::FromMinExtent(begin, extent, range->loc));
      }
      return TensorRegion(source, ranges, TensorRegionType(), region->loc);
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(region, mode);
  }

  UnchangedOr<Expr> Mutate_(const CallObj *call, InplaceMode mode) final {
    // Keep the original public tensor as the raw storage origin of a packed
    // alias. This also makes repeated application idempotent.
    if (call->op.same_as(tirx::tensor_data_ptr_op()) &&
        call->args[0].as<VarObj>() &&
        parameters_.count(call->args[0].as<VarObj>()))
      return Unchanged();
    bool packed_access = !call->args.empty() && IsFp4StorageType(call->ty) &&
                         Genuine(call->args[0]);
    Array<Expr> args;
    for (const Expr &arg : call->args)
      args.push_back(Mutate(arg, mode).ValueOrUnchanged(arg));
    Type type = call->ty;
    auto ty_args = call->ty_args;
    if (call->op.same_as(tirx::tensor_data_ptr_op())) {
      type = args[0]->ty.as_or_throw<TensorType>()->DataPtrType();
    } else if (call->op.same_as(tirx::address_of_op())) {
      auto load = args[0].as_or_throw<TensorLoad>();
      type = load->source->ty.as_or_throw<TensorType>()->DataPtrType();
    } else if (packed_access && call->op.same_as(tl::access_ptr())) {
      auto load = args[0].as_or_throw<TensorLoad>();
      type = load->source->ty.as_or_throw<TensorType>()->DataPtrType();
      args.Set(1, Half(call->args[1].as_or_throw<PrimExpr>(), "access extent"));
    } else if (packed_access && call->op.same_as(tl::ptr_access())) {
      type = PackType(call->ty);
      args.Set(0, PointerCast(args[0], type.as_or_throw<PtrType>()));
      args.Set(1, Half(call->args[1].as_or_throw<PrimExpr>(),
                       "pointer element offset"));
      args.Set(2, Half(call->args[2].as_or_throw<PrimExpr>(),
                       "pointer element extent"));
    } else if (packed_access &&
               (call->op.same_as(tirx::reinterpret_op()) ||
                call->op.same_as(tirx::ptr_byte_offset_op()))) {
      type = PackType(call->ty);
      if (call->op.same_as(tirx::ptr_byte_offset_op()))
        args.Set(0, PointerCast(args[0], type.as_or_throw<PtrType>()));
      // ptr_byte_offset operands are already bytes, including nonzero origins.
    }
    if (!ffi::StructuralEqual()(type, call->ty)) {
      ty_args = {};
      if (call->op.same_as(tirx::reinterpret_op()))
        ty_args = {type};
    }
    return Call(type, call->op, args, call->attrs, ty_args, call->loc);
  }

  PhysicalRootAnalysis roots_;
  std::unordered_set<const VarObj *> genuine_roots_;
  std::unordered_set<const VarObj *> parameters_;
  Map<Var, Var> replacements_;
  Array<Stmt> parameter_views_;
};
} // namespace

namespace transform {
tvm::transform::Pass RewriteFp4ToFp4x2() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false)) {
      auto analyzer = CreateTileLangAnalyzer();
      auto rewriter = ffi::make_object<Fp4StorageRewriter>(function, analyzer);
      function.CopyOnWrite()->body = rewriter->Rewrite(function->body);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.RewriteFp4ToFp4x2");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.RewriteFp4ToFp4x2",
                                   RewriteFp4ToFp4x2);
}
} // namespace transform
} // namespace tvm::tl

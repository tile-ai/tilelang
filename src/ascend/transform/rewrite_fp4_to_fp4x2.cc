#include "compat/analyzer.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "ascend/transform/native_visitor.h"
#include "compat/expr.h"
#include "op/builtin.h"

namespace tvm::tl {
namespace {

bool IsScalarFP4(PrimType dtype) {
  return dtype.MatchesElementType(kDLFloat4_e2m1fn, 4) && dtype.lanes() == 1;
}

bool HasFP4StorageType(const Type &type) {
  if (const auto *tensor = type.as<tirx::TensorTypeNode>()) {
    return IsScalarFP4(tensor->dtype);
  }
  if (const auto *pointer = type.as<PtrTypeNode>()) {
    auto element = pointer->element_type.as<PrimType>();
    return element && IsScalarFP4(element.value());
  }
  return false;
}

class FP4StorageCollector : public ascend::AnalysisVisitor {
public:
  void Collect(const tirx::Function &function) {
    for (const Var &parameter : function->params) {
      if (HasFP4StorageType(parameter->ty))
        storage_.insert(parameter.get());
    }
    Visit(function->body);
  }

  bool IsFP4Storage(const Expr &expression) const {
    if (const auto *variable = expression.as<VarNode>()) {
      return storage_.count(variable);
    }
    if (const auto *load = expression.as<TensorLoadNode>()) {
      return IsFP4Storage(load->source);
    }
    if (const auto *call = expression.as<CallNode>()) {
      if (call->op.same_as(tirx::alloc_tensor_op())) {
        return HasFP4StorageType(call->ty);
      }
      if (call->op.same_as(tirx::decl_tensor_op()) ||
          call->op.same_as(tirx::tensor_data_ptr_op()) ||
          call->op.same_as(tirx::reinterpret_op()) ||
          call->op.same_as(tirx::ptr_byte_offset_op()) ||
          call->op.same_as(tirx::address_of_op())) {
        return !call->args.empty() && IsFP4Storage(call->args[0]);
      }
    }
    return false;
  }

private:
  void VisitStmt_(const BindNode *node) final {
    Visit(node->value);
    if (IsFP4Storage(node->value))
      storage_.insert(node->var.get());
  }

  std::unordered_set<const VarNode *> storage_;
};

class FP4StorageRewriter : public ascend::NativeMutator {
public:
  explicit FP4StorageRewriter(FP4StorageCollector *storage)
      : storage_(storage) {}

  tirx::Function Rewrite(tirx::Function function) {
    auto parameters = function->params.Map([&](const Var &parameter) {
      Type type = storage_->IsFP4Storage(parameter) ? RewriteType(parameter->ty)
                                                    : parameter->ty;
      Var rewritten = type.same_as(parameter->ty)
                          ? parameter
                          : parameter.CopyWithType(type);
      VarRemapSet(parameter, rewritten);
      return rewritten;
    });
    auto body = Mutate(function->body, ffi::InplaceMode::kDisallow)
                    .ValueOrUnchanged(function->body);
    auto attrs = Mutate(function->attrs, ffi::InplaceMode::kDisallow)
                     .as_or_throw<ffi::UnchangedOr<DictAttrs>>()
                     .ValueOrUnchanged(function->attrs);
    auto *node = function.CopyOnWrite();
    node->params = std::move(parameters);
    node->body = std::move(body);
    node->attrs = std::move(attrs);
    node->ty = node->func_type_annotation();
    return function;
  }

private:
  PrimExpr Halve(const PrimExpr &value, const char *description) {
    PrimExpr pair = IntImm(value.ty(), 2);
    TVM_FFI_CHECK(tl::arith::CanProve(analyzer_, FloorMod(value, pair) == 0),
                  ValueError)
        << "RewriteFp4ToFp4x2: " << description
        << " must be even (two fp4 codes per byte), got " << value;
    return tl::arith::Simplify(analyzer_, FloorDiv(value, pair));
  }

  Type RewriteType(const Type &type) {
    if (!HasFP4StorageType(type))
      return type;
    if (auto found = types_.find(type.get()); found != types_.end()) {
      return found->second;
    }
    Type rewritten = type;
    if (const auto *pointer = type.as<PtrTypeNode>()) {
      rewritten =
          PtrType(pointer->element_type.as_or_throw<PrimType>().WithLanes(2),
                  pointer->scope);
    } else {
      auto tensor = type.as_or_throw<tirx::TensorType>();
      auto geometry = tirx::MatchShapeStridePattern(tensor);
      TVM_FFI_CHECK(geometry && !tensor->shape.empty(), ValueError)
          << "RewriteFp4ToFp4x2 requires a strided tensor with an innermost "
             "pair axis";
      auto shape = tensor->shape;
      auto strides = geometry.value().get<1>();
      TVM_FFI_CHECK(tl::arith::CanProve(analyzer_, strides.back() == 1),
                    ValueError)
          << "RewriteFp4ToFp4x2 requires contiguous innermost fp4 elements";
      shape.Set(shape.size() - 1, Halve(shape.back(), "fp4 shape"));
      ffi::Optional<tirx::Layout> layout;
      if (tensor->layout) {
        ffi::Array<tirx::Iter> iterators;
        auto memory = tirx::Axis::Get("m");
        for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
          PrimExpr stride = dimension + 1 == shape.size()
                                ? strides[dimension]
                                : Halve(strides[dimension], "fp4 stride");
          iterators.push_back(tirx::Iter(shape[dimension], stride, memory));
        }
        layout = tirx::TileLayout(
            iterators, {},
            {{memory, Halve(geometry.value().get<2>(), "fp4 layout offset")}});
      }
      rewritten =
          tirx::TensorType(shape, tensor->dtype.WithLanes(2), tensor->scope,
                           tensor->data_alignment, layout, tensor->loc);
    }
    types_.emplace(type.get(), rewritten);
    return rewritten;
  }

  ffi::Array<PrimExpr> RewriteIndices(const ffi::Array<PrimExpr> &indices,
                                      bool address) {
    TVM_FFI_CHECK(indices.size() == 1, ValueError)
        << "RewriteFp4ToFp4x2 expects flat accesses; run FlattenBuffer first";
    PrimExpr index = VisitExpr(indices[0]).as_or_throw<PrimExpr>();
    if (const auto *ramp = index.as<RampNode>()) {
      TVM_FFI_CHECK(tl::arith::CanProve(analyzer_, ramp->stride == 1),
                    ValueError)
          << "RewriteFp4ToFp4x2 requires contiguous fp4 vector accesses";
      PrimExpr base = Halve(ramp->base, "fp4 vector base");
      PrimExpr lanes = Halve(ramp->lanes, "fp4 vector lanes");
      if (IsOne(lanes))
        return {base};
      return {Ramp(base, IntImm(base.ty(), 1), lanes)};
    }
    TVM_FFI_CHECK(address, ValueError) << "RewriteFp4ToFp4x2 cannot treat one "
                                          "scalar fp4 value as a packed pair";
    return {Halve(index, "fp4 address")};
  }

  PrimExpr VisitExpr_(const TensorLoadNode *node) final {
    if (!HasFP4StorageType(node->source->ty) ||
        !storage_->IsFP4Storage(node->source)) {
      return ascend::NativeMutator::VisitExpr_(node);
    }
    return TensorLoad(VisitExpr(node->source),
                      RewriteIndices(node->indices, false), node->loc);
  }

  Stmt VisitStmt_(const TensorStoreNode *node) final {
    if (!HasFP4StorageType(node->dest->ty) ||
        !storage_->IsFP4Storage(node->dest)) {
      return ascend::NativeMutator::VisitStmt_(node);
    }
    return TensorStore(
        VisitExpr(node->dest), RewriteIndices(node->indices, false),
        VisitExpr(node->value).as_or_throw<PrimExpr>(), node->loc);
  }

  Expr VisitExpr_(const CallNode *node) final {
    bool rewrite_storage = storage_->IsFP4Storage(ffi::GetRef<Expr>(node));
    if (rewrite_storage && node->op.same_as(tirx::address_of_op())) {
      if (const auto *load = node->args[0].as<TensorLoadNode>();
          load && HasFP4StorageType(load->source->ty)) {
        TensorLoad address(VisitExpr(load->source),
                           RewriteIndices(load->indices, true), load->loc);
        return Call(RewriteType(node->ty), node->op, {address}, node->attrs,
                    node->ty_args, node->loc);
      }
    }
    Call rewritten =
        ascend::NativeMutator::VisitExpr_(node).as_or_throw<Call>();
    if (!rewrite_storage || !HasFP4StorageType(node->ty))
      return rewritten;
    auto *result = rewritten.CopyOnWrite();
    result->ty = RewriteType(node->ty);
    result->ty_args = result->ty_args.Map(
        [&](const Type &type) { return RewriteType(type); });
    if (node->op.same_as(tirx::ptr_byte_offset_op())) {
      if (auto attrs = result->attrs.as<DictAttrs>()) {
        auto values = attrs.value()->dict;
        if (auto extent = values.Get(attr::kAccessExtent)) {
          PrimExpr count = extent.value().as<PrimExpr>().value_or(IntImm(
              PrimType::Int(64), GetIntAnnotation(extent.value()).value_or(0)));
          TVM_FFI_CHECK(extent.value().as<PrimExpr>() ||
                            GetIntAnnotation(extent.value()),
                        TypeError)
              << "fp4 access extent must be an integer expression";
          values.Set(attr::kAccessExtent, Halve(count, "fp4 access extent"));
          result->attrs = DictAttrs(values);
        }
      }
    }
    return rewritten;
  }

  FP4StorageCollector *storage_;
  sym::Analyzer analyzer_;
  std::unordered_map<const TypeNode *, Type> types_;
};

} // namespace

namespace transform {

tvm::transform::Pass RewriteFp4ToFp4x2() {
  auto pass = [](tirx::Function function, const IRModule &,
                 tvm::transform::PassContext) {
    if (!function->body)
      return function;
    auto storage = ffi::make_object<FP4StorageCollector>();
    storage->Collect(function);
    return ffi::make_object<FP4StorageRewriter>(storage.get())
        ->Rewrite(function);
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.RewriteFp4ToFp4x2");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.RewriteFp4ToFp4x2",
                                   RewriteFp4ToFp4x2);
}

} // namespace transform
} // namespace tvm::tl

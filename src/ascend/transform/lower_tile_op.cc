/*!
 * \file tl/ascend/transform/lower_tile_op.cc
 * \brief Ascend fork of LowerTileOp (VF region scopes, buffer-version remap).
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "ascend/layout/ascend_layouts.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "ascend/transform/buffer_alias.h"
#include "ascend/transform/buffer_version.h"
#include "compat/expr.h"
#include "compat/visit.h"
#include "op/gemm_blockscaled.h"
#include "transform/lower_tile_op.h"

namespace tvm::tl::ascend {
namespace {

Expr ApplyL0AccessPitch(const Call &pointer, sym::Analyzer analyzer) {
  if (!pointer->op.same_as(access_ptr()) || !pointer->attrs)
    return pointer;
  auto attributes = pointer->attrs.value().as_or_throw<DictAttrs>();
  auto pitch = attributes.GetAttr<PrimExpr>(attr::kL0AccessPitch);
  if (!pitch)
    return pointer;
  auto region = NormalizeToBufferRegion(pointer->args[0]);
  auto tensor = region->source.as_or_throw<tirx::TensorVar>();
  const auto &shape = tensor.shape();
  TVM_FFI_CHECK((IsL0ABuffer(tensor) || IsL0BBuffer(tensor)) &&
                    shape.size() >= 4,
                ValueError)
      << "L0 pitch requires an L0A/L0B fractal layout";
  size_t rank = shape.size();
  PrimType index_type = PrimType::Int(64);
  auto index = [&](const PrimExpr &expression) {
    return prim::cast(expression, index_type);
  };
  PrimExpr extra_elements = index(region->region[rank - 4]->begin) *
                            (index(pitch.value()) - index(shape[rank - 3])) *
                            index(shape[rank - 2]) * index(shape[rank - 1]);
  PrimExpr extra_bits =
      extra_elements *
      IntImm(index_type, tensor.dtype().bits() * tensor.dtype().lanes());
  TVM_FFI_CHECK(analyzer->CanProveEqual(
                    prim::FloorMod(extra_bits, IntImm(index_type, 8)), 0),
                ValueError)
      << "L0 pitch displacement must be byte aligned";
  attributes.CopyOnWrite()->dict.erase(attr::kL0AccessPitch);
  Call base(pointer->ty, pointer->op, pointer->args, attributes,
            pointer->ty_args, pointer->loc);
  auto offset =
      analyzer->Simplify(prim::FloorDiv(extra_bits, IntImm(index_type, 8)));
  if (prim::IsZero(offset))
    return base;
  return Call(pointer->ty, tirx::ptr_byte_offset_op(), {base, offset},
              std::nullopt, {}, pointer->loc);
}

class ScaleHandleCleanup : public tirx::StmtExprMutator {
public:
  ffi::UnchangedOr<Stmt> Mutate_(const BindNode *node,
                                 ffi::InplaceMode mode) final {
    if (auto tensor = node->var.as<tirx::TensorVar>();
        tensor && IsL0SFBuffer(tensor.value())) {
      const auto *allocation = node->value.as<CallNode>();
      TVM_FFI_CHECK(allocation &&
                        allocation->op.same_as(tirx::alloc_tensor_op()),
                    ValueError)
          << "L0 SF handles must be declared by an allocation binding";
      return SeqStmt({});
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                                 ffi::InplaceMode mode) final {
    auto region = StmtExprMutator::Mutate_(node, mode)
                      .ValueOrUnchanged(ffi::GetRef<Stmt>(node))
                      .as_or_throw<RegionStmt>();
    auto attributes = region->attrs;
    attributes.CopyOnWrite()->dict.erase(kL0SFBindings);
    attributes.CopyOnWrite()->dict.erase("tl.simd_pointer_dtypes");
    region.CopyOnWrite()->attrs = attributes;
    return region;
  }
};

} // namespace

tvm::transform::Pass AscendLowerTileOp() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    if (!function->body)
      return function;
    auto bindings = CollectL0SFBindings(function->body.value());
    LowerTileOpOptions options;
    options.rewrite_access_pointer = ApplyL0AccessPitch;
    options.rewrite_operator =
        [bindings](const TileOperator &operation,
                   sym::Analyzer analyzer) -> TileOperator {
      if (auto gemm = operation.as<GemmBlockScaled>())
        ValidateL0SFGemm(gemm.value(), bindings, analyzer);
      if (const auto *copy = operation.as<CopyNode>();
          copy && IsL0SFBuffer(copy->dst)) {
        auto data = bindings.Get(copy->dst);
        TVM_FFI_CHECK(data.has_value(), ValueError)
            << "L0 SF copy has no allocation binding";
        AscendCopy bound =
            operation.as<AscendCopy>().value_or(AscendCopy(*copy));
        bound.CopyOnWrite()->mx_sf_data =
            data.value().as_or_throw<tirx::TensorVar>();
        return bound;
      }
      return operation;
    };
    function = LowerTileOperators(std::move(function), std::move(options));
    auto cleanup = ffi::make_object<ScaleHandleCleanup>();
    auto original = function->body.value();
    function.CopyOnWrite()->body =
        cleanup->Mutate(original).ValueOrUnchanged(original);
    auto attributes = function->attrs;
    attributes.CopyOnWrite()->dict.erase("tl.simd_pointer_dtypes");
    function.CopyOnWrite()->attrs = std::move(attributes);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0, "tl.AscendLowerTileOp");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.AscendLowerTileOp",
                                   AscendLowerTileOp);
}

} // namespace tvm::tl::ascend

/*!
 * \file lower_blackwell_2sm.cc
 * \brief Carry two-CTA GEMM requirements in their native kernel scope.
 */
#include "cuda/target_utils.h"
#include "ir/annotations.h"
#include <tvm/tirx/stmt_functor.h>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
using namespace ffi;
using namespace tirx;

Optional<int64_t> ConstantMetadata(const Any &value) {
  if (auto integer = value.as<int64_t>())
    return integer;
  if (auto boolean = value.as<bool>())
    return *boolean ? 1 : 0;
  if (const auto *integer = value.as<IntImmObj>())
    return integer->value.as<int64_t>();
  return std::nullopt;
}

bool ValidClusterDims(const Map<String, Any> &metadata) {
  auto value = metadata.Get("cluster_dims");
  if (!value)
    return false;
  auto dims = value->cast<Array<Any>>();
  if (dims.size() != 3)
    return false;
  auto x = ConstantMetadata(dims[0]);
  auto y = ConstantMetadata(dims[1]);
  auto z = ConstantMetadata(dims[2]);
  return x && y && z && *z == 1 &&
         ((*x == 2 && *y == 1) || (*x == 1 && *y == 2));
}

class Kernel2SmLowerer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *region,
                            InplaceMode mode) final {
    const bool launch = region->op->name == "tl.launch_kernel";
    Map<String, Any> fields;
    if (region->op->name == "tl.attr") {
      fields = ReadAnnotations(region);
      if (fields.at("key").cast<String>() != "tl.kernel")
        return StmtExprMutator::Mutate_(region, mode);
    } else if (!launch) {
      return StmtExprMutator::Mutate_(region, mode);
    }
    Map<String, Any> metadata;
    if (launch) {
      metadata = DecodeAnnotations(
                     region->args[4].as_or_throw<tvm::Tuple>(),
                     region->attrs.GetAttr<Array<Any>>("hint_schema").value())
                     .cast<Map<String, Any>>();
    } else {
      metadata = fields.at("value").cast<Map<String, Any>>();
    }
    KernelState state{ValidClusterDims(metadata), false};
    auto *saved = kernel_;
    kernel_ = &state;
    SeqStmt body = Mutate(region->body, mode).ValueOrUnchanged(region->body);
    kernel_ = saved;
    if (state.requested)
      metadata.Set("use_2cta", IntImm(PrimType::Int(32), 1));
    if (launch) {
      auto encoded = EncodeAnnotations(metadata);
      auto arguments = region->args;
      arguments.Set(4, encoded.payload);
      auto attrs = region->attrs->dict;
      attrs.Set("hint_schema", encoded.schema);
      return RegionStmt(region->op, arguments, region->body_params,
                        DictAttrs(attrs), body, region->result_vars,
                        region->loc);
    }
    return MakeAttribute(fields.at("node"), "tl.kernel", metadata, body,
                         region->loc);
  }

  UnchangedOr<Expr> Mutate_(const CallObj *call, InplaceMode mode) final {
    auto op = call->op.as<Op>();
    // Dense and block-scaled GEMM are the two native GemmNode families.
    // Keep the metadata on the original Call for instruction selection; only
    // publish the allocation requirement to the enclosing kernel's consumers.
    if (op && (op.value()->name == "tl.tileop.gemm" ||
               op.value()->name == "tl.tileop.gemm_blockscaled" ||
               op.value()->name == "tl.tileop.tcgen05_gemm" ||
               op.value()->name == "tl.tileop.wgmma_gemm")) {
      auto args = DecodeTileOpArguments(call);
      if (auto value = args.annotations.Get("use_2cta")) {
        auto enabled = ConstantMetadata(*value);
        TVM_FFI_CHECK(enabled, ValueError)
            << "use_2cta must be a constant integer or bool";
        if (*enabled != 0) {
          TVM_FFI_CHECK(kernel_ && kernel_->valid_cluster, ValueError)
              << "T.tcgen05_gemm(use_2cta=True) requires cluster_dims (2,1,1) "
                 "or (1,2,1) on its own kernel; a one-CTA fallback is not "
                 "valid "
                 "for rank-sharded operands";
          kernel_->requested = true;
        }
      }
    }
    return StmtExprMutator::Mutate_(call, mode);
  }

private:
  struct KernelState {
    bool valid_cluster;
    bool requested;
  };
  KernelState *kernel_{nullptr};
};
} // namespace

tvm::transform::Pass LowerBlackwell2SM() {
  auto rewrite = [](tirx::Function function, IRModule,
                    tvm::transform::PassContext) {
    auto target = function->GetAttr<Target>(tvm::attr::kTarget);
    if (!target || !TargetIsSm100(target.value()) ||
        function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto lowerer = make_object<Kernel2SmLowerer>();
    SeqStmt body = function->body;
    function.CopyOnWrite()->body = lowerer->Mutate(body).ValueOrUnchanged(body);
    return function;
  };
  return tirx::transform::CreateFunctionPass(rewrite, 0,
                                             "tl.LowerBlackwell2SM");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.LowerBlackwell2SM",
                                   LowerBlackwell2SM);
}
} // namespace tvm::tl

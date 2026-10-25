#include "backend/common/target_utils.h"
#include "compat/expr.h"
#include "op/gemm.h"
#include "transform/common/attr.h"

#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
using namespace ffi;

bool ValidCluster(const DictAttrs &attributes) {
  auto dimensions = attributes.GetAttr<Array<Any>>("cluster_dims");
  if (!dimensions || dimensions->size() != 3)
    return false;
  auto width = GetIntAnnotation(dimensions.value()[0]);
  auto height = GetIntAnnotation(dimensions.value()[1]);
  auto depth = GetIntAnnotation(dimensions.value()[2]);
  return width && height && depth && *depth == 1 &&
         ((*width == 2 && *height == 1) || (*width == 1 && *height == 2));
}

class Blackwell2SM : public tirx::StmtExprMutator {
public:
  explicit Blackwell2SM(bool valid_cluster) : valid_cluster_(valid_cluster) {}
  bool requested = false;

private:
  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *node,
                            InplaceMode mode) final {
    if (!IsDeviceMainBlock(node))
      return StmtExprMutator::Mutate_(node, mode);
    bool saved_cluster = valid_cluster_;
    bool saved_requested = requested;
    valid_cluster_ = ValidCluster(node->attrs);
    requested = false;
    RegionStmt original = GetRef<RegionStmt>(node);
    RegionStmt result = StmtExprMutator::Mutate_(node, mode)
                            .ValueOrUnchanged(original)
                            .as_or_throw<RegionStmt>();
    if (requested) {
      auto attributes = result->attrs->dict;
      attributes.Set("use_2cta", true);
      result.CopyOnWrite()->attrs = DictAttrs(attributes);
    }
    valid_cluster_ = saved_cluster;
    requested = saved_requested;
    return result;
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateNode *node, InplaceMode mode) final {
    const auto *call = node->value.as<CallNode>();
    if (call && call->attrs) {
      auto annotations = call->attrs.value().as_or_throw<DictAttrs>()->dict;
      auto flag = annotations.Get("use_2cta");
      auto enabled = flag ? GetIntAnnotation(*flag) : std::nullopt;
      if (enabled && *enabled != 0) {
        TileOperator operation = ParseOperator(GetRef<Stmt>(node));
        if (operation.as<GemmNode>()) {
          TVM_FFI_CHECK(valid_cluster_, ValueError)
              << "T.tcgen05_gemm(use_2cta=True) requires cluster_dims (2,1,1) "
                 "or (1,2,1) "
                 "on the same kernel; a one-CTA fallback is not valid for "
                 "rank-sharded operands";
          requested = true;
        }
      }
    }
    return StmtExprMutator::Mutate_(node, mode);
  }

  bool valid_cluster_;
};
} // namespace

tvm::transform::Pass LowerBlackwell2SM() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) -> ffi::Optional<tirx::Function> {
    auto target = function->GetAttr<Target>("target");
    if (!function->body || !target || !TargetIsSm100(*target))
      return function;
    auto lowerer =
        ffi::make_object<Blackwell2SM>(ValidCluster(function->attrs));
    SeqStmt body = function->body.value();
    function.CopyOnWrite()->body =
        lowerer->Mutate(body).ValueOrUnchanged(body).as_or_throw<SeqStmt>();
    if (lowerer->requested)
      function = WithAttr(std::move(function), "use_2cta", true);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.cuda.LowerBlackwell2SM");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.cuda.transform.LowerBlackwell2SM",
                                   LowerBlackwell2SM);
}
} // namespace tvm::tl

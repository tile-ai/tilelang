/*!
 * \file annotate_device_bound_tma_copies.cc
 * \brief Mark copies whose TensorMap base pointer is defined in the body.
 */

#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>
#include <utility>

#include "cuda/op/builtin.h"
#include "op/copy.h"
#include "op/operator.h"
#include "op/utils.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

class BodyBoundHandleCollector : public StmtExprVisitor {
public:
  static VarSet Collect(const Stmt &stmt) {
    BodyBoundHandleCollector collector;
    collector(stmt);
    return std::move(collector.vars_);
  }

private:
  void VisitStmt_(const BindNode *op) final {
    if (op->var->dtype.is_handle()) {
      vars_.insert(op->var);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  VarSet vars_;
};

class DeviceBoundTmaCopyAnnotator : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc f) {
    VarSet body_bound_handles = BodyBoundHandleCollector::Collect(f->body);
    if (body_bound_handles.empty()) {
      return f;
    }
    DeviceBoundTmaCopyAnnotator annotator(body_bound_handles);
    Stmt body = annotator.VisitStmt(f->body);
    f.CopyOnWrite()->body = std::move(body);
    return f;
  }

private:
  explicit DeviceBoundTmaCopyAnnotator(const VarSet &body_bound_handles)
      : body_bound_handles_(body_bound_handles) {}

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    static const Op &copy_op = Op::Get("tl.tileop.copy");
    static const Op &async_copy_op = Op::Get("tl.tileop.async_copy");
    static const Op &tma_copy_op = Op::Get("tl.tileop.tma_copy");
    if (!call->op.same_as(copy_op) && !call->op.same_as(async_copy_op) &&
        !call->op.same_as(tma_copy_op)) {
      return call;
    }

    TileOperator tile_op = ParseOperator(call);
    const auto *copy = tile_op.as<CopyNode>();
    ICHECK(copy != nullptr);
    bool device_bound_global_base =
        (IsGlobalBuffer(copy->src) &&
         body_bound_handles_.count(copy->src->data)) ||
        (IsGlobalBuffer(copy->dst) &&
         body_bound_handles_.count(copy->dst->data));
    if (!device_bound_global_base ||
        call->annotations.count(attr::kTmaDescriptorBaseIsDeviceBound)) {
      return call;
    }

    Map<String, ObjectRef> annotations = call->annotations;
    annotations.Set(attr::kTmaDescriptorBaseIsDeviceBound,
                    IntImm(DataType::Int(32), 1));
    return Call(call->dtype, call->op, call->args, annotations, call->span);
  }

  const VarSet &body_bound_handles_;
};

} // namespace

tvm::transform::Pass AnnotateDeviceBoundTmaCopies() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc f, const IRModule &m, PassContext ctx) {
    return DeviceBoundTmaCopyAnnotator::Rewrite(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnnotateDeviceBoundTmaCopies",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.cuda.transform.AnnotateDeviceBoundTmaCopies",
                        AnnotateDeviceBoundTmaCopies);
}

} // namespace tl
} // namespace tvm

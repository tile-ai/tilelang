/*!
 * \file infer_instructions.cc
 * \brief Infer the specific instructions for tileop (currently only for Gemm)
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "common/gemm.h"

namespace tvm {
namespace tl {

using namespace tir;

class InferInstructionsPass : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "InferInstructions: Require the target attribute";

    InferInstructionsPass infer(target.value());
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = infer.VisitStmt(f->body);
    return f;
  }

  explicit InferInstructionsPass(Target target) : target_(target) {}

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ICHECK_NE(iv->thread_tag.length(), 0U);
      if (iv->thread_tag == "threadIdx.x") {
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_block_size_ = iv->dom->extent.as<IntImmNode>()->value;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      // Only handle Gemm operator for now
      // Copy instruction inference requires layout information, which is not
      // available yet
      auto gemm_node = TryParseGemmNode(*call);
      if (!gemm_node.has_value()) {
        return StmtExprMutator::VisitStmt_(op);
      }
      if (call->annotations.count("instruction")) {
        // Instruction is specified by user, skip inference
        return StmtExprMutator::VisitStmt_(op);
      }
      Map<String, ObjectRef> new_annotations = call->annotations;
      ICHECK(thread_block_size_ > 0)
          << "Thread block size not set, ensure AttrStmt with thread_extent is "
             "visited first";
      GemmInst gemm_inst = std::visit(
          [this](const auto *node) {
            return node->getGemmInst(thread_block_size_, target_);
          },
          *gemm_node);
      new_annotations.Set("instruction", Integer(static_cast<int>(gemm_inst)));
      // LOG(INFO) << "Inferred GEMM instruction: " <<
      // static_cast<int>(gemm_inst);
      Call new_call =
          Call(call->dtype, call->op, call->args, new_annotations, call->span);
      return Evaluate(new_call);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Target target_;
  int thread_block_size_ = 0;
};

using namespace tir::transform;

tvm::transform::Pass InferInstructions() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return InferInstructionsPass::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InferInstructions", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InferInstructions", InferInstructions);
}

} // namespace tl
} // namespace tvm

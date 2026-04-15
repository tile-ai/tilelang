/*!
 * \file instruction_annotation.cc
 * \brief Annotate tile operations with coarse-grained instruction kind.
 *
 * This pass runs **before** LayoutInference and LowerTileOp.  It inspects
 * every `tl.tileop.*` Call node and determines the instruction category that
 * will eventually be selected during lowering.  The result is stored as a
 * string annotation (`tl_instruction_kind`) on the Call node so that later
 * passes (e.g. warp specialization) can make structural decisions without
 * needing the full lowered IR.
 *
 * For copy operations the classification is:
 *   - "tma"      : will use TMA bulk load/store (descriptor or 1-D)
 *   - "cp_async" : will use cp.async
 *   - "sync"     : synchronous copy (SIMT / LDSM / STSM / TMem / normal)
 *
 * For gemm operations the classification is:
 *   - "wgmma"      : Hopper warp-group MMA
 *   - "tcgen5mma"  : Blackwell TCGEN5 MMA
 *   - "mma"        : Volta/Ampere tensor-core MMA
 *   - "mfma"       : AMD CDNA matrix fused multiply-add
 *   - "scalar"     : scalar fallback
 *
 * Because this pass runs before layout inference it intentionally uses only
 * coarse checks (target arch, buffer scopes, shape alignment) that do not
 * depend on the inferred memory layout.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/gemm.h"
#include "../op/operator.h"
#include "../op/utils.h"
#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

/// Annotation key written by this pass.
static constexpr const char *kInstructionKind = "tl_instruction_kind";

static bool IsAutoAsyncCopyEnabled(Target target, bool default_enabled = true) {
  using namespace tvm::transform;
  PassContext pass_ctx = PassContext::Current();
  return TargetHasAsyncCopy(target) &&
         pass_ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(default_enabled))
             .value();
}

static bool CanUseAutoCPAsyncCopy(const CopyNode *copy, Target target,
                                  arith::Analyzer *analyzer,
                                  bool default_enabled = true) {
  return copy != nullptr && !copy->GetIsTmaCopy() && !copy->GetIsAsyncCopy() &&
         IsAutoAsyncCopyEnabled(target, default_enabled) &&
         copy->CheckCPAsyncCopy(target, LayoutMap(), analyzer);
}

// ---------------------------------------------------------------------------
// Classify copy ops
// ---------------------------------------------------------------------------

/*!
 * \brief Determine the coarse instruction kind for a CopyNode.
 *
 * The classification does **not** depend on layout_map (which is unavailable
 * at this point).  It mirrors the priority order in CopyNode::GetCopyInst but
 * collapses BulkLoad/BulkLoad1D/BulkStore/BulkStore1D into "tma" and skips
 * checks that require layout information.
 */
std::string ClassifyCopy(const CopyNode *copy, Target target, bool in_pipeline,
                         arith::Analyzer *analyzer) {
  // Explicit T.tma_copy() — always TMA.
  if (copy->GetIsTmaCopy()) {
    // Verify target can do TMA at all.
    if (copy->CheckBulkLoad(target, analyzer, /*check_last_dim=*/false) ||
        copy->CheckBulkStore(target, analyzer, /*check_last_dim=*/false)) {
      return "tma";
    }
    // User asked for TMA but target doesn't support it — leave unannotated
    // so that LowerTileOp can produce a proper error later.
    return "sync";
  }

  // Explicit T.async_copy() — always cp.async.
  if (copy->GetIsAsyncCopy()) {
    return "cp_async";
  }

  // Generic T.copy() stays synchronous here. Auto-TMA is only introduced by
  // warp-specialized rewriting, which rewrites the op to explicit T.tma_copy.

  // LDSM / STSM / TMem — these are synchronous from the WS perspective.
  if (copy->CheckLDSMCopy(target) || copy->CheckSTSMCopy(target) ||
      copy->CheckTMemLoad(target) || copy->CheckTMemStore(target)) {
    return "sync";
  }

  // Inside a pipelined loop, eligible copies may be lowered to cp.async.
  if (in_pipeline && CanUseAutoCPAsyncCopy(copy, target, analyzer,
                                           /*default_enabled=*/false)) {
    return "cp_async";
  }

  return "sync";
}

// ---------------------------------------------------------------------------
// Classify gemm ops
// ---------------------------------------------------------------------------

std::string ClassifyGemm(const GemmNode *gemm, int block_size, Target target) {
  GemmInst inst = gemm->getGemmInst(block_size, target);
  switch (inst) {
  case GemmInst::kWGMMA:
    return "wgmma";
  case GemmInst::kTCGEN5MMA:
    return "tcgen5mma";
  case GemmInst::kMMA:
    return "mma";
  case GemmInst::kMFMA:
    return "mfma";
  case GemmInst::kScalar:
    return "scalar";
  default:
    return "unknown";
  }
}

// ---------------------------------------------------------------------------
// IR rewriter
// ---------------------------------------------------------------------------

class InstructionAnnotator : public StmtExprMutator {
public:
  static PrimFunc Annotate(PrimFunc f) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "InstructionAnnotation: target attribute is required";

    InstructionAnnotator annotator;
    annotator.target_ = target.value();
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = annotator.VisitStmt(f->body);
    return f;
  }

private:
  // Track threadIdx.x extent for gemm instruction selection.
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        if (auto *int_imm = op->value.as<IntImmNode>()) {
          block_size_ = static_cast<int>(int_imm->value);
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // Track whether we are inside a pipelined loop.
  Stmt VisitStmt_(const ForNode *op) final {
    bool old_in_pipeline = in_pipeline_;
    if (op->annotations.Get("num_stages")) {
      in_pipeline_ = true;
    }
    Stmt result = StmtExprMutator::VisitStmt_(op);
    in_pipeline_ = old_in_pipeline;
    return result;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));

    // Only process tile operators.
    auto tile_op = ParseOperator(call);
    if (!tile_op.defined())
      return call;

    // Skip if already annotated.
    if (call->annotations.count(kInstructionKind))
      return call;

    std::string kind;

    if (auto *copy_node = tile_op.as<CopyNode>()) {
      kind = ClassifyCopy(copy_node, target_, in_pipeline_, &analyzer_);
    } else if (auto *gemm_node = tile_op.as<GemmNode>()) {
      kind = ClassifyGemm(gemm_node, block_size_, target_);
    } else {
      // Other tile ops (reduce, fill, etc.) are synchronous.
      kind = "sync";
    }

    // Create a new Call with the annotation added.
    auto new_annotations = call->annotations;
    new_annotations.Set(kInstructionKind, StringImm(kind));
    return Call(call->dtype, call->op, call->args, new_annotations, call->span);
  }

  Target target_;
  bool in_pipeline_{false};
  int block_size_{0};
  arith::Analyzer analyzer_;
};

} // namespace

// ---------------------------------------------------------------------------
// Pass registration
// ---------------------------------------------------------------------------

tvm::transform::Pass InstructionAnnotation() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return InstructionAnnotator::Annotate(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InstructionAnnotation", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.InstructionAnnotation",
                        InstructionAnnotation);
}

} // namespace tl
} // namespace tvm

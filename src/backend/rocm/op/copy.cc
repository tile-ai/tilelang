/*!
 * \file tl/backend/rocm/op/copy.cc
 * \brief ROCm implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "op/builtin.h"
#include "op/utils.h"
#include "target/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"
#include "transform/ptx_async_copy_injector.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/transform.h>

#include <cstdint>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace rocm {

enum class CopyInst : uint8_t {
  kNormal = 0,
  kCPAsync = 1,
};

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                               InferLevel level) {
    SelectInst(op, T.target, T.layout_map, T.analyzer);
    return op.InferSIMTLayout(T, level);
  }

  static CopyInst SelectInst(const CopyNode &op, Target target,
                             const LayoutMap &layout_map,
                             arith::Analyzer *analyzer) {
    if (op.GetIsTmaCopy()) {
      LOG(FATAL) << "T.tma_copy() is not supported on ROCm target "
                 << target->ToDebugString();
    }

    if (op.GetIsAsyncCopy() || op.GetNoImplicitAsyncCommitWait()) {
      bool cp_async_supported =
          CheckCPAsyncCopy(op, target, layout_map, analyzer);
      ICHECK(cp_async_supported)
          << "Explicit async copy semantics require ROCm async copy lowering, "
             "but constraints were not satisfied. Got src="
          << op.src->name << " (scope=" << op.src.scope()
          << ", dtype=" << op.src->dtype << "), dst=" << op.dst->name
          << " (scope=" << op.dst.scope() << ", dtype=" << op.dst->dtype
          << ").";
      return CopyInst::kCPAsync;
    }

    return CopyInst::kNormal;
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    auto copy_inst = SelectInst(op, T.target, T.layout_map, analyzer);
    if (copy_inst == CopyInst::kCPAsync) {
      return LowerCPAsync(op, T, analyzer);
    }
    if (copy_inst == CopyInst::kNormal) {
      return LowerNormalCopy(op, T, analyzer);
    }
    LOG(FATAL) << "Unsupported ROCm copy inst " << static_cast<int>(copy_inst);
  }

private:
  static Stmt LowerCPAsync(const CopyNode &op, const LowerArgs &T,
                           arith::Analyzer *analyzer) {
    using namespace tvm::transform;

    PassContext pass_ctx = PassContext::Current();
    bool enable_async_copy =
        pass_ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(true)).value();
    bool no_implicit_commit_wait = op.GetNoImplicitAsyncCommitWait();
    bool explicit_async_semantics =
        no_implicit_commit_wait || op.GetIsAsyncCopy();
    if (!enable_async_copy && !explicit_async_semantics) {
      return LowerNormalCopy(op, T, analyzer);
    }

    auto simt_loop = op.MakeSIMTLoop(analyzer);
    auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
    auto par_op = ParallelOp(fused_loop);

    std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                      InferLevel::kFree};
    for (auto level : levels) {
      par_op->InferLayout({T.target,
                           T.thread_bounds,
                           T.layout_map,
                           analyzer,
                           false,
                           T.buffer_remap,
                           {}},
                          level);
    }
    auto loop_layout = par_op->GetLoopLayout();
    Stmt lowered_loop = LowerParallelLoop(par_op->GetRoot(), loop_layout,
                                          T.thread_var, analyzer, T.layout_map,
                                          par_op->GetPredicate(T.thread_var));

    auto inject_result =
        InjectPTXAsyncCopy(lowered_loop, /*enable_auto_async_copy=*/true,
                           /*async_without_async_commit_wait=*/
                           no_implicit_commit_wait || op.GetIsAsyncCopy());
    Stmt cp_async_loop = inject_result.stmt;
    if (!inject_result.injected_ptx_async_copy) {
      LOG(WARNING) << "cp.async rewrite miss for copy src=" << op.src->name
                   << " (scope=" << op.src.scope()
                   << ", dtype=" << op.src->dtype << "), dst=" << op.dst->name
                   << " (scope=" << op.dst.scope()
                   << ", dtype=" << op.dst->dtype
                   << "), no_implicit_async_commit_wait="
                   << no_implicit_commit_wait
                   << ", is_async_copy=" << op.GetIsAsyncCopy();
      if (no_implicit_commit_wait) {
        LOG(WARNING)
            << "Pipeline-managed async copy fallback to normal copy because "
               "cp.async rewrite found no eligible global->shared store.";
        return lowered_loop;
      }
      if (explicit_async_semantics) {
        LOG(FATAL)
            << "Explicit async copy semantics require cp.async lowering, "
               "but no eligible global->shared store was rewritten.";
      }
      LOG(WARNING) << "Fallback to normal copy because cp.async rewrite found "
                      "no eligible global->shared store.";
      return LowerNormalCopy(op, T, analyzer);
    }
    if (no_implicit_commit_wait) {
      return cp_async_loop;
    }
    if (op.GetIsAsyncCopy()) {
      Stmt commit_group =
          Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
      return SeqStmt({cp_async_loop, commit_group});
    }
    return cp_async_loop;
  }

  static bool CheckCPAsyncCopyPreconditions(const CopyNode &op) {
    if (!IsGlobalBuffer(op.src) || !IsSharedBuffer(op.dst)) {
      return false;
    }
    if (op.src->dtype != op.dst->dtype) {
      return false;
    }
    return true;
  }

  static bool CheckCPAsyncCopy(const CopyNode &op, Target target,
                               const LayoutMap &layout_map,
                               arith::Analyzer *analyzer) {
    if (!TargetHasAsyncCopy(target)) {
      return false;
    }
    return CheckCPAsyncCopyPreconditions(op);
  }
};

} // namespace rocm

namespace {

bool MatchROCmCopyTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmCopy() {
  RegisterCopyImpl(CopyImpl{
      "rocm.Copy",
      MatchROCmCopyTarget,
      100,
      rocm::Copy::InferLayout,
      rocm::Copy::Lower,
  });
  return true;
}

const bool rocm_copy_registered = RegisterROCmCopy();

} // namespace

} // namespace tl
} // namespace tvm

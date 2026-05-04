/*!
 * \file tl/backend/rocm/op/copy.cc
 * \brief ROCm implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace rocm {

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                               InferLevel level) {
    return CopyLoweringAccess::InferLayoutImpl(op, T, level);
  }

  static CopyInst SelectInst(const CopyNode &op, Target target,
                             const LayoutMap &layout_map,
                             arith::Analyzer *analyzer, bool buffer_oob) {
    if (op.GetIsTmaCopy()) {
      LOG(FATAL) << "T.tma_copy() is not supported on ROCm target "
                 << target->ToDebugString();
    }

    if (op.GetIsAsyncCopy() || op.GetNoImplicitAsyncCommitWait()) {
      bool cp_async_supported =
          op.CheckCPAsyncCopy(target, layout_map, analyzer);
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
    auto copy_inst =
        SelectInst(op, T.target, T.layout_map, analyzer, /*buffer_oob=*/false);
    if (copy_inst == CopyInst::kCPAsync) {
      return LowerCPAsyncCopy(op, T, analyzer);
    }
    if (copy_inst == CopyInst::kNormal) {
      return LowerNormalCopy(op, T, analyzer);
    }
    LOG(FATAL) << "Unsupported ROCm copy inst " << static_cast<int>(copy_inst);
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
      rocm::Copy::SelectInst,
      rocm::Copy::Lower,
  });
  return true;
}

const bool rocm_copy_registered = RegisterROCmCopy();

} // namespace

} // namespace tl
} // namespace tvm

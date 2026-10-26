#include "compat/analyzer.h"
/*!
 * \file tl/cpu/op/fill.cc
 * \brief CPU implementation for tl.fill lowering.
 */

#include "op/fill.h"
#include <tvm/runtime/logging.h>

#include "backend/common/op/fill.h"
#include "backend/common/target_utils.h"
#include "op/utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

namespace tvm {
namespace tl {

namespace cpu {

struct Fill {
  static Stmt Lower(const FillNode &op, const LowerArgs &lower_args,
                    sym::Analyzer analyzer) {
    if (IsFragmentBuffer(op.dst)) {
      TVM_FFI_CHECK(tl::arith::CanProveEqual(
                        analyzer, lower_args.thread_bounds->extent, 1),
                    ValueError)
          << "CPU fragment fill requires one logical participant";
      return backend::Fill::Lower(op, lower_args, analyzer);
    }
    if (IsLocalBuffer(op.dst, true) || IsGlobalBuffer(op.dst)) {
      auto init_loop = op.MakeSIMTLoop(analyzer);
      if (lower_args.target->kind->name == "c") {
        return PragmaUnrollLoop(init_loop);
      }
      auto vectorized_loop =
          VectorizeLoop(init_loop, analyzer, lower_args.layout_map);
      return PragmaUnrollLoop(vectorized_loop);
    }

    TVM_FFI_THROW(ValueError)
        << "CPU fill only supports local and global tensors, but got "
        << "dst scope `" << op.dst.scope().name() << "`.";
  }
};

} // namespace cpu

namespace {

bool MatchCPUFillTarget(Target target) { return TargetIsCPU(target); }

bool RegisterCPUFill() {
  RegisterFillImpl(FillImpl{
      "cpu.Fill",
      MatchCPUFillTarget,
      cpu::Fill::Lower,
  });
  return true;
}

const bool cpu_fill_registered = RegisterCPUFill();

} // namespace

} // namespace tl
} // namespace tvm

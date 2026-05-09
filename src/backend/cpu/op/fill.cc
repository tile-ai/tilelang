/*!
 * \file tl/backend/cpu/op/fill.cc
 * \brief CPU implementation for tl.fill lowering.
 */

#include "op/fill.h"

#include "op/utils.h"
#include "target/utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

namespace tvm {
namespace tl {

namespace cpu {

struct Fill {
  static Stmt Lower(const FillNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    if (IsLocalBuffer(op.dst, true) || IsGlobalBuffer(op.dst)) {
      auto init_loop = op.MakeSIMTLoop(analyzer);
      auto vectorized_loop = VectorizeLoop(init_loop, analyzer, T.layout_map);
      return PragmaUnrollLoop(vectorized_loop);
    }

    LOG(FATAL) << "CPU fill only supports local and global buffers, but got "
               << "dst scope `" << op.dst.scope() << "`.";
    return Stmt();
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

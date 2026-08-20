/*!
 * \file tl/cuda/op/reduce.cc
 * \brief CUDA implementation for tl.reduce AllReduce lowering.
 */

#include "backend/common/op/reduce.h"

#include "backend/common/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace cuda {

struct Reduce : backend::ReduceLowerer<Reduce> {
  static bool IsFAdd2Enabled(const ReduceOpNode &op) {
    constexpr const char *kEnableFAdd2 = "enable_fadd2";
    if (auto value = op.annotations.Get(kEnableFAdd2)) {
      if (auto enabled = value.value().as<Bool>()) {
        return enabled.value();
      }
      if (auto enabled = value.value().as<IntImm>()) {
        return enabled.value()->value != 0;
      }
      LOG(FATAL) << "CUDA ReduceOp annotation `" << kEnableFAdd2
                 << "` must be a boolean";
    }
    return true;
  }

  static bool SupportsFp16Bf16NanReduce(Target target) {
    return TargetIsCuda(target);
  }

  static int GetPreferredVectorizedSize(const ReduceOpNode &op, Target target) {
    if (!TargetIsCuda(target)) {
      return 1;
    }
    bool supports_fp32x2 = TargetHasSMVersionGE(target, 100);
    int vsize = backend::reduce::GetPreferredVectorizedSize(op.dst->dtype,
                                                            supports_fp32x2);
    if (vsize == 2 && op.dst->dtype.is_float() && op.dst->dtype.bits() == 32 &&
        (op.type->IsSum() || op.type->IsAbsSum()) && !IsFAdd2Enabled(op)) {
      return 1;
    }
    return vsize;
  }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset,
                                        PrimExpr all_threads, int batch,
                                        int workspace_stride, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (TargetSupportsNamedBarrier(target)) {
      ss << ", tl::NamedBarrier<" << all_threads << ">";
    } else {
      ss << ", tl::SyncThreadsBarrier";
    }
    ss << ", " << batch << ", " << workspace_stride << ">::run_batch";
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset,
                                         PrimExpr all_threads, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (TargetSupportsNamedBarrier(target)) {
      ss << ", tl::NamedBarrier<" << all_threads << ">";
    }
    ss << ">::run";
    return ss.str();
  }
};

} // namespace cuda

namespace {

bool MatchCudaReduceTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaReduce() {
  RegisterReduceImpl(ReduceImpl{
      "cuda.Reduce",
      MatchCudaReduceTarget,
      cuda::Reduce::Lower,
  });
  return true;
}

const bool cuda_reduce_registered = RegisterCudaReduce();

} // namespace

} // namespace tl
} // namespace tvm

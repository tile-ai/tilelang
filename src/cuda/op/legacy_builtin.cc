/*!
 * \file tl/cuda/op/legacy_builtin.cc
 * \brief TileLang-owned contracts for source spellings absent from native TIRx.
 */

#include "builtin.h"
#include "op/native_intrinsic.h"

namespace tvm::tl {
namespace {

void ValidateLegacyCudaIntrinsic(Call call) {
  const auto name = call->op.as_or_throw<Op>()->name;
  const auto args = NativeIntrinsicOperands(call.get());
  if (name == "tl.create_barriers") {
    const auto *count = args[0].as<IntImmObj>();
    TVM_FFI_CHECK(count && count->value > 0 && count->value <= INT32_MAX,
                  ValueError)
        << "create_barriers requires a positive constant barrier count";
  } else if (name == "tl.ptx_wait_group") {
    const auto *count = args[0].as<IntImmObj>();
    TVM_FFI_CHECK(count && count->value >= 0 && count->value < 8, ValueError)
        << "ptx_wait_group requires an immediate in [0, 7]";
  } else if (name == "tl.ptx_arrive_barrier_expect_tx") {
    TVM_FFI_CHECK(args.size() == 2 || args.size() == 4, ValueError)
        << "ptx_arrive_barrier_expect_tx expects barrier/bytes and optional "
           "CTA/predicate";
  } else if (name == "tl.ptx_cp_async_bytes") {
    const auto *count = args[2].as<IntImmObj>();
    TVM_FFI_CHECK(
        count && (count->value == 4 || count->value == 8 || count->value == 16),
        ValueError)
        << "ptx_cp_async_bytes requires a constant byte width in {4, 8, 16}";
  }
}

void DefineLegacyCuda(const char *name, int minimum, int maximum) {
  OpDef def(name);
  DefineNativeIntrinsic(def, minimum, maximum, CallEffectKind::kOpaque);
  def.set_attr(kNativeIntrinsicExtraValidator,
               NativeIntrinsicExtraValidator(ValidateLegacyCudaIntrinsic));
}

} // namespace

#define TL_LEGACY_CUDA_GETTER(Name)                                            \
  const Op &Name() {                                                           \
    static const Op op = Op::Get("tl." #Name);                                 \
    return op;                                                                 \
  }

TL_LEGACY_CUDA_GETTER(ptx_mma_sp)
TL_LEGACY_CUDA_GETTER(ptx_cp_async_bulk)
TL_LEGACY_CUDA_GETTER(ptx_wait_barrier)
TL_LEGACY_CUDA_GETTER(create_barriers)
TL_LEGACY_CUDA_GETTER(ptx_arrive_barrier)
TL_LEGACY_CUDA_GETTER(ptx_init_barrier_thread_count)
TL_LEGACY_CUDA_GETTER(ptx_arrive_barrier_expect_tx)
TL_LEGACY_CUDA_GETTER(ptx_cp_async_barrier)
TL_LEGACY_CUDA_GETTER(ptx_commit_group)
TL_LEGACY_CUDA_GETTER(ptx_wait_group)
TL_LEGACY_CUDA_GETTER(ptx_cp_async_bytes)
TL_LEGACY_CUDA_GETTER(ptx_ldg32)

#undef TL_LEGACY_CUDA_GETTER

TVM_FFI_STATIC_INIT_BLOCK() {
  DefineLegacyCuda("tl.ptx_mma_sp", 16, 16);
  // This legacy op intentionally retains a constant barrier ID.  Native
  // pointer-based TIRx PTX ops keep their own registration and are not aliases.
  DefineLegacyCuda("tl.ptx_cp_async_bulk", 6, 6);
  DefineLegacyCuda("tl.ptx_wait_barrier", 1, 1);
  DefineLegacyCuda("tl.create_barriers", 1, 1);
  DefineLegacyCuda("tl.ptx_arrive_barrier", 1, 1);
  DefineLegacyCuda("tl.ptx_init_barrier_thread_count", 2, 2);
  DefineLegacyCuda("tl.ptx_arrive_barrier_expect_tx", 2, 4);
  DefineLegacyCuda("tl.ptx_cp_async_barrier", 1, 1);
  DefineLegacyCuda("tl.ptx_commit_group", 0, 0);
  DefineLegacyCuda("tl.ptx_wait_group", 1, 1);
  DefineLegacyCuda("tl.ptx_cp_async_bytes", 3, 4);
  DefineLegacyCuda("tl.ptx_ldg32", 4, 4);
}

} // namespace tvm::tl

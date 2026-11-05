/*!
 * \file tl/metal/op/fill.cc
 * \brief Metal implementation for tl.fill lowering.
 */

#include "op/fill.h"
#include "backend/common/op/fill.h"
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>

#include "backend/common/target_utils.h"
#include "metal/op/builtin.h"
#include "metal/op/utils.h"
#include "op/utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;

namespace metal {

struct Fill {
  static Stmt Lower(const FillNode &op, const LowerArgs &lower_args,
                    sym::AnalyzerObj *analyzer) {
    if (IsCooperativeTensorBuffer(op.dst)) {
      int region_elements = 1;
      for (auto r : op.region) {
        auto imm = r->extent.as<IntImmObj>();
        TVM_FFI_ICHECK(imm)
            << "cooperative_tensor fill region must have constant extents";
        region_elements *= RequireStaticInt(r->extent, "Metal fill region");
      }
      constexpr int kTileM = 16;
      constexpr int kTileN = 32;
      constexpr int kTileElems = kTileM * kTileN;
      TVM_FFI_ICHECK(region_elements % kTileElems == 0)
          << "cooperative_tensor buffer size must be multiple of " << kTileElems
          << ", got " << region_elements;
      int num_tiles = region_elements / kTileElems;
      PrimExpr fill_value = cast(op.value, op.dst.dtype());
      Array<Stmt> stmts;
      for (int i = 0; i < num_tiles; i++) {
        stmts.push_back(
            Evaluate(Call(PrimType::Void(), cooperative_tensor_fill(),
                          {op.dst, IntImm(PrimType::Int(32), i), fill_value,
                           IntImm(PrimType::Int(32), kTileM),
                           IntImm(PrimType::Int(32), kTileN)})));
      }
      if (stmts.size() == 1) {
        return stmts[0];
      }
      return SeqStmt(stmts);
    }

    if (IsSIMDGroupBuffer(op.dst)) {
      int region_elements = 1;
      for (auto r : op.region) {
        auto imm = r->extent.as<IntImmObj>();
        TVM_FFI_ICHECK(imm)
            << "simdgroup fill region must have constant extents";
        region_elements *= RequireStaticInt(r->extent, "Metal fill region");
      }
      TVM_FFI_ICHECK(region_elements % 64 == 0)
          << "simdgroup buffer size must be multiple of 64 (8x8), got "
          << region_elements;

      int num_matrices = region_elements / 64;
      if (auto physical = lower_args.buffer_remap.Get(op.dst)) {
        // A promoted fragment denotes a logical matrix distributed over warps.
        // Its typed replacement contains only the matrices owned by one warp.
        for (size_t axis = 0; axis < op.region.size(); ++axis) {
          TVM_FFI_CHECK(IsZero(op.region[axis]->begin) &&
                            analyzer->CanProveEqual(op.region[axis]->extent,
                                                    op.dst.shape()[axis]),
                        ValueError)
              << "Metal SIMD-group fill requires a full accumulator region";
        }
        int elements = 1;
        for (const auto &extent : physical.value().shape())
          elements *=
              RequireStaticInt(extent, "Metal per-warp accumulator shape");
        TVM_FFI_CHECK_EQ(elements % 64, 0, ValueError);
        num_matrices = elements / 64;
      }
      PrimExpr fill_value = cast(op.value, op.dst.dtype());
      Array<Stmt> stmts;
      for (int i = 0; i < num_matrices; i++) {
        stmts.push_back(Evaluate(Call(
            PrimType::Void(),
            Op::Get("tirx.metal.make_filled_simdgroup_matrix"),
            {op.dst, IntImm(PrimType::Int(32), i), fill_value,
             IntImm(PrimType::Int(32), 8), IntImm(PrimType::Int(32), 8)})));
      }
      if (stmts.size() == 1) {
        return stmts[0];
      }
      return SeqStmt(stmts);
    }

    return backend::Fill::Lower(op, lower_args, analyzer);
  }
};

} // namespace metal

namespace {

bool MatchMetalFillTarget(Target target) { return TargetIsMetal(target); }

bool RegisterMetalFill() {
  RegisterFillImpl(FillImpl{
      "metal.Fill",
      MatchMetalFillTarget,
      metal::Fill::Lower,
  });
  return true;
}

const bool metal_fill_registered = RegisterMetalFill();

} // namespace

} // namespace tl
} // namespace tvm

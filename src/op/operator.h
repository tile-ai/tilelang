/*!
 * \file tl/op/op.h
 * \brief Tile library operations.
 *
 */

#ifndef TVM_TL_OP_OP_H_
#define TVM_TL_OP_OP_H_

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/target/target.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>

#include "../layout/layout.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using AddWorkspaceCallback = std::function<PrimExpr(int, DataType)>;
// Allocate a compiler-generated shared mbarrier slot. The optional hint names
// the backing barrier buffer when the first slot is created; later slots share
// the same buffer and may ignore the hint.
using AllocMBarrierCallback =
    std::function<int(int arrive_count, std::optional<std::string> name)>;
using UpdateBarrierArriveCallback = std::function<void(tirx::Var, PrimExpr)>;
// Record a minimum shared-memory base alignment (bytes) required for the
// buffer backed by the given data Var (e.g. TMA/MMA swizzle constraints).
using RequireSmemAlignmentCallback = std::function<void(tirx::Var, int)>;
using LayoutMap = ffi::Map<tirx::Buffer, Layout>;
using BufferMap = ffi::Map<tirx::Var, tirx::Buffer>;
using BlockAnnotations = ffi::Map<ffi::String, ffi::Any>;

enum AccessMask : int {
  kAccessRead = 1,
  kAccessWrite = 2,
  kAccessReadWrite = kAccessRead | kAccessWrite,
};

struct AccessRegion {
  tirx::BufferRegion region;
  int access_mask{kAccessReadWrite};
};

struct AccessRegions {
  ffi::Array<tirx::BufferRegion> reads;
  ffi::Array<tirx::BufferRegion> writes;
};

inline void AppendAccessRegionByMask(const AccessRegion &access,
                                     ffi::Array<tirx::BufferRegion> *reads,
                                     ffi::Array<tirx::BufferRegion> *writes) {
  if (!access.region.defined()) {
    return;
  }
  if (access.access_mask & kAccessRead) {
    reads->push_back(access.region);
  }
  if (access.access_mask & kAccessWrite) {
    writes->push_back(access.region);
  }
}

enum class InferLevel : uint8_t {
  kFree = 0,
  kCommon = 1,
  kStrict = 2,
};

/// Convert InferLevel enum to string for debugging
inline const char *InferLevelToString(InferLevel level) {
  switch (level) {
  case InferLevel::kFree:
    return "Free";
  case InferLevel::kCommon:
    return "Common";
  case InferLevel::kStrict:
    return "Strict";
  default:
    return "Unknown";
  }
}

struct LowerArgs {
  Target target;
  Range thread_bounds;
  // Logical thread index consumed by lowering helpers. This is an expression
  // rather than a Var: GPU lowering passes the real threadIdx.x Var (bound by
  // a thread_extent AttrStmt), while targets without thread bindings (e.g.
  // CPU) pass constant 0. It must never be an unbound synthetic Var.
  PrimExpr thread_index;
  LayoutMap layout_map;
  ffi::Map<tirx::Buffer, tirx::Buffer> buffer_remap;
  // Map from Bind variable to its bound expression, for resolving
  // fragment buffer accesses through Bind values
  ffi::Map<tirx::Var, PrimExpr> bind_var_to_expr;
  // Fallback mbarrier parity for ops that do not carry an explicit
  // tl.pipeline_mbar_phase_expr annotation. LowerTileOp derives this from the
  // nearest enclosing serial loop so non-pipelined TMA loops still alternate
  // barrier phase correctly.
  PrimExpr mbar_phase_expr = IntImm(DataType::Int(32), 0);
  // Pointer to the shared.barrier buffer for compiler-generated mbarriers.
  // Points to the LowerTileOpPass member so copy.cc sees the buffer
  // even when created lazily by the alloc_mbarrier callback.
  ffi::Optional<tirx::Buffer> *mbarrier_buffer = nullptr;
  // Product of cluster_dims (from block annotation). Defaults to 1 (no
  // cluster). Used by TMA copy lowering to scale expect_tx bytes for cluster
  // barriers.
  int cluster_size = 1;

  // Callbacks used by op lowerings to request pass-owned resources or report
  // metadata that later passes need.
  AddWorkspaceCallback add_workspace = nullptr;
  AllocMBarrierCallback alloc_mbarrier = nullptr;
  UpdateBarrierArriveCallback update_barrier_arrive = nullptr;
  // Optional callback to record a minimum shared-memory base alignment for a
  // buffer's data Var. Lowerings that bind a buffer to a swizzle-sensitive
  // instruction (TMA bulk copy, wgmma/tcgen05 descriptors) report the
  // alignment implied by the chosen swizzle mode here; LowerTileOp collects
  // the results into the kSmemAlignmentMap PrimFunc attribute.
  RequireSmemAlignmentCallback require_smem_alignment = nullptr;
};

struct LayoutInferArgs {
  Target target;
  Range thread_bounds;
  LayoutMap layout_map;
  arith::Analyzer *analyzer;
  ffi::Map<tirx::Buffer, tirx::Buffer> buffer_remap;
  // Map from Bind variable to its bound expression, for resolving
  // fragment buffer accesses through Bind values
  ffi::Map<tirx::Var, PrimExpr> bind_var_to_expr;
  // Whether the current TileOp is nested inside a pipelined loop
  // (i.e. a surrounding loop annotated with num_stages > 0).
  bool in_pipeline = false;
};

class TileOperator;

class TileOperatorNode : public ffi::Object {
public:
  virtual tirx::Stmt Lower(const LowerArgs &lower_args,
                           arith::Analyzer *analyzer) const = 0;

  virtual LayoutMap InferLayout(const LayoutInferArgs &layout_args,
                                InferLevel level) const = 0;

  virtual TileOperator Clone() const = 0;

  virtual AccessRegions GetAccessRegions() const {
    AccessRegions result;
    for (const auto &access : access_regions_) {
      AppendAccessRegionByMask(access, &result.reads, &result.writes);
    }
    return result;
  }

  void SetAccessRegions(std::vector<AccessRegion> access_regions) {
    access_regions_ = std::move(access_regions);
  }

  TVM_FFI_DECLARE_OBJECT_INFO("tl.TileOperator", TileOperatorNode, ffi::Object);

protected:
  std::vector<AccessRegion> access_regions_;
};

class TileOperator : public ffi::ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TileOperator, ffi::ObjectRef,
                                             TileOperatorNode);
};

tirx::Var GetVarFromAccessPtr(const PrimExpr &expr);

TileOperator
ParseOperator(const tirx::Call &call,
              const BlockAnnotations &block_annotations = BlockAnnotations());
TileOperator
ParseOperator(const tirx::Stmt &stmt,
              const BlockAnnotations &block_annotations = BlockAnnotations());

using OpBuilderFunc = ffi::TypedFunction<TileOperator(
    ffi::Array<PrimExpr>, ffi::Map<ffi::String, ffi::ObjectRef>)>;
using OpBlockAnnotationHandlerFunc =
    ffi::TypedFunction<TileOperator(TileOperator, BlockAnnotations)>;

static constexpr const char *kTLOpBlockAnnotationHandler =
    "TLOpBlockAnnotationHandler";

#define TIR_REGISTER_TL_TILE_OP(Entry, OpName)                                 \
  const Op &Entry::Get() {                                                     \
    static const Op &op = Op::Get("tl.tileop." #OpName);                       \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl.tileop." #OpName)                                        \
      .set_attr<tirx::TScriptPrinterName>("TScriptPrinterName", #OpName)       \
      .set_attr<OpBuilderFunc>(                                                \
          "TLOpBuilder",                                                       \
          [](ffi::Array<PrimExpr> args,                                        \
             ffi::Map<ffi::String, ffi::ObjectRef> annotations) {              \
            return Entry(args, annotations);                                   \
          })

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_OP_H_

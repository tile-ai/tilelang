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

#include <tvm/ir/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/target/target.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/stmt.h>

#include "../layout/layout.h"
#include "builtin_registry.h"
#include "support/check.h"

namespace tvm {
namespace tl {

using AddWorkspaceCallback = std::function<Expr(int, PrimType)>;
// Allocate a compiler-generated shared mbarrier slot. The optional hint names
// the backing barrier buffer when the first slot is created; later slots share
// the same buffer and may ignore the hint.
using AllocMBarrierCallback =
    std::function<int(int arrive_count, std::optional<std::string> name)>;
using UpdateBarrierArriveCallback = std::function<void(Var, PrimExpr)>;
// Record a minimum shared-memory base alignment (bytes) required for the
// buffer backed by the given data Var (e.g. TMA/MMA swizzle constraints).
using RequireSmemAlignmentCallback = std::function<void(Var, int)>;
using LayoutMap = ffi::Map<tirx::TensorVar, Layout>;
using BufferMap = ffi::Map<Var, tirx::TensorVar>;
using BlockAnnotations = ffi::Map<ffi::String, ffi::Any>;

enum AccessMask : int {
  kAccessRead = 1,
  kAccessWrite = 2,
  kAccessReadWrite = kAccessRead | kAccessWrite,
};

// Preserve known read/write bits and conservatively assume both when a mask
// cannot be resolved at compile time.
inline int GetConservativeAccessMask(const PrimExpr &rw_mask) {
  if (auto mask = GetConstInt(rw_mask)) {
    return static_cast<int>(*mask & kAccessReadWrite);
  }
  return kAccessReadWrite;
}

struct AccessRegion {
  TensorRegion region;
  int access_mask{kAccessReadWrite};
};

struct AccessRegions {
  ffi::Array<TensorRegion> reads;
  ffi::Array<TensorRegion> writes;
};

inline void AppendAccessRegionByMask(const AccessRegion &access,
                                     ffi::Array<TensorRegion> *reads,
                                     ffi::Array<TensorRegion> *writes) {
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
  ffi::Map<tirx::TensorVar, tirx::TensorVar> buffer_remap;
  // Map from Bind variable to its bound expression, for resolving
  // fragment buffer accesses through Bind values
  ffi::Map<Var, Expr> bind_var_to_expr;
  // Fallback mbarrier parity for ops that do not carry an explicit
  // tl.pipeline_mbar_phase_expr annotation. LowerTileOp derives this from the
  // zero-based epoch of the nearest enclosing serial loop.
  PrimExpr mbar_phase_expr = IntImm(PrimType::Int(32), 0);
  // Pointer to the shared.barrier buffer for compiler-generated mbarriers.
  // Points to the LowerTileOpPass member so copy.cc sees the buffer
  // even when created lazily by the alloc_mbarrier callback.
  ffi::Optional<tirx::TensorVar> *mbarrier_buffer = nullptr;
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

/*! \brief One reducer_update site, as seen by layout analysis: the update
 *  loop's solved layout (undefined Fragment while the loop is still
 *  unsolved), the parallel nest vars, the update target's logical indices
 *  and the contribution expression. Assembled by ParallelOp for its own
 *  partial-layout proposal and by ReducerPlanAndMaterialize for lowering
 *  (both feed AnalyzeReducerUpdateSite). */
struct ReducerUpdateSiteHint {
  Fragment loop_layout; // undefined while the update loop is unsolved
  ffi::Array<PrimVar> loop_vars;
  ffi::Array<PrimExpr> indices;
  PrimExpr value{ffi::UnsafeInit{}};
};

struct LayoutInferArgs {
  Target target;
  Range thread_bounds;
  LayoutMap layout_map;
  sym::Analyzer analyzer;
  ffi::Map<tirx::TensorVar, tirx::TensorVar> buffer_remap;
  // Map from Bind variable to its bound expression, for resolving
  // fragment buffer accesses through Bind values
  ffi::Map<Var, Expr> bind_var_to_expr;
  // Whether the current TileOp is nested inside a pipelined loop
  // (i.e. a surrounding loop annotated with num_stages > 0).
  bool in_pipeline = false;
  // Snapshot of the layouts fixed at kStrict (annotations included), taken
  // by the inference engine after the strict pass. Ops use it to recognize
  // authoritative constraints — e.g. an annotated reducer PartialFragment
  // that update nests must satisfy rather than widen. Empty at
  // lowering-time re-inference call sites.
  LayoutMap strict_layout_map;
  // Per-call cap for generating a free-mode parallel candidate. Zero preserves
  // the vectorizer's width; scalar reducer-root attempts pass one. This is
  // a search option, not operator state or a constraint on inferred layouts.
  int candidate_vector_size_limit = 0;
};

class TileOperator;

class TileOperatorNode : public ffi::Object {
public:
  virtual Stmt Lower(const LowerArgs &lower_args,
                     sym::Analyzer analyzer) const = 0;

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

  /*!
   * \brief Regions whose preexisting contents this op consumes before
   *        establishing a value of its own.
   *
   * A subset of GetAccessRegions().reads, which answers the weaker may-read
   * question ("could this op read this region?") and is deliberately
   * conservative for dependency analysis. Ops whose read set depends on an
   * argument value narrow it here; the default reports every read.
   */
  virtual ffi::Array<TensorRegion> GetReadBeforeWriteRegions() const {
    return GetAccessRegions().reads;
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

TILELANG_DLL Var GetVarFromAccessPtr(const Expr &expr);

TILELANG_DLL TileOperator
ParseOperator(const Call &call,
              const BlockAnnotations &block_annotations = BlockAnnotations());
TILELANG_DLL TileOperator
ParseOperator(const Stmt &stmt,
              const BlockAnnotations &block_annotations = BlockAnnotations());

using OpBuilderFunc =
    ffi::TypedFunction<TileOperator(ffi::Array<Expr>, BlockAnnotations)>;
using OpBlockAnnotationHandlerFunc =
    ffi::TypedFunction<TileOperator(TileOperator, BlockAnnotations)>;

static constexpr const char *kTLOpBlockAnnotationHandler =
    "TLOpBlockAnnotationHandler";

template <typename Entry, int NumInputs>
void RegisterTileOperator(const ffi::String &name) {
  RegisterBuiltin<NumInputs>(name, CallEffectKind::kOpaque);
  OpDef(name)
      .set_attr(tvm::op_attr::kFixedReturnType, TupleType(ffi::Array<Type>{}))
      .set_attr<OpBuilderFunc>("TLOpBuilder", [](ffi::Array<Expr> args,
                                                 BlockAnnotations annotations) {
        return Entry(std::move(args), std::move(annotations));
      });
}

#define TIR_REGISTER_TL_TILE_OP(Entry, OpName, NumInputs)                      \
  const Op &Entry::Get() {                                                     \
    static const Op op = Op::Get("tl.tileop." #OpName);                        \
    return op;                                                                 \
  }                                                                            \
  TVM_FFI_STATIC_INIT_BLOCK() {                                                \
    RegisterTileOperator<Entry, NumInputs>("tl.tileop." #OpName);              \
  }

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_OP_H_

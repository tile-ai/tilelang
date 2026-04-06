/*!
 * \file tl/op/op.h
 * \brief Tile library operations.
 *
 */

#ifndef TVM_TL_OP_OP_H_
#define TVM_TL_OP_OP_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/target/target.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/stmt.h>
#include <utility>
#include <vector>

#include "../layout/layout.h"

namespace tvm {
namespace tl {

using namespace tir;

using AddWorkspaceCallback = std::function<PrimExpr(int, DataType)>;
using AllocMBarrierCallback = std::function<int(int arrive_count)>;
using LayoutMap = Map<Buffer, Layout>;
using BufferMap = Map<Var, Buffer>;

enum AccessMask : int {
  kAccessRead = 1,
  kAccessWrite = 2,
  kAccessReadWrite = kAccessRead | kAccessWrite,
};

struct AccessRegion {
  BufferRegion region;
  int access_mask{kAccessReadWrite};
};

struct AccessRegions {
  Array<BufferRegion> reads;
  Array<BufferRegion> writes;
};

inline void AppendAccessRegionByMask(const AccessRegion &access,
                                     Array<BufferRegion> *reads,
                                     Array<BufferRegion> *writes) {
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
  Var thread_var;
  AddWorkspaceCallback AddWorkspace;
  AllocMBarrierCallback AllocMBarrier;
  LayoutMap layout_map;
  Map<Buffer, Buffer> buffer_remap;
  // Map from LetStmt variable to its bound expression, for resolving
  // fragment buffer accesses through let bindings
  Map<Var, PrimExpr> let_var_to_expr;
  // Fallback mbarrier parity for ops that do not carry an explicit
  // tl.pipeline_mbar_phase_expr annotation. LowerTileOp derives this from the
  // nearest enclosing serial loop so non-pipelined TMA loops still alternate
  // barrier phase correctly.
  PrimExpr mbar_phase_expr = IntImm(DataType::Int(32), 0);
  // Pointer to the shared.barrier buffer for compiler-generated mbarriers.
  // Points to the LowerTileOpPass member so copy.cc sees the buffer
  // even when created lazily by the AllocMBarrier callback.
  Optional<Buffer> *mbarrier_buffer = nullptr;
  // Product of cluster_dims (from block annotation). Defaults to 1 (no
  // cluster). Used by TMA copy lowering to scale expect_tx bytes for cluster
  // barriers.
  int cluster_size = 1;
};

struct LayoutInferArgs {
  Target target;
  Range thread_bounds;
  LayoutMap layout_map;
  arith::Analyzer *analyzer;
  bool buffer_oob = false;
  Map<Buffer, Buffer> buffer_remap;
  // Map from LetStmt variable to its bound expression, for resolving
  // fragment buffer accesses through let bindings
  Map<Var, PrimExpr> let_var_to_expr;
  // Whether the current TileOp is nested inside a pipelined loop
  // (i.e. a surrounding loop annotated with num_stages > 0).
  bool in_pipeline = false;
};

class TileOperator;

class TileOperatorNode : public Object {
public:
  virtual Stmt Lower(const LowerArgs &T, arith::Analyzer *analyzer) const = 0;

  virtual LayoutMap InferLayout(const LayoutInferArgs &T,
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

  TVM_FFI_DECLARE_OBJECT_INFO("tl.TileOperator", TileOperatorNode, Object);

protected:
  std::vector<AccessRegion> access_regions_;
};

class TileOperator : public ObjectRef {
public:
  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NULLABLE(TileOperator, ObjectRef,
                                             TileOperatorNode);
};

Var GetVarFromAccessPtr(const PrimExpr &expr);

TileOperator ParseOperator(Call call);
TileOperator ParseOperator(Stmt stmt);

using OpBuilderFunc =
    ffi::TypedFunction<TileOperator(Array<PrimExpr>, Map<String, ObjectRef>)>;

#define TIR_REGISTER_TL_TILE_OP(Entry, OpName)                                 \
  const Op &Entry::Get() {                                                     \
    static const Op &op = Op::Get("tl.tileop." #OpName);                       \
    return op;                                                                 \
  }                                                                            \
  TVM_REGISTER_OP("tl.tileop." #OpName)                                        \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", #OpName)             \
      .set_attr<OpBuilderFunc>(                                                \
          "TLOpBuilder",                                                       \
          [](Array<PrimExpr> args, Map<String, ObjectRef> annotations) {       \
            return Entry(args, annotations);                                   \
          })

} // namespace tl
} // namespace tvm

#endif // TVM_TL_OP_OP_H_

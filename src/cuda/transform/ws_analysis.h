/*!
 * \file ws_analysis.h
 * \brief Statement analysis shared by the warp-specialization passes.
 *
 * Extracted from producer_consumer_ws.cc so automatic schedulers reuse the
 * same classification the legacy producer/consumer rewriter has been
 * validated with.
 */
#pragma once

#include <tvm/ir/stmt.h>
#include <tvm/target/target.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "layout/layout.h"
#include "op/copy.h"
#include "op/operator.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;
using namespace ffi;

using BufferRemap =
    std::unordered_map<TensorVar, TensorVar, ObjectPtrHash, ObjectPtrEqual>;
using BufferLayoutMap = std::unordered_map<Var, std::pair<TensorVar, Layout>,
                                           ObjectPtrHash, ObjectPtrEqual>;

enum class TileStmtKind {
  kTmaProducer,     // TMA load (global->shared tile-op copy)
  kCpAsyncProducer, // tile-op copy selecting cp.async
  kCpAsyncRaw,      // raw ptx_cp_async / commit_group / wait_group statement
  kSimtProducer, // Non-tile-op SIMT copy: For loop writing shared from global
  kTmaStore,     // TMA bulk store (shared->global tile-op copy)
  kTcgen05Mma,   // tcgen05 GEMM accumulating in tensor memory
  kConsumer,     // Everything else (compute, wgmma, plain copies)
};

/// The Evaluate(Call) inside else-less if / attribute / block wrappers, if
/// the statement has that shape.
Optional<Call> GetEvaluateCallInSimpleWrapper(const Stmt &stmt);

/// Hand-written synchronization or TMA control: barriers, arrives, waits,
/// raw TMA instructions, and block-wide syncs. A kernel using these
/// manages a protocol the schedule cannot see.
bool IsBarrierOrTmaControlCall(const CallObj *call);

/// Classify a tile-op copy as TMA load producer, cp.async producer, TMA
/// store, or consumer using coarse pre-layout checks.
TileStmtKind ClassifyCopy(const CopyNode *copy, Target target);

/// Classify a single statement in a pipeline loop body.
TileStmtKind ClassifyStmt(const Stmt &stmt, Target target);

struct GemmInfo {
  TensorVar accumulator{ffi::UnsafeInit{}};
  int wg_wait;
};

/// The accumulator and wg_wait of a dense or sparse gemm tile op;
/// nullopt for any other operator.
std::optional<GemmInfo> GetGemmInfo(const TileOperator &op);

bool IsProducer(TileStmtKind kind);

/// Both the T.ws() language-level attr ("warp_specialize") and the
/// compiler-level attr (kWarpSpecializationScope).
bool HasManualWarpSpecialization(const Stmt &stmt);

/// Check whether a layout annotation on a shared buffer is compatible with
/// TMA. TMA supports identity (linear) layouts and the three standard
/// swizzle modes (32B / 64B / 128B). Any other layout (e.g. padded,
/// Volta-style) cannot be used with TMA.
bool IsTmaCompatibleLayout(const Layout &layout, const TensorVar &buffer);

/// Read explicit native annotation payloads, keyed by the TensorVar identity.
/// Metadata stays in operands so type/shape/layout substitutions remain
/// visible.
void CollectAnnotatedLayouts(const Stmt &body, BufferLayoutMap &layouts);

/// Rewrite every use of a remapped buffer (loads, stores, region roots) to
/// the replacement, including typed TensorVar uses in addresses and metadata
/// payloads.
Stmt RemapBuffers(Stmt stmt, const BufferRemap &remap);

/// Decode an annotated call without treating the metadata Tuple as an operand.
TileOpArguments ReadWSCall(const CallObj *call);

/// A typed call whose user spelling carries annotations retains explicit
/// payload uses.
Call RewriteWSCall(const CallObj *call, const Map<String, Any> &annotations,
                   Optional<Op> replacement_op = std::nullopt);

} // namespace tl
} // namespace tvm

namespace tvm::tl {
// A marker labels the following statement without introducing a definition
// scope.
inline constexpr const char *kWSNextOpIdKey = "tl.ws_next_op_id";
Optional<String> WSNextOpId(const Stmt &statement);
Optional<Map<String, Any>> WSAttribute(const RegionStmtObj *region);
PrimExpr WSPipelineIterationModulo(const std::vector<const ForObj *> &loops,
                                   int stages);
} // namespace tvm::tl

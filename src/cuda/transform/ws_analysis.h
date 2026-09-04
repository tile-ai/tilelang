/*!
 * \file ws_analysis.h
 * \brief Statement analysis shared by the warp-specialization passes.
 *
 * Extracted from producer_consumer_ws.cc so automatic schedulers reuse the
 * same classification the legacy producer/consumer rewriter has been
 * validated with.
 */
#pragma once

#include <tvm/target/target.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/stmt.h>

#include <optional>
#include <unordered_map>
#include <utility>

#include "layout/layout.h"
#include "op/copy.h"
#include "op/operator.h"

namespace tvm {
namespace tl {

using namespace tirx;

using BufferRemap =
    std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>;
using BufferLayoutMap = std::unordered_map<Var, std::pair<Buffer, Layout>,
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
bool IsBarrierOrTmaControlCall(const CallNode *call);

/// Classify a tile-op copy as TMA load producer, cp.async producer, TMA
/// store, or consumer using coarse pre-layout checks.
TileStmtKind ClassifyCopy(const CopyNode *copy, Target target);

/// Classify a single statement in a pipeline loop body.
TileStmtKind ClassifyStmt(const Stmt &stmt, Target target);

struct GemmInfo {
  Buffer accumulator;
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
bool IsTmaCompatibleLayout(const Layout &layout, const Buffer &buffer);

/// Merge one block's "layout_map" annotation into `layouts`, keyed by the
/// buffer's data var. Var-keyed entries are resolved against the block's
/// allocations.
void CollectAnnotatedLayouts(const SBlock &block, BufferLayoutMap &layouts);

/// Rewrite every use of a remapped buffer (loads, stores, region roots) to
/// the replacement, including bare uses of the old data vars.
Stmt RemapBuffers(Stmt stmt, const BufferRemap &remap);

} // namespace tl
} // namespace tvm

#pragma once

#include <tvm/target/target.h>
#include <tvm/tirx/stmt.h>

#include <optional>
#include <unordered_map>
#include <utility>

#include "layout/layout.h"
#include "op/copy.h"
#include "op/operator.h"

namespace tvm::tl {

using BufferRemap = std::unordered_map<tirx::TensorVar, tirx::TensorVar,
                                       ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using BufferLayoutMap =
    std::unordered_map<Var, std::pair<tirx::TensorVar, Layout>,
                       ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

enum class TileStmtKind {
  kTmaProducer,
  kCpAsyncProducer,
  kCpAsyncRaw,
  kSimtProducer,
  kTmaStore,
  kTcgen05Mma,
  kConsumer,
};

struct GemmInfo {
  tirx::TensorVar accumulator;
  int wg_wait;
};

ffi::Optional<Call> GetEvaluateCallInSimpleWrapper(const Stmt &statement);
bool IsBarrierOrTmaControlCall(const CallNode *call);
TileStmtKind ClassifyCopy(const CopyNode *copy, Target target);
TileStmtKind ClassifyStmt(const Stmt &statement, Target target);
std::optional<GemmInfo> GetGemmInfo(const TileOperator &operation);
bool IsProducer(TileStmtKind kind);
bool HasManualWarpSpecialization(const Stmt &statement);
bool IsTmaCompatibleLayout(const Layout &layout, const tirx::TensorVar &tensor);
void CollectAnnotatedLayouts(const RegionStmt &region,
                             BufferLayoutMap &layouts);
Stmt RemapBuffers(Stmt statement, const BufferRemap &remap);

} // namespace tvm::tl

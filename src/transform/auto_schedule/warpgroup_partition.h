#pragma once

#include <tvm/tir/buffer.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "./barrier.h"
#include "./ir_structure.h"

namespace tvm {
namespace tl {

using namespace tir;
using ffi::Map;

struct WarpSpecializeConfig;

class VarRefCollector : public StmtExprVisitor {
public:
  std::unordered_set<const VarNode *> vars;
  void VisitExpr_(const VarNode *op) override { vars.insert(op); }
};

bool IsLetDeclTask(const TaskNode *task);
bool IsLetDeclNode(const IRStructure *node);
bool ContainsLetDecl(const IRStructure *node);

std::shared_ptr<IRStructure>
CloneIRStructureWithWarpgroupFilter(IRStructure *node, int warpgroup_id,
                                    Map<Var, PrimExpr> &var_remap);
std::shared_ptr<IRStructure>
CloneIRStructureWithWarpgroupFilter(IRStructure *node, int warpgroup_id);

std::shared_ptr<IRStructure>
RemoveUnusedLetDecls(std::shared_ptr<IRStructure> root);

class SimtCopyDetector;

Stmt ConvertIRStructureToStmt(IRStructure *root, const bool outer_enable_epi);

Stmt ApplyWarpgroupPartitionToIRStructure(
    IRStructure *root, IterVar thread_var, std::vector<Buffer> &barrier_buffers,
    Map<ObjectRef, ObjectRef> &barrier_map, const bool enable_epi,
    PrimExpr thread_count[2], bool producer_consumer,
    const WarpSpecializeConfig &config, Buffer neutral_sync_shared_barrier);

Stmt ReNestLetStmts(const Stmt &stmt);

} // namespace tl
} // namespace tvm

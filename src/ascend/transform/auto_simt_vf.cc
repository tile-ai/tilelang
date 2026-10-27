/*!
 * \file auto_simt_vf.cc
 * \brief Form Ascend SIMT_VF regions around eligible Parallel computation.
 *
 * One planner follows M1 Analyze, M2 PlanRegions, M3 PlanStorage, and
 * M4 Rewrite/VerifyResult.
 *
 * Uniform serial/unrolled loops and branches join a structured region before
 * storage planning. Cross-invocation values use shared.dyn; existing VF is
 * opaque. Scalar state uses local.var, not local.fragment[1].
 */
#include "ascend/op/ascend_mte_plan.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "ascend/target_utils.h"
#include "op/copy.h"
#include "op/fill.h"
#include "op/reduce.h"
#include "op/reducer.h"
#include "op/utils.h"
#include "support/check.h"
#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tvm/ir/attrs.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/var.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {
tvm::transform::Pass Simplify(bool simplify_arguments);
namespace transform {
tvm::transform::Pass UnrollLoopSkipVF();
}
namespace ascend {
tvm::transform::Pass AscendLayoutInference();
}
using namespace tirx;
using namespace ffi;
namespace {

enum class Role { kSeed, kFusible, kBoundary };
enum class ErrorCode {
  kInputContract,
  kThreadEnvelope,
  kUnsupportedStatement,
  kUnsupportedCopy,
  kUnsupportedReduce,
  kRegionConflict,
  kPlanInvariant,
  kRewriteInvariant,
};

const char *ErrorCodeName(ErrorCode code) {
  static const char *kNames[] = {"input_contract",        "thread_envelope",
                                 "unsupported_statement", "unsupported_copy",
                                 "unsupported_reduce",    "region_conflict",
                                 "plan_invariant",        "rewrite_invariant"};
  return kNames[static_cast<int>(code)];
}

[[noreturn]] void Reject(ErrorCode code, const char *detail) {
  TVM_FFI_THROW(InternalError)
      << "[AutoSimtVF] " << ErrorCodeName(code) << ": " << detail;
}
struct CopyInfo {
  Buffer src, dst;
};

void AddUnique(std::vector<Buffer> *buffers, const Buffer &buffer) {
  auto same = [&](const Buffer &item) { return item.same_as(buffer); };
  if (std::none_of(buffers->begin(), buffers->end(), same))
    buffers->push_back(buffer);
}

struct UnitSemantics {
  Role role;
  std::optional<CopyInfo> copy;
  std::optional<Buffer> reduce_src_fragment;
  bool fragment_dst_full_def{
      false}; // This unit replaces the prior fragment value.
  UnitSemantics(Role role, std::optional<CopyInfo> copy = std::nullopt)
      : role(role), copy(std::move(copy)) {}
};

struct UnitAccess {
  std::vector<Buffer> reads;
  std::vector<Buffer> writes;
  bool Accesses(const Buffer &buffer) const {
    auto same = [&](const Buffer &item) {
      return item->data.same_as(buffer->data);
    };
    return std::any_of(reads.begin(), reads.end(), same) ||
           std::any_of(writes.begin(), writes.end(), same);
  }
  // Flow analysis matches data aliases; substitutions match exact Buffers.
  bool Writes(const Buffer &buffer) const {
    return std::any_of(writes.begin(), writes.end(), [&](const Buffer &item) {
      return item->data.same_as(buffer->data);
    });
  }
};

// No performance migration rule is published: candidates remain on the host.
enum class ScalarClass { kS1, kS2, kS3, kS4, kS5, kS6 };

struct ScalarDecision {
  std::vector<int> units;
  std::vector<int> local_tail_begins;
  ScalarClass category;
  std::string position;
  std::string reason;
  bool reads_left{false};
};

// Summarize the entire block, including initialization and loop-carried state.
// Tile operations, Parallel and existing VF are not host scalar candidates.
class ScalarFacts : public StmtExprVisitor {
public:
  bool scalar{true}, loop{false}, read_memory{false}, shared_write{false};
  bool unknown_effect{false};
  UnitAccess access;
  std::vector<Var> bindings;

  ScalarClass Classify(bool host_consumer) const {
    if (unknown_effect)
      return ScalarClass::kS6;
    if (host_consumer)
      return ScalarClass::kS5;
    if (shared_write)
      return ScalarClass::kS4;
    if (loop)
      return ScalarClass::kS3;
    return read_memory ? ScalarClass::kS2 : ScalarClass::kS1;
  }

private:
  void VisitStmt_(const ForNode *op) final {
    scalar &= op->kind == ForKind::kSerial || op->kind == ForKind::kUnrolled;
    loop = true;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const SBlockNode *op) final { scalar = false; }
  void VisitStmt_(const BindNode *op) final {
    bindings.push_back(op->var);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const CallNode *op) final {
    const auto callee = op->op.as<Op>();
    static const auto effects =
        Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
    if (callee && callee.value()->name.find("tl.tileop.") == 0)
      scalar = false;
    unknown_effect |= !callee || !effects.count(callee.value()) ||
                      effects[callee.value()]->value >
                          static_cast<int>(CallEffectKind::kPure);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    AddUnique(&access.reads, op->buffer);
    read_memory |= !IsLocalVarBuffer(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    AddUnique(&access.writes, op->buffer);
    shared_write |= !IsLocalVarBuffer(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }
};

// This is deliberately a closed policy, not an op/length heuristic or a user
// switch that can bypass the offline publication and legality requirements.
const char *SelectScalarPlacement(ScalarClass category) {
  switch (category) {
  case ScalarClass::kS4:
    return "shared_write";
  case ScalarClass::kS5:
    return "host_consumer";
  case ScalarClass::kS6:
    return "unknown_effect";
  default:
    return "no_published_rule";
  }
}

enum class AccessEffect { kRead, kFullDef, kUpdate };
struct AccessEvent {
  int unit{-1};
  AccessEffect effect{AccessEffect::kRead};
  bool Reads() const { return effect != AccessEffect::kFullDef; }
  bool Writes() const { return effect != AccessEffect::kRead; }
};

struct Unit : UnitSemantics {
  int id{-1}, scope{-1}, order{-1};
  Stmt stmt;
  UnitAccess access;
  bool host_scalar{false};
  std::vector<int> child_scopes;
  Unit(int id, int scope, int order, Stmt stmt, UnitSemantics semantics)
      : UnitSemantics(std::move(semantics)), id(id), scope(scope), order(order),
        stmt(std::move(stmt)) {}
};

enum class ChildSlot { kRoot, kLoopBody, kThen, kElse, kAttrBody };
struct Scope {
  int parent{-1};
  ChildSlot slot{ChildSlot::kRoot};
  std::vector<int> units; // units[order] directly yields the unit id.
};

struct BufferFlow {
  Buffer original;
  std::vector<AccessEvent> events;
  std::unordered_map<int, int> event_by_unit;
  std::vector<Buffer> aliases;
};

struct Region {
  int id{-1}, scope{-1}, begin{-1}, end{-1};
};
using BufferMap = Map<Buffer, Buffer>;
struct RegionAction {
  std::vector<int> relocated_copies; // Emitted before direct region members.
  std::vector<Buffer> allocations;   // Allocations owned by this VF.
  std::vector<Stmt> prologue;        // Shared-state reloads.
  std::vector<Stmt> epilogue;        // Shared-state writebacks.
};

struct RewritePlan {
  std::vector<Region> regions;
  std::vector<int> owners; // Includes descendants absorbed by a region.
  std::unordered_set<int> relocated;
  std::vector<RegionAction> region_actions;
  std::vector<BufferMap> substitutions;
  std::vector<Buffer> add_allocations, remove_allocations;
  PrimExpr tx_extent;
};

struct PlannerState {
  PrimFunc original;
  SBlock root_block;
  int root_scope{-1};
  std::vector<Unit> units;
  std::vector<Scope> scopes;
  std::vector<BufferFlow> fragment_flows;
  std::unordered_set<int64_t> occupied_source_indices;
  int original_vf_count{0};
  RewritePlan plan;
  std::vector<ScalarDecision> scalar_decisions;
};

bool IsVFBlock(const String &name_hint) {
  return name_hint == "SIMT_VF" || name_hint == "SIMD_VF";
}

bool IsScopedAttr(const String &key) {
  return key == "tl.ascend_stage" || key == "tl.assume" ||
         key == "tl.assume_requires_runtime_check" || key == "tl.ascend_task" ||
         key == "tl.ascend_per_core_task";
}

const SBlockNode *GetBlockNode(const Stmt &stmt) {
  if (const auto *block = stmt.as<SBlockNode>())
    return block;
  if (const auto *realize = stmt.as<SBlockRealizeNode>())
    return realize->block.get();
  return nullptr;
}

bool IsStaticPositiveInt(const PrimExpr &extent) {
  const auto *imm = extent.as<IntImmNode>();
  return imm != nullptr && imm->value > 0;
}

Array<PrimExpr> HeaderExprs(const Stmt &stmt) {
  if (const auto *loop = stmt.as<ForNode>()) {
    Array<PrimExpr> exprs{loop->min, loop->extent};
    if (loop->step.defined())
      exprs.push_back(loop->step.value());
    return exprs;
  }
  if (const auto *branch = stmt.as<IfThenElseNode>())
    return {branch->condition};
  if (const auto *bind = stmt.as<BindNode>())
    return {bind->value};
  return {};
}

std::vector<std::pair<ChildSlot, Stmt>> GetControlChildren(const Stmt &stmt) {
  std::vector<std::pair<ChildSlot, Stmt>> children;
  if (const auto *loop = stmt.as<ForNode>()) {
    children.emplace_back(ChildSlot::kLoopBody, loop->body);
  } else if (const auto *if_stmt = stmt.as<IfThenElseNode>()) {
    children.emplace_back(ChildSlot::kThen, if_stmt->then_case);
    if (if_stmt->else_case.defined())
      children.emplace_back(ChildSlot::kElse, if_stmt->else_case.value());
  } else if (const auto *attr = stmt.as<AttrStmtNode>()) {
    if (IsScopedAttr(attr->attr_key))
      children.emplace_back(ChildSlot::kAttrBody, attr->body);
  }
  return children;
}

Stmt ReplaceControlChild(const Stmt &owner, ChildSlot slot,
                         const Stmt &new_body) {
  if (const auto *loop = owner.as<ForNode>()) {
    ICHECK(slot == ChildSlot::kLoopBody);
    return For(loop->loop_var, loop->min, loop->extent, loop->kind, new_body,
               loop->thread_binding, loop->annotations, loop->step, loop->span);
  }
  if (const auto *if_stmt = owner.as<IfThenElseNode>()) {
    if (slot == ChildSlot::kThen)
      return IfThenElse(if_stmt->condition, new_body, if_stmt->else_case,
                        if_stmt->span);
    ICHECK(slot == ChildSlot::kElse);
    return IfThenElse(if_stmt->condition, if_stmt->then_case, new_body,
                      if_stmt->span);
  }
  const auto *attr = owner.as<AttrStmtNode>();
  ICHECK(attr && slot == ChildSlot::kAttrBody);
  return AttrStmt(attr->node, attr->attr_key, attr->value, new_body,
                  attr->span);
}

// Capture the thread envelope, root, and candidates; existing VFs stay opaque.
bool IsStage1Candidate(const Stmt &stmt);
class InputReader : public StmtExprVisitor {
public:
  PrimExpr tx_extent{IntImm(DataType::Int(32), 512)};
  std::vector<Var> thread_vars;
  Optional<Var> subcore_var;
  Optional<SBlockRealize> root;
  bool has_candidate{false}, outer_thread_var_used{false};

private:
  bool in_root_{false};
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>()) {
        if (iv->thread_tag == "cthread") {
          subcore_var = iv->var;
        } else if (iv->thread_tag == "threadIdx.x") {
          tx_extent = op->value;
          thread_vars.push_back(iv->var);
        } else if (iv->thread_tag == "threadIdx.y" ||
                   iv->thread_tag == "threadIdx.z") {
          thread_vars.push_back(iv->var);
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const SBlockNode *op) final {
    if (!IsVFBlock(op->name_hint))
      StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const SBlockRealizeNode *op) final {
    if (IsVFBlock(op->block->name_hint))
      return;
    bool is_root = op->block->name_hint == "tilelang_root";
    if (is_root)
      root = GetRef<SBlockRealize>(op);
    bool previous = in_root_;
    in_root_ |= is_root;
    StmtExprVisitor::VisitStmt_(op);
    in_root_ = previous;
  }
  void VisitStmt_(const ForNode *op) final {
    has_candidate |= op->kind == ForKind::kParallel;
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitStmt_(const EvaluateNode *op) final {
    has_candidate |= IsStage1Candidate(GetRef<Stmt>(op));
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const VarNode *op) final {
    if (in_root_) {
      Var var = GetRef<Var>(op);
      outer_thread_var_used |=
          std::any_of(thread_vars.begin(), thread_vars.end(),
                      [&](const Var &thread) { return var.same_as(thread); });
    }
  }
};

template <typename TileOp>
std::optional<TileOp> MatchTileOp(const Stmt &stmt, const char *name) {
  const auto *eval = stmt.as<EvaluateNode>();
  const auto *call = eval ? eval->value.as<CallNode>() : nullptr;
  if (call == nullptr)
    return std::nullopt;
  if (!call->op.same_as(Op::Get(name)))
    return std::nullopt;
  return TileOp(call->args, call->annotations);
}

std::optional<Copy> MatchTileCopy(const Stmt &stmt) {
  if (auto copy = MatchTileOp<AscendCopy>(stmt, "tl.tileop.ascend_copy"))
    return *copy;
  return MatchTileOp<Copy>(stmt, "tl.tileop.copy");
}

// Whole-buffer operations require static, zero-based ranges.
bool CoversWholeBuffer(const Array<Range> &ranges, const Buffer &buffer) {
  if (ranges.size() != buffer->shape.size())
    return false;
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto *min = ranges[i]->min.as<IntImmNode>();
    const auto *extent = ranges[i]->extent.as<IntImmNode>();
    const auto *dim = buffer->shape[i].as<IntImmNode>();
    if (min == nullptr || extent == nullptr || dim == nullptr)
      return false;
    if (min->value != 0 || extent->value != dim->value)
      return false;
  }
  return true;
}

bool IsFillSeed(const Fill &fill) {
  return CoversWholeBuffer(fill->region, fill->dst) && !IsL1Buffer(fill->dst) &&
         !IsLocalVarBuffer(fill->dst);
}

bool IsFragmentEndpointPair(const Buffer &src, const Buffer &dst) {
  return (IsFragmentBuffer(src) &&
          (IsSharedBuffer(dst) || IsGlobalBuffer(dst))) ||
         (IsFragmentBuffer(dst) &&
          (IsSharedBuffer(src) || IsGlobalBuffer(src)));
}

bool IsUbGmPair(const Buffer &src, const Buffer &dst) {
  return (IsSharedBuffer(src) && IsGlobalBuffer(dst)) ||
         (IsGlobalBuffer(src) && IsSharedBuffer(dst));
}

// MTE/Cube copy paths are hard region boundaries rather than SIMT work.
bool IsDedicatedEnginePair(const Buffer &src, const Buffer &dst) {
  return (IsGlobalBuffer(src) && IsL1Buffer(dst)) ||
         (IsL1Buffer(src) && (IsL0ABuffer(dst) || IsL0BBuffer(dst))) ||
         (IsL0CBuffer(src) && (IsSharedBuffer(dst) || IsGlobalBuffer(dst))) ||
         (IsSharedBuffer(src) && IsL1Buffer(dst));
}

// Kernels without Stage1 work retain their original Ascend IR.
bool IsStage1Candidate(const Stmt &stmt) {
  if (auto copy = MatchTileCopy(stmt)) {
    return IsFragmentEndpointPair((*copy)->src, (*copy)->dst) ||
           (IsUbGmPair((*copy)->src, (*copy)->dst) &&
            (*copy)->src->dtype != (*copy)->dst->dtype);
  }
  if (auto fill = MatchTileOp<Fill>(stmt, "tl.tileop.fill"))
    return IsFragmentBuffer((*fill)->dst) || IsFillSeed(*fill);
  return MatchTileOp<ReduceOp>(stmt, "tl.tileop.reduce").has_value();
}

// Fallback access summary for units without explicit semantics.
class BufferReadWriteCollector : public StmtExprVisitor {
public:
  UnitAccess summary;
  void VisitExpr_(const CallNode *op) final {
    static const Op &region_op = Op::Get("tl.region");
    const auto *load =
        op->args.empty() ? nullptr : op->args[0].as<BufferLoadNode>();
    const auto *mask =
        op->args.size() < 2 ? nullptr : op->args[1].as<IntImmNode>();
    if (op->op.same_as(region_op) && load != nullptr && mask != nullptr) {
      if (mask->value & 1)
        AddUnique(&summary.reads, load->buffer);
      if (mask->value & 2)
        AddUnique(&summary.writes, load->buffer);
      const BufferLoadNode *saved = region_carrier_;
      region_carrier_ = load;
      StmtExprVisitor::VisitExpr_(op);
      region_carrier_ = saved;
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    if (op != region_carrier_)
      AddUnique(&summary.reads, op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    AddUnique(&summary.writes, op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  const BufferLoadNode *region_carrier_{nullptr};
};

// A dominating element definition permits private storage, including local.var.
// Require identical indices at every access: a neighbour read needs
// communication.
bool IsParallelDefinition(Stmt stmt, const Buffer &buffer) {
  Array<PrimExpr> indices;
  while (const auto *loop = stmt.as<ForNode>()) {
    if (loop->kind != ForKind::kParallel)
      return false;
    indices.push_back(loop->loop_var);
    if (!IsLocalVarBuffer(buffer) &&
        (indices.size() > buffer->shape.size() || !is_zero(loop->min) ||
         !StructuralEqual()(loop->extent, buffer->shape[indices.size() - 1])))
      return false;
    stmt = loop->body;
  }
  if (IsLocalVarBuffer(buffer))
    indices = {Integer(0)};
  if (indices.size() != buffer->shape.size())
    return false;
  Stmt body = stmt;
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    for (const Stmt &child : seq->seq) {
      BufferReadWriteCollector access;
      access(child);
      if (access.summary.Accesses(buffer)) {
        stmt = child;
        break;
      }
    }
  }
  const auto *def = stmt.as<BufferStoreNode>();
  if (!def || !def->buffer.same_as(buffer) || def->predicate.defined() ||
      !StructuralEqual()(indices, def->indices))
    return false;
  BufferReadWriteCollector inputs;
  inputs(def->value);
  bool valid = !inputs.summary.Accesses(buffer);
  PostOrderVisit(body, [&](const ObjectRef &obj) {
    if (const auto *load = obj.as<BufferLoadNode>()) {
      if (load->buffer->data.same_as(buffer->data))
        valid &= load->buffer.same_as(buffer) &&
                 StructuralEqual()(load->indices, indices);
    } else if (const auto *store = obj.as<BufferStoreNode>()) {
      if (store->buffer->data.same_as(buffer->data))
        valid &= store->buffer.same_as(buffer) &&
                 StructuralEqual()(store->indices, indices);
    }
  });
  return valid;
}

// Headers may use captured host values and uniform induction variables, but do
// not re-evaluate mutable memory or unknown calls independently in every lane.
bool IsUniformHeader(const Stmt &stmt, const Optional<Var> &subcore_var) {
  bool uniform = true;
  for (const PrimExpr &expr : HeaderExprs(stmt))
    PostOrderVisit(expr, [&](const ObjectRef &obj) {
      // Core participation stays visible to AssignCore. Projecting a false
      // subcore branch inside a VF would leave an empty, invalid VF block.
      uniform &= !obj.as<BufferLoadNode>() && !obj.as<CallNode>() &&
                 (!subcore_var.defined() || !obj.same_as(subcore_var.value()));
    });
  if (const auto *loop = stmt.as<ForNode>()) {
    return uniform &&
           (loop->kind == ForKind::kSerial ||
            loop->kind == ForKind::kUnrolled) &&
           loop->annotations.empty();
  }
  return stmt.as<IfThenElseNode>() && uniform;
}

// Do not lift a loop whose Parallel hides an engine op or opaque side effect.
// Pure math calls remain eligible even when their operands load GM/UB values.
bool HasOnlyPureCalls(const Stmt &stmt) {
  static const auto effects =
      Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
  bool safe = true;
  PostOrderVisit(stmt, [&](const ObjectRef &obj) {
    if (const auto *call = obj.as<CallNode>()) {
      auto op = call->op.as<Op>();
      safe &=
          op && effects.count(op.value()) &&
          effects[op.value()]->value <= static_cast<int>(CallEffectKind::kPure);
    }
  });
  return safe;
}

bool HeaderReadsFragment(const Stmt &stmt) {
  BufferReadWriteCollector reads;
  for (const PrimExpr &expr : HeaderExprs(stmt))
    reads(expr);
  return std::any_of(reads.summary.reads.begin(), reads.summary.reads.end(),
                     IsFragmentBuffer);
}

// M3 helpers for buffer cloning and synthetic copies.
Buffer CloneBufferWithScope(const Buffer &original, const String &name,
                            const String &scope) {
  Buffer result = original;
  result.CopyOnWrite()->data =
      Var(name, PointerType(PrimType(original->dtype), scope), original->span);
  result.CopyOnWrite()->name = name;
  return result;
}

PrimExpr MakeRegionCall(const Buffer &buffer, int64_t access_mask) {
  static const Op &region_op = Op::Get("tl.region");
  std::vector<PrimExpr> indices;
  Array<PrimExpr> args;
  for (const PrimExpr &dim : buffer->shape)
    indices.push_back(make_zero(dim.dtype()));
  args.push_back(BufferLoad(buffer, indices));
  args.push_back(IntImm(DataType::Int(32), access_mask));
  for (const PrimExpr &dim : buffer->shape)
    args.push_back(dim);
  return Call(DataType::Handle(), region_op, args);
}

Stmt MakeCopyStmt(const Buffer &src, const Buffer &dst) {
  static const Op &copy_op = Op::Get("tl.tileop.copy");
  return Evaluate(Call(DataType::Handle(), copy_op,
                       {MakeRegionCall(src, 1), MakeRegionCall(dst, 2)},
                       Map<String, ObjectRef>{}));
}

// Match SimtVFFrame structure while retaining the outer thread envelope.
Stmt BuildSimtVF(Stmt body, const PrimExpr &tx_extent, int64_t source_index,
                 Span span, const Array<Buffer> &allocations) {
  Stmt inner = AttrStmt(StringImm("simtvf"), "tl.simtvf_scope",
                        IntImm(DataType::Int(32), 1), std::move(body));
  const char *names[] = {"simtvf_tx", "simtvf_ty", "simtvf_tz"};
  const char *tags[] = {"threadIdx.x", "threadIdx.y", "threadIdx.z"};
  DataType dtype = tx_extent.dtype();
  for (int axis = 2; axis >= 0; --axis) {
    PrimExpr extent = axis == 0 ? tx_extent : IntImm(dtype, 1);
    IterVar iv(Range::FromMinExtent(make_zero(dtype), extent),
               Var(names[axis], dtype), IterVarType::kThreadIndex, tags[axis]);
    inner = AttrStmt(iv, tirx::attr::thread_extent, extent, std::move(inner));
  }
  Map<String, Any> annotations;
  annotations.Set("tl.vf_source_index",
                  IntImm(DataType::Int(64), source_index));
  SBlock block({}, {}, {}, "SIMT_VF", std::move(inner), std::nullopt,
               allocations, {}, annotations, span);
  return block;
}

// Applies buffer substitutions to loads and stores during the M4 IR rewrite.
class LocalBufferRewriter : public StmtExprMutator {
public:
  explicit LocalBufferRewriter(const BufferMap &substitutions)
      : substitutions_(substitutions) {}
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (auto to = substitutions_.Get(load->buffer))
      load.CopyOnWrite()->buffer = *to;
    return load;
  }
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (auto to = substitutions_.Get(store->buffer))
      store.CopyOnWrite()->buffer = *to;
    return store;
  }

private:
  const BufferMap &substitutions_;
};

// Auto Parallel stores describe logical iterations. Mark only stores whose
// address, value and control are replica-independent. Fragment stores remain
// unmarked so every private replica is initialized. Explicit VFs stay opaque.
class LogicalStoreMarker : public StmtExprMutator {
public:
  explicit LogicalStoreMarker(const PlannerState &state) {
    const PrimFunc &func = state.original;
    for (const Var &param : func->params)
      uniform_.insert(param.get());
    for (const auto &[param, buffer] : func->buffer_map)
      uniform_.insert(buffer->data.get());
    for (const Buffer &buffer : state.plan.add_allocations)
      uniform_.insert(buffer->data.get());
    std::vector<const ForNode *> loops;
    std::vector<const BindNode *> binds;
    std::unordered_map<const VarNode *, std::vector<Copy>> input_copies;
    std::unordered_set<const VarNode *> modified_fragments;
    PostOrderVisit(func->body, [&](const ObjectRef &obj) {
      auto escape = [&](const PrimExpr &expr) {
        if (!expr.dtype().is_handle())
          return;
        PostOrderVisit(expr, [&](const ObjectRef &value) {
          if (const auto *var = value.as<VarNode>())
            modified_fragments.insert(var);
          if (const auto *load = value.as<BufferLoadNode>())
            modified_fragments.insert(load->buffer->data.get());
        });
      };
      if (const auto *call = obj.as<CallNode>()) {
        auto op = call->op.as<Op>();
        std::string name = op ? std::string(op.value()->name) : "";
        if (name != "tl.region" && name.rfind("tl.tileop.", 0) != 0)
          for (const PrimExpr &arg : call->args)
            escape(arg);
      }
      if (const auto *bind = obj.as<BindNode>())
        escape(bind->value);
      if (const auto *block = obj.as<SBlockNode>()) {
        for (const Buffer &buffer : block->alloc_buffers) {
          if (IsGlobalBuffer(buffer) || IsSharedBuffer(buffer))
            uniform_.insert(buffer->data.get());
        }
      } else if (const auto *attr = obj.as<AttrStmtNode>()) {
        const auto *iv = attr->node.as<IterVarNode>();
        if (iv &&
            (iv->thread_tag == "cthread" || iv->thread_tag == "blockIdx.x" ||
             iv->thread_tag == "blockIdx.y" || iv->thread_tag == "blockIdx.z"))
          uniform_.insert(iv->var.get());
      } else if (const auto *loop = obj.as<ForNode>()) {
        if (loop->kind != ForKind::kThreadBinding)
          loops.push_back(loop);
      } else if (const auto *bind = obj.as<BindNode>()) {
        binds.push_back(bind);
      }
    });
    // M1 already summarized every write, including aliases and opaque VFs.
    // Its copy contract guarantees a whole fragment destination.
    for (const BufferFlow &flow : state.fragment_flows)
      for (const AccessEvent &event : flow.events) {
        if (!event.Writes())
          continue;
        const Unit &unit = state.units[event.unit];
        if (unit.copy &&
            (IsGlobalBuffer(unit.copy->src) || IsSharedBuffer(unit.copy->src)))
          input_copies[flow.original->data.get()].push_back(
              *MatchTileCopy(unit.stmt));
        else
          modified_fragments.insert(flow.original->data.get());
      }
    // SSA aliases and nested loop bounds may depend on earlier bindings.
    auto infer_bindings = [&] {
      size_t previous;
      do {
        previous = uniform_.size();
        for (const auto *loop : loops)
          if (Safe(loop->min) && Safe(loop->extent) &&
              (!loop->step.defined() || Safe(loop->step.value())))
            uniform_.insert(loop->loop_var.get());
        for (const auto *bind : binds)
          if (Safe(bind->value))
            uniform_.insert(bind->var.get());
      } while (uniform_.size() != previous);
    };
    infer_bindings();
    for (const auto &[data, copies] : input_copies) {
      bool safe = !modified_fragments.count(data);
      for (const Copy &copy : copies) {
        safe &= SafeAddress(copy->src) && SafeAddress(copy->dst, true);
        for (const auto &ranges : {copy->src_range, copy->dst_range})
          for (const Range &range : ranges)
            safe &= Safe(range->min) && Safe(range->extent);
      }
      if (safe)
        replica_inputs_.insert(data);
    }
    infer_bindings();
  }

private:
  bool Safe(const PrimExpr &expr) const {
    bool safe = true;
    PostOrderVisit(expr, [&](const ObjectRef &obj) {
      if (const auto *var = obj.as<VarNode>())
        safe &= uniform_.count(var) != 0;
      // Purity alone does not exclude lane/shuffle intrinsics. Calls and
      // unproven private loads deliberately do not establish uniformity.
      if (obj.as<CallNode>())
        safe = false;
      if (const auto *load = obj.as<BufferLoadNode>()) {
        safe &= IsSharedBuffer(load->buffer) || IsGlobalBuffer(load->buffer) ||
                replica_inputs_.count(load->buffer->data.get());
        safe &= SafeAddress(load->buffer);
      }
    });
    return safe;
  }

  bool SafeAddress(const Buffer &buffer, bool copy_dst = false) const {
    const auto *data = buffer->data.get();
    if ((!uniform_.count(data) && !replica_inputs_.count(data) && !copy_dst) ||
        !Safe(buffer->elem_offset))
      return false;
    for (const PrimExpr &dim : buffer->shape)
      if (!Safe(dim))
        return false;
    for (const PrimExpr &stride : buffer->strides)
      if (!Safe(stride))
        return false;
    return true;
  }

  Stmt VisitStmt(const Stmt &stmt) final {
    // Unknown control constructs do not establish replica-independent paths.
    if (!stmt.as<SeqStmtNode>() && !stmt.as<ForNode>() &&
        !stmt.as<IfThenElseNode>() && !stmt.as<AttrStmtNode>() &&
        !stmt.as<SBlockNode>() && !stmt.as<SBlockRealizeNode>() &&
        !stmt.as<BufferStoreNode>() && !stmt.as<BindNode>() &&
        !stmt.as<EvaluateNode>())
      return stmt;
    return StmtExprMutator::VisitStmt(stmt);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kThreadBinding || !Safe(op->min) ||
        !Safe(op->extent) || (op->step.defined() && !Safe(op->step.value())))
      return GetRef<Stmt>(op);
    parallel_depth_ += op->kind == ForKind::kParallel;
    Stmt result = StmtExprMutator::VisitStmt_(op);
    parallel_depth_ -= op->kind == ForKind::kParallel;
    return result;
  }
  Stmt VisitStmt_(const IfThenElseNode *op) final {
    return Safe(op->condition) ? StmtExprMutator::VisitStmt_(op)
                               : GetRef<Stmt>(op);
  }
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (!IsScopedAttr(op->attr_key) || !Safe(op->value))
      return GetRef<Stmt>(op);
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const SBlockNode *op) final {
    return IsVFBlock(op->name_hint) || !op->annotations.empty()
               ? GetRef<Stmt>(op)
               : StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const SBlockRealizeNode *op) final {
    if (!Safe(op->predicate))
      return GetRef<Stmt>(op);
    for (const PrimExpr &value : op->iter_values)
      if (!Safe(value))
        return GetRef<Stmt>(op);
    return StmtExprMutator::VisitStmt_(op);
  }
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Stmt stmt = GetRef<Stmt>(op);
    if (!parallel_depth_ ||
        (!IsGlobalBuffer(op->buffer) && !IsSharedBuffer(op->buffer)) ||
        !Safe(op->value) ||
        (op->predicate.defined() && !Safe(op->predicate.value())))
      return stmt;
    for (const PrimExpr &index : op->indices)
      if (!Safe(index))
        return stmt;
    if (!SafeAddress(op->buffer))
      return stmt;
    return AttrStmt(op->buffer->data, attr::kParallelMultiplicity,
                    IntImm(DataType::Int(32), 1), stmt);
  }
  std::unordered_set<const VarNode *> uniform_;
  std::unordered_set<const VarNode *> replica_inputs_;
  int parallel_depth_{0};
};

// Validate a temporary candidate through existing pass interfaces. Only a
// solver with no legal layout triggers replanning; other diagnostics propagate.
bool CanInferStructuredLayout(PrimFunc func) {
  try {
    With<Target> target_scope(
        func->GetAttr<Target>(tvm::attr::kTarget).value());
    IRModule mod(Map<GlobalVar, BaseFunc>{{GlobalVar("main"), func}});
    mod = transform::UnrollLoopSkipVF()(mod);
    mod = tl::Simplify(true)(mod);
    ascend::AscendLayoutInference()(mod);
    return true;
  } catch (const Error &error) {
    if (error.message().find("no available layout found") == std::string::npos)
      throw;
    return false;
  }
}

class AutoSimtVFPlanner {
public:
  explicit AutoSimtVFPlanner(PrimFunc func, bool fuse_serial)
      : fuse_serial_(fuse_serial) {
    state_.original = std::move(func);
  }

  // M1-M3 only build a plan; M4 rewrites after verification, so failures never
  // expose a partially transformed function.
  PrimFunc Run(const InputReader &reader) {
    Analyze(reader);
    PlanRegions();
    PlanStorage();
    PlanPrivateScalars();
    PrimFunc result = Rewrite();
    VerifyResult(result);
    return result;
  }

  bool fused_control() const { return fused_control_; }

  Array<Map<String, Any>> ScalarReport() {
    PlanScalarScope(state_.root_scope);
    Array<Map<String, Any>> report;
    for (const ScalarDecision &decision : state_.scalar_decisions) {
      Map<String, Any> row;
      Array<Integer> units;
      for (int unit : decision.units)
        units.push_back(Integer(unit));
      row.Set("units", units);
      row.Set("category",
              String("S" +
                     std::to_string(1 + static_cast<int>(decision.category))));
      row.Set("position", String(decision.position));
      row.Set("strategy", String("split"));
      row.Set("reason", String(decision.reason));
      row.Set("reads_left", Bool(decision.reads_left));
      row.Set("whole_candidate", Bool(decision.category <= ScalarClass::kS3 &&
                                      decision.position != "isolated"));
      row.Set("local_candidate", String("record_only"));
      Array<Integer> cuts;
      for (int unit : decision.local_tail_begins)
        cuts.push_back(Integer(unit));
      row.Set("local_tail_begin_units", cuts);
      report.push_back(row);
    }
    return report;
  }

private:
  const Unit &U(int id) const { return state_.units[id]; }
  const Scope &S(int id) const { return state_.scopes[id]; }
  bool IsRelocated(int unit_id) const {
    int region = plan.owners[unit_id];
    return region >= 0 && plan.relocated.count(unit_id);
  }

  // ---- M1: Statement analysis ----
  // Preserve the statement tree and leaf effects. Uniform control can be
  // absorbed as a whole; host cuts and existing VFs keep their child scopes.
  void Analyze(const InputReader &reader) {
    subcore_var_ = reader.subcore_var;
    if (!IsStaticPositiveInt(reader.tx_extent)) {
      Reject(ErrorCode::kThreadEnvelope,
             "threadIdx.x extent is not a positive static integer");
    }
    plan.tx_extent = reader.tx_extent;
    Optional<SBlockRealize> root = reader.root;
    if (!root.defined()) {
      Reject(ErrorCode::kInputContract, "tilelang_root block not found");
    }
    state_.root_block = root.value()->block;

    if (reader.outer_thread_var_used)
      Reject(ErrorCode::kInputContract,
             "outer thread variable is outside AutoSimtVF Stage1 "
             "input contract (rewrite the thread-id guard as an "
             "unconditional MainScalar statement)");

    state_.root_scope = AddScope(/*parent=*/-1, ChildSlot::kRoot);
    for (const Buffer &buffer : state_.root_block->alloc_buffers) {
      if (!IsFragmentBuffer(buffer))
        continue;
      if (buffer->shape.size() == 1) {
        const auto *dim = buffer->shape[0].as<IntImmNode>();
        if (dim != nullptr && dim->value == 1) {
          Reject(
              ErrorCode::kInputContract,
              "local.fragment[1] scalar storage is unsupported; use alloc_var");
        }
      }
      state_.fragment_flows.push_back(BufferFlow{buffer});
    }
    VisitScope(state_.root_block->body, state_.root_scope);
    // Freeze host scalar boundaries before region and storage planning. A later
    // vector-control decision must not silently absorb a scalar-only block.
    for (Unit &unit : state_.units) {
      ScalarFacts facts;
      facts(unit.stmt);
      unit.host_scalar = facts.scalar;
    }
    state_.fragment_flows.erase(std::remove_if(state_.fragment_flows.begin(),
                                               state_.fragment_flows.end(),
                                               [](const BufferFlow &flow) {
                                                 return flow.events.empty();
                                               }),
                                state_.fragment_flows.end());
  }

  int AddScope(int parent, ChildSlot slot) {
    int id = static_cast<int>(state_.scopes.size());
    state_.scopes.push_back(Scope{parent, slot, {}});
    return id;
  }

  int AddUnit(Stmt stmt, int scope_id, int order, UnitSemantics semantics) {
    int id = static_cast<int>(state_.units.size());
    state_.units.emplace_back(id, scope_id, order, std::move(stmt),
                              std::move(semantics));
    state_.scopes[scope_id].units.push_back(id);
    return id;
  }

  // Dynamic engine ranges are allowed; fragment endpoints must be whole.
  UnitSemantics ClassifyCopy(const Copy &copy) {
    const Buffer &src = copy->src;
    const Buffer &dst = copy->dst;
    if (IsDedicatedEnginePair(src, dst)) {
      return UnitSemantics(Role::kBoundary, CopyInfo{src, dst});
    }
    if (IsFragmentEndpointPair(src, dst)) {
      Buffer fragment = IsFragmentBuffer(src) ? src : dst;
      auto ranges = IsFragmentBuffer(src) ? copy->src_range : copy->dst_range;
      if (!CoversWholeBuffer(ranges, fragment)) {
        Reject(ErrorCode::kUnsupportedCopy,
               "partial fragment copy is unsupported; the fragment "
               "endpoint must cover the whole allocation");
      }
      return UnitSemantics(Role::kFusible, CopyInfo{src, dst});
    }
    if (IsUbGmPair(src, dst)) {
      return UnitSemantics(src->dtype == dst->dtype ? Role::kBoundary
                                                    : Role::kSeed,
                           CopyInfo{src, dst});
    }
    Reject(ErrorCode::kUnsupportedCopy,
           "unsupported copy scope/dtype combination");
  }

  UnitSemantics ClassifyLeaf(const Stmt &stmt) {
    if (auto copy = MatchTileCopy(stmt))
      return ClassifyCopy(*copy);
    if (auto fill = MatchTileOp<Fill>(stmt, "tl.tileop.fill")) {
      bool full_def = CoversWholeBuffer((*fill)->region, (*fill)->dst);
      if (!full_def && IsFragmentBuffer((*fill)->dst)) {
        Reject(ErrorCode::kUnsupportedStatement,
               "fill must be static and cover the whole buffer");
      }
      bool seed = IsFillSeed(*fill);
      if (seed && !IsFragmentBuffer((*fill)->dst)) {
        // Preserve ordinary fill lowering when VF partitioning would
        // require a padded, non-bijective physical layout.
        int64_t elements = 1;
        for (const Range &range : (*fill)->region)
          elements *= range->extent.as<IntImmNode>()->value;
        int64_t threads = plan.tx_extent.as<IntImmNode>()->value;
        seed = elements > 0 &&
               (elements % threads == 0 || threads % elements == 0);
      }
      UnitSemantics leaf(seed ? Role::kSeed : Role::kBoundary);
      leaf.fragment_dst_full_def = full_def;
      return leaf;
    }
    if (auto reduce = MatchTileOp<ReduceOp>(stmt, "tl.tileop.reduce")) {
      const ReduceOp &reduce_ref = *reduce;
      if (reduce_ref->src->dtype != DataType::Float(32) ||
          (!reduce_ref->type->IsSum() && !reduce_ref->type->IsMax()) ||
          !IsFragmentBuffer(reduce_ref->src) ||
          !IsFragmentBuffer(reduce_ref->dst) ||
          !CoversWholeBuffer(reduce_ref->srcRegion_->region, reduce_ref->src) ||
          !CoversWholeBuffer(reduce_ref->dstRegion_->region, reduce_ref->dst)) {
        Reject(ErrorCode::kUnsupportedReduce,
               "reduce must be static float32 sum/max on whole fragments");
      }
      UnitSemantics leaf(Role::kFusible);
      leaf.reduce_src_fragment = reduce_ref->src;
      leaf.fragment_dst_full_def = reduce_ref->clear;
      return leaf;
    }
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (const auto *call = eval->value.as<CallNode>()) {
        if (const auto *op = call->op.as<OpNode>();
            op != nullptr && op->name.find("tl.tileop.") == 0) {
          if (op->name == "tl.tileop.gemm")
            return UnitSemantics(Role::kBoundary);
          Reject(ErrorCode::kUnsupportedStatement,
                 "tile-op call is outside the Stage1 whitelist");
        }
      }
      return UnitSemantics(Role::kBoundary);
    }
    if (stmt.as<BindNode>()) {
      if (HeaderReadsFragment(stmt))
        Reject(ErrorCode::kUnsupportedStatement,
               "fragment load in a scalar binding is unsupported");
      return UnitSemantics(Role::kBoundary);
    }
    if (stmt.as<BufferStoreNode>())
      return UnitSemantics(Role::kBoundary);
    Reject(ErrorCode::kUnsupportedStatement,
           "statement is outside the Stage1 whitelist");
  }

  void VisitScope(const Stmt &body, int scope_id) {
    std::vector<Stmt> children{body};
    if (const auto *seq = body.as<SeqStmtNode>())
      children.assign(seq->seq.begin(), seq->seq.end());
    for (int order = 0; order < static_cast<int>(children.size()); ++order) {
      const Stmt &stmt = children[order];
      const auto *block = GetBlockNode(stmt);
      bool existing_vf = block != nullptr && IsVFBlock(block->name_hint);
      const auto *loop = stmt.as<ForNode>();
      bool parallel = loop != nullptr && loop->kind == ForKind::kParallel;
      const auto *attr = stmt.as<AttrStmtNode>();
      bool scoped_attr = attr != nullptr && IsScopedAttr(attr->attr_key);
      bool control =
          !existing_vf && !parallel &&
          (loop != nullptr || stmt.as<IfThenElseNode>() || scoped_attr);
      if (control && HeaderReadsFragment(stmt))
        Reject(ErrorCode::kInputContract,
               "fragment load in a control header is unsupported");
      UnitSemantics semantics = control || existing_vf
                                    ? UnitSemantics(Role::kBoundary)
                                : parallel ? UnitSemantics(Role::kSeed)
                                           : ClassifyLeaf(stmt);
      int id = AddUnit(stmt, scope_id, order, std::move(semantics));
      if (control) {
        for (auto [slot, child_body] : GetControlChildren(stmt)) {
          int child = AddScope(scope_id, slot);
          state_.units[id].child_scopes.push_back(child);
          VisitScope(child_body, child);
          for (int unit_id : S(child).units) {
            for (const Buffer &buffer : U(unit_id).access.reads)
              AddUnique(&state_.units[id].access.reads, buffer);
            for (const Buffer &buffer : U(unit_id).access.writes)
              AddUnique(&state_.units[id].access.writes, buffer);
          }
        }
        bool fusible = fuse_serial_ && IsUniformHeader(stmt, subcore_var_);
        bool has_work = false;
        for (int child : U(id).child_scopes)
          for (int member : S(child).units) {
            fusible &= U(member).role != Role::kBoundary;
            const auto *nested = U(member).stmt.as<ForNode>();
            if (nested && nested->kind == ForKind::kParallel)
              fusible &= HasOnlyPureCalls(U(member).stmt);
            has_work = true;
          }
        if (fusible && has_work) {
          state_.units[id].role = Role::kSeed;
          fused_control_ = true;
        }
        continue;
      }
      SummarizeUnitAccess(id);

      if (existing_vf) {
        ++state_.original_vf_count;
        auto it = block->annotations.find("tl.vf_source_index");
        if (it != block->annotations.end()) {
          if (const auto *imm = (*it).second.as<IntImmNode>()) {
            state_.occupied_source_indices.insert(imm->value);
          }
        }
      }
    }
  }

  void SummarizeUnitAccess(int unit_id) {
    Unit &unit = state_.units[unit_id];
    BufferReadWriteCollector collector;
    collector(unit.stmt);
    unit.access = std::move(collector.summary);
    RecordFragmentEvents(unit);
  }

  void RecordFragmentEvents(const Unit &unit) {
    for (BufferFlow &flow : state_.fragment_flows) {
      auto find_access =
          [&](const std::vector<Buffer> &buffers) -> const Buffer * {
        const Buffer *first = nullptr;
        for (const Buffer &buffer : buffers) {
          if (buffer->data.same_as(flow.original->data)) {
            if (!buffer.same_as(flow.original))
              AddUnique(&flow.aliases, buffer);
            if (first == nullptr)
              first = &buffer;
          }
        }
        return first;
      };
      const Buffer *read = find_access(unit.access.reads);
      const Buffer *write = find_access(unit.access.writes);
      if (!write && !read)
        continue;
      int index = static_cast<int>(flow.events.size());
      AccessEffect effect = AccessEffect::kRead;
      if (write != nullptr) {
        bool full_def = read == nullptr &&
                        ((unit.copy && IsFragmentBuffer(unit.copy->dst)) ||
                         unit.fragment_dst_full_def);
        const auto *loop = unit.stmt.as<ForNode>();
        full_def |= loop && loop->kind == ForKind::kParallel &&
                    IsParallelDefinition(unit.stmt, *write);
        // A view may cover only part of its root storage.
        full_def &= write->same_as(flow.original);
        effect = full_def ? AccessEffect::kFullDef : AccessEffect::kUpdate;
      }
      flow.events.push_back(AccessEvent{unit.id, effect});
      flow.event_by_unit.emplace(unit.id, index);
    }
  }

  // ---- M2: Structured region planning ----
  // A region owns its direct interval AND every absorbed control descendant.
  // Storage is deliberately planned only after this partition is final.
  std::optional<int> EventIndex(const BufferFlow &flow, int unit_id) const {
    auto it = flow.event_by_unit.find(unit_id);
    return it == flow.event_by_unit.end() ? std::nullopt
                                          : std::optional<int>(it->second);
  }

  void OwnSubtree(int id, int region) {
    ICHECK_EQ(plan.owners[id], -1) << "operation has two VF owners";
    plan.owners[id] = region;
    for (int child : U(id).child_scopes)
      for (int member : S(child).units)
        OwnSubtree(member, region);
  }

  void PlanScope(int scope_id) {
    const Scope &scope = S(scope_id);
    int first = -1;
    for (int order = 0; order <= static_cast<int>(scope.units.size());
         ++order) {
      bool boundary = order == static_cast<int>(scope.units.size()) ||
                      U(scope.units[order]).role == Role::kBoundary ||
                      U(scope.units[order]).host_scalar;
      if (!boundary) {
        if (first < 0)
          first = order;
        continue;
      }
      if (first >= 0) {
        int region = static_cast<int>(plan.regions.size());
        plan.regions.push_back({region, scope_id, first, order - 1});
        plan.region_actions.emplace_back();
        for (int i = first; i < order; ++i)
          OwnSubtree(scope.units[i], region);
        first = -1;
      }
      if (order < static_cast<int>(scope.units.size()))
        for (int child : U(scope.units[order]).child_scopes)
          PlanScope(child);
    }
  }

  // Close read-only fragment inputs across unrelated host operations. Moving
  // only within the same scope preserves zero-trip paths and copy multiplicity.
  // Unlike the Stage1 ancestor-loop special case, no copy is duplicated per VF.
  void CloseDefiners() {
    for (const BufferFlow &flow : state_.fragment_flows) {
      if (!flow.aliases.empty() || flow.events.empty())
        continue;
      const Unit &copy = U(flow.events.front().unit);
      if (!copy.copy || !IsFragmentBuffer(copy.copy->dst) ||
          flow.events.front().effect != AccessEffect::kFullDef)
        continue;
      int old = plan.owners[copy.id], target = -1;
      if (old < 0 || plan.regions[old].begin != plan.regions[old].end)
        continue;
      bool safe = true;
      for (size_t i = 1; i < flow.events.size(); ++i) {
        const AccessEvent &event = flow.events[i];
        int owner = plan.owners[event.unit];
        safe &= !event.Writes() && owner >= 0 && owner != old &&
                (target < 0 || owner == target);
        target = owner;
      }
      if (!safe || target < 0)
        continue;
      const Region &region = plan.regions[target];
      if (copy.scope != region.scope || copy.order >= region.begin)
        continue;
      for (int i = copy.order + 1; i < region.begin; ++i) {
        const Unit &crossed = U(S(copy.scope).units[i]);
        // Only known copy operations may be crossed; unknown effects are cuts.
        safe &= crossed.copy.has_value();
        for (const Buffer &read : copy.access.reads)
          safe &= !crossed.access.Writes(read);
        if (!state_.original->HasNonzeroAttr(tirx::attr::kNoAlias))
          for (const Buffer &write : crossed.access.writes)
            safe &= !IsGlobalBuffer(write);
      }
      if (safe) {
        plan.owners[copy.id] = target;
        plan.relocated.insert(copy.id);
        plan.region_actions[target].relocated_copies.push_back(copy.id);
      }
    }
    // The old singleton regions are now empty; compact once before M3.
    std::vector<int> remap(plan.regions.size(), -1);
    std::vector<Region> regions;
    std::vector<RegionAction> actions;
    for (Region region : plan.regions) {
      if (region.begin == region.end &&
          IsRelocated(S(region.scope).units[region.begin]))
        continue;
      remap[region.id] = static_cast<int>(regions.size());
      actions.push_back(std::move(plan.region_actions[region.id]));
      region.id = remap[region.id];
      regions.push_back(region);
    }
    for (int &owner : plan.owners)
      if (owner >= 0)
        owner = remap[owner];
    plan.regions = std::move(regions);
    plan.region_actions = std::move(actions);
  }

  void PlanRegions() {
    plan.owners.assign(state_.units.size(), -1);
    PlanScope(state_.root_scope);
    CloseDefiners();
    VerifyRegions();
  }

  // A host scalar block never joins a vector-control region implicitly. Group
  // adjacent scalar statements so initialization and recurrences stay together.
  void CollectSubtree(int id, std::unordered_set<int> *ids) const {
    ids->insert(id);
    for (int child : U(id).child_scopes)
      for (int member : S(child).units)
        CollectSubtree(member, ids);
  }

  bool HasHostConsumer(const ScalarFacts &facts,
                       const std::unordered_set<int> &members) const {
    for (const Unit &consumer : state_.units) {
      if (members.count(consumer.id) || plan.owners[consumer.id] >= 0)
        continue;
      const auto *block = GetBlockNode(consumer.stmt);
      if (block && IsVFBlock(block->name_hint))
        continue;
      // A control unit includes its descendants; inspect only its header here.
      // Leaf consumers are checked once, including MTE addresses and call args.
      auto headers = HeaderExprs(consumer.stmt);
      if (headers.empty() && !consumer.child_scopes.empty())
        continue;
      bool used = false;
      auto check = [&](const ObjectRef &obj) {
        if (const auto *load = obj.as<BufferLoadNode>())
          used |= facts.access.Writes(load->buffer);
        if (const auto *var = obj.as<VarNode>()) {
          for (const Var &binding : facts.bindings)
            used |= binding.same_as(GetRef<Var>(var));
          // A raw pointer escape is an unproved host consumer too.
          for (const Buffer &written : facts.access.writes)
            used |= written->data.same_as(GetRef<Var>(var));
        }
      };
      if (headers.empty() && consumer.child_scopes.empty())
        PostOrderVisit(consumer.stmt, check);
      else
        for (const PrimExpr &header : headers)
          PostOrderVisit(header, check);
      if (used)
        return true;
    }
    return false;
  }

  bool IsVFUnit(int id) const {
    // Scalar positions describe M2 ownership, before MTE legalization.
    if (plan.owners[id] >= 0)
      return true;
    const auto *block = GetBlockNode(U(id).stmt);
    return block && IsVFBlock(block->name_hint);
  }

  bool VFWrites(int id, const Buffer &buffer) const {
    int owner = plan.owners[id];
    if (owner < 0)
      return U(id).access.Writes(buffer);
    const Region &region = plan.regions[owner];
    for (int order = region.begin; order <= region.end; ++order)
      if (U(S(region.scope).units[order]).access.Writes(buffer))
        return true;
    for (int relocated : plan.region_actions[owner].relocated_copies)
      if (U(relocated).access.Writes(buffer))
        return true;
    return false;
  }

  void PlanScalarScope(int scope_id) {
    const Scope &scope = S(scope_id);
    for (size_t cursor = 0; cursor < scope.units.size();) {
      int id = scope.units[cursor];
      if (plan.owners[id] >= 0 || !U(id).host_scalar) {
        if (plan.owners[id] < 0)
          for (int child : U(id).child_scopes)
            PlanScalarScope(child);
        ++cursor;
        continue;
      }
      size_t begin = cursor;
      ScalarFacts facts;
      ScalarDecision decision;
      std::unordered_set<int> members;
      while (cursor < scope.units.size()) {
        int member = scope.units[cursor];
        if (plan.owners[member] >= 0 || !U(member).host_scalar)
          break;
        facts(U(member).stmt);
        decision.units.push_back(member);
        CollectSubtree(member, &members);
        ++cursor;
      }
      bool left = begin > 0 && IsVFUnit(scope.units[begin - 1]);
      bool right = cursor < scope.units.size() && IsVFUnit(scope.units[cursor]);
      decision.position = left && right ? "bridge"
                          : right       ? "prefix"
                          : left        ? "postfix"
                                        : "isolated";
      if (left)
        for (const Buffer &read : facts.access.reads)
          decision.reads_left |= VFWrites(scope.units[begin - 1], read);
      decision.category = facts.Classify(HasHostConsumer(facts, members));
      decision.reason = SelectScalarPlacement(decision.category);
      // Only record mathematical tails after the complete scalar recurrence.
      // These are unproved boundaries, never permissions to migrate a slice.
      if (decision.category <= ScalarClass::kS3) {
        size_t last_loop = 0;
        for (size_t i = 0; i < decision.units.size(); ++i) {
          ScalarFacts part;
          part(U(decision.units[i]).stmt);
          if (part.loop)
            last_loop = i;
        }
        for (size_t i = last_loop + 1; i < decision.units.size(); ++i)
          decision.local_tail_begins.push_back(decision.units[i]);
      }
      state_.scalar_decisions.push_back(std::move(decision));
    }
  }

  void VerifyRegions() {
    // Validate owners before M3 consumes the region index.
    for (const Region &region : plan.regions) {
      for (int order = region.begin; order <= region.end; ++order) {
        const Unit &covered = U(S(region.scope).units[order]);
        if (covered.role == Role::kBoundary)
          Reject(ErrorCode::kRegionConflict,
                 "continuous SIMT region crosses a boundary");
      }
    }
    for (const Unit &unit : state_.units) {
      if (unit.host_scalar && plan.owners[unit.id] >= 0)
        Reject(ErrorCode::kPlanInvariant,
               "host scalar migration has no published structural rule");
      if (unit.role != Role::kBoundary && plan.owners[unit.id] < 0)
        Reject(ErrorCode::kPlanInvariant,
               "fusible statement has no region owner");
    }
  }

  // ---- M3: Storage legalization ----
  // Convert fragment flow and region ownership into VF-local,
  // shared.dyn-backed, or preserved storage plus the substitutions and
  // reload/writeback actions.
  struct RegionFlow {
    bool writes{false}, reload{false}, reduce_source{false};
  };

  // Definite assignment is composed over Seq/If/For, not lexical last_def.
  // A loop body is checked at entry; only a proven nonempty loop exports defs.
  // This monotone transfer reaches its fixed point in one pass: definitions
  // persist, and every possible first-iteration entry read is accounted for.
  void Transfer(const BufferFlow &flow, int id, bool *defined,
                RegionFlow *result) const {
    const Unit &unit = U(id);
    if (unit.child_scopes.empty()) {
      auto index = EventIndex(flow, id);
      if (!index)
        return;
      const AccessEvent &event = flow.events[*index];
      result->reload |= event.Reads() && !*defined;
      result->writes |= event.Writes();
      result->reduce_source |=
          unit.reduce_src_fragment &&
          flow.original->data.same_as((*unit.reduce_src_fragment)->data);
      *defined |= event.effect == AccessEffect::kFullDef;
      return;
    }
    bool after = true;
    for (int child : unit.child_scopes) {
      bool branch_defined = *defined;
      for (int member : S(child).units)
        Transfer(flow, member, &branch_defined, result);
      after &= branch_defined;
    }
    if (const auto *loop = unit.stmt.as<ForNode>()) {
      if (IsStaticPositiveInt(loop->extent))
        *defined = after;
    } else if (const auto *branch = unit.stmt.as<IfThenElseNode>()) {
      *defined = after && (branch->else_case.defined() || *defined);
    }
  }

  std::map<int, RegionFlow> SummarizeFlowRegions(const BufferFlow &flow) const {
    std::map<int, RegionFlow> regions;
    for (const AccessEvent &event : flow.events)
      regions.try_emplace(plan.owners[event.unit]);
    for (auto &[id, result] : regions) {
      if (id < 0)
        continue;
      const Region &region = plan.regions[id];
      bool defined = false;
      for (int copy : plan.region_actions[id].relocated_copies)
        Transfer(flow, copy, &defined, &result);
      for (int i = region.begin; i <= region.end; ++i) {
        int member = S(region.scope).units[i];
        if (!IsRelocated(member))
          Transfer(flow, member, &defined, &result);
      }
      // The epilogue reads the whole fragment, including paths that skip a
      // conditional definition or execute a loop zero times.
      result.reload |=
          regions.size() > 1 && !regions.count(-1) && result.writes && !defined;
    }
    return regions;
  }

  // Cross-VF fragment reuse materializes shared state. Regions needing
  // fragment-only operations reload locally; writing regions also commit.
  void PlanStateBacked(const BufferFlow &flow,
                       const std::map<int, RegionFlow> &regions) {
    Buffer state = CloneBufferWithScope(
        flow.original, flow.original->name + "_simtvf_state", "shared.dyn");
    plan.remove_allocations.push_back(flow.original);
    plan.add_allocations.push_back(state);
    std::map<int, Buffer> targets;
    for (const auto &[region, region_flow] : regions) {
      Buffer target = state;
      if (region_flow.writes || region_flow.reduce_source) {
        target = CloneBufferWithScope(flow.original,
                                      flow.original->name + "_simtvf_r" +
                                          std::to_string(region),
                                      "local.fragment");
        plan.region_actions[region].allocations.push_back(target);
        if (region_flow.reload)
          plan.region_actions[region].prologue.push_back(
              MakeCopyStmt(state, target));
        if (region_flow.writes)
          plan.region_actions[region].epilogue.push_back(
              MakeCopyStmt(target, state));
      }
      targets.emplace(region, target);
    }
    for (const AccessEvent &event : flow.events) {
      Buffer target = targets.at(plan.owners[event.unit]);
      plan.substitutions[event.unit].Set(flow.original, target);
      // Every view follows the same region-local storage as its root. Retain
      // its shape, strides and offset, rather than replacing only one access.
      for (const Buffer &alias : flow.aliases) {
        Buffer view = alias;
        view.CopyOnWrite()->data = target->data;
        plan.substitutions[event.unit].Set(alias, view);
      }
    }
  }

  // A standalone fragment copy may become an ordinary MTE copy after its
  // read-only endpoint is replaced by shared state.
  bool CanEmitPlainEngineCopy(const Region &region) const {
    if (region.begin != region.end)
      return false;
    const RegionAction &action = plan.region_actions[region.id];
    if (!action.relocated_copies.empty() || !action.allocations.empty() ||
        !action.prologue.empty() || !action.epilogue.empty()) {
      return false;
    }
    const Unit &unit = U(S(region.scope).units[region.begin]);
    if (!unit.copy || !IsFragmentBuffer(unit.copy->src))
      return false;
    Buffer src = plan.substitutions[unit.id]
                     .Get(unit.copy->src)
                     .value_or(unit.copy->src);
    Buffer dst = plan.substitutions[unit.id]
                     .Get(unit.copy->dst)
                     .value_or(unit.copy->dst);
    if (!IsUbGmPair(src, dst) || src->dtype != dst->dtype)
      return false;
    Copy copy = *MatchTileCopy(unit.stmt);
    arith::Analyzer analyzer;
    return copy->dst_range.size() == 1 ||
           TryNormalizeMTE2DLayout(copy->dst, copy->dst_range, &analyzer)
               .has_value();
  }

  void PlanStorage() {
    plan.substitutions.assign(state_.units.size(), {});
    for (const BufferFlow &flow : state_.fragment_flows) {
      // Summarize each region in its eventual emission order before choosing
      // storage.
      auto regions = SummarizeFlowRegions(flow);
      if (regions.empty() || regions.count(-1))
        continue;
      if (regions.size() == 1 && !regions.begin()->second.reload) {
        plan.remove_allocations.push_back(flow.original);
        plan.region_actions[regions.begin()->first].allocations.push_back(
            flow.original);
      } else if (std::any_of(
                     regions.begin(), regions.end(),
                     [](const auto &entry) { return entry.second.writes; })) {
        PlanStateBacked(flow, regions);
      }
    }
  }

  void PlanPrivateScalars() {
    for (const Buffer &buffer : state_.root_block->alloc_buffers) {
      if (!IsLocalVarBuffer(buffer))
        continue;
      int user = -1;
      for (const Unit &unit : state_.units) {
        if (!unit.child_scopes.empty())
          continue; // Control units duplicate their child access summaries.
        if (!unit.access.Accesses(buffer))
          continue;
        if (user >= 0) {
          user = -2;
          break;
        }
        user = unit.id;
      }
      if (user < 0 || plan.owners[user] < 0)
        continue;
      const auto *loop = U(user).stmt.as<ForNode>();
      if (loop == nullptr || loop->kind != ForKind::kParallel)
        continue;
      if (!IsParallelDefinition(U(user).stmt, buffer))
        continue;
      plan.remove_allocations.push_back(buffer);
      plan.region_actions[plan.owners[user]].allocations.push_back(buffer);
    }
  }

  // ---- M4: IR rewrite and result verification ----
  int64_t NextSourceIndex() {
    while (state_.occupied_source_indices.count(next_source_index_))
      ++next_source_index_;
    return next_source_index_++;
  }

  PrimFunc Rewrite() {
    LogicalStoreMarker marker(state_);
    Stmt body = SpliceTilelangRoot(state_.original->body,
                                   RewriteScope(state_.root_scope, &marker));
    PrimFunc result = state_.original;
    result.CopyOnWrite()->body = std::move(body);
    return result;
  }

  Stmt RewriteUnit(const Unit &unit, LogicalStoreMarker *marker) {
    Stmt stmt = unit.stmt;
    for (int child : unit.child_scopes)
      stmt =
          ReplaceControlChild(stmt, S(child).slot, RewriteScope(child, marker));
    const BufferMap &replacements = plan.substitutions[unit.id];
    return replacements.empty() ? stmt
                                : LocalBufferRewriter(replacements)(stmt);
  }

  Stmt RewriteScope(int scope_id, LogicalStoreMarker *marker) {
    const Scope &scope = S(scope_id);
    std::vector<Stmt> output;
    for (int cursor = 0; cursor < static_cast<int>(scope.units.size());) {
      int unit_id = scope.units[cursor];
      const Unit &unit = U(unit_id);
      if (IsRelocated(unit_id)) {
        ++cursor;
        continue;
      }
      int owner = plan.owners[unit_id];
      if (owner < 0 || plan.regions[owner].scope != scope_id) {
        output.push_back(RewriteUnit(unit, marker));
        ++cursor;
        continue;
      }
      const Region &region = plan.regions[owner];
      ICHECK_EQ(cursor, region.begin);
      const RegionAction &action = plan.region_actions[owner];
      if (CanEmitPlainEngineCopy(region)) {
        output.push_back(RewriteUnit(unit, marker));
      } else {
        std::vector<Stmt> body = action.prologue;
        for (int copy : action.relocated_copies)
          body.push_back(RewriteUnit(U(copy), marker));
        for (int order = region.begin; order <= region.end; ++order) {
          int member = scope.units[order];
          if (!IsRelocated(member))
            body.push_back(RewriteUnit(U(member), marker));
        }
        body.insert(body.end(), action.epilogue.begin(), action.epilogue.end());
        Stmt logical_body = (*marker)(SeqStmt::Flatten(body));
        output.push_back(BuildSimtVF(std::move(logical_body), plan.tx_extent,
                                     NextSourceIndex(), unit.stmt->span,
                                     action.allocations));
      }
      cursor = region.end + 1;
    }
    return SeqStmt::Flatten(output);
  }

  Stmt SpliceTilelangRoot(const Stmt &body, const Stmt &new_body) {
    struct Splicer : StmtExprMutator {
      Splicer(const Stmt &replacement, const std::vector<Buffer> &add,
              const std::vector<Buffer> &remove)
          : replacement(replacement), add(add), remove(remove) {}
      const Stmt &replacement;
      const std::vector<Buffer> &add;
      const std::vector<Buffer> &remove;
      bool replaced{false};
      Stmt VisitStmt_(const SBlockRealizeNode *op) final {
        if (op->block->name_hint == "tilelang_root") {
          ICHECK(!replaced) << "AutoSimtVF: duplicate tilelang_root block";
          replaced = true;
          SBlock block = op->block;
          auto *node = block.CopyOnWrite();
          node->body = replacement;
          Array<Buffer> kept;
          for (const Buffer &alloc : node->alloc_buffers)
            if (std::none_of(remove.begin(), remove.end(),
                             [&](const Buffer &r) { return alloc.same_as(r); }))
              kept.push_back(alloc);
          ICHECK(kept.size() + remove.size() == node->alloc_buffers.size())
              << "AutoSimtVF: removing unknown allocation";
          for (const Buffer &buffer : add)
            kept.push_back(buffer);
          node->alloc_buffers = std::move(kept);
          return SBlockRealize(op->iter_values, op->predicate, block, op->span);
        }
        return StmtExprMutator::VisitStmt_(op);
      }
    } splicer{new_body, plan.add_allocations, plan.remove_allocations};
    return splicer(body);
  }

  struct VFStructureScan : StmtVisitor {
    int vf_depth{0};
    int vf_count{0};
    int nested_vf{0};
    int parallel_outside_vf{0};
    std::unordered_set<int64_t> source_indices;
    bool duplicate_index{false};
    void VisitStmt(const Stmt &stmt) final {
      const SBlockNode *block = GetBlockNode(stmt);
      if (block && IsVFBlock(block->name_hint))
        ScanOneVF(block);
      else
        StmtVisitor::VisitStmt(stmt);
    }
    void VisitStmt_(const ForNode *op) final {
      if (op->kind == ForKind::kParallel && vf_depth == 0)
        ++parallel_outside_vf;
      StmtVisitor::VisitStmt_(op);
    }

  private:
    void ScanOneVF(const SBlockNode *block) {
      ++vf_count;
      if (vf_depth > 0)
        ++nested_vf;
      ++vf_depth;
      auto it = block->annotations.find("tl.vf_source_index");
      if (it != block->annotations.end()) {
        if (const auto *imm = (*it).second.as<IntImmNode>()) {
          duplicate_index |= !source_indices.insert(imm->value).second;
        }
      }
      StmtVisitor::VisitStmt(block->body);
      --vf_depth;
    }
  };

  void VerifyResult(const PrimFunc &result) {
    VFStructureScan scan;
    scan(result->body);
    if (scan.parallel_outside_vf != 0 || scan.nested_vf != 0) {
      Reject(ErrorCode::kRewriteInvariant,
             "Parallel coverage or nested VF invariant failed");
    }
    int generated_vf_count = 0;
    for (const Region &region : plan.regions)
      generated_vf_count += !CanEmitPlainEngineCopy(region);
    if (scan.vf_count != state_.original_vf_count + generated_vf_count) {
      Reject(ErrorCode::kRewriteInvariant,
             "existing VF count changed after rewrite");
    }
    if (scan.duplicate_index) {
      Reject(ErrorCode::kRewriteInvariant,
             "duplicate tl.vf_source_index after rewrite");
    }
  }

  Optional<Var> subcore_var_;
  bool fuse_serial_, fused_control_{false};
  int64_t next_source_index_{0};
  PlannerState state_;
  RewritePlan &plan{state_.plan};
};
} // namespace
namespace transform {
using namespace tirx::transform;
TVM_REGISTER_PASS_CONFIG_OPTION("tl.ascend_auto_simt_fuse_serial", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tl.ascend_auto_simt_report_scalar", Bool);
Pass AutoSimtVF() {
  auto pass_func = [](PrimFunc func, const IRModule &, PassContext) {
    auto target_opt = func->GetAttr<Target>(tvm::attr::kTarget);
    if (!target_opt.defined() || !TargetIsAscend(target_opt.value())) {
      return func;
    }
    InputReader reader;
    reader(func->body);
    if (!reader.has_candidate)
      return func;
    bool fuse_serial =
        PassContext::Current()
            ->GetConfig<Bool>("tl.ascend_auto_simt_fuse_serial", Bool(true))
            .value();
    auto planner = std::make_unique<AutoSimtVFPlanner>(func, fuse_serial);
    PrimFunc result = planner->Run(reader);
    if (planner->fused_control() && !CanInferStructuredLayout(result)) {
      // Replan from untouched logical IR; never undo materialized state copies.
      planner = std::make_unique<AutoSimtVFPlanner>(func, false);
      result = planner->Run(reader);
    }
    if (PassContext::Current()
            ->GetConfig<Bool>("tl.ascend_auto_simt_report_scalar", Bool(false))
            .value())
      result = WithAttr(result, "tl.auto_simt_scalar_decisions",
                        planner->ScalarReport());
    return result;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AutoSimtVF", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.AutoSimtVF", AutoSimtVF);
}
} // namespace transform
} // namespace tl
} // namespace tvm

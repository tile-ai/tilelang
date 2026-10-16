/*!
 * \file auto_simt_vf.cc
 * \brief Form Ascend SIMT_VF regions around eligible Parallel computation.
 *
 * One planner follows M1 Analyze, M2 PlanRegions, M3 PlanStorage, and
 * M4 VerifyPlan/Rewrite/VerifyResult.
 *
 * Fragment definers join consuming regions when safe. A fragment owned by one
 * VF moves there; cross-VF reuse uses shared.dyn state. Existing VF is opaque.
 * Scalar state uses local.var, not local.fragment[1].
 */
#include "ascend/op/ascend_mte_plan.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "ascend/target_utils.h"
#include "op/copy.h"
#include "op/fill.h"
#include "op/reduce.h"
#include "op/utils.h"
#include "support/check.h"
#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <tvm/ir/attrs.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
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
  int index = static_cast<int>(code);
  return index >= 0 &&
                 index < static_cast<int>(sizeof(kNames) / sizeof(*kNames))
             ? kNames[index]
             : "<unknown>";
}

struct Failure {
  ErrorCode code;
  std::string detail;
};
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
  std::optional<Buffer> fragment_dst, reduce_src_fragment;
  bool fragment_dst_full_def{
      false}; // This unit replaces the prior fragment value.
  UnitSemantics(Role role, std::optional<CopyInfo> copy = std::nullopt)
      : role(role), copy(std::move(copy)) {}
};

struct UnitAccess {
  std::vector<Buffer> reads;
  std::vector<Buffer> writes;
  // Flow analysis matches data aliases; substitutions match exact Buffers.
  bool Writes(const Buffer &buffer) const {
    return std::any_of(writes.begin(), writes.end(), [&](const Buffer &item) {
      return item->data.same_as(buffer->data);
    });
  }
};

enum class AccessEffect { kRead, kFullDef, kUpdate };
struct AccessEvent {
  int unit{-1};
  Buffer buffer;
  AccessEffect effect{AccessEffect::kRead};
  int reaching_def{-1}; // Index in BufferFlow::events; -1 is the entry value.
  bool Reads() const { return effect != AccessEffect::kFullDef; }
  bool Writes() const { return effect != AccessEffect::kRead; }
};

struct Unit : UnitSemantics {
  int id{-1}, scope{-1}, order{-1};
  Stmt stmt;
  UnitAccess access;
  std::vector<int> child_scopes;
  Span span;
  Unit(int id, int scope, int order, Stmt stmt, UnitSemantics semantics)
      : UnitSemantics(std::move(semantics)), id(id), scope(scope), order(order),
        stmt(std::move(stmt)), span(this->stmt->span) {}
};

enum class ChildSlot { kRoot, kLoopBody, kThen, kElse, kAttrBody };
struct Scope {
  int id{-1}, parent{-1};
  ChildSlot slot{ChildSlot::kRoot};
  std::vector<int> units; // units[order] directly yields the unit id.
};

struct BufferFlow {
  Buffer original;
  std::vector<AccessEvent> events;
  std::unordered_map<int, int> event_by_unit;
  int last_def{-1};
  bool has_alias_view{false}; // Alias/view access retains the root allocation.
};

struct Region {
  int id{-1}, scope{-1}, begin{-1}, end{-1};
};
struct Substitution {
  Buffer from, to;
};
struct RegionAction {
  std::vector<int> relocated_copies; // Emitted before direct region members.
  std::vector<Buffer> allocations;   // Fragment allocations owned by this VF.
  std::vector<Stmt> prologue;        // Shared-state reloads.
  std::vector<Stmt> epilogue;        // Shared-state writebacks.
};

struct RewritePlan {
  std::vector<Region> regions;
  std::vector<int>
      owners; // -1 means unowned; differing scopes mean relocation.
  std::vector<RegionAction> region_actions;
  std::vector<std::vector<Substitution>> substitutions;
  std::vector<Buffer> add_allocations, remove_allocations;
  PrimExpr tx_extent;
};

struct PlannerState {
  PrimFunc original;
  Stmt logical_body;
  SBlock root_block;
  int root_scope{-1};
  std::vector<Unit> units;
  std::vector<Scope> scopes;
  std::vector<BufferFlow> fragment_flows;
  std::unordered_set<int64_t> occupied_source_indices;
  int original_vf_count{0};
  RewritePlan plan;
  std::optional<Failure> failure;
};

bool IsVFBlock(const String &name_hint) {
  return name_hint == "SIMT_VF" || name_hint == "SIMD_VF";
}

bool IsScopedAttr(const String &key) {
  return key == "tl.ascend_stage" || key == "tl.assume" ||
         key == "tl.assume_requires_runtime_check";
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

std::optional<Stmt> ReplaceControlChild(const Stmt &owner, ChildSlot slot,
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
  if (const auto *attr = owner.as<AttrStmtNode>()) {
    ICHECK(slot == ChildSlot::kAttrBody && IsScopedAttr(attr->attr_key));
    return AttrStmt(attr->node, attr->attr_key, attr->value, new_body,
                    attr->span);
  }
  return std::nullopt;
}

class SourceIndexAllocator {
public:
  explicit SourceIndexAllocator(std::unordered_set<int64_t> occupied)
      : occupied_(std::move(occupied)) {}
  int64_t Next() {
    while (occupied_.count(next_) != 0)
      ++next_;
    return next_++;
  }

private:
  int64_t next_{0};
  std::unordered_set<int64_t> occupied_;
};

// Capture the thread envelope, root, and candidates; existing VFs stay opaque.
bool IsStage1Candidate(const Stmt &stmt);
class InputReader : public StmtExprVisitor {
public:
  PrimExpr tx_extent{IntImm(DataType::Int(32), 128)};
  std::vector<Var> thread_vars;
  Optional<SBlockRealize> root;
  bool has_candidate{false}, outer_thread_var_used{false};

private:
  bool in_root_{false};
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>()) {
        if (iv->thread_tag == "threadIdx.x") {
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

bool HeaderReadsFragment(const Stmt &stmt) {
  BufferReadWriteCollector reads;
  if (const auto *branch = stmt.as<IfThenElseNode>()) {
    reads(branch->condition);
  } else if (const auto *loop = stmt.as<ForNode>()) {
    reads(loop->min);
    reads(loop->extent);
    if (loop->step.defined())
      reads(loop->step.value());
  } else if (const auto *bind = stmt.as<BindNode>()) {
    reads(bind->value);
  }
  return std::any_of(reads.summary.reads.begin(), reads.summary.reads.end(),
                     IsFragmentBuffer);
}

// M3 helpers for buffer cloning and synthetic copies.
Buffer CloneBufferWithScope(const Buffer &original, const String &name,
                            const String &scope) {
  const auto *ptr = original->data->type_annotation.as<PointerTypeNode>();
  ICHECK(ptr != nullptr) << "fragment data var must have a pointer type";
  Var data(name, PointerType(PrimType(original->dtype), scope), original->span);
  return Buffer(data, original->dtype, original->shape, original->strides,
                original->elem_offset, name, original->data_alignment,
                original->offset_factor, original->buffer_type,
                original->axis_separators, original->span);
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
  explicit LocalBufferRewriter(const std::vector<Substitution> &substitutions)
      : substitutions_(substitutions) {}
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    const auto *load = expr.as<BufferLoadNode>();
    if (load == nullptr)
      return expr;
    auto to = Lookup(load->buffer);
    if (!to.has_value())
      return expr;
    return BufferLoad(*to, load->indices, load->predicate, load->span);
  }
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    const auto *store = stmt.as<BufferStoreNode>();
    if (store == nullptr)
      return stmt;
    auto to = Lookup(store->buffer);
    if (!to.has_value())
      return stmt;
    return BufferStore(*to, store->value, store->indices, store->predicate,
                       store->span);
  }

private:
  std::optional<Buffer> Lookup(const Buffer &from) const {
    for (const Substitution &substitution : substitutions_) {
      if (substitution.from.same_as(from))
        return substitution.to;
    }
    return std::nullopt;
  }
  const std::vector<Substitution> &substitutions_;
};

class AutoSimtVFPlanner {
public:
  explicit AutoSimtVFPlanner(PrimFunc func) {
    state_.original = std::move(func);
  }

  // M1-M3 only build a plan; M4 rewrites after verification, so failures never
  // expose a partially transformed function.
  std::optional<PrimFunc> Run(const InputReader &reader) {
    if (!Analyze(reader))
      return std::nullopt;
    if (!PlanRegions())
      return std::nullopt;
    PlanStorage();
    if (!VerifyPlan())
      return std::nullopt;
    auto result = Rewrite();
    if (!result.has_value())
      return std::nullopt;
    if (!VerifyResult(*result))
      return std::nullopt;
    return result;
  }

  const Failure &failure() const { return *state_.failure; }

private:
  bool Fail(ErrorCode code, std::string detail) {
    if (!state_.failure.has_value())
      state_.failure = Failure{code, std::move(detail)};
    return false;
  }

  std::optional<UnitSemantics> Reject(ErrorCode code, const char *detail) {
    Fail(code, detail);
    return std::nullopt;
  }

  const Unit &U(int id) const { return state_.units[id]; }
  const Scope &S(int id) const { return state_.scopes[id]; }
  bool IsRelocated(int unit_id) const {
    int region = plan.owners[unit_id];
    return region >= 0 && U(unit_id).scope != plan.regions[region].scope;
  }

  // ---- M1: Statement analysis ----
  // Build lexical Scope/Unit tables and per-fragment reaching-definition
  // chains. Existing VFs stay opaque; control bodies become child scopes.
  bool Analyze(const InputReader &reader) {
    if (!IsStaticPositiveInt(reader.tx_extent)) {
      return Fail(ErrorCode::kThreadEnvelope,
                  "threadIdx.x extent is not a positive static integer");
    }
    plan.tx_extent = reader.tx_extent;
    Optional<SBlockRealize> root = reader.root;
    if (!root.defined()) {
      return Fail(ErrorCode::kInputContract, "tilelang_root block not found");
    }
    state_.root_block = root.value()->block;
    state_.logical_body = state_.root_block->body;

    if (reader.outer_thread_var_used)
      return Fail(ErrorCode::kInputContract,
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
          return Fail(
              ErrorCode::kInputContract,
              "local.fragment[1] scalar storage is unsupported; use alloc_var");
        }
      }
      state_.fragment_flows.push_back(BufferFlow{buffer});
    }
    if (!VisitScope(state_.logical_body, state_.root_scope))
      return false;
    state_.fragment_flows.erase(std::remove_if(state_.fragment_flows.begin(),
                                               state_.fragment_flows.end(),
                                               [](const BufferFlow &flow) {
                                                 return flow.events.empty();
                                               }),
                                state_.fragment_flows.end());
    return true;
  }

  int AddScope(int parent, ChildSlot slot) {
    int id = static_cast<int>(state_.scopes.size());
    state_.scopes.push_back(Scope{id, parent, slot, {}});
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
  std::optional<UnitSemantics> ClassifyCopy(const Copy &copy) {
    const Buffer &src = copy->src;
    const Buffer &dst = copy->dst;
    if (IsDedicatedEnginePair(src, dst)) {
      return UnitSemantics(Role::kBoundary, CopyInfo{src, dst});
    }
    if (IsFragmentEndpointPair(src, dst)) {
      Buffer fragment = IsFragmentBuffer(src) ? src : dst;
      auto ranges = IsFragmentBuffer(src) ? copy->src_range : copy->dst_range;
      if (!CoversWholeBuffer(ranges, fragment)) {
        return Reject(ErrorCode::kUnsupportedCopy,
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
    return Reject(ErrorCode::kUnsupportedCopy,
                  "unsupported copy scope/dtype combination");
  }

  std::optional<UnitSemantics> ClassifyLeaf(const Stmt &stmt) {
    if (auto copy = MatchTileCopy(stmt))
      return ClassifyCopy(*copy);
    if (auto fill = MatchTileOp<Fill>(stmt, "tl.tileop.fill")) {
      bool full_def = CoversWholeBuffer((*fill)->region, (*fill)->dst);
      if (!full_def && IsFragmentBuffer((*fill)->dst)) {
        return Reject(ErrorCode::kUnsupportedStatement,
                      "fill must be static and cover the whole buffer");
      }
      UnitSemantics leaf(IsFillSeed(*fill) ? Role::kSeed : Role::kBoundary);
      leaf.fragment_dst = (*fill)->dst;
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
        return Reject(
            ErrorCode::kUnsupportedReduce,
            "reduce must be static float32 sum/max on whole fragments");
      }
      UnitSemantics leaf(Role::kFusible);
      leaf.fragment_dst = reduce_ref->dst;
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
          return Reject(ErrorCode::kUnsupportedStatement,
                        "tile-op call is outside the Stage1 whitelist");
        }
      }
      return UnitSemantics(Role::kBoundary);
    }
    if (stmt.as<BindNode>()) {
      if (HeaderReadsFragment(stmt))
        return Reject(ErrorCode::kUnsupportedStatement,
                      "fragment load in a scalar binding is unsupported");
      return UnitSemantics(Role::kBoundary);
    }
    if (stmt.as<BufferStoreNode>())
      return UnitSemantics(Role::kBoundary);
    return Reject(ErrorCode::kUnsupportedStatement,
                  "statement is outside the Stage1 whitelist");
  }

  bool VisitScope(const Stmt &body, int scope_id) {
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
        return Fail(ErrorCode::kInputContract,
                    "fragment load in a control header is unsupported");
      std::optional<UnitSemantics> semantics =
          control || existing_vf ? std::optional{UnitSemantics(Role::kBoundary)}
          : parallel             ? std::optional{UnitSemantics(Role::kSeed)}
                                 : ClassifyLeaf(stmt);
      if (!semantics)
        return false;
      int id = AddUnit(stmt, scope_id, order, std::move(*semantics));
      if (control) {
        for (auto [slot, child_body] : GetControlChildren(stmt)) {
          int child = AddScope(scope_id, slot);
          state_.units[id].child_scopes.push_back(child);
          if (!VisitScope(child_body, child))
            return false;
          for (int unit_id : S(child).units) {
            for (const Buffer &buffer : U(unit_id).access.reads)
              AddUnique(&state_.units[id].access.reads, buffer);
            for (const Buffer &buffer : U(unit_id).access.writes)
              AddUnique(&state_.units[id].access.writes, buffer);
          }
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
    return true;
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
        for (const Buffer &buffer : buffers) {
          if (buffer->data.same_as(flow.original->data))
            return &buffer;
        }
        return nullptr;
      };
      const Buffer *read = find_access(unit.access.reads);
      const Buffer *write = find_access(unit.access.writes);
      const Buffer *accessed = write != nullptr ? write : read;
      if (accessed == nullptr)
        continue;
      flow.has_alias_view |= !accessed->same_as(flow.original);
      int index = static_cast<int>(flow.events.size());
      AccessEffect effect = AccessEffect::kRead;
      if (write != nullptr) {
        bool full_def = read == nullptr &&
                        ((unit.copy && IsFragmentBuffer(unit.copy->dst)) ||
                         unit.fragment_dst_full_def);
        effect = full_def ? AccessEffect::kFullDef : AccessEffect::kUpdate;
      }
      flow.events.push_back(
          AccessEvent{unit.id, *accessed, effect, flow.last_def});
      flow.event_by_unit.emplace(unit.id, index);
      if (flow.events.back().Writes())
        flow.last_def = index;
    }
  }

  // ---- M2: Region planning ----
  // Split scopes into boundary-free windows, grow seeds through fragment flow,
  // then commit intervals, owners, and safe ancestor-definer relocations.
  bool IsFragmentCopy(const Unit &unit) const {
    return unit.copy && IsFragmentEndpointPair(unit.copy->src, unit.copy->dst);
  }

  std::optional<int> EventIndex(const BufferFlow &flow, int unit_id) const {
    auto it = flow.event_by_unit.find(unit_id);
    return it == flow.event_by_unit.end() ? std::nullopt
                                          : std::optional<int>(it->second);
  }

  bool UnitsShareValue(int lhs_unit, int rhs_unit) const {
    for (const BufferFlow &flow : state_.fragment_flows) {
      auto lhs_index = EventIndex(flow, lhs_unit);
      auto rhs_index = EventIndex(flow, rhs_unit);
      if (!lhs_index || !rhs_index)
        continue;
      const AccessEvent &lhs = flow.events[*lhs_index];
      const AccessEvent &rhs = flow.events[*rhs_index];
      if (lhs.reaching_def == *rhs_index || rhs.reaching_def == *lhs_index ||
          (lhs.Reads() && rhs.Reads() &&
           lhs.reaching_def == rhs.reaching_def)) {
        return true;
      }
    }
    return false;
  }

  // Relocation preserves reaching definitions and source address dependencies.
  bool PreservesFlowWhenMoved(const Unit &copy, const Region &target) const {
    const CopyInfo &info = *copy.copy;
    if (!IsFragmentBuffer(info.dst))
      return false;
    for (int cur = target.scope; cur != copy.scope; cur = S(cur).parent) {
      if (cur < 0 || S(cur).parent < 0 || S(cur).slot != ChildSlot::kLoopBody)
        return false;
    }

    auto it =
        std::find_if(state_.fragment_flows.begin(), state_.fragment_flows.end(),
                     [&](const BufferFlow &flow) {
                       return info.dst->data.same_as(flow.original->data);
                     });
    if (it == state_.fragment_flows.end() || it->has_alias_view)
      return false;
    auto definer = EventIndex(*it, copy.id);
    if (!definer || it->events[*definer].effect != AccessEffect::kFullDef)
      return false;
    bool has_consumer = false;
    for (const AccessEvent &event : it->events) {
      if (event.unit == copy.id)
        continue;
      has_consumer = true;
      const Unit &consumer = U(event.unit);
      if (event.Writes() || event.reaching_def != *definer ||
          consumer.scope != target.scope || consumer.order < target.begin ||
          consumer.order > target.end) {
        return false;
      }
    }
    if (!has_consumer)
      return false;

    BufferReadWriteCollector dependencies;
    dependencies(copy.stmt);
    const Scope &home = S(copy.scope);
    for (const Buffer &buffer : dependencies.summary.reads) {
      if (buffer->data.same_as(info.dst->data))
        continue;
      for (int id : home.units) {
        if (U(id).order > copy.order && U(id).access.Writes(buffer))
          return false;
      }
      for (int cur = target.scope; cur != copy.scope; cur = S(cur).parent) {
        for (int id : S(cur).units) {
          if (U(id).access.Writes(buffer))
            return false;
        }
      }
    }
    return true;
  }

  void ClaimAncestorDefinerCopies(const Region &target) {
    for (const Unit &copy : state_.units) {
      if (!IsFragmentCopy(copy) || !IsFragmentBuffer(copy.copy->dst) ||
          copy.scope == target.scope || plan.owners[copy.id] >= 0 ||
          !PreservesFlowWhenMoved(copy, target))
        continue;
      plan.owners[copy.id] = target.id;
      plan.region_actions[target.id].relocated_copies.push_back(copy.id);
    }
  }

  bool PlanWindow(const Scope &scope, int first, int last) {
    std::vector<Region> local;
    for (int order = first; order <= last; ++order) {
      const Unit &unit = U(scope.units[order]);
      if (IsRelocated(unit.id) || unit.role != Role::kSeed)
        continue;
      local.push_back(Region{/*id=*/-1, scope.id, order, order});
    }
    bool has_seed = !local.empty();
    std::vector<bool> attached(scope.units.size(), false);
    // Breadth-first growth follows the transitive value chain from each seed.
    for (Region &region : local) {
      std::vector<int> members{region.begin};
      for (size_t cursor = 0; cursor < members.size(); ++cursor) {
        for (int order = first; order <= last; ++order) {
          const Unit &unit = U(scope.units[order]);
          if (IsRelocated(unit.id) || unit.role != Role::kFusible ||
              std::find(members.begin(), members.end(), order) !=
                  members.end() ||
              !UnitsShareValue(unit.id, scope.units[members[cursor]]))
            continue;
          members.push_back(order);
          attached[order] = true;
          region.begin = std::min(region.begin, order);
          region.end = std::max(region.end, order);
        }
      }
    }

    // Copies and reductions that cannot join a seed form standalone regions.
    for (int order = first; order <= last; ++order) {
      const Unit &unit = U(scope.units[order]);
      if (IsRelocated(unit.id) || unit.role != Role::kFusible ||
          attached[order])
        continue;
      if (IsFragmentCopy(unit) || unit.reduce_src_fragment) {
        local.push_back(Region{/*id=*/-1, scope.id, order, order});
        continue;
      }
      return Fail(ErrorCode::kPlanInvariant,
                  "fusible statement has no region owner");
    }

    // Normalize connected and adjacent intervals before assigning ownership.
    std::sort(local.begin(), local.end(), [](const Region &a, const Region &b) {
      return a.begin != b.begin ? a.begin < b.begin : a.end < b.end;
    });
    size_t first_region = plan.regions.size();
    for (Region &region : local) {
      if (plan.regions.size() > first_region &&
          region.begin <= plan.regions.back().end + 1) {
        plan.regions.back().end = std::max(plan.regions.back().end, region.end);
      } else {
        region.id = static_cast<int>(plan.regions.size());
        plan.regions.push_back(region);
        plan.region_actions.emplace_back();
      }
    }
    // Commit intervals, then claim legal ancestor definers.
    for (size_t i = first_region; i < plan.regions.size(); ++i) {
      const Region &region = plan.regions[i];
      for (int order = region.begin; order <= region.end; ++order) {
        int unit_id = scope.units[order];
        if (IsRelocated(unit_id))
          continue;
        int &owner = plan.owners[unit_id];
        if (owner >= 0) {
          return Fail(ErrorCode::kPlanInvariant,
                      "unit has multiple region owners");
        }
        owner = region.id;
      }
      if (has_seed)
        ClaimAncestorDefinerCopies(region);
    }
    return true;
  }

  bool PlanRegions() {
    plan.owners.assign(state_.units.size(), -1);
    for (size_t i = state_.scopes.size(); i > 0; --i) {
      const Scope &scope = S(static_cast<int>(i - 1));
      int first = -1;
      for (int order = 0; order <= static_cast<int>(scope.units.size());
           ++order) {
        bool boundary = order == static_cast<int>(scope.units.size());
        if (!boundary)
          boundary = U(scope.units[order]).role == Role::kBoundary;
        if (boundary) {
          if (first >= 0 && !PlanWindow(scope, first, order - 1))
            return false;
          first = -1;
        } else if (first < 0) {
          first = order;
        }
      }
    }
    return FinalizeRegionIndexAndVerify();
  }

  bool FinalizeRegionIndexAndVerify() {
    // Validate owners before M3 consumes the region index.
    for (const Region &region : plan.regions) {
      if (region.scope < 0 ||
          region.scope >= static_cast<int>(state_.scopes.size()) ||
          region.begin < 0 || region.end < region.begin ||
          region.end >= static_cast<int>(S(region.scope).units.size())) {
        return Fail(ErrorCode::kPlanInvariant, "invalid region interval");
      }
      for (int order = region.begin; order <= region.end; ++order) {
        const Unit &covered = U(S(region.scope).units[order]);
        if (covered.role == Role::kBoundary)
          return Fail(ErrorCode::kRegionConflict,
                      "continuous SIMT region crosses a boundary");
      }
    }
    for (const Unit &unit : state_.units) {
      if (unit.role == Role::kSeed && plan.owners[unit.id] < 0) {
        return Fail(ErrorCode::kPlanInvariant, "seed has no region owner");
      }
      if (unit.role == Role::kFusible && plan.owners[unit.id] < 0) {
        return Fail(ErrorCode::kPlanInvariant,
                    "fusible statement has no region owner");
      }
    }
    return true;
  }

  // ---- M3: Storage legalization ----
  // Convert fragment flow and region ownership into VF-local,
  // shared.dyn-backed, or preserved storage plus the substitutions and
  // reload/writeback actions.
  struct RegionFlow {
    bool writes{false}, reload{false}, reduce_source{false};
  };

  std::map<int, RegionFlow> SummarizeFlowRegions(const BufferFlow &flow) const {
    std::map<int, RegionFlow> regions;
    auto consume = [&](const AccessEvent &event) {
      auto [it, first] = regions.try_emplace(plan.owners[event.unit]);
      if (first)
        it->second.reload = event.Reads();
      it->second.writes |= event.Writes();
      const Unit &unit = U(event.unit);
      it->second.reduce_source |=
          unit.reduce_src_fragment &&
          event.buffer->data.same_as((*unit.reduce_src_fragment)->data);
    };
    // Emission order within each region is relocated copies, then members.
    for (bool relocated : {true, false})
      for (const AccessEvent &event : flow.events)
        if (IsRelocated(event.unit) == relocated)
          consume(event);
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
    for (const AccessEvent &event : flow.events)
      plan.substitutions[event.unit].push_back(
          {event.buffer, targets.at(plan.owners[event.unit])});
  }

  Buffer RewrittenBuffer(int unit_id, const Buffer &buffer) const {
    for (const Substitution &substitution : plan.substitutions[unit_id]) {
      if (substitution.from.same_as(buffer))
        return substitution.to;
    }
    return buffer;
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
    Buffer src = RewrittenBuffer(unit.id, unit.copy->src);
    Buffer dst = RewrittenBuffer(unit.id, unit.copy->dst);
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
      if (flow.has_alias_view)
        continue;
      // Summarize each region in its eventual emission order before choosing
      // storage.
      auto regions = SummarizeFlowRegions(flow);

      if (regions.size() == 1 && !regions.count(-1) &&
          !regions.begin()->second.reload) {
        plan.remove_allocations.push_back(flow.original);
        plan.region_actions[regions.begin()->first].allocations.push_back(
            flow.original);
      } else if (!regions.empty() && !regions.count(-1) &&
                 std::any_of(
                     regions.begin(), regions.end(),
                     [](const auto &entry) { return entry.second.writes; })) {
        PlanStateBacked(flow, regions);
      }
    }
  }

  // ---- M4: Plan verification, IR rewrite, and result verification ----
  // Verify the complete plan before rebuilding, then check Parallel coverage,
  // VF nesting/counts, and source-index uniqueness on the emitted PrimFunc.
  bool VerifyPlan() {
    if (!IsStaticPositiveInt(plan.tx_extent)) {
      return Fail(ErrorCode::kThreadEnvelope,
                  "tx extent must be a positive static integer");
    }
    if (plan.region_actions.size() != plan.regions.size() ||
        plan.substitutions.size() != state_.units.size()) {
      return Fail(ErrorCode::kPlanInvariant,
                  "action vectors do not match the state");
    }

    for (const Region &region : plan.regions) {
      const RegionAction &action = plan.region_actions[region.id];
      for (const Buffer &buffer : action.allocations) {
        if (!IsFragmentBuffer(buffer)) {
          return Fail(ErrorCode::kPlanInvariant,
                      "VF-owned allocation is not a fragment");
        }
      }
    }
    for (int unit_id = 0; unit_id < static_cast<int>(plan.substitutions.size());
         ++unit_id) {
      const auto &sites = plan.substitutions[unit_id];
      for (size_t i = 0; i < sites.size(); ++i) {
        for (size_t j = i + 1; j < sites.size(); ++j) {
          if (sites[i].from.same_as(sites[j].from) &&
              !sites[i].to.same_as(sites[j].to)) {
            return Fail(ErrorCode::kPlanInvariant,
                        "conflicting buffer substitutions in one unit");
          }
        }
      }
    }
    return true;
  }

  std::optional<PrimFunc> Rewrite() {
    SourceIndexAllocator indices(state_.occupied_source_indices);
    auto logical_body = RewriteScope(state_.root_scope, &indices);
    if (!logical_body.has_value())
      return std::nullopt;
    Stmt body = SpliceTilelangRoot(state_.original->body, *logical_body);
    PrimFunc result = state_.original;
    result.CopyOnWrite()->body = std::move(body);
    return result;
  }

  Stmt RewriteUnit(const Unit &unit, Stmt stmt) {
    const auto &sites = plan.substitutions[unit.id];
    if (!sites.empty()) {
      stmt = LocalBufferRewriter(sites)(std::move(stmt));
    }
    return stmt;
  }

  std::optional<Stmt> RewriteScope(int scope_id,
                                   SourceIndexAllocator *indices) {
    const Scope &scope = S(scope_id);
    std::vector<Stmt> output;
    for (int cursor = 0; cursor < static_cast<int>(scope.units.size());) {
      int unit_id = scope.units[cursor];
      const Unit &unit = U(unit_id);
      if (IsRelocated(unit_id)) {
        ++cursor; // emitted at the head of its target VF.
        continue;
      }
      int region_id = plan.owners[unit_id];
      if (region_id >= 0 && plan.regions[region_id].begin != unit.order)
        region_id = -1;
      if (region_id < 0) {
        Stmt stmt = unit.stmt;
        for (int child_id : unit.child_scopes) {
          auto child = RewriteScope(child_id, indices);
          if (!child.has_value())
            return std::nullopt;
          auto replaced = ReplaceControlChild(stmt, S(child_id).slot, *child);
          if (!replaced.has_value()) {
            Fail(ErrorCode::kRewriteInvariant,
                 "control child slot has no rebuild rule");
            return std::nullopt;
          }
          stmt = *replaced;
        }
        output.push_back(RewriteUnit(unit, std::move(stmt)));
        ++cursor;
        continue;
      }
      const Region &region = plan.regions[region_id];
      const RegionAction &action = plan.region_actions[region_id];
      if (CanEmitPlainEngineCopy(region)) {
        ICHECK(region.begin == region.end);
        const Unit &member = U(scope.units[region.begin]);
        output.push_back(RewriteUnit(member, member.stmt));
        cursor = region.end + 1;
        continue;
      }
      std::vector<Stmt> body;
      body.insert(body.end(), action.prologue.begin(), action.prologue.end());
      for (int copy_id : action.relocated_copies) {
        body.push_back(RewriteUnit(U(copy_id), U(copy_id).stmt));
      }
      for (int order = region.begin; order <= region.end; ++order) {
        int member_id = scope.units[order];
        if (IsRelocated(member_id))
          continue;
        const Unit &member = U(member_id);
        body.push_back(RewriteUnit(member, member.stmt));
      }
      body.insert(body.end(), action.epilogue.begin(), action.epilogue.end());
      Span span = body.empty() ? state_.logical_body->span : body.front()->span;
      Array<Buffer> allocations = action.allocations;
      output.push_back(BuildSimtVF(SeqStmt::Flatten(body), plan.tx_extent,
                                   indices->Next(), span, allocations));
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

  bool VerifyResult(const PrimFunc &result) {
    VFStructureScan scan;
    scan(result->body);
    if (scan.parallel_outside_vf != 0 || scan.nested_vf != 0) {
      return Fail(ErrorCode::kRewriteInvariant,
                  "Parallel coverage or nested VF invariant failed");
    }
    int generated_vf_count = 0;
    for (const Region &region : plan.regions)
      generated_vf_count += !CanEmitPlainEngineCopy(region);
    if (scan.vf_count != state_.original_vf_count + generated_vf_count) {
      return Fail(ErrorCode::kRewriteInvariant,
                  "existing VF count changed after rewrite");
    }
    if (scan.duplicate_index) {
      return Fail(ErrorCode::kRewriteInvariant,
                  "duplicate tl.vf_source_index after rewrite");
    }
    return true;
  }

  PlannerState state_;
  RewritePlan &plan{state_.plan};
};
} // namespace
namespace transform {
using namespace tirx::transform;
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
    AutoSimtVFPlanner planner(func);
    auto result = planner.Run(reader);
    if (!result.has_value()) {
      const Failure &failure = planner.failure();
      LOG(FATAL) << "[AutoSimtVF] " << ErrorCodeName(failure.code) << ": "
                 << failure.detail;
      return func; // unreachable: LOG(FATAL) throws InternalError.
    }
    return *result;
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

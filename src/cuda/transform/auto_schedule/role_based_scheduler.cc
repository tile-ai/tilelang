/*!
 * \file role_based_scheduler.cc
 * \brief The "role_based" automatic warp-specialization scheduler.
 *
 * Four steps:
 *  1. Classify: each op gets a fixed role from its lowering eligibility
 *     (ws_analysis.h) — Load (TMA / cp.async global->shared copy), MMA
 *     (tcgen05 GEMM), Store (shared->global TMA copy), Worker (the
 *     rest). Ops touching only warp-private state start roleless.
 *  2. Place: a role's body is the backward slice of its fixed ops — the
 *     defs of every scalar and private-buffer read, plus the bounds and
 *     guards of every entered scope. Slices may overlap (the
 *     materializer SSA-freshens shared private chains per role); a slice
 *     reaching an op fixed to another role declines. Roleless ops in no
 *     slice run in every active role.
 *  3. Pipelines: synchronization is ownership movement. Each touch owns
 *     a storage from an initial to a final role. Bottom-up, a scope
 *     hosts a pipeline iff consecutive touches mismatch (fin != next
 *     init, wrapping to the first touch); an unbroken chain passes
 *     upward as one opaque touch. Program order names the producer (the
 *     first in-body transfer is its handoff), transfers alternate
 *     handoff/recycle, and the wrap recycle — consumer release at the
 *     end of the body, producer acquire at its start — rides the
 *     barrier's cyclic phases. Nested bindings multiply a storage's
 *     versions; bindings below the outermost stay single-buffered.
 *  4. Emit the typed WSSchedule; versioning, barriers, and arrive
 *     counts are MaterializeWSSchedule's job.
 *
 * TODO: pipeline merging — one barrier pair for pipelines of one scope
 * whose consumer-side brackets coincide, producer sides unioned per
 * cycle (producer_consumer_ws's positional criterion; merging is then
 * free by construction). Software-pipeline stage offsets (every sync is
 * stage 0); if-else branches; sibling-pipeline chaining; multi-consumer
 * fan-out; a multi-warp role for register-staged SIMT copies; delayed
 * wgmma waits (wg_wait != 0 declines).
 */

#include "./role_based_scheduler.h"

#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "./memory_detector.h"
#include "cuda/op/builtin.h"
#include "cuda/target_utils.h"
#include "cuda/transform/ws_analysis.h"
#include "op/copy.h"
#include "op/operator.h"
#include "op/utils.h"
#include "transform/common/constr_visitor.h"
#include "transform/common/warp_specialize.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// Fixed heuristic roles, in warp order.
enum class Role : uint8_t { kWorker = 0, kLoad = 1, kMma = 2, kStore = 3 };
constexpr int kNumRoles = 4;
constexpr std::array<Role, kNumRoles> kAllRoles = {Role::kWorker, Role::kLoad,
                                                   Role::kMma, Role::kStore};

constexpr int kAuxiliaryRoleNumRegisters = 24;

// Registers for a widened cp.async Load warpgroup's addressing work.
constexpr int kCpAsyncLoadNumRegisters = 48;

// Occupancy estimate cap, and the per-SM capacities bounding it.
constexpr int kHeuristicBlocksPerSM = 4;
constexpr int kMaxThreadsPerSM = 2048;
constexpr int kSmemBytesPerSM = 228 * 1024; // sm90 / sm100
constexpr int kTmemBytesPerSM = 256 * 1024; // 512 b32 columns of 128 rows

// The occupancy annotation from the T.annotate_min_blocks_per_sm
// wrapper, 0 when absent.
int AnnotatedMinBlocksPerSM(const Stmt &body) {
  int min_blocks = 0;
  PostOrderVisit(body, [&](const ObjectRef &node) {
    if (const auto *attr = node.as<AttrStmtNode>()) {
      if (attr->attr_key == attr::kMinBlocksPerSM) {
        if (const auto *imm = attr->value.as<IntImmNode>()) {
          if (imm->value <= 0) {
            LOG(FATAL) << "min_blocks_per_sm must be positive, got: "
                       << imm->value;
          }
          min_blocks = static_cast<int>(imm->value);
        }
      }
    }
  });
  return min_blocks;
}

// A SET of roles: an op is replicated into several roles when roleless,
// and a scope's participants span its ops' roles. Single-role facts use
// `Role` directly.
struct RoleMask {
  uint8_t bits{0};

  static constexpr RoleMask Of(Role role) {
    return {static_cast<uint8_t>(1u << static_cast<unsigned>(role))};
  }
  bool Empty() const { return bits == 0; }
  int Count() const { return __builtin_popcount(bits); }
  bool Contains(Role role) const { return bits & Of(role).bits; }
  bool ContainsAll(RoleMask other) const { return !(other.bits & ~bits); }
  void Add(Role role) { bits |= Of(role).bits; }
  void Add(RoleMask other) { bits |= other.bits; }
  // The single role of a one-role set.
  Role TheRole() const {
    ICHECK_EQ(Count(), 1);
    return static_cast<Role>(__builtin_ctz(bits));
  }
  bool operator==(const RoleMask &other) const { return bits == other.bits; }
};

const char *RoleName(Role role) {
  switch (role) {
  case Role::kWorker:
    return "Worker";
  case Role::kLoad:
    return "Load";
  case Role::kMma:
    return "MMA";
  case Role::kStore:
    return "Store";
  }
  LOG(FATAL) << "invalid role";
  return "";
}

// Warp-private storage: values there cannot be handed between roles.
bool IsPrivateBuffer(const Buffer &buffer) {
  return IsLocalBuffer(buffer, true) || IsFragmentBuffer(buffer);
}

// Storage that may carry a cross-role handoff through a pipeline.
bool IsPipelineBuffer(const Buffer &buffer) {
  return IsSharedBuffer(buffer) || IsTmemBuffer(buffer);
}

// The pipeline depth a scope contributes: its loop's num_stages, else 1.
int ScopeStages(const For &loop) {
  if (!loop.defined())
    return 1;
  if (auto value = loop->annotations.Get("num_stages")) {
    if (auto integer = value.value().try_cast<Integer>())
      return std::max<int64_t>(1, integer.value()->value);
    if (const auto *imm = value.value().as<IntImmNode>())
      return std::max<int64_t>(1, imm->value);
  }
  return 1;
}

struct SchedScope;

// The roles owning a storage when an entry begins and when it ends: a
// leaf owns in its role throughout; a hosting scope receives as its
// producer and surrenders as its consumer.
struct Ownership {
  Role initial;
  Role final;
};

// One schedulable statement, reduced to the facts scheduling needs (the
// statement stays in the kernel body, addressed by id). A scope IS an
// op: one opaque entry at its parent's level, `roles` = participants,
// `reads` = bound/guard reads.
struct SchedOp {
  String id;
  TileStmtKind kind{TileStmtKind::kConsumer};
  TileOperator tile_op;        // parsed once for a direct tile-op call
  SchedScope *parent{nullptr}; // scope whose entries reference this
  bool is_bind{false};  // scalar Bind: duplicable even when it reads globals
  bool guarded{false};  // under an if: a skipped write carries state
  bool roleless{false}; // touches only warp-private state
  RoleMask roles;       // a fixed single role, or the grown set when roleless
  std::vector<Buffer> reads, writes;      // buffer accesses, guards included
  std::vector<Var> read_vars, write_vars; // scalar uses / Bind defs
  // Ownership per resolved storage (nullopt = untouched): leaves seeded
  // by BuildPipelines, scopes written by ResolveOwnership.
  std::vector<std::optional<Ownership>> ownership;

  virtual ~SchedOp() = default;
  virtual SchedScope *AsScope() { return nullptr; }
  const SchedScope *AsScope() const {
    return const_cast<SchedOp *>(this)->AsScope();
  }
  bool PlacedIn(Role role) const { return roles.Contains(role); }
};

// The single placing role of an op touching shared storage.
Role RoleOf(const SchedOp &op) { return op.roles.TheRole(); }

struct SchedScope : SchedOp {
  For loop;         // a sequential-loop (serial / unrolled) scope
  While while_loop; // a T.ws_op-wrapped while scope; both undefined = root
  std::vector<SchedOp *> entries; // ops and child scopes, in source order
  // T.annotate_ws_pipeline_depth in this scope's body: buffer var -> depth.
  std::unordered_map<Var, int64_t, ObjectPtrHash, ObjectPtrEqual>
      pipeline_depth;

  SchedScope *AsScope() override { return this; }

  bool IsRoot() const { return parent == nullptr; }

  // Whether this scope is `ancestor` or nested somewhere below it.
  bool IsNestedIn(const SchedScope *ancestor) const {
    for (const SchedScope *s = this; s != nullptr; s = s->parent)
      if (s == ancestor)
        return true;
    return false;
  }
};

std::string ScopeName(const SchedScope *scope) {
  return scope->IsRoot() ? "root" : std::string(scope->id);
}

struct Pipeline {
  String name;
  Buffer allocation;
  int depth{1};
  Role producer{Role::kWorker};
  Role consumer{Role::kWorker};
  // The single scope hosting this pipeline's brackets, keyed by entry
  // boundary (boundary b sits just before entry b). At a boundary,
  // closes emit before opens, in push order.
  const SchedScope *scope{nullptr};
  std::map<int, std::vector<WSSyncKind>> closes, opens;
};

// Append the detector's results for `stmt` to the given vectors.
void CollectAccess(const Stmt &stmt, std::vector<Buffer> *reads,
                   std::vector<Buffer> *writes, std::vector<Var> *read_vars,
                   std::vector<Var> *write_vars) {
  MemoryAccessDetector detector;
  detector.Analyze(stmt);
  for (const BufferRegion &region : detector.GetReadRegions())
    reads->push_back(region->buffer);
  if (writes) {
    for (const BufferRegion &region : detector.GetWriteRegions())
      writes->push_back(region->buffer);
  }
  for (const Var &var : detector.GetReadVars())
    read_vars->push_back(var);
  if (write_vars) {
    for (const Var &var : detector.GetWriteVars())
      write_vars->push_back(var);
  }
}

// Builds the op/scope structure from the normalized body. ConstrVisitor
// maintains the lexical constraint stack, so an op's guards are the
// conditions on the stack where it appears. Ops are leaves: recorded,
// not recursed into.
class StructureBuilder : public ConstrVisitor {
public:
  explicit StructureBuilder(Target target) : target_(std::move(target)) {}

  std::vector<std::unique_ptr<SchedOp>> ops;
  std::vector<std::unique_ptr<SchedScope>> scopes; // [0] is the root

  bool Build(const Stmt &body) {
    auto root = std::make_unique<SchedScope>();
    root->id = kWSRootScopeId;
    scope_stack_.push_back(root.get());
    scopes.push_back(std::move(root));
    VisitStmt(body);
    return ok_;
  }

private:
  // Conditions evaluated on every control path to the current statement;
  // assumes are facts, not evaluated code.
  void ChargeConditions(std::vector<Buffer> *reads,
                        std::vector<Var> *read_vars) {
    for (const Constr &constr : constr_stack_) {
      if (constr.kind == Constr::kConstr && !constr.is_assume)
        CollectAccess(Evaluate(constr.value), reads, nullptr, read_vars,
                      nullptr);
    }
  }

  void MakeOp(const String &id, const Stmt &stmt) {
    ICHECK(op_ids_.insert(id).second)
        << "two statements carry ws op id '" << id << "'";
    auto op = std::make_unique<SchedOp>();
    op->id = id;
    op->parent = scope_stack_.back();
    op->is_bind = stmt.as<BindNode>() != nullptr;
    op->guarded = if_depth_ > 0;
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (const auto *call = eval->value.as<CallNode>())
        op->tile_op = ParseOperator(GetRef<Call>(call));
    }
    op->kind = ClassifyStmt(stmt, target_);
    CollectAccess(stmt, &op->reads, &op->writes, &op->read_vars,
                  &op->write_vars);
    ChargeConditions(&op->reads, &op->read_vars);
    scope_stack_.back()->entries.push_back(op.get());
    ops.push_back(std::move(op));
  }

  SchedScope *OpenScope(const String &id) {
    auto scope = std::make_unique<SchedScope>();
    SchedScope *child = scope.get();
    child->id = id;
    child->parent = scope_stack_.back();
    scope_stack_.back()->entries.push_back(child);
    scopes.push_back(std::move(scope));
    scope_stack_.push_back(child);
    return child;
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    if (!ok_)
      return;
    if (op->else_case.defined()) {
      LOG(WARNING)
          << "AutoSchedule skipped: if-else branches are not supported";
      ok_ = false;
      return;
    }
    ++if_depth_;
    ConstrVisitor::VisitStmt_(op); // guards the branch with its condition
    --if_depth_;
  }

  void VisitStmt_(const ForNode *op) final {
    if (!ok_)
      return;
    auto id = op->annotations.Get(kWSOpIdKey);
    ICHECK(id.has_value()) << "AutoSchedule normalized loop is missing "
                           << kWSOpIdKey;
    // Sequential loops are scopes; parallel / vectorized loops are one op.
    if (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled) {
      MakeOp(ExtractOpId(id.value()), GetRef<For>(op));
      return;
    }
    SchedScope *child = OpenScope(ExtractOpId(id.value()));
    child->loop = GetRef<For>(op);
    CollectAccess(Evaluate(op->min), &child->reads, nullptr, &child->read_vars,
                  nullptr);
    CollectAccess(Evaluate(op->extent), &child->reads, nullptr,
                  &child->read_vars, nullptr);
    if (op->step.defined())
      CollectAccess(Evaluate(op->step.value()), &child->reads, nullptr,
                    &child->read_vars, nullptr);
    ChargeConditions(&child->reads, &child->read_vars);
    ConstrVisitor::VisitStmt_(op); // guards the body with the loop range
    scope_stack_.pop_back();
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    if (!ok_)
      return;
    if (op->attr_key == kWSOpIdKey) {
      // A wrapped while loop is a scope; any other wrapped statement is
      // one opaque op.
      if (const auto *wl = op->body.as<WhileNode>()) {
        SchedScope *child = OpenScope(ExtractOpId(Any(op->value)));
        child->while_loop = GetRef<While>(wl);
        CollectAccess(Evaluate(wl->condition), &child->reads, nullptr,
                      &child->read_vars, nullptr);
        ChargeConditions(&child->reads, &child->read_vars);
        ConstrVisitor::VisitStmt_(wl); // guards the body with the condition
        scope_stack_.pop_back();
        return;
      }
      MakeOp(ExtractOpId(Any(op->value)), op->body);
      return;
    }
    if (op->attr_key == kWSPipelineDepthKey) {
      auto gmap = op->node.as<Map<ObjectRef, ObjectRef>>();
      ICHECK(gmap.has_value())
          << "T.annotate_ws_pipeline_depth must carry a map";
      for (const auto &[key, val] : gmap.value()) {
        auto var = key.as<Var>();
        const auto *depth = val.as<IntImmNode>();
        ICHECK(var && depth && depth->value >= 1)
            << "T.annotate_ws_pipeline_depth entries must map a buffer to a "
               "positive depth";
        scope_stack_.back()->pipeline_depth[var.value()] = depth->value;
      }
    }
    // Assumptions enter the constraint stack; other wrappers (kernel
    // metadata) are transparent.
    ConstrVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const EvaluateNode *op) final {
    if (!ok_)
      return;
    if (const auto *call = op->value.as<CallNode>()) {
      if (auto id = call->annotations.Get(kWSOpIdKey)) {
        MakeOp(ExtractOpId(Any(id.value())), GetRef<Stmt>(op));
        return;
      }
    }
    LOG(FATAL) << "AutoSchedule: Evaluate carries no ws op id";
  }

  // The normalizer wraps these statement forms with an id; reaching one
  // bare is its bug.
  void VisitStmt_(const BufferStoreNode *op) final {
    LOG(FATAL) << "AutoSchedule: BufferStore carries no ws op id";
  }
  void VisitStmt_(const BindNode *op) final {
    LOG(FATAL) << "AutoSchedule: Bind carries no ws op id";
  }
  void VisitStmt_(const WhileNode *op) final {
    LOG(FATAL) << "AutoSchedule: while loop carries no ws op id";
  }
  void VisitStmt_(const SBlockNode *op) final {
    LOG(FATAL) << "AutoSchedule: block carries no ws op id";
  }

  Target target_;
  bool ok_ = true;   // cleared when the kernel shape is unsupported
  int if_depth_ = 0; // enclosing IfThenElse frames
  std::vector<SchedScope *> scope_stack_;
  std::set<String> op_ids_;
};

// Plans the schedule for one normalized kernel body. Every unsupported
// shape declines with a warning: auto scheduling is opt-in, so the user
// expects it to fire.
class RoleBasedScheduler {
public:
  RoleBasedScheduler(Target target, int worker_threads)
      : target_(std::move(target)), worker_threads_(worker_threads) {}

  Optional<WSSchedule> Run(const SBlock &block, const Stmt &body) {
    if (!TargetHasBulkCopy(target_)) {
      LOG(WARNING) << "AutoSchedule skipped: target has no bulk-copy (TMA) "
                      "support";
      return std::nullopt;
    }
    if (worker_threads_ % 128 != 0) {
      LOG(WARNING) << "AutoSchedule skipped: worker threads must be a multiple "
                      "of 128 so issuer warps start a fresh warpgroup";
      return std::nullopt;
    }
    CollectAnnotatedLayouts(block, layout_map_);
    annotated_min_blocks_ = AnnotatedMinBlocksPerSM(body);
    alloc_buffers_ = block->alloc_buffers;
    if (!BuildStructure(body) || !ClassifyOps())
      return std::nullopt;
    IndexBufferUses();
    if (!PlaceSlices())
      return std::nullopt;
    // A scope's roles are its participants.
    for (const auto &op : ops_) {
      for (SchedScope *s = op->parent; s != nullptr; s = s->parent)
        s->roles.Add(op->roles);
    }
    if (!BuildPipelines(block))
      return std::nullopt;
    if (pipelines_.empty()) {
      LOG(WARNING) << "AutoSchedule skipped: no cross-role on-chip handoff to "
                      "pipeline";
      return std::nullopt;
    }
    return Emit();
  }

private:
  // ---- structure ------------------------------------------------------------

  bool BuildStructure(const Stmt &body) {
    StructureBuilder builder(target_);
    if (!builder.Build(body))
      return false;
    ops_ = std::move(builder.ops);
    scopes_ = std::move(builder.scopes);
    root_ = scopes_.front().get();
    return true;
  }

  // ---- classification -------------------------------------------------------

  // A TMA-classified copy joins its one-warp role only when its shared
  // side's annotated layout (if any) is TMA-expressible; otherwise
  // lowering falls back to a normal copy, which one warp would serialize.
  bool TmaSharedLayoutCompatible(const SchedOp &op, bool store) const {
    const auto *copy = op.tile_op.as<CopyNode>();
    if (copy == nullptr)
      return true; // Im2Col carries no annotated shared layout
    const Buffer &shared = store ? copy->src : copy->dst;
    auto it = layout_map_.find(shared->data);
    if (it == layout_map_.end())
      return true;
    const auto &[buffer, layout] = it->second;
    return layout.defined() && IsTmaCompatibleLayout(layout, buffer);
  }

  bool ClassifyOps() {
    for (const auto &op_ptr : ops_) {
      SchedOp &op = *op_ptr;
      switch (op.kind) {
      case TileStmtKind::kTmaProducer:
        // TODO: support warp specialization for cluster kernels — the
        // materialized barriers are CTA-local, so cluster-wide multicast
        // backpressure cannot be expressed yet.
        if (const auto *copy = op.tile_op.as<CopyNode>()) {
          if (auto mask = copy->annotations.Get("cluster_mask")) {
            if (const auto *imm = mask.value().as<IntImmNode>();
                imm == nullptr || imm->value != 0) {
              LOG(WARNING) << "AutoSchedule skipped: op '" << op.id
                           << "' is a cluster multicast copy";
              return false;
            }
          }
        }
        op.roles = RoleMask::Of(TmaSharedLayoutCompatible(op, /*store=*/false)
                                    ? Role::kLoad
                                    : Role::kWorker);
        continue;
      case TileStmtKind::kCpAsyncProducer:
        // The pipeline commit arrives through a deferred
        // cp.async.mbarrier.arrive.
        op.roles = RoleMask::Of(Role::kLoad);
        continue;
      case TileStmtKind::kCpAsyncRaw:
        // Raw cp.async carries its own thread-local completion protocol.
        LOG(WARNING) << "AutoSchedule skipped: op '" << op.id
                     << "': raw cp.async statements carry their own "
                        "thread-local completion protocol";
        return false;
      case TileStmtKind::kSimtProducer:
        // Register-staged and blocking; a one-warp Load role would
        // serialize it. TODO: multi-warp copy role.
        op.roles = RoleMask::Of(Role::kWorker);
        continue;
      case TileStmtKind::kTmaStore:
        op.roles = RoleMask::Of(TmaSharedLayoutCompatible(op, /*store=*/true)
                                    ? Role::kStore
                                    : Role::kWorker);
        continue;
      case TileStmtKind::kTcgen05Mma:
        op.roles = RoleMask::Of(Role::kMma);
        continue;
      case TileStmtKind::kConsumer:
        break;
      }

      if (op.tile_op.defined()) {
        // wg_wait != 0 returns with MMAs still reading the operands; a
        // release right after the gemm would race with them.
        if (auto gemm = GetGemmInfo(op.tile_op)) {
          if (gemm->wg_wait != 0) {
            LOG(WARNING) << "AutoSchedule skipped: op '" << op.id
                         << "': gemm with wg_wait != 0 completes "
                            "asynchronously; delayed wgmma waits are not "
                            "supported yet";
            return false;
          }
        }
        op.roles =
            RoleMask::Of(Role::kWorker); // any other tile op is SIMT work
        continue;
      }

      // Compound or plain statement op (PreprocessIR enforced no hosted
      // asynchronous tile op).
      bool private_writes = true;
      for (const Buffer &buffer : op.writes)
        private_writes = private_writes && IsPrivateBuffer(buffer);
      bool duplicable_reads = true;
      for (const Buffer &buffer : op.reads) {
        duplicable_reads =
            duplicable_reads &&
            (IsPrivateBuffer(buffer) || (op.is_bind && IsGlobalBuffer(buffer)));
      }
      if (private_writes && duplicable_reads)
        op.roleless = true; // placed by demand propagation
      else
        op.roles = RoleMask::Of(Role::kWorker);
    }
    return true;
  }

  // ---- role placement: backward slices --------------------------------------

  // A role's body is the backward slice of its fixed ops: the reaching
  // definition of each scalar read, every writer of each private-buffer
  // read, and — once a slice enters a scope — the reads of that scope's
  // bounds and guards.
  bool PlaceSlices() {
    std::unordered_map<Var, SchedOp *, ObjectPtrHash, ObjectPtrEqual> bind_def;
    for (const auto &op : ops_) {
      for (const Var &var : op->write_vars)
        bind_def.emplace(var, op.get());
    }

    auto slice = [&](Role role, const std::vector<SchedOp *> &seeds) {
      std::vector<SchedOp *> queue;
      for (SchedOp *op : seeds) {
        op->roles.Add(role);
        queue.push_back(op);
      }
      auto enqueue = [&](SchedOp *def, const String &user) {
        if (def->roles.Contains(role))
          return true;
        if (!def->roleless) {
          LOG(WARNING) << "AutoSchedule skipped: '" << user << "' needs op '"
                       << def->id << "' in role " << RoleName(role)
                       << ", but that op is fixed to role "
                       << RoleName(RoleOf(*def));
          return false;
        }
        def->roles.Add(role);
        queue.push_back(def);
        return true;
      };
      auto enqueue_reads = [&](const std::vector<Buffer> &reads,
                               const std::vector<Var> &read_vars,
                               const SchedOp *self, const String &user) {
        for (const Var &var : read_vars) {
          auto it = bind_def.find(var);
          if (it != bind_def.end() && it->second != self &&
              !enqueue(it->second, user))
            return false;
        }
        for (const Buffer &buffer : reads) {
          if (!IsPrivateBuffer(buffer))
            continue;
          auto it = buffer_uses_.find(buffer->data);
          if (it == buffer_uses_.end())
            continue;
          for (SchedOp *writer : it->second.writers) {
            if (writer != self && !enqueue(writer, user))
              return false;
          }
        }
        return true;
      };
      std::unordered_set<const SchedScope *> entered;
      while (!queue.empty()) {
        SchedOp *op = queue.back();
        queue.pop_back();
        // An entered scope implies every ancestor is entered; the root
        // has no bounds to charge.
        for (SchedScope *s = op->parent; s != nullptr && !entered.count(s);
             s = s->parent) {
          entered.insert(s);
          if (!enqueue_reads(s->reads, s->read_vars, /*self=*/nullptr, s->id))
            return false;
        }
        if (!enqueue_reads(op->reads, op->read_vars, op, op->id))
          return false;
      }
      return true;
    };

    for (Role role : kAllRoles) {
      std::vector<SchedOp *> seeds;
      for (const auto &op : ops_) {
        if (!op->roleless && op->roles == RoleMask::Of(role))
          seeds.push_back(op.get());
      }
      if (!seeds.empty() && !slice(role, seeds))
        return false;
    }

    // Private state in no slice runs in every active role.
    RoleMask active;
    for (const auto &op : ops_)
      active.Add(op->roles);
    if (active.Empty()) {
      LOG(WARNING) << "AutoSchedule skipped: kernel has no schedulable work";
      return false;
    }
    std::vector<SchedOp *> leftovers;
    for (const auto &op : ops_) {
      if (op->roles.Empty())
        leftovers.push_back(op.get());
    }
    for (Role role : kAllRoles) {
      if (leftovers.empty())
        break;
      if (active.Contains(role) && !slice(role, leftovers))
        return false;
    }
    return true;
  }

  // ---- def-use index --------------------------------------------------------

  // Program-ordered users of one buffer, indexed once; later phases query
  // these lists instead of re-scanning the IR.
  struct BufferUses {
    std::vector<SchedOp *> touches; // readers and writers, program order
    std::vector<SchedOp *> writers;
    std::vector<SchedOp *> readers;
  };

  void IndexBufferUses() {
    auto touch = [&](const Buffer &buffer, SchedOp *op) -> BufferUses & {
      BufferUses &uses = buffer_uses_[buffer->data];
      if (uses.touches.empty() || uses.touches.back() != op)
        uses.touches.push_back(op);
      return uses;
    };
    for (const auto &op : ops_) {
      for (const Buffer &buffer : op->reads)
        touch(buffer, op.get()).readers.push_back(op.get());
      for (const Buffer &buffer : op->writes)
        touch(buffer, op.get()).writers.push_back(op.get());
    }
  }

  // ---- pipelines ------------------------------------------------------------

  // Resolves one storage into pipelines, collected innermost first.
  struct StorageResolution {
    Buffer allocation;
    std::vector<Pipeline> pipelines;
    bool failed{false};
  };

  // Synchronization is ownership movement: a transfer is needed exactly
  // where one touch's final owner differs from the next touch's initial
  // owner, cyclically (wrap included). A scope with transfers hosts a
  // pipeline; an unbroken chain passes upward as one opaque touch.
  void ResolveOwnership(std::vector<StorageResolution> *resolutions,
                        SchedScope &scope) const {
    for (SchedOp *entry : scope.entries)
      if (SchedScope *child = entry->AsScope())
        ResolveOwnership(resolutions, *child);

    for (size_t k = 0; k < resolutions->size(); ++k) {
      StorageResolution &resolution = (*resolutions)[k];
      if (resolution.failed)
        continue;
      // Positions of the entries touching this storage.
      std::vector<int> pos;
      for (int e = 0; e < static_cast<int>(scope.entries.size()); ++e)
        if (scope.entries[e]->ownership[k].has_value())
          pos.push_back(e);
      if (pos.empty())
        continue;
      auto own = [&](int e) -> const Ownership & {
        return *scope.entries[e]->ownership[k];
      };
      Ownership level{own(pos.front()).initial, own(pos.back()).final};
      // A lone touch covers even the wraparound with its own handshake.
      if (pos.size() == 1) {
        scope.ownership[k] = level;
        continue;
      }
      // A loop runs back-to-back: before the first touch the storage is
      // owned by whoever ends the iteration, and it returns there after
      // the last. Host iff ownership must move between consecutive
      // touches, the wraparound included; an unbroken chain passes
      // upward as one opaque touch.
      size_t m = pos.size();
      bool moves = false;
      for (size_t i = 0; i < m; ++i)
        moves = moves || own(pos[i]).final != own(pos[(i + 1) % m]).initial;
      if (!moves) {
        scope.ownership[k] = level;
        continue;
      }

      // Production precedes consumption (the pattern of every schedule
      // in examples/aws): the first ownership movement between touches
      // is the producer's handoff. A body whose only movement is the
      // wraparound recycles — whoever takes the storage back produces.
      Role producer, consumer;
      {
        size_t i = 1;
        while (i < m && own(pos[i - 1]).final == own(pos[i]).initial)
          ++i;
        producer = i < m ? own(pos[i - 1]).final : own(pos.front()).initial;
        consumer = i < m ? own(pos[i]).initial : own(pos.back()).final;
      }
      bool two_roles = true;
      for (size_t i = 0; i < m; ++i) {
        for (Role role : {own(pos[i]).initial, own(pos[i]).final})
          two_roles = two_roles && (role == producer || role == consumer);
      }
      if (!two_roles) {
        LOG(WARNING) << "AutoSchedule skipped: storage '"
                     << resolution.allocation->name
                     << "' is handed between more than two roles in scope '"
                     << scope.id << "'";
        resolution.failed = true;
        continue;
      }

      Pipeline pipeline;
      pipeline.allocation = resolution.allocation;
      pipeline.scope = &scope;
      pipeline.producer = producer;
      pipeline.consumer = consumer;
      // The wraparound transfer returns the storage to the producer. Its
      // two halves are unconditional: the producer acquires at the first
      // touch and the consumer releases after the last (parity-only when
      // ownership does not move at the boundary — barrier elision TODO).
      pipeline.opens[pos[0]].push_back(WSSyncKind::ProducerAcquire());

      // Walk the touches with the owed transfer direction. A transfer
      // between touches alternates the producer's handoff (commit/wait)
      // with the recycle (release/acquire); a touch that surrenders
      // ownership to the consumer internally completes an owed handoff —
      // the producer commits right after it and the consumer waits at
      // its next touch, or drains with a bare wait;release at the end.
      bool ok = true;
      bool handoff = true; // the producer's handoff is owed first
      const char *reason = nullptr;
      for (size_t i = 0; ok && i < m; ++i) {
        const Ownership &o = own(pos[i]);
        if (i > 0 && own(pos[i - 1]).final != o.initial) {
          Role giver = handoff ? producer : consumer;
          // A recycle mid-body must come from the consumer's own touch:
          // an empty consumer span is only the end-of-body drain.
          if (own(pos[i - 1]).final != giver ||
              (!handoff && own(pos[i - 1]).initial != consumer)) {
            reason = "ownership transfers do not alternate between one "
                     "producer and one consumer";
            ok = false;
            break;
          }
          pipeline.closes[pos[i - 1] + 1].push_back(
              handoff ? WSSyncKind::ProducerCommit()
                      : WSSyncKind::ConsumerRelease());
          pipeline.opens[pos[i]].push_back(handoff
                                               ? WSSyncKind::ConsumerWait()
                                               : WSSyncKind::ProducerAcquire());
          handoff = !handoff;
        }
        if (o.initial == o.final) {
          if (o.initial == consumer && handoff) {
            reason = "the consumer touches the storage before the "
                     "producer's handoff";
            ok = false;
          }
          continue;
        }
        // A touch moving ownership is bracketed by the producer; inside
        // the consumer's span it would leave its producer-side interior
        // unheld.
        if (!handoff) {
          reason = "ownership moves inside the consumer's span";
          ok = false;
          break;
        }
        if (o.final == consumer) {
          pipeline.closes[pos[i] + 1].push_back(WSSyncKind::ProducerCommit());
          if (i + 1 < m)
            pipeline.opens[pos[i + 1]].push_back(WSSyncKind::ConsumerWait());
          else
            pipeline.closes[pos[m - 1] + 1].push_back(
                WSSyncKind::ConsumerWait());
          handoff = false;
        }
      }
      if (ok && handoff) {
        // A trailing producer run would need its span to cross the
        // wraparound (software-pipeline stages, unsupported).
        reason = "the body ends in the producer's span";
        ok = false;
      }
      if (!ok) {
        LOG(WARNING) << "AutoSchedule skipped: storage '"
                     << resolution.allocation->name << "' in scope '"
                     << scope.id << "': " << reason;
        resolution.failed = true;
        continue;
      }
      pipeline.closes[pos[m - 1] + 1].push_back(WSSyncKind::ConsumerRelease());

      scope.ownership[k] = level;
      resolution.pipelines.push_back(std::move(pipeline));
    }
  }

  // Depth of one binding. Multi-versioning is sound only when the
  // storage carries no LOOP-CARRIED DEPENDENCY across the scope's
  // iterations — each cycle must fully redefine the value. Conservative
  // approximation: the consumer writing or the producer reading inside
  // the scope may carry the previous cycle's value (TODO: a dependence
  // analyzer, e.g. proving a clear_accum resets the recurrence; today
  // T.annotate_ws_pipeline_depth asserts it). Accesses from OUTSIDE the
  // scope do not pin: the materializer resolves them to the adjacent
  // version (reads the last-completed, writes the next-produced).
  bool FinalizePipeline(Pipeline *pipeline,
                        const SchedScope *inner_scope) const {
    bool loop_carried = false;
    bool guarded_writer = false;
    std::optional<int64_t> annotated;
    if (auto vit =
            pipeline->scope->pipeline_depth.find(pipeline->allocation->data);
        vit != pipeline->scope->pipeline_depth.end())
      annotated = vit->second;
    const BufferUses &uses = buffer_uses_.at(pipeline->allocation->data);
    for (const SchedOp *op : uses.writers) {
      if (!op->parent->IsNestedIn(pipeline->scope)) {
        // A skipped outside write would leave the adjacent slot stale.
        guarded_writer = guarded_writer || op->guarded;
        continue;
      }
      loop_carried = loop_carried || RoleOf(*op) == pipeline->consumer;
      // A write inside a deeper binding's scope cycles that binding's
      // slot; skipping it cannot skew this ring's versions.
      if (!(inner_scope && op->parent->IsNestedIn(inner_scope)))
        guarded_writer = guarded_writer || op->guarded;
    }
    for (const SchedOp *op : uses.readers) {
      if (op->parent->IsNestedIn(pipeline->scope))
        loop_carried = loop_carried || RoleOf(*op) == pipeline->producer;
    }
    if (annotated)
      pipeline->depth = static_cast<int>(*annotated);
    else if (!loop_carried)
      pipeline->depth = ScopeStages(pipeline->scope->loop);
    // Under versioning, a guard-skipped write would expose the slot from
    // `depth` iterations ago instead of the previous value.
    if (guarded_writer && pipeline->depth > 1) {
      LOG(WARNING) << "AutoSchedule skipped: storage '" << pipeline->name
                   << "' is written under a guard and would be "
                   << pipeline->depth
                   << "-way versioned; a skipped write would expose a "
                      "stale version instead of the previous value";
      return false;
    }
    return true;
  }

  // Name, nesting, and depth for one storage's pipelines (innermost
  // first), then hand them to pipelines_.
  bool FinalizeResolution(StorageResolution *resolution) {
    // Sibling bindings of one storage (barrier chaining, a TODO) are the
    // materializer's strictly-nested mandate to reject.
    for (size_t i = 0; i < resolution->pipelines.size(); ++i) {
      Pipeline &pipeline = resolution->pipelines[i];
      // The innermost pipeline keeps the storage name; enclosing ones
      // carry their scope's id.
      std::string storage(resolution->allocation->name);
      pipeline.name =
          i == 0 ? storage : storage + "_" + ScopeName(pipeline.scope);
      const SchedScope *inner_scope =
          i == 0 ? nullptr : resolution->pipelines[i - 1].scope;
      if (!FinalizePipeline(&pipeline, inner_scope))
        return false;
      pipelines_.push_back(std::move(pipeline));
    }
    return true;
  }

  bool BuildPipelines(const SBlock &block) {
    // A global buffer written by one role must not be touched by another;
    // globals have no cross-role synchronization mechanism.
    for (const auto &writer : ops_) {
      for (const Buffer &buffer : writer->writes) {
        if (!IsGlobalBuffer(buffer))
          continue;
        RoleMask others;
        for (const SchedOp *op : buffer_uses_[buffer->data].touches)
          others.Add(op->roles);
        if (!writer->roles.ContainsAll(others)) {
          LOG(WARNING) << "AutoSchedule skipped: global buffer '"
                       << buffer->name << "' is written by op '" << writer->id
                       << "' and touched by another role";
          return false;
        }
      }
    }

    std::vector<StorageResolution> resolutions;
    for (const Buffer &allocation : block->alloc_buffers) {
      if (!IsPipelineBuffer(allocation))
        continue;
      auto it = buffer_uses_.find(allocation->data);
      if (it == buffer_uses_.end() || it->second.writers.empty())
        continue; // never written by a scheduled op
      RoleMask roles;
      for (const SchedOp *op : it->second.touches) {
        ICHECK_EQ(op->roles.Count(), 1)
            << "op '" << op->id << "' touches shared storage '"
            << allocation->name << "' but is placed by several roles";
        roles.Add(op->roles);
      }
      if (roles.Count() < 2)
        continue; // role-private storage needs no pipeline
      StorageResolution resolution;
      resolution.allocation = allocation;
      resolutions.push_back(std::move(resolution));
    }
    if (resolutions.empty())
      return true; // Run declines when nothing needs a pipeline

    for (const auto &op : ops_)
      op->ownership.assign(resolutions.size(), std::nullopt);
    for (const auto &scope : scopes_)
      scope->ownership.assign(resolutions.size(), std::nullopt);
    for (size_t k = 0; k < resolutions.size(); ++k) {
      const Buffer &allocation = resolutions[k].allocation;
      for (SchedOp *op : buffer_uses_.at(allocation->data).touches) {
        Role role = RoleOf(*op);
        op->ownership[k] = Ownership{role, role};
      }
    }
    ResolveOwnership(&resolutions, *root_);
    for (StorageResolution &resolution : resolutions) {
      if (resolution.failed)
        return false;
      // Two-role ownership must alternate at some scope, or the walk
      // fails the resolution.
      ICHECK(!resolution.pipelines.empty());
      if (!FinalizeResolution(&resolution))
        return false;
    }
    // Nested bindings' derived names may collide with real buffer names.
    std::set<std::string> names;
    for (Pipeline &pipeline : pipelines_) {
      std::string base = pipeline.name;
      for (int i = 2; !names.insert(pipeline.name).second; ++i)
        pipeline.name = base + "_" + std::to_string(i);
    }
    return true;
  }

  // ---- emission -------------------------------------------------------------

  struct RolePlan {
    Role role;
    int warp_lo;
    int warp_hi;
    int nreg;
  };

  // Use more threads if async producer is cp.async.
  bool HasCpAsyncLoad() const {
    for (const auto &op : ops_)
      if (op->kind == TileStmtKind::kCpAsyncProducer &&
          op->PlacedIn(Role::kLoad))
        return true;
    return false;
  }

  // The occupancy-target average minus the auxiliary requests, split
  // over the worker warpgroups. 0 = emit no setmaxnreg anywhere: the
  // split fell below the average or outside setmaxnreg's [24, 256].
  int WorkerNreg(const std::vector<RolePlan> &plans, int num_warps,
                 int avg) const {
    int worker_warps = worker_threads_ / 32;
    if (worker_warps % 4 != 0 || num_warps == worker_warps)
      return 0;
    int budget = num_warps / 4 * avg;
    for (int wg = worker_warps / 4; wg < num_warps / 4; ++wg) {
      int nreg = kAuxiliaryRoleNumRegisters; // idle warps donate too
      for (const RolePlan &p : plans)
        if (p.role != Role::kWorker && p.warp_lo < (wg + 1) * 4 &&
            wg * 4 < p.warp_hi)
          nreg = std::max(nreg, p.nreg);
      budget -= nreg;
    }
    int nreg = budget / (worker_warps / 4) / 8 * 8;
    if (nreg < std::max(avg, 24) || nreg > 256)
      return 0;
    return nreg;
  }

  // Occupancy estimate from the scheduled resource usage: shared and
  // tensor memory.
  int EstimateBlocksPerSM(int num_threads) const {
    if (annotated_min_blocks_ > 0)
      return annotated_min_blocks_;
    int64_t smem_bytes = 0;
    int64_t tmem_bytes = 0;
    for (const Buffer &buf : alloc_buffers_) {
      if (!IsPipelineBuffer(buf))
        continue;
      int64_t bytes = (buf->dtype.bits() * buf->dtype.lanes() + 7) / 8;
      for (const PrimExpr &extent : buf->shape) {
        const auto *imm = extent.as<IntImmNode>();
        if (imm == nullptr)
          return 1;
        bytes *= imm->value;
      }
      int64_t versions = 1;
      for (const Pipeline &pipeline : pipelines_)
        if (pipeline.allocation.same_as(buf))
          versions = std::max<int64_t>(versions, pipeline.depth);
      if (IsSharedBuffer(buf))
        smem_bytes += bytes * versions;
      else
        tmem_bytes += bytes * versions;
    }
    int64_t blocks = std::min<int64_t>(kHeuristicBlocksPerSM,
                                       kMaxThreadsPerSM / num_threads);
    if (smem_bytes > 0)
      blocks = std::min<int64_t>(blocks, kSmemBytesPerSM / smem_bytes);
    if (tmem_bytes > 0)
      blocks = std::min<int64_t>(blocks, kTmemBytesPerSM / tmem_bytes);
    return std::max<int64_t>(1, blocks);
  }

  // Plan the active roles in warp order: worker low, one warp per
  // auxiliary role, a cp.async Load widened to a full warpgroup.
  std::vector<RolePlan> PlanRoles(RoleMask active) const {
    int worker_warps = worker_threads_ / 32;
    std::vector<RolePlan> plans;
    int cursor = active.Contains(Role::kWorker) ? worker_warps : 0;
    for (Role role : kAllRoles) {
      if (!active.Contains(role))
        continue;
      if (role == Role::kWorker) {
        plans.push_back({role, 0, worker_warps, 0});
        continue;
      }
      bool wide = role == Role::kLoad && HasCpAsyncLoad();
      plans.push_back(
          {role, cursor, cursor + (wide ? 4 : 1),
           wide ? kCpAsyncLoadNumRegisters : kAuxiliaryRoleNumRegisters});
      cursor = plans.back().warp_hi;
    }
    int num_warps = (cursor + 3) / 4 * 4;
    // Registers per thread at the occupancy target.
    int num_threads = num_warps * 32;
    int avg = 65536 / (EstimateBlocksPerSM(num_threads) * num_threads) / 8 * 8;
    int worker_nreg = 0;
    for (RolePlan &p : plans)
      if (p.role == Role::kWorker)
        p.nreg = worker_nreg = WorkerNreg(plans, num_warps, avg);
    // Donations pay off only when the worker receives.
    if (worker_nreg == 0)
      for (RolePlan &p : plans)
        p.nreg = 0;
    return plans;
  }

  Optional<WSSchedule> Emit() const {
    RoleMask active;
    for (const auto &op : ops_)
      active.Add(op->roles);

    std::vector<RolePlan> plans = PlanRoles(active);
    int num_warps = 0;
    for (const RolePlan &p : plans)
      num_warps = std::max(num_warps, p.warp_hi);
    num_warps = (num_warps + 3) / 4 * 4;
    int max_threads = static_cast<int>(
        target_->GetAttr<Integer>("max_num_threads").value_or(1024)->value);
    if (num_warps * 32 > max_threads) {
      LOG(WARNING) << "AutoSchedule skipped: " << num_warps
                   << " warps exceed the target's " << max_threads
                   << "-thread block limit";
      return std::nullopt;
    }

    Array<WSRole> roles;
    for (const RolePlan &p : plans)
      roles.push_back(WSRole(RoleName(p.role), p.warp_lo, p.warp_hi, p.nreg));

    Array<WSPipeline> ws_pipelines;
    for (const Pipeline &pipeline : pipelines_) {
      ws_pipelines.push_back(WSPipeline(
          pipeline.name, Array<Buffer>{pipeline.allocation}, pipeline.depth));
    }

    // Scopes, innermost first, root last.
    Array<WSScope> scopes;
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it)
      scopes.push_back(EmitScope(**it, active));

    return WSSchedule(num_warps, roles, ws_pipelines, scopes);
  }

  // A bracket belongs to the producer or the consumer by its kind.
  static Role SyncRole(const Pipeline &pipeline, const WSSyncKind &kind) {
    return kind.IsProducerAcquire() || kind.IsProducerCommit()
               ? pipeline.producer
               : pipeline.consumer;
  }

  WSScope EmitScope(const SchedScope &scope, RoleMask active) const {
    int num_entries = static_cast<int>(scope.entries.size());
    Map<String, Array<WSInstr>> bodies;
    for (Role role : kAllRoles) {
      if (!active.Contains(role))
        continue;
      std::vector<char> mine(num_entries, 0);
      int count = 0;
      for (int e = 0; e < num_entries; ++e) {
        mine[e] = scope.entries[e]->PlacedIn(role);
        count += mine[e];
      }
      if (count == 0)
        continue;

      // At every boundary: this role's closes, then its opens, then the
      // entry itself when it is ours.
      Array<WSInstr> body;
      auto append = [&](int boundary, bool close) {
        for (const Pipeline &pipeline : pipelines_) {
          if (pipeline.scope != &scope)
            continue;
          const auto &at = close ? pipeline.closes : pipeline.opens;
          auto it = at.find(boundary);
          if (it == at.end())
            continue;
          for (const WSSyncKind &kind : it->second) {
            if (SyncRole(pipeline, kind) == role)
              body.push_back(WSSync(kind, pipeline.name, /*stage=*/0));
          }
        }
      };
      for (int b = 0; b <= num_entries; ++b) {
        append(b, /*close=*/true);
        append(b, /*close=*/false);
        if (b < num_entries && mine[b])
          body.push_back(WSOpRef(scope.entries[b]->id));
      }
      bodies.Set(RoleName(role), body);
    }
    return WSScope(scope.id, bodies);
  }

  Target target_;
  int worker_threads_;
  int annotated_min_blocks_ = 0;
  Array<Buffer> alloc_buffers_;
  BufferLayoutMap layout_map_;
  std::vector<std::unique_ptr<SchedOp>> ops_;
  std::vector<std::unique_ptr<SchedScope>> scopes_; // [0] is the root
  SchedScope *root_ = nullptr;
  std::unordered_map<Var, BufferUses, ObjectPtrHash, ObjectPtrEqual>
      buffer_uses_;
  std::vector<Pipeline> pipelines_;
};

} // namespace

ffi::Optional<WSSchedule> RoleBasedSchedule(const SBlock &block,
                                            const Stmt &body,
                                            int worker_threads,
                                            const Target &target) {
  return RoleBasedScheduler(target, worker_threads).Run(block, body);
}

} // namespace tl
} // namespace tvm

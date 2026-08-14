/*!
 * \file layout_cost_model.cc
 * \brief Statement-level bottleneck traffic model for free-mode layout
 *        attempts (layout RFC, design B2). See layout_cost_model.h.
 */

#include "layout_cost_model.h"

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../layout/layout.h"
#include "../../layout/utils.h"
#include "../../op/copy.h"
#include "../../op/parallel.h"
#include "../../op/utils.h"
#include "../../span_utils.h"
#include "../loop_vectorize.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

// ---------------------------------------------------------------------------
// Statement-level bottleneck traffic model (layout RFC, design B2)
//
// The unit of account is one STATEMENT that touches global memory — a
// fragment<->global tl.copy or a parallel loop with direct global accesses.
// The statement is measured in the FORWARD direction: enumerate every
// (logical point, replica) pair, evaluate the layout's own forward maps
//     thread = forward_thread(x, r)        slot = forward_index(x)
// plus the trivially affine global address addr(x), and materialize the
// exact (thread, slot) -> address table the lowered code will execute.
// No layout inversion (the forward maps are total and already built) and
// no witness sampling: the full walk is exact, with each point folded to
// a constant by the Analyzer.
//
// Execution time is bounded by two resources, both measured in BYTES so a
// fully busy block streaming maximal-width lanes scores exactly its
// logical byte count on either. The hardware geometry is parameterized:
// lane width from the vectorizer's shared MaxVectorLoadBits policy, warp
// size and coalescing-segment granularity from the target (see
// BindMemoryGeometry).
//     bw    = coalescing segments touched, counted exactly per (vector
//             step, warp); intra-warp broadcast merges into one segment,
//             store lanes holding a replica != 0 sleep behind the
//             replication guard and count nothing
//     issue = per-thread instruction depth x what a fully busy block
//             would stream at max lane width over that many steps:
//             steps x threads x lane_bytes. Idle lanes do not shorten the
//             depth, so thread-collapse pathologies surface here (#1729).
//     time(S) ~ max(bw, issue);      cost = sum over statements.
// ---------------------------------------------------------------------------

using VarBindings = std::unordered_map<const VarNode *, int64_t>;

/*! \brief Evaluate an index expression at a concrete point through the
 *  Analyzer: substitute every var — bound vars with their value, unbound
 *  vars (block indices inside region offsets) with zero, since they shift
 *  every address of the statement equally and cancel out of both
 *  contiguity and segment geometry — then constant-fold with Simplify.
 *  Anything that does not fold to an integer is outside the model. */
std::optional<int64_t> EvalIndexExpr(const PrimExpr &e,
                                     const VarBindings &bindings,
                                     arith::Analyzer *analyzer) {
  PrimExpr bound =
      Substitute(e, [&](const Var &var) -> ffi::Optional<PrimExpr> {
        if (!var->dtype.is_int() && !var->dtype.is_uint()) {
          return ffi::Optional<PrimExpr>(); // not foldable: outside the model
        }
        auto it = bindings.find(var.get());
        int64_t value = it == bindings.end() ? 0 : it->second;
        return IntImm(var->dtype, value);
      });
  PrimExpr folded = analyzer->Simplify(bound);
  if (const auto *imm = folded.as<IntImmNode>()) {
    return imm->value;
  }
  return std::nullopt;
}

/*! \brief One global-memory access stream of a statement. `addr` is the
 *  flat element index into the global buffer, written in the probe's
 *  point_vars — replica-independent by construction. */
struct GlobalAccessProbe {
  PrimExpr addr;
  int64_t elem_bytes{4};
  bool is_store{false};
  int64_t repeat{1}; // enclosing serial trip count: the address pattern
                     // replays (shifted) that many times per step
};

/*! \brief A statement prepared for scoring. Three states, told apart by
 *  two fields (every consumer must honor this protocol):
 *    accesses.empty()   — the statement touches no global memory: charge 0;
 *    !measurable        — geometry outside the model: charge WorstCaseBytes
 *                         (sized by `worst_elements`);
 *    otherwise          — measure with ScoreStatement (which may still
 *                         fail and fall back to WorstCaseBytes). */
struct StatementProbe {
  std::vector<GlobalAccessProbe> accesses;
  bool measurable{false};
  // Forward-walk geometry, valid when `measurable`:
  std::vector<int64_t> extents; // logical iteration space, outermost first
  std::vector<Var> point_vars;  // one per extent; addr/thread/slot
                                // expressions are written in these
  Var rep_var;                  // replica index var of thread_expr
  int64_t rep{1};
  int64_t threads{1};
  int64_t slots{1};     // per-thread serial slots
  PrimExpr thread_expr; // forward thread map: point_vars + rep_var
  PrimExpr slot_expr;   // forward slot map: point_vars
  // Widest vector access in bits the vectorizer will plan for this
  // statement's memory mix (MaxVectorLoadBits — the shared policy).
  int64_t vector_bits{128};
  // Memory-system geometry of the target: how many threads issue one
  // coalesced request together, and the byte granularity each request is
  // charged at.
  int64_t warp_size{32};
  int64_t segment_bytes{128};
  int64_t worst_elements{0};
};

struct StatementTraffic {
  int64_t bw{0};
  int64_t issue{0};
  int64_t Time() const { return std::max(bw, issue); }
};

/*! \brief The conservative charge for a statement outside the model:
 *  every element its own full-segment transaction. An attempt must never
 *  profit from opacity — evaluability depends on the layout under test. */
int64_t WorstCaseBytes(const StatementProbe &probe) {
  int64_t total = 0;
  for (const auto &access : probe.accesses) {
    total += probe.worst_elements * access.repeat * probe.segment_bytes;
  }
  return total;
}

/*! \brief Table cap: statements whose (thread, slot) table would exceed
 *  this fall back to the worst case. Generous — a 128-thread x 128-slot
 *  fragment is 16K entries — while bounding pathological replications. */
constexpr int64_t kMaxTableEntries = int64_t(1) << 20;
constexpr int64_t kAbsentAddr = std::numeric_limits<int64_t>::min();

/*! \brief Measure one prepared statement.
 *
 *  Walk every (logical point, replica) pair, evaluate the forward maps,
 *  and fill the (thread, slot) -> address table; any unevaluable or
 *  out-of-range value, any collision (a non-injective candidate), or a
 *  table over the cap aborts to nullopt and the caller charges the
 *  conservative worst case. From the table, exactly:
 *    vector = widest power-of-two width every thread's aligned slot
 *             blocks sustain contiguously (the vectorizer's question)
 *    issue  = steps x repeat x threads x lane_bytes  (instruction depth)
 *    bw     = repeat x segments x segment_bytes      (transaction bytes)
 *  with segments summed over every (vector step, warp). */
std::optional<StatementTraffic>
ScoreStatementImpl(const StatementProbe &probe) {
  if (probe.accesses.empty()) {
    return StatementTraffic{};
  }
  if (!probe.measurable || probe.threads <= 0 || probe.slots <= 0 ||
      probe.rep <= 0 || probe.warp_size <= 0 || probe.segment_bytes <= 0 ||
      !probe.rep_var.defined()) {
    return std::nullopt;
  }

  int64_t points = 1;
  for (int64_t extent : probe.extents) {
    if (extent <= 0 || points > kMaxTableEntries / extent) {
      return std::nullopt;
    }
    points *= extent;
  }
  int64_t table_size = probe.threads * probe.slots;
  // A fragment the lowering accepts is a bijection between
  // (logical point, replica) and (thread, slot); anything else is
  // outside the model. The size identity is the cheap necessary half;
  // the collision check during the walk is the sufficient half.
  if (table_size > kMaxTableEntries || points > kMaxTableEntries / probe.rep ||
      points * probe.rep != table_size) {
    return std::nullopt;
  }

  size_t naccess = probe.accesses.size();
  std::vector<std::vector<int64_t>> addr_table(
      naccess, std::vector<int64_t>(table_size, kAbsentAddr));
  // Cells whose entry is the lead replica (r == 0): the only lanes that
  // stay active for stores under the replication guard.
  std::vector<uint8_t> lead_replica(table_size, 0);

  arith::Analyzer analyzer;
  VarBindings bindings;
  std::vector<int64_t> point(probe.extents.size(), 0);
  std::vector<int64_t> point_addr(naccess, 0);
  for (int64_t flat = 0; flat < points; ++flat) {
    int64_t rem = flat;
    for (int d = static_cast<int>(probe.extents.size()) - 1; d >= 0; --d) {
      point[d] = rem % probe.extents[d];
      rem /= probe.extents[d];
    }
    for (size_t d = 0; d < probe.point_vars.size(); ++d) {
      bindings[probe.point_vars[d].get()] = point[d];
    }
    bindings[probe.rep_var.get()] = 0;
    // Address and slot are replica-independent: evaluate once per point.
    for (size_t a = 0; a < naccess; ++a) {
      auto addr = EvalIndexExpr(probe.accesses[a].addr, bindings, &analyzer);
      if (!addr) {
        return std::nullopt;
      }
      point_addr[a] = *addr;
    }
    auto slot = EvalIndexExpr(probe.slot_expr, bindings, &analyzer);
    if (!slot || *slot < 0 || *slot >= probe.slots) {
      return std::nullopt;
    }
    for (int64_t r = 0; r < probe.rep; ++r) {
      bindings[probe.rep_var.get()] = r;
      auto thread = EvalIndexExpr(probe.thread_expr, bindings, &analyzer);
      if (!thread || *thread < 0 || *thread >= probe.threads) {
        return std::nullopt;
      }
      int64_t cell = *thread * probe.slots + *slot;
      if (addr_table[0][cell] != kAbsentAddr) {
        return std::nullopt; // collision: candidate is not injective
      }
      for (size_t a = 0; a < naccess; ++a) {
        addr_table[a][cell] = point_addr[a];
      }
      if (r == 0) {
        lead_replica[cell] = 1;
      }
    }
  }
  // points * rep == table_size and no collisions: every cell is filled.

  // Widest power-of-two vector width every access of the statement
  // sustains. This is the numeric mirror of IndicesCanVectorize (the
  // vectorizer's symbolic predicate), checked exhaustively on the table:
  //   - width cap: the shared MaxVectorLoadBits policy per access dtype;
  //   - extent divisibility: slots % cand == 0;
  //   - base alignment: every block base divisible by the width;
  //   - contiguity: slot + 1 advances every address by exactly 1 inside
  //     aligned width-sized blocks.
  int64_t vector_lane_bytes = probe.vector_bits / 8;
  int64_t max_vector = probe.slots;
  for (const auto &access : probe.accesses) {
    max_vector = std::min<int64_t>(max_vector,
                                   vector_lane_bytes /
                                       std::max<int64_t>(1, access.elem_bytes));
  }
  int64_t vector = 1;
  for (int64_t cand = 32; cand >= 2; cand /= 2) {
    if (cand > max_vector || probe.slots % cand != 0) {
      continue;
    }
    bool contiguous = true;
    for (size_t a = 0; a < naccess && contiguous; ++a) {
      for (int64_t t = 0; t < probe.threads && contiguous; ++t) {
        const int64_t *row = addr_table[a].data() + t * probe.slots;
        for (int64_t q = 0; q < probe.slots && contiguous; q += cand) {
          if (row[q] % cand != 0) {
            contiguous = false; // misaligned base: vectorizer would reject
            break;
          }
          for (int64_t offset = 1; offset < cand; ++offset) {
            if (row[q + offset] != row[q] + offset) {
              contiguous = false;
              break;
            }
          }
        }
      }
    }
    if (contiguous) {
      vector = cand;
      break;
    }
  }
  int64_t steps = probe.slots / vector;

  StatementTraffic traffic;
  int64_t num_warps = (probe.threads + probe.warp_size - 1) / probe.warp_size;
  for (size_t a = 0; a < naccess; ++a) {
    const auto &access = probe.accesses[a];
    traffic.issue += steps * access.repeat * probe.threads * vector_lane_bytes;

    int64_t segments_total = 0;
    for (int64_t q = 0; q < steps; ++q) {
      for (int64_t w = 0; w < num_warps; ++w) {
        std::set<int64_t> segments;
        for (int64_t lane = 0; lane < probe.warp_size; ++lane) {
          int64_t t = w * probe.warp_size + lane;
          if (t >= probe.threads) {
            break;
          }
          int64_t cell = t * probe.slots + q * vector;
          if (access.is_store && !lead_replica[cell]) {
            continue; // guarded replica: this lane is idle for stores
          }
          int64_t first_byte = addr_table[a][cell] * access.elem_bytes;
          int64_t last_byte = first_byte + vector * access.elem_bytes - 1;
          for (int64_t seg = first_byte / probe.segment_bytes;
               seg <= last_byte / probe.segment_bytes; ++seg) {
            segments.insert(seg);
          }
        }
        segments_total += static_cast<int64_t>(segments.size());
      }
    }
    traffic.bw += access.repeat * segments_total * probe.segment_bytes;
  }
  return traffic;
}

/*! \brief Exception-safe wrapper: printing/substitution on pathological
 *  candidate layouts can throw deep inside the layout stack; every such
 *  case is simply outside the model. */
std::optional<StatementTraffic> ScoreStatement(const StatementProbe &probe) {
  try {
    return ScoreStatementImpl(probe);
  } catch (const std::exception &e) {
    DLOG(INFO) << "[LayoutCost] statement scoring threw: " << e.what();
    return std::nullopt;
  }
}

/*! \brief Row-major element strides; nullopt when the shape is symbolic
 *  or the buffer exceeds the int32 address algebra. */
std::optional<std::vector<int64_t>> RowMajorStrides(const Buffer &buffer) {
  size_t ndim = buffer->shape.size();
  std::vector<int64_t> strides(ndim, 1);
  for (int d = static_cast<int>(ndim) - 2; d >= 0; --d) {
    const int64_t *extent = as_const_int(buffer->shape[d + 1]);
    if (!extent) {
      return std::nullopt;
    }
    strides[d] = strides[d + 1] * *extent;
  }
  const int64_t *outer = as_const_int(buffer->shape[0]);
  if (!outer || strides[0] * *outer >= (int64_t(1) << 31)) {
    return std::nullopt;
  }
  return strides;
}

/*! \brief Bind the target's memory-system geometry on a probe. The warp
 *  size comes from the target (32 CUDA lanes, 64 ROCm wavefront lanes);
 *  the segment granularity stays at 128B — the L1-line coalescing unit on
 *  NVIDIA and a serviceable approximation elsewhere — until calibration
 *  demands a per-target dispatch here. */
void BindMemoryGeometry(StatementProbe *probe, const Target &target) {
  probe->warp_size = target->GetAttr<Integer>("thread_warp_size", Integer(32))
                         .value()
                         .IntValue();
  probe->segment_bytes = 128;
}

/*! \brief Fresh point vars for a probe's logical space. */
std::vector<Var> MakePointVars(size_t ndim) {
  std::vector<Var> vars;
  vars.reserve(ndim);
  for (size_t d = 0; d < ndim; ++d) {
    vars.push_back(Var("_cost_x" + std::to_string(d), DataType::Int(32)));
  }
  return vars;
}

/*! \brief Install a fragment's forward maps on a probe: point_vars must
 *  already be set. Returns false (probe untouched otherwise) when the
 *  fragment's geometry is not constant-sized. */
bool BindForwardMaps(StatementProbe *probe, const Fragment &layout) {
  if (layout->OutputDim() != 1) {
    return false;
  }
  // Keep the expressions alive: as_const_int returns a pointer into the
  // node, and OutputShape()/ThreadExtent() build fresh temporaries.
  PrimExpr slots_expr = layout->OutputShape()[0];
  PrimExpr threads_expr = layout->ThreadExtent();
  PrimExpr rep_expr = layout->ReplicateExtent();
  const int64_t *slots_ptr = as_const_int(slots_expr);
  const int64_t *threads_ptr = as_const_int(threads_expr);
  const int64_t *rep_ptr = as_const_int(rep_expr);
  if (!slots_ptr || !threads_ptr || !rep_ptr || *slots_ptr <= 0 ||
      *threads_ptr <= 0 || *rep_ptr <= 0) {
    return false;
  }
  Array<PrimExpr> points;
  for (const Var &v : probe->point_vars) {
    points.push_back(v);
  }
  probe->rep_var = Var("_cost_rep", DataType::Int(32));
  probe->thread_expr = layout->ForwardThread(points, probe->rep_var);
  probe->slot_expr = layout->Forward(points)[0];
  probe->slots = *slots_ptr;
  probe->threads = *threads_ptr;
  probe->rep = *rep_ptr;
  return true;
}

/*! \brief Prepare a fragment<->global tl.copy for scoring: the logical
 *  walk is the fragment's full logical shape, the thread/slot maps are
 *  the fragment layout's own forward expressions, and the single access
 *  is the global side's affine address in the same logical point. */
StatementProbe BuildCopyProbe(const CopyNode *copy, const Fragment &frag_layout,
                              bool frag_is_src, const Target &target) {
  const Buffer &global = frag_is_src ? copy->dst : copy->src;
  const Array<Range> &frag_range =
      frag_is_src ? copy->src_range : copy->dst_range;
  const Array<Range> &global_range =
      frag_is_src ? copy->dst_range : copy->src_range;

  // Sized worst-case fallback for every shape outside the model: logical
  // elements from the fragment-side range (symbolic extents count as 1 —
  // worst geometry, optimistic count).
  auto worst_only = [&]() {
    StatementProbe probe;
    BindMemoryGeometry(&probe, target);
    probe.worst_elements = 1;
    for (const Range &r : frag_range) {
      const int64_t *extent = as_const_int(r->extent);
      probe.worst_elements *= extent ? *extent : 1;
    }
    GlobalAccessProbe access;
    access.is_store = frag_is_src;
    probe.accesses.push_back(std::move(access));
    return probe;
  };

  size_t ndim = frag_layout->InputShape().size();
  if (frag_range.size() != ndim || global_range.size() != ndim) {
    return worst_only();
  }
  // Whole-fragment copies only: the iteration space is the fragment's full
  // logical shape.
  std::vector<int64_t> extents;
  int64_t logical_elements = 1;
  for (size_t d = 0; d < ndim; ++d) {
    if (!is_zero(frag_range[d]->min)) {
      return worst_only();
    }
    const int64_t *frag_extent = as_const_int(frag_range[d]->extent);
    const int64_t *global_extent = as_const_int(global_range[d]->extent);
    const int64_t *shape_extent = as_const_int(frag_layout->InputShape()[d]);
    if (!frag_extent || !global_extent || !shape_extent ||
        *frag_extent != *shape_extent || *global_extent != *shape_extent) {
      return worst_only();
    }
    extents.push_back(*shape_extent);
    logical_elements *= *shape_extent;
  }

  int64_t elem_bits = global->dtype.bits() * global->dtype.lanes();
  if (elem_bits < 8) {
    return worst_only();
  }
  auto strides = RowMajorStrides(global);
  if (!strides.has_value()) {
    return worst_only();
  }

  StatementProbe probe;
  probe.worst_elements = logical_elements;
  probe.extents = std::move(extents);
  probe.point_vars = MakePointVars(ndim);
  if (!BindForwardMaps(&probe, frag_layout)) {
    return worst_only();
  }
  // A fragment<->global copy touches global memory and no shared memory.
  probe.vector_bits = MaxVectorLoadBits(target, /*global_only_access=*/true);
  BindMemoryGeometry(&probe, target);

  PrimExpr addr = make_zero(DataType::Int(32));
  for (size_t d = 0; d < ndim; ++d) {
    addr = addr + (global_range[d]->min + probe.point_vars[d]) *
                      IntImm(DataType::Int(32), (*strides)[d]);
  }
  GlobalAccessProbe access;
  access.addr = std::move(addr);
  access.elem_bytes = elem_bits / 8;
  access.is_store = frag_is_src;
  probe.accesses.push_back(std::move(access));
  probe.measurable = true;
  return probe;
}

/*! \brief Collect the direct global-memory accesses of a parallel loop
 *  body, with the trip count of any enclosing serial loops (the access
 *  pattern replays, shifted, once per serial iteration). */
class LoopGlobalAccessCollector : public StmtExprVisitor {
public:
  struct RawAccess {
    Buffer buffer;
    Array<PrimExpr> indices;
    bool is_store;
    int64_t repeat;
    bool symbolic_repeat;
  };
  std::vector<RawAccess> accesses;
  // Whether the loop body also touches shared memory: feeds the shared
  // width-cap policy (256-bit loads are global-only).
  bool touches_shared{false};

  void Collect(const Stmt &stmt) { VisitStmt(stmt); }

private:
  void VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kParallel) {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    const int64_t *extent = as_const_int(op->extent);
    serial_stack_.push_back(extent ? *extent : -1);
    StmtExprVisitor::VisitStmt_(op);
    serial_stack_.pop_back();
  }
  void VisitStmt_(const BufferStoreNode *op) final {
    Record(op->buffer, op->indices, /*is_store=*/true);
    StmtExprVisitor::VisitStmt_(op);
  }
  void VisitExpr_(const BufferLoadNode *op) final {
    Record(op->buffer, op->indices, /*is_store=*/false);
    StmtExprVisitor::VisitExpr_(op);
  }
  void Record(const Buffer &buffer, const Array<PrimExpr> &indices,
              bool is_store) {
    if (IsSharedBuffer(buffer)) {
      touches_shared = true;
    }
    if (!IsGlobalBuffer(buffer)) {
      return;
    }
    int64_t repeat = 1;
    bool symbolic = false;
    for (int64_t extent : serial_stack_) {
      if (extent < 0) {
        symbolic = true;
      } else {
        repeat *= extent;
      }
    }
    accesses.push_back(RawAccess{buffer, indices, is_store, repeat, symbolic});
  }
  std::vector<int64_t> serial_stack_;
};

/*! \brief Prepare a parallel loop with direct global accesses for scoring
 *  (issue #1729, extension E1): the logical walk is the loop nest itself,
 *  the thread/slot maps are the loop layout's forward expressions bound
 *  to the loop vars, and each access keeps its raw index expressions —
 *  they are already written in the loop vars, so no substitution at all.
 *  Returns: probe with empty accesses (charge zero) when the loop touches
 *  no global memory; a full probe otherwise; nullopt when the loop cannot
 *  even be sized (skip — nothing sensible to charge). */
std::optional<StatementProbe> BuildLoopProbe(const ParallelOpNode *loop,
                                             const Target &target) {
  LoopGlobalAccessCollector collector;
  collector.Collect(loop->GetRoot());
  StatementProbe probe;
  if (collector.accesses.empty()) {
    return probe; // pure fragment/shared loop: no global traffic to model
  }

  // Loop nest geometry, taken from the For nest so it exists even when
  // the loop has no layout: an attempt must never profit from an unsolved
  // (opaque) loop.
  std::vector<int64_t> extents;
  std::vector<Var> nest_vars;
  int64_t domain = 1;
  {
    const ForNode *cur = loop->GetRoot().get();
    while (cur != nullptr && cur->kind == ForKind::kParallel) {
      const int64_t *value = as_const_int(cur->extent);
      if (!value) {
        return std::nullopt; // truly unsizeable: nothing sensible to charge
      }
      extents.push_back(*value);
      nest_vars.push_back(cur->loop_var);
      domain *= *value;
      cur = cur->body.as<ForNode>();
    }
  }
  probe.worst_elements = domain;

  // Mark the probe un-scoreable but sized (worst-case) when any structural
  // requirement fails. Symbolic serial trip counts fall back to repeat=1:
  // worst geometry, optimistic count.
  auto worst_only = [&]() -> std::optional<StatementProbe> {
    StatementProbe worst;
    BindMemoryGeometry(&worst, target);
    worst.worst_elements = probe.worst_elements;
    for (const auto &raw : collector.accesses) {
      GlobalAccessProbe access;
      access.elem_bytes = std::max<int64_t>(
          1, raw.buffer->dtype.bits() * raw.buffer->dtype.lanes() / 8);
      access.is_store = raw.is_store;
      access.repeat = raw.symbolic_repeat ? 1 : raw.repeat;
      worst.accesses.push_back(std::move(access));
    }
    return worst;
  };

  Fragment layout = loop->GetLoopLayout();
  if (!layout.defined()) {
    DLOG(INFO) << "[LayoutCost] loop layout undefined; charged worst-case";
    return worst_only();
  }
  if (nest_vars.size() != layout->InputShape().size()) {
    return worst_only();
  }
  probe.extents = std::move(extents);
  probe.point_vars = std::move(nest_vars);
  if (!BindForwardMaps(&probe, layout)) {
    return worst_only();
  }
  probe.vector_bits = MaxVectorLoadBits(
      target, /*global_only_access=*/!collector.touches_shared);
  BindMemoryGeometry(&probe, target);

  for (const auto &raw : collector.accesses) {
    if (raw.symbolic_repeat) {
      return worst_only();
    }
    int64_t elem_bits = raw.buffer->dtype.bits() * raw.buffer->dtype.lanes();
    auto strides = RowMajorStrides(raw.buffer);
    if (elem_bits < 8 || !strides.has_value() ||
        raw.indices.size() != raw.buffer->shape.size()) {
      return worst_only();
    }
    PrimExpr addr = make_zero(DataType::Int(32));
    for (size_t d = 0; d < raw.indices.size(); ++d) {
      addr = addr + raw.indices[d] * IntImm(DataType::Int(32), (*strides)[d]);
    }
    GlobalAccessProbe access;
    access.addr = std::move(addr);
    access.elem_bytes = elem_bits / 8;
    access.is_store = raw.is_store;
    access.repeat = raw.repeat;
    probe.accesses.push_back(std::move(access));
  }
  probe.measurable = true;
  return probe;
}

/*! \brief Total per-thread register slots across every fragment layout of
 *  the attempt. Shared by every cost model: it is the legacy score and the
 *  IO-aware model's tiebreak. */
int64_t CountRegisterSlots(const LayoutMap &tmp_layout_map) {
  int64_t regs = 0;
  for (const auto &[buffer, layout] : tmp_layout_map) {
    if (auto frag = layout.as<Fragment>()) {
      int64_t frag_reg_num = 1;
      for (auto i : frag.value()->OutputShape()) {
        auto pci = as_const_int(i);
        ICHECK(pci != nullptr) << "Can not use non-constant range to "
                                  "iterate over a fragment/local "
                                  "buffer. Non-constant shape expr is: "
                               << i
                               << ". This is possibly because you use "
                                  "symbolic shape when "
                                  "accessing a fragment/local buffer."
                               << SpanHintSuffix(buffer->span);
        frag_reg_num *= *pci;
      }
      regs += frag_reg_num;
    }
  }
  return regs;
}

/*! \brief Legacy policy: total register slots, nothing else. `mem` stays 0
 *  so the ordering is byte-identical to the historical register-count
 *  selection. */
class RegisterCountCostModel final : public LayoutCostModel {
public:
  AttemptCost Score(const std::vector<int> &members,
                    const std::vector<TileOperator> &infer_list,
                    const LayoutMap &tmp_layout_map) const final {
    (void)members;
    (void)infer_list;
    AttemptCost cost;
    cost.regs = CountRegisterSlots(tmp_layout_map);
    return cost;
  }
  const char *Name() const final { return "register-count"; }
};

/*! \brief IO-aware policy (layout RFC, design B2): every global-touching
 *  statement is charged max(bandwidth bytes, issue-equivalent bytes) under
 *  the attempt's tentative layouts; statements outside the model are
 *  charged a conservative worst case (an attempt must never profit from
 *  opacity). Registers remain the lexicographic tiebreak. */
class IOAwareCostModel final : public LayoutCostModel {
public:
  explicit IOAwareCostModel(Target target) : target_(std::move(target)) {}

  AttemptCost Score(const std::vector<int> &members,
                    const std::vector<TileOperator> &infer_list,
                    const LayoutMap &tmp_layout_map) const final {
    AttemptCost cost;
    cost.regs = CountRegisterSlots(tmp_layout_map);

    auto charge = [&](const std::optional<StatementProbe> &probe,
                      const char *what) {
      if (!probe.has_value() || probe->accesses.empty()) {
        return; // no global traffic to model / nothing sensible to charge
      }
      std::optional<StatementTraffic> traffic;
      if (probe->measurable) {
        traffic = ScoreStatement(*probe);
      }
      if (traffic.has_value()) {
        cost.mem += traffic->Time();
        DLOG(INFO) << "[LayoutCost] " << what << ": bw=" << traffic->bw
                   << " issue=" << traffic->issue;
      } else {
        cost.mem += WorstCaseBytes(*probe);
        DLOG(INFO) << "[LayoutCost] " << what
                   << " outside the model; charged worst-case.";
      }
    };

    for (int idx : members) {
      if (const auto *copy = infer_list[idx].as<CopyNode>()) {
        bool src_frag = IsFragmentBuffer(copy->src);
        bool dst_frag = IsFragmentBuffer(copy->dst);
        Buffer frag;
        bool frag_is_src = false;
        if (src_frag && IsGlobalBuffer(copy->dst)) {
          frag = copy->src;
          frag_is_src = true;
        } else if (dst_frag && IsGlobalBuffer(copy->src)) {
          frag = copy->dst;
        } else {
          continue; // register moves / shared staging: out of the model
        }
        auto layout = tmp_layout_map.Get(frag);
        if (!layout.has_value()) {
          continue;
        }
        auto frag_layout = layout.value().as<Fragment>();
        if (!frag_layout.has_value()) {
          continue;
        }
        std::optional<StatementProbe> probe;
        try {
          probe =
              BuildCopyProbe(copy, frag_layout.value(), frag_is_src, target_);
        } catch (const std::exception &e) {
          probe = std::nullopt; // charge() skips; builder-side fallbacks
                                // cover every non-throwing failure
        }
        charge(probe, "copy");
      } else if (const auto *loop = infer_list[idx].as<ParallelOpNode>()) {
        std::optional<StatementProbe> probe;
        try {
          probe = BuildLoopProbe(loop, target_);
        } catch (const std::exception &e) {
          probe = std::nullopt;
        }
        charge(probe, "parallel loop");
      }
    }
    return cost;
  }

  const char *Name() const final { return "io-aware"; }

private:
  Target target_;
};

} // namespace

std::unique_ptr<LayoutCostModel> LayoutCostModel::Create(bool io_aware,
                                                         Target target) {
  if (io_aware) {
    return std::make_unique<IOAwareCostModel>(std::move(target));
  }
  return std::make_unique<RegisterCountCostModel>();
}

} // namespace tl
} // namespace tvm

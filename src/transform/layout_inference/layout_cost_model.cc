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
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../config.h"
#include "../../layout/cute_layout.h"
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

using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

template <typename T> std::string FormatVector(const std::vector<T> &values) {
  std::ostringstream os;
  os << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << values[i];
  }
  os << ']';
  return os.str();
}

// ---------------------------------------------------------------------------
// Statement-level bottleneck traffic model (layout RFC, design B2)
//
// The unit of account is one STATEMENT that touches global memory — a
// fragment<->global tl.copy or a parallel loop with direct global accesses.
// The statement is scored SYMBOLICALLY on the in-tree CuTe layout algebra
// (src/layout/cute_layout.h). The forward maps are packed into one plain
// multi-output layout whose row-major output serialization is exactly the
// physical cell index `thread * slots + slot`; LayoutFromTileLang recovers
// its (shape, stride) normal form (probe-then-prove, so the conversion is
// self-certifying), RightInverse + Composition derive the per-cell address
// layout, and the questions become mode arithmetic:
//     vector = the innermost stride-1 run of the coalesced slot axis,
//              alignment-checked against every other mode stride (the
//              vectorizer's question, answered on the normal form)
//     bw     = coalescing segments touched, counted exactly per (vector
//              step, warp) by evaluating the DERIVED layout once per issued
//              vector lane rather than once per logical point and replica.
//              Intra-warp broadcast merges into one segment; store lanes
//              holding a replica != 0 sleep behind the replication guard
//              (the replica index is read back through the inverse) and
//              count nothing.
//     issue  = per-thread instruction depth x what a fully busy block
//              would stream at max lane width over that many steps:
//              steps x threads x lane_bytes. Idle lanes do not shorten
//              the depth, so thread-collapse pathologies surface here
//              (#1729).
//     time(S) ~ max(bw, issue);      cost = sum over statements.
//
// Anything the algebra cannot express (non-affine indices, swizzle,
// non-bijective candidates) is charged the conservative worst case — an
// attempt must never profit from opacity. The retired exact enumerator
// survives as a debug oracle behind `tl.layout_cost_model_verify`
// (VerifyByEnumeration), rebuilt from the same derived layouts.
// Hardware geometry stays parameterized: lane width from the vectorizer's
// shared MaxVectorLoadBits policy, warp size and coalescing-segment
// granularity from the target (see BindMemoryGeometry).
// ---------------------------------------------------------------------------

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

void LogProbe(int member_idx, const char *what, const StatementProbe &probe) {
  const char *state = probe.accesses.empty()
                          ? "no-global-access"
                          : (probe.measurable ? "measurable" : "worst-case");
  DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
             << " probe: state=" << state
             << " accesses=" << probe.accesses.size()
             << " extents=" << FormatVector(probe.extents)
             << " worst_elements=" << probe.worst_elements
             << " threads=" << probe.threads << " slots=" << probe.slots
             << " replicas=" << probe.rep
             << " vector_bits=" << probe.vector_bits
             << " warp_size=" << probe.warp_size
             << " segment_bytes=" << probe.segment_bytes;
  if (probe.measurable) {
    DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
               << " maps: thread=" << probe.thread_expr
               << " slot=" << probe.slot_expr;
  }
  for (size_t i = 0; i < probe.accesses.size(); ++i) {
    const GlobalAccessProbe &access = probe.accesses[i];
    std::ostringstream os;
    os << "[LayoutCost] member " << member_idx << ' ' << what << " access[" << i
       << "]: kind=" << (access.is_store ? "store" : "load")
       << " elem_bytes=" << access.elem_bytes << " repeat=" << access.repeat
       << " addr=";
    if (access.addr.defined()) {
      os << access.addr;
    } else {
      os << "<unavailable>";
    }
    DLOG(INFO) << os.str();
  }
}

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

/*! \brief Entry cap for the optional exact-enumeration debug oracle. The
 *  production CuTe path does not materialize a table and is not subject to
 *  this limit. */
constexpr int64_t kMaxOracleEntries = int64_t(1) << 20;

bool TryMultiplyPositive(int64_t lhs, int64_t rhs, int64_t *product) {
  if (lhs <= 0 || rhs <= 0 || lhs > std::numeric_limits<int64_t>::max() / rhs) {
    return false;
  }
  *product = lhs * rhs;
  return true;
}

/*! \brief Zero out vars outside the probe's own coordinate set: block
 *  indices inside region offsets shift every address of the statement
 *  equally and cancel out of both contiguity and segment geometry.
 *  Non-int foreign vars are kept and fail the affine recovery, marking the
 *  statement outside the model. */
PrimExpr ZeroForeignVars(const PrimExpr &e, const VarSet &own) {
  return Substitute(e, [&](const Var &var) -> ffi::Optional<PrimExpr> {
    if (own.count(var) || (!var->dtype.is_int() && !var->dtype.is_uint())) {
      return ffi::Optional<PrimExpr>();
    }
    return make_zero(var->dtype);
  });
}

/*! \brief The probe's iteration space as IterVars: (point_vars..., rep) —
 *  the canonical packing of FragmentNode::InverseWithLevel, replication as
 *  a trailing ordinary dimension. */
Array<IterVar> ProbeIterVars(const StatementProbe &probe) {
  Array<IterVar> ivs;
  for (size_t d = 0; d < probe.point_vars.size(); ++d) {
    ivs.push_back(IterVar(Range(IntImm(DataType::Int(32), 0),
                                IntImm(DataType::Int(32), probe.extents[d])),
                          probe.point_vars[d], IterVarType::kDataPar));
  }
  ivs.push_back(IterVar(
      Range(IntImm(DataType::Int(32), 0), IntImm(DataType::Int(32), probe.rep)),
      probe.rep_var, IterVarType::kDataPar));
  return ivs;
}

/*! \brief Recover expressions over the probe's iteration space as ONE
 *  plain strided CuTe layout. Multi-output layouts are serialized
 *  row-major by the recovery probe, so outputs [thread, slot] yield the
 *  physical cell index `thread * slots + slot` directly. Nullopt when the
 *  expressions are not affine-recoverable (the conversion proves its own
 *  equivalence, so a wrong recovery cannot slip through). */
Optional<cute::Layout> ProbeExprsToCute(const StatementProbe &probe,
                                        Array<PrimExpr> outputs,
                                        const VarSet &own) {
  outputs =
      outputs.Map([&](const PrimExpr &e) { return ZeroForeignVars(e, own); });
  Layout packed(ProbeIterVars(probe), outputs);
  return cute::LayoutFromTileLang(packed);
}

/*! \brief Evaluate a plain strided CuTe layout at a linear coordinate. */
int64_t EvalCute(const cute::Layout &layout, int64_t coord) {
  return cute::AsConst(layout(cute::IntTuple(coord)));
}

/*! \brief CuTe spelling of a layout, for diagnostics. */
std::string CuteToString(const cute::Layout &layout) {
  std::ostringstream os;
  layout.Print(os);
  return os.str();
}

/*! \brief Coalesced (extent, stride) mode pairs, innermost first; nullopt
 *  when any leaf is not a constant (dynamic strides are outside the
 *  model). */
std::optional<std::vector<std::pair<int64_t, int64_t>>>
FlatModes(const cute::Layout &layout) {
  cute::Layout lay = cute::Coalesce(layout);
  cute::IntTuple shape = cute::Flatten(lay->shape);
  cute::IntTuple stride = cute::Flatten(lay->stride);
  Array<cute::IntTuple> shapes = cute::Wrap(shape)->fields;
  Array<cute::IntTuple> strides = cute::Wrap(stride)->fields;
  if (shapes.size() != strides.size()) {
    return std::nullopt;
  }
  std::vector<std::pair<int64_t, int64_t>> modes;
  modes.reserve(shapes.size());
  for (size_t i = 0; i < shapes.size(); ++i) {
    if (!cute::IsConst(shapes[i]) || !cute::IsConst(strides[i])) {
      return std::nullopt;
    }
    modes.emplace_back(cute::AsConst(shapes[i]), cute::AsConst(strides[i]));
  }
  return modes;
}

/*! \brief Debug oracle behind `tl.layout_cost_model_verify`: rebuild the
 *  full (thread, slot) -> address table by evaluating the DERIVED layouts
 *  at every cell, and recompute vector width and segment counts with the
 *  retired exact-enumeration formulas. Validates the mode arithmetic (the
 *  conversion already proves itself); a disagreement warns loudly and the
 *  caller falls back to the conservative worst case. */
bool VerifyByEnumeration(const StatementProbe &probe, const cute::Layout &inv,
                         const std::vector<cute::Layout> &addr_layouts,
                         int64_t points, int64_t table_size, int64_t vector,
                         const StatementTraffic &traffic, int member_idx,
                         const char *what) {
  if (table_size > kMaxOracleEntries) {
    LOG(WARNING) << "[LayoutCost] member " << member_idx << ' ' << what
                 << " enumeration oracle needs " << table_size
                 << " entries, above its " << kMaxOracleEntries
                 << "-entry safety cap; charging the conservative worst case.";
    return false;
  }
  size_t naccess = probe.accesses.size();
  std::vector<std::vector<int64_t>> addr_table(
      naccess, std::vector<int64_t>(table_size, 0));
  std::vector<uint8_t> lead_replica(table_size, 0);
  for (int64_t cell = 0; cell < table_size; ++cell) {
    lead_replica[cell] = EvalCute(inv, cell) / points == 0;
    for (size_t a = 0; a < naccess; ++a) {
      addr_table[a][cell] = EvalCute(addr_layouts[a], cell);
    }
  }

  // Retired enumerator formulas, verbatim.
  int64_t vector_lane_bytes = probe.vector_bits / 8;
  int64_t max_vector = probe.slots;
  for (const auto &access : probe.accesses) {
    max_vector = std::min<int64_t>(max_vector,
                                   vector_lane_bytes /
                                       std::max<int64_t>(1, access.elem_bytes));
  }
  int64_t enum_vector = 1;
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
            contiguous = false;
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
      enum_vector = cand;
      break;
    }
  }
  int64_t steps = probe.slots / enum_vector;

  StatementTraffic enum_traffic;
  int64_t num_warps = (probe.threads + probe.warp_size - 1) / probe.warp_size;
  std::vector<int64_t> segments;
  segments.reserve(2 * probe.warp_size);
  for (size_t a = 0; a < naccess; ++a) {
    const auto &access = probe.accesses[a];
    enum_traffic.issue +=
        steps * access.repeat * probe.threads * vector_lane_bytes;
    int64_t segments_total = 0;
    for (int64_t q = 0; q < steps; ++q) {
      for (int64_t w = 0; w < num_warps; ++w) {
        segments.clear();
        for (int64_t lane = 0; lane < probe.warp_size; ++lane) {
          int64_t t = w * probe.warp_size + lane;
          if (t >= probe.threads) {
            break;
          }
          int64_t cell = t * probe.slots + q * enum_vector;
          if (access.is_store && !lead_replica[cell]) {
            continue;
          }
          int64_t first_byte = addr_table[a][cell] * access.elem_bytes;
          int64_t last_byte = first_byte + enum_vector * access.elem_bytes - 1;
          for (int64_t seg = first_byte / probe.segment_bytes;
               seg <= last_byte / probe.segment_bytes; ++seg) {
            if (std::find(segments.begin(), segments.end(), seg) ==
                segments.end()) {
              segments.push_back(seg);
            }
          }
        }
        segments_total += static_cast<int64_t>(segments.size());
      }
    }
    enum_traffic.bw += access.repeat * segments_total * probe.segment_bytes;
  }

  if (enum_vector != vector || enum_traffic.bw != traffic.bw ||
      enum_traffic.issue != traffic.issue) {
    LOG(WARNING) << "[LayoutCost] member " << member_idx << ' ' << what
                 << " CuTe scoring disagrees with the enumeration oracle: "
                 << "cute(vector=" << vector << " bw=" << traffic.bw
                 << " issue=" << traffic.issue
                 << ") vs enum(vector=" << enum_vector
                 << " bw=" << enum_traffic.bw << " issue=" << enum_traffic.issue
                 << "); charging the conservative worst case.";
    return false;
  }
  return true;
}

/*! \brief Score one prepared statement on the CuTe layout algebra.
 *
 *  Pack (coords, rep) -> [thread, slot] and recover it as the plain
 *  strided cell layout; RightInverse (bijectivity checked by size) and
 *  per-access Composition derive `cell -> element address` layouts. The
 *  vector width is read off the coalesced (slot, thread) mode split; the
 *  segment count evaluates the derived layouts at (step, warp, lane)
 *  granularity only. Every failure — non-affine expressions, dynamic
 *  modes, algebra ICHECKs (caught by the ScoreStatement wrapper) — lands
 *  on nullopt and the caller charges the conservative worst case. */
std::optional<StatementTraffic> ScoreStatementImpl(const StatementProbe &probe,
                                                   int member_idx,
                                                   const char *what) {
  if (probe.accesses.empty()) {
    return StatementTraffic{};
  }
  if (!probe.measurable || probe.threads <= 0 || probe.slots <= 0 ||
      probe.rep <= 0 || probe.warp_size <= 0 || probe.segment_bytes <= 0 ||
      !probe.rep_var.defined()) {
    DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
               << " cannot be measured: invalid probe geometry";
    return std::nullopt;
  }

  int64_t points = 1;
  for (int64_t extent : probe.extents) {
    if (!TryMultiplyPositive(points, extent, &points)) {
      DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
                 << " cannot be measured: invalid or overflowing logical "
                    "domain at extent="
                 << extent;
      return std::nullopt;
    }
  }
  int64_t table_size = 0;
  int64_t logical_size = 0;
  // A fragment the lowering accepts is a bijection between
  // (logical point, replica) and (thread, slot); anything else is
  // outside the model. The size identity is the cheap necessary half;
  // the RightInverse size check below is the sufficient half.
  if (!TryMultiplyPositive(probe.threads, probe.slots, &table_size) ||
      !TryMultiplyPositive(points, probe.rep, &logical_size) ||
      logical_size != table_size) {
    DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
               << " cannot be measured: forward map is not a bounded "
                  "logical-to-(thread,slot) bijection";
    return std::nullopt;
  }

  VarSet own;
  for (const Var &v : probe.point_vars) {
    own.insert(v);
  }
  own.insert(probe.rep_var);

  // Recover the packed forward map as the plain strided cell layout.
  Optional<cute::Layout> flat =
      ProbeExprsToCute(probe, {probe.thread_expr, probe.slot_expr}, own);
  if (!flat.defined()) {
    DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
               << " cannot be measured: forward map is not affine-"
                  "recoverable (thread="
               << probe.thread_expr << " slot=" << probe.slot_expr << ")";
    return std::nullopt;
  }
  cute::Layout inv = cute::RightInverse(flat.value());
  if (cute::AsConst(cute::Size(inv)) != table_size) {
    DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
               << " cannot be measured: candidate is not injective "
                  "(right-inverse covers "
               << cute::AsConst(cute::Size(inv)) << " of " << table_size
               << " cells)";
    return std::nullopt;
  }

  // Derive one `cell -> element address` layout per access and its
  // (slot, thread) mode split.
  cute::IntTuple st_shape = cute::IntTupleTuple(
      {cute::IntTuple(probe.slots), cute::IntTuple(probe.threads)});
  size_t naccess = probe.accesses.size();
  std::vector<cute::Layout> addr_layouts;
  std::vector<std::vector<std::pair<int64_t, int64_t>>> slot_modes;
  std::vector<std::vector<std::pair<int64_t, int64_t>>> thread_modes;
  addr_layouts.reserve(naccess);
  for (size_t a = 0; a < naccess; ++a) {
    const auto &access = probe.accesses[a];
    if (access.elem_bytes <= 0 ||
        probe.segment_bytes % access.elem_bytes != 0) {
      DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
                 << " cannot be measured: element size does not divide the "
                    "segment granularity";
      return std::nullopt;
    }
    Optional<cute::Layout> g = ProbeExprsToCute(probe, {access.addr}, own);
    if (!g.defined()) {
      DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
                 << " has a non-affine address expression; charged "
                    "worst-case: "
                 << access.addr;
      return std::nullopt;
    }
    cute::Layout addr = cute::Composition(g.value(), inv);
    cute::Layout split = addr.WithShape(st_shape);
    auto smodes = FlatModes(split[0]);
    auto tmodes = FlatModes(split[1]);
    if (!smodes || !tmodes) {
      DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
                 << " cannot be measured: derived address modes are not "
                    "constant";
      return std::nullopt;
    }
    DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
               << " access[" << a
               << "] address layout: " << CuteToString(split);
    addr_layouts.push_back(std::move(addr));
    slot_modes.push_back(std::move(*smodes));
    thread_modes.push_back(std::move(*tmodes));
  }

  // Widest power-of-two vector width every access sustains, read off the
  // mode decomposition (the vectorizer's question on the normal form):
  //   - width cap: the shared MaxVectorLoadBits policy per access dtype;
  //   - extent divisibility: slots % cand == 0;
  //   - contiguity: the innermost slot mode must be a stride-1 run whose
  //     extent the width divides;
  //   - base alignment: every other nonzero mode stride (higher slot
  //     modes and all thread modes) divisible by the width.
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
    bool sustained = true;
    for (size_t a = 0; a < naccess && sustained; ++a) {
      const auto &smodes = slot_modes[a];
      int64_t run =
          (!smodes.empty() && smodes[0].second == 1) ? smodes[0].first : 1;
      if (run % cand != 0) {
        sustained = false;
        break;
      }
      for (size_t m = 1; m < smodes.size() && sustained; ++m) {
        if (smodes[m].second != 0 && smodes[m].second % cand != 0) {
          sustained = false;
        }
      }
      for (const auto &mode : thread_modes[a]) {
        if (mode.second != 0 && mode.second % cand != 0) {
          sustained = false;
          break;
        }
      }
    }
    if (sustained) {
      vector = cand;
      break;
    }
  }
  int64_t steps = probe.slots / vector;
  DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
             << " vectorization: vector_lanes=" << vector
             << " issue_bytes_per_thread_step=" << vector_lane_bytes
             << " steps_per_thread=" << steps;

  // Segments per (vector step, warp): evaluate the derived address layout
  // at warp/step granularity — steps x warps x warp_size points, bounded
  // by the machine shape. The replica index of a cell is read back through
  // the inverse (rep is the slowest packed input, so it is the flat
  // logical index divided by the logical point count).
  StatementTraffic traffic;
  int64_t num_warps = (probe.threads + probe.warp_size - 1) / probe.warp_size;
  std::vector<int64_t> segments;
  segments.reserve(2 * probe.warp_size);
  for (size_t a = 0; a < naccess; ++a) {
    const auto &access = probe.accesses[a];
    int64_t issue_contribution =
        steps * access.repeat * probe.threads * vector_lane_bytes;
    traffic.issue += issue_contribution;

    int64_t seg_elems = probe.segment_bytes / access.elem_bytes;
    int64_t segments_total = 0;
    for (int64_t q = 0; q < steps; ++q) {
      for (int64_t w = 0; w < num_warps; ++w) {
        segments.clear();
        for (int64_t lane = 0; lane < probe.warp_size; ++lane) {
          int64_t t = w * probe.warp_size + lane;
          if (t >= probe.threads) {
            break;
          }
          int64_t cell = t * probe.slots + q * vector;
          if (access.is_store && probe.rep > 1 &&
              EvalCute(inv, cell) / points != 0) {
            continue; // guarded replica: this lane is idle for stores
          }
          int64_t first = EvalCute(addr_layouts[a], cell);
          int64_t last = first + vector - 1;
          for (int64_t seg = first / seg_elems; seg <= last / seg_elems;
               ++seg) {
            if (std::find(segments.begin(), segments.end(), seg) ==
                segments.end()) {
              segments.push_back(seg);
            }
          }
        }
        segments_total += static_cast<int64_t>(segments.size());
      }
    }
    int64_t bw_contribution =
        access.repeat * segments_total * probe.segment_bytes;
    traffic.bw += bw_contribution;
    DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
               << " access[" << a
               << "] contribution: segments=" << segments_total
               << " bw=" << bw_contribution << " issue=" << issue_contribution;
  }

  if (tl_config::LayoutCostModelVerifyEnabled() &&
      !VerifyByEnumeration(probe, inv, addr_layouts, points, table_size, vector,
                           traffic, member_idx, what)) {
    return std::nullopt;
  }
  return traffic;
}

/*! \brief Exception-safe wrapper: printing/substitution on pathological
 *  candidate layouts can throw deep inside the layout stack; every such
 *  case is simply outside the model. */
std::optional<StatementTraffic>
ScoreStatement(const StatementProbe &probe, int member_idx, const char *what) {
  try {
    return ScoreStatementImpl(probe, member_idx, what);
  } catch (const std::exception &e) {
    DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
               << " scoring threw; charged worst-case: " << e.what();
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
StatementProbe BuildCopyProbe(const Copy &copy, const Fragment &frag_layout,
                              bool frag_is_src, const Target &target) {
  const Buffer &global = frag_is_src ? copy->dst : copy->src;
  const Array<Range> &frag_range =
      frag_is_src ? copy->src_range : copy->dst_range;
  const Array<Range> &global_range =
      frag_is_src ? copy->dst_range : copy->src_range;

  // Sized worst-case fallback for every shape outside the model: logical
  // elements from the fragment-side range (symbolic extents count as 1 —
  // worst geometry, optimistic count).
  auto worst_only = [&](const char *reason) {
    DLOG(INFO) << "[LayoutCost] copy probe falls back to worst-case: "
               << reason;
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
    return worst_only("region rank differs from fragment layout rank");
  }
  // Whole-fragment copies only: the iteration space is the fragment's full
  // logical shape.
  std::vector<int64_t> extents;
  int64_t logical_elements = 1;
  for (size_t d = 0; d < ndim; ++d) {
    if (!is_zero(frag_range[d]->min)) {
      return worst_only("copy covers a nonzero fragment-side offset");
    }
    const int64_t *frag_extent = as_const_int(frag_range[d]->extent);
    const int64_t *global_extent = as_const_int(global_range[d]->extent);
    const int64_t *shape_extent = as_const_int(frag_layout->InputShape()[d]);
    if (!frag_extent || !global_extent || !shape_extent ||
        *frag_extent != *shape_extent || *global_extent != *shape_extent) {
      return worst_only("copy is symbolic, partial, or shape-mismatched");
    }
    extents.push_back(*shape_extent);
    logical_elements *= *shape_extent;
  }

  int64_t elem_bits = global->dtype.bits() * global->dtype.lanes();
  if (elem_bits < 8) {
    return worst_only("global element width is below one byte");
  }
  auto strides = RowMajorStrides(global);
  if (!strides.has_value()) {
    return worst_only("global buffer shape has unsupported address geometry");
  }

  StatementProbe probe;
  probe.worst_elements = logical_elements;
  probe.extents = std::move(extents);
  probe.point_vars = MakePointVars(ndim);
  if (!BindForwardMaps(&probe, frag_layout)) {
    return worst_only("fragment forward maps have unsupported geometry");
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
std::optional<StatementProbe> BuildLoopProbe(const ParallelOp &loop,
                                             const Target &target) {
  LoopGlobalAccessCollector collector;
  collector.Collect(loop->GetRoot());
  StatementProbe probe;
  if (collector.accesses.empty()) {
    DLOG(INFO) << "[LayoutCost] parallel-loop probe has no direct global "
                  "accesses";
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
        DLOG(INFO) << "[LayoutCost] parallel-loop probe is unavailable: "
                      "symbolic parallel extent";
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
  auto worst_only = [&](const char *reason) -> std::optional<StatementProbe> {
    DLOG(INFO) << "[LayoutCost] parallel-loop probe falls back to worst-case: "
               << reason;
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
    return worst_only("loop layout is undefined");
  }
  if (nest_vars.size() != layout->InputShape().size()) {
    return worst_only("loop nest rank differs from loop layout rank");
  }
  probe.extents = std::move(extents);
  probe.point_vars = std::move(nest_vars);
  if (!BindForwardMaps(&probe, layout)) {
    return worst_only("loop forward maps have unsupported geometry");
  }
  probe.vector_bits = MaxVectorLoadBits(
      target, /*global_only_access=*/!collector.touches_shared);
  BindMemoryGeometry(&probe, target);

  for (const auto &raw : collector.accesses) {
    if (raw.symbolic_repeat) {
      return worst_only("global access has a symbolic serial repeat count");
    }
    int64_t elem_bits = raw.buffer->dtype.bits() * raw.buffer->dtype.lanes();
    auto strides = RowMajorStrides(raw.buffer);
    if (elem_bits < 8 || !strides.has_value() ||
        raw.indices.size() != raw.buffer->shape.size()) {
      return worst_only("global access has unsupported dtype, shape, or rank");
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
  DLOG(INFO) << "[LayoutCost] register count: layout_entries="
             << tmp_layout_map.size();
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
      DLOG(INFO) << "[LayoutCost] register count: buffer=" << buffer
                 << " output_shape=" << frag.value()->OutputShape()
                 << " contribution=" << frag_reg_num
                 << " running_total=" << regs;
    }
  }
  DLOG(INFO) << "[LayoutCost] register count total=" << regs;
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
    (void)infer_list;
    DLOG(INFO) << "[LayoutCost] register-count score begin: members="
               << FormatVector(members);
    AttemptCost cost;
    cost.regs = CountRegisterSlots(tmp_layout_map);
    DLOG(INFO) << "[LayoutCost] register-count score end: mem=" << cost.mem
               << " regs=" << cost.regs;
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
    DLOG(INFO) << "[LayoutCost] io-aware score begin: members="
               << FormatVector(members)
               << " layout_entries=" << tmp_layout_map.size();
    AttemptCost cost;
    cost.regs = CountRegisterSlots(tmp_layout_map);

    for (int idx : members) {
      const TileOperator &op = infer_list[idx];
      DLOG(INFO) << "[LayoutCost] member " << idx
                 << " begin: type=" << op->GetTypeKey();
      if (const auto *copy = infer_list[idx].as<CopyNode>()) {
        Copy copy_op = GetRef<Copy>(copy);
        DLOG(INFO) << "[LayoutCost] member " << idx
                   << " copy: src=" << copy->src
                   << " (scope=" << copy->src.scope() << ") dst=" << copy->dst
                   << " (scope=" << copy->dst.scope() << ')';
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
          DLOG(INFO) << "[LayoutCost] member " << idx
                     << " copy ignored: not a fragment<->global transfer";
          continue; // register moves / shared staging: out of the model
        }
        DLOG(INFO) << "[LayoutCost] member " << idx
                   << " copy modeled as global "
                   << (frag_is_src ? "store" : "load")
                   << " through fragment=" << frag;
        auto layout = tmp_layout_map.Get(frag);
        if (!layout.has_value()) {
          DLOG(INFO) << "[LayoutCost] member " << idx
                     << " copy ignored: fragment has no tentative layout";
          continue;
        }
        auto frag_layout = layout.value().as<Fragment>();
        if (!frag_layout.has_value()) {
          DLOG(INFO) << "[LayoutCost] member " << idx
                     << " copy ignored: tentative layout is not a Fragment";
          continue;
        }
        DLOG(INFO) << "[LayoutCost] member " << idx << " copy fragment layout: "
                   << frag_layout.value()->DebugOutput();
        int64_t statement_mem =
            CachedStatementMem(idx, frag_layout.value(), [&]() {
              std::optional<StatementProbe> probe;
              try {
                probe = BuildCopyProbe(copy_op, frag_layout.value(),
                                       frag_is_src, target_);
              } catch (const std::exception &e) {
                DLOG(INFO) << "[LayoutCost] member " << idx
                           << " copy probe construction threw: " << e.what();
                probe = std::nullopt; // skipped below; builder-side fallbacks
                                      // cover every non-throwing failure
              }
              return ChargeStatement(probe, idx, "copy");
            });
        cost.mem += statement_mem;
        DLOG(INFO) << "[LayoutCost] member " << idx
                   << " copy contribution=" << statement_mem
                   << " running_mem=" << cost.mem;
      } else if (const auto *loop = infer_list[idx].as<ParallelOpNode>()) {
        ParallelOp loop_op = GetRef<ParallelOp>(loop);
        Fragment loop_layout = loop_op->GetLoopLayout();
        if (loop_layout.defined()) {
          DLOG(INFO) << "[LayoutCost] member " << idx
                     << " parallel-loop layout: " << loop_layout->DebugOutput();
        } else {
          DLOG(INFO) << "[LayoutCost] member " << idx
                     << " parallel-loop layout is undefined";
        }
        auto compute = [&]() {
          std::optional<StatementProbe> probe;
          try {
            probe = BuildLoopProbe(loop_op, target_);
          } catch (const std::exception &e) {
            DLOG(INFO) << "[LayoutCost] member " << idx
                       << " parallel-loop probe construction threw: "
                       << e.what();
            probe = std::nullopt;
          }
          return ChargeStatement(probe, idx, "parallel-loop");
        };
        int64_t statement_mem =
            loop_layout.defined()
                ? CachedStatementMem(idx, loop_layout, compute)
                : compute();
        cost.mem += statement_mem;
        DLOG(INFO) << "[LayoutCost] member " << idx
                   << " parallel-loop contribution=" << statement_mem
                   << " running_mem=" << cost.mem;
      } else {
        DLOG(INFO) << "[LayoutCost] member " << idx
                   << " ignored: type=" << op->GetTypeKey()
                   << " is outside the IO-aware statement model";
      }
    }
    DLOG(INFO) << "[LayoutCost] io-aware score end: mem=" << cost.mem
               << " regs=" << cost.regs;
    return cost;
  }

  const char *Name() const final { return "io-aware"; }

private:
  /*! \brief Final charge of one prepared statement, honoring the probe's
   *  three-state protocol (zero / worst-case / measured). */
  static int64_t ChargeStatement(const std::optional<StatementProbe> &probe,
                                 int member_idx, const char *what) {
    if (!probe.has_value()) {
      DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
                 << " probe unavailable; contribution=0";
      return 0; // nothing sensible to charge
    }
    LogProbe(member_idx, what, *probe);
    if (probe->accesses.empty()) {
      DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
                 << " has no direct global traffic; contribution=0";
      return 0;
    }
    std::optional<StatementTraffic> traffic;
    if (probe->measurable) {
      traffic = ScoreStatement(*probe, member_idx, what);
    }
    if (traffic.has_value()) {
      DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
                 << " measured: bw=" << traffic->bw
                 << " issue=" << traffic->issue
                 << " contribution=max(bw,issue)=" << traffic->Time();
      return traffic->Time();
    }
    int64_t worst_case = WorstCaseBytes(*probe);
    DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
               << " outside the measurable model; worst-case contribution="
               << worst_case;
    return worst_case;
  }

  /*! \brief Memoize a statement's charge by (op index, layout): the charge
   *  depends only on the op (fixed per index) and the layout under test,
   *  and different attempt roots frequently converge to the same layout —
   *  those attempts then score the statement for free. Structural layout
   *  equality; entries per index stay tiny (one per distinct layout). */
  template <typename F>
  int64_t CachedStatementMem(int idx, const Fragment &layout,
                             F &&compute) const {
    auto &entries = stmt_cache_[idx];
    for (const auto &[cached_layout, mem] : entries) {
      if (cached_layout->IsEqual(layout.get())) {
        DLOG(INFO) << "[LayoutCost] member " << idx
                   << " statement cache hit: contribution=" << mem;
        return mem;
      }
    }
    DLOG(INFO) << "[LayoutCost] member " << idx
               << " statement cache miss: cached_layouts=" << entries.size();
    int64_t mem = compute();
    entries.emplace_back(layout, mem);
    DLOG(INFO) << "[LayoutCost] member " << idx
               << " statement cache store: contribution=" << mem;
    return mem;
  }

  Target target_;
  mutable std::unordered_map<int, std::vector<std::pair<Fragment, int64_t>>>
      stmt_cache_;
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

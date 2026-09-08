/*!
 * \file layout_cost_model.cc
 * \brief Statement-level bottleneck traffic model for free-mode layout
 *        attempts (layout RFC, design B2). See layout_cost_model.h.
 */

#include "layout_cost_model.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../config.h"
#include "../../cuda/target_utils.h"
#include "../../layout/cute_layout.h"
#include "../../layout/layout.h"
#include "../../layout/utils.h"
#include "../../op/copy.h"
#include "../../op/parallel.h"
#include "../../op/reducer.h"
#include "../../op/utils.h"
#include "../../span_utils.h"
#include "../loop_partition.h"
#include "../loop_vectorize.h"
#include "../reducer_plan.h"

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
// attempt must never profit from opacity. The mode arithmetic was audited
// once against a full exact-enumeration oracle across the test corpus
// (zero disagreements) before that oracle was removed; its Python mirror
// survives in maint/layout_inference (`run.py --cute`).
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

/*! \brief Evaluate flattened (extent, stride) modes at a linear coordinate
 *  (column-major digit decomposition — the same function the layout
 *  computes, minus the ObjectRef machinery). The scoring loops evaluate at
 *  most one coordinate per physical cell; plain int64 arithmetic keeps that
 *  walk at nanoseconds per point. */
int64_t EvalModes(const std::vector<std::pair<int64_t, int64_t>> &modes,
                  int64_t coord) {
  int64_t value = 0;
  for (const auto &[extent, stride] : modes) {
    value += (coord % extent) * stride;
    coord /= extent;
  }
  return value;
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
  int64_t physical_cells = 0;
  int64_t logical_size = 0;
  // A fragment the lowering accepts is a bijection between
  // (logical point, replica) and (thread, slot); anything else is
  // outside the model. The size identity is the cheap necessary half;
  // the RightInverse size check below is the sufficient half.
  if (!TryMultiplyPositive(probe.threads, probe.slots, &physical_cells) ||
      !TryMultiplyPositive(points, probe.rep, &logical_size) ||
      logical_size != physical_cells) {
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
  if (cute::AsConst(cute::Size(inv)) != physical_cells) {
    DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
               << " cannot be measured: candidate is not injective "
                  "(right-inverse covers "
               << cute::AsConst(cute::Size(inv)) << " of " << physical_cells
               << " cells)";
    return std::nullopt;
  }
  auto inv_modes = FlatModes(inv);
  if (!inv_modes) {
    return std::nullopt; // unreachable: a right inverse is const-strided
  }

  // Derive one `cell -> element address` layout per access and its
  // (slot, thread) mode split.
  cute::IntTuple st_shape = cute::IntTupleTuple(
      {cute::IntTuple(probe.slots), cute::IntTuple(probe.threads)});
  size_t naccess = probe.accesses.size();
  std::vector<std::vector<std::pair<int64_t, int64_t>>> addr_modes;
  std::vector<std::vector<std::pair<int64_t, int64_t>>> slot_modes;
  std::vector<std::vector<std::pair<int64_t, int64_t>>> thread_modes;
  addr_modes.reserve(naccess);
  slot_modes.reserve(naccess);
  thread_modes.reserve(naccess);
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
    auto amodes = FlatModes(addr);
    auto smodes = FlatModes(split[0]);
    auto tmodes = FlatModes(split[1]);
    if (!amodes || !smodes || !tmodes) {
      DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
                 << " cannot be measured: derived address modes are not "
                    "constant";
      return std::nullopt;
    }
    DLOG(INFO) << "[LayoutCost] member " << member_idx << ' ' << what
               << " access[" << a
               << "] address layout: " << CuteToString(split);
    addr_modes.push_back(std::move(*amodes));
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
              EvalModes(*inv_modes, cell) / points != 0) {
            continue; // guarded replica: this lane is idle for stores
          }
          int64_t first = EvalModes(addr_modes[a], cell);
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
class LoopMemoryAccessCollector : public StmtExprVisitor {
public:
  struct RawAccess {
    Buffer buffer;
    Array<PrimExpr> indices;
    bool is_store;
    int64_t repeat;
    bool symbolic_repeat;
  };
  std::vector<RawAccess> accesses;
  std::vector<RawAccess> shared_accesses;

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
    bool shared = IsSharedBuffer(buffer);
    if (!shared && !IsGlobalBuffer(buffer)) {
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
    (shared ? shared_accesses : accesses)
        .push_back(RawAccess{buffer, indices, is_store, repeat, symbolic});
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
  LoopMemoryAccessCollector collector;
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
      target, /*global_only_access=*/collector.shared_accesses.empty());
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

/*! \brief Estimated local-memory traffic an attempt's layouts would force:
 *  for every member loop (plain parallel nests and copies' inner SIMT
 *  nests), any fragment access whose PHYSICAL slot index depends on the
 *  thread variable demotes the whole per-thread register array to local
 *  memory, turning every executed access into a local load/store. Charged
 *  in bytes as `max(array bytes, 2 x threads x per-thread iterations x
 *  element bytes)` — the round-trip traffic of touching the demoted array
 *  once per iteration — so it competes honestly with the io-aware model's
 *  global-byte estimates (an 8-byte array spill must not veto kilobytes of
 *  parallel global bandwidth) while still burying any register-count
 *  difference in the spill-blind model. */
int64_t CountSpilledBytes(const std::vector<int> &members,
                          const std::vector<TileOperator> &infer_list,
                          const LayoutMap &layout_map) {
  int64_t spilled = 0;
  auto charge_loop = [&](const ParallelOpNode *loop_op) {
    if (loop_op == nullptr) {
      return;
    }
    Fragment loop_layout = loop_op->GetLoopLayout();
    if (!loop_layout.defined()) {
      return;
    }
    arith::Analyzer analyzer;
    FragmentThreadIndexProbe probe(loop_layout, loop_op->GetLoopVars(),
                                   &analyzer);
    if (!probe.valid()) {
      return;
    }
    auto const_product = [](const Array<PrimExpr> &dims) -> int64_t {
      int64_t product = 1;
      for (const PrimExpr &dim : dims) {
        const int64_t *extent = as_const_int(dim);
        if (extent == nullptr) {
          return -1;
        }
        product *= *extent;
      }
      return product;
    };
    for (const Buffer &buffer : loop_op->GetAccessOrder()) {
      if (!layout_map.count(buffer)) {
        continue;
      }
      auto fragment = layout_map[buffer].as<Fragment>();
      if (!fragment.has_value()) {
        continue;
      }
      const auto &access = loop_op->GetIndiceMap().at(buffer);
      if (!probe.AccessUsesThread(access.indices, fragment.value())) {
        continue;
      }
      int64_t elem_bytes =
          (buffer->dtype.bits() * buffer->dtype.lanes() + 7) / 8;
      // The whole per-thread array is demoted, not just the touched slot.
      int64_t array_slots = const_product(fragment.value()->OutputShape());
      int64_t array_bytes = (array_slots < 0 ? 1 : array_slots) * elem_bytes;
      // Every executed iteration touches the demoted array through local
      // memory (round trip).
      int64_t per_thread_iters = const_product(loop_layout->OutputShape());
      const int64_t *threads = as_const_int(loop_layout->ThreadExtent());
      int64_t traffic_bytes =
          (per_thread_iters < 0 || threads == nullptr)
              ? array_bytes
              : 2 * (*threads) * per_thread_iters * elem_bytes;
      spilled += std::max(array_bytes, traffic_bytes);
    }
  };
  for (int idx : members) {
    const TileOperator &op = infer_list[idx];
    if (const auto *loop_op = op.as<ParallelOpNode>()) {
      charge_loop(loop_op);
    } else if (const auto *copy = op.as<CopyNode>()) {
      // Bulk/TMA/TMem copies never build a SIMT nest; nothing to probe.
      if (copy->par_op_.defined()) {
        charge_loop(copy->par_op_.get());
      }
    }
  }
  return spilled;
}

/*! \brief Legacy policy: total register slots, with spill traffic as the
 *  leading `mem` charge (this model estimates no other memory, so mem is
 *  exactly the spill term). Among spill-free attempts the ordering is
 *  byte-identical to the historical register-count selection. */
class RegisterCountCostModel final : public LayoutCostModel {
public:
  AttemptCost Score(const std::vector<int> &members,
                    const std::vector<TileOperator> &infer_list,
                    const LayoutMap &tmp_layout_map) const final {
    DLOG(INFO) << "[LayoutCost] register-count score begin: members="
               << FormatVector(members);
    AttemptCost cost;
    cost.mem = CountSpilledBytes(members, infer_list, tmp_layout_map);
    cost.regs = CountRegisterSlots(tmp_layout_map);
    DLOG(INFO) << "[LayoutCost] register-count score end: mem(spill)="
               << cost.mem << " regs=" << cost.regs;
    return cost;
  }
  const char *Name() const final { return "register-count"; }
  ReducerVectorSearch GetReducerVectorSearch() const final {
    return ReducerVectorSearch::kScalar;
  }
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
    // Spill traffic enters the same byte-denominated channel as the global
    // estimates below, competing rather than vetoing.
    cost.mem = CountSpilledBytes(members, infer_list, tmp_layout_map);
    cost.regs = CountRegisterSlots(tmp_layout_map);

    for (int idx : members) {
      cost.mem += GlobalMemoryCost(idx, infer_list[idx], tmp_layout_map);
    }
    DLOG(INFO) << "[LayoutCost] io-aware score end: mem=" << cost.mem
               << " regs=" << cost.regs;
    return cost;
  }

  int64_t GlobalMemoryCost(int idx, const TileOperator &op,
                           const LayoutMap &tmp_layout_map) const {
    DLOG(INFO) << "[LayoutCost] member " << idx
               << " begin: type=" << op->GetTypeKey();
    if (const auto *copy = op.as<CopyNode>()) {
      Copy copy_op = GetRef<Copy>(copy);
      DLOG(INFO) << "[LayoutCost] member " << idx << " copy: src=" << copy->src
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
        return 0;
      }
      DLOG(INFO) << "[LayoutCost] member " << idx << " copy modeled as global "
                 << (frag_is_src ? "store" : "load")
                 << " through fragment=" << frag;
      auto layout = tmp_layout_map.Get(frag);
      if (!layout.has_value()) {
        DLOG(INFO) << "[LayoutCost] member " << idx
                   << " copy ignored: fragment has no tentative layout";
        return 0;
      }
      auto frag_layout = layout.value().as<Fragment>();
      if (!frag_layout.has_value()) {
        DLOG(INFO) << "[LayoutCost] member " << idx
                   << " copy ignored: tentative layout is not a Fragment";
        return 0;
      }
      DLOG(INFO) << "[LayoutCost] member " << idx << " copy fragment layout: "
                 << frag_layout.value()->DebugOutput();
      int64_t statement_mem =
          CachedStatementMem(idx, frag_layout.value(), [&]() {
            std::optional<StatementProbe> probe;
            try {
              probe = BuildCopyProbe(copy_op, frag_layout.value(), frag_is_src,
                                     target_);
            } catch (const std::exception &e) {
              DLOG(INFO) << "[LayoutCost] member " << idx
                         << " copy probe construction threw: " << e.what();
              probe = std::nullopt; // skipped below; builder-side fallbacks
                                    // cover every non-throwing failure
            }
            return ChargeStatement(probe, idx, "copy");
          });
      DLOG(INFO) << "[LayoutCost] member " << idx
                 << " copy contribution=" << statement_mem;
      return statement_mem;
    } else if (const auto *loop = op.as<ParallelOpNode>()) {
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
                     << " parallel-loop probe construction threw: " << e.what();
          probe = std::nullopt;
        }
        return ChargeStatement(probe, idx, "parallel-loop");
      };
      int64_t statement_mem =
          loop_layout.defined() ? CachedStatementMem(idx, loop_layout, compute)
                                : compute();
      DLOG(INFO) << "[LayoutCost] member " << idx
                 << " parallel-loop contribution=" << statement_mem;
      return statement_mem;
    } else {
      DLOG(INFO) << "[LayoutCost] member " << idx
                 << " ignored: type=" << op->GetTypeKey()
                 << " is outside the IO-aware statement model";
    }
    return 0;
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

constexpr int64_t kUnknownReductionCost =
    std::numeric_limits<int64_t>::max() / 16;

int64_t CeilDiv(int64_t numerator, int64_t denominator) {
  return numerator / denominator + (numerator % denominator != 0);
}

int64_t AddCost(int64_t left, int64_t right) {
  return left >= kUnknownReductionCost - right ? kUnknownReductionCost
                                               : left + right;
}

int64_t MultiplyCost(int64_t left, int64_t right) {
  if (left == 0 || right == 0) {
    return 0;
  }
  return left > kUnknownReductionCost / right ? kUnknownReductionCost
                                              : left * right;
}

std::optional<int64_t> ConstantCount(const PrimExpr &expression) {
  arith::Analyzer analyzer;
  PrimExpr simplified = analyzer.Simplify(expression);
  const int64_t *value = as_const_int(simplified);
  if (value == nullptr || *value < 0 || *value >= kUnknownReductionCost) {
    return std::nullopt;
  }
  return *value;
}

std::optional<int64_t> ConstantProduct(const Array<PrimExpr> &extents) {
  int64_t result = 1;
  for (const PrimExpr &extent : extents) {
    auto value = ConstantCount(extent);
    if (!value.has_value()) {
      return std::nullopt;
    }
    result = MultiplyCost(result, value.value());
  }
  return result < kUnknownReductionCost ? std::optional<int64_t>(result)
                                        : std::nullopt;
}

class ExecutionCountCollector : public StmtExprVisitor {
public:
  std::unordered_map<ObjectRef, int64_t, ObjectPtrHash, ObjectPtrEqual> counts;
  bool known{true};

  void Collect(const PrimFunc &function) { VisitStmt(function->body); }

private:
  void VisitStmt_(const ForNode *loop) final {
    counts[GetRef<For>(loop)] = count_;
    int64_t previous = count_;
    if (loop->kind != ForKind::kParallel) {
      auto extent = ConstantCount(loop->extent);
      count_ = count_ < 0 || !extent.has_value()
                   ? -1
                   : MultiplyCost(count_, extent.value());
      known = known && count_ >= 0 && count_ < kUnknownReductionCost;
    }
    StmtExprVisitor::VisitStmt_(loop);
    count_ = previous;
  }

  void VisitExpr_(const CallNode *call) final {
    counts[GetRef<Call>(call)] = count_;
    StmtExprVisitor::VisitExpr_(call);
  }

  int64_t count_{1};
};

struct ReducerCostFeatures {
  int64_t local_issues{0};
  int64_t shared_issues{0};
  int64_t shuffle_issues{0};
  int64_t shared_rounds{0};
  int64_t barriers{0};
  int64_t combine_issues{0};
  int64_t slots{0};
  std::vector<int> vector_widths;
};

int PartitionedVectorWidth(const For &loop, const Fragment &layout,
                           const LayoutMap &layouts) {
  arith::Analyzer analyzer;
  Var thread("reduction_cost_thread", DataType::Int(32));
  Range bounds = layout->ThreadRange();
  if (!bounds.defined()) {
    bounds = Range::FromMinExtent(0, layout->ThreadExtent());
  }
  analyzer.Bind(thread, bounds);
  For partitioned = PartitionLoop(loop, thread, &analyzer, layout);
  return GetVectorizeSize(partitioned, &analyzer, layouts);
}

class ReducerCostRewriter : public StmtExprMutator {
public:
  ReducerCostRewriter(const std::vector<ReducerPlanInfo> &plans,
                      const LayoutMap &layouts)
      : layouts(layouts) {
    for (const ReducerPlanInfo &plan : plans) {
      Array<PrimExpr> shape = plan.reducer->shape;
      Fragment storage = plan.storage_layout;
      if (plan.packed_layout.defined()) {
        shape.push_back(Integer(2));
        storage = plan.packed_layout.value();
      }
      Var data("reduction_cost_acc",
               PointerType(PrimType(plan.reducer->dtype), "local.fragment"));
      Buffer buffer(data, plan.reducer->dtype, shape, {}, Integer(0),
                    "reduction_cost_acc", plan.reducer->data_alignment,
                    plan.reducer->offset_factor, plan.reducer->buffer_type);
      this->layouts.Set(buffer, storage);
      updates_.emplace(
          plan.reducer->data,
          Update{buffer, plan.op, plan.pack_lane_var, plan.narrow});
    }
  }

  LayoutMap layouts;

private:
  struct Update {
    Buffer buffer;
    ReducerV2OpType op;
    Optional<Var> pack_lane;
    bool narrow;
  };

  Stmt VisitStmt_(const EvaluateNode *statement) final {
    const auto *call = statement->value.as<CallNode>();
    if (call == nullptr || !call->op.same_as(reducer_update())) {
      return StmtExprMutator::VisitStmt_(statement);
    }
    ReducerUpdateArgs args = ParseReducerUpdate(call);
    const Update &update = updates_.at(args.reducer->data);
    Array<PrimExpr> indices = args.indices;
    if (update.pack_lane.defined()) {
      indices.push_back(
          FloorMod(update.pack_lane.value(), IntImm(DataType::Int(32), 2)));
    }
    Stmt store = BufferStore(
        update.buffer,
        ReducerV2Combine(update.op, BufferLoad(update.buffer, indices),
                         args.value),
        indices);
    if (!update.narrow) {
      store = AttrStmt(update.buffer->data, attr::kParallelMultiplicity,
                       IntImm(DataType::Int(32), 1), store);
    }
    return store;
  }

  std::unordered_map<Var, Update, ObjectPtrHash, ObjectPtrEqual> updates_;
};

Map<For, Integer> ReducerVectorWidths(const std::vector<ReducerPlanInfo> &plans,
                                      const LayoutMap &layouts) {
  ReducerCostRewriter rewriter(plans, layouts);
  Map<For, Integer> widths;
  for (const ReducerPlanInfo &plan : plans) {
    for (const ReducerUpdatePlanSite &site : plan.updates) {
      if (!widths.count(site.loop)) {
        For loop = Downcast<For>(rewriter(site.loop));
        widths.Set(site.loop, Integer(PartitionedVectorWidth(
                                  loop, site.loop_layout, rewriter.layouts)));
      }
    }
  }
  return widths;
}

std::optional<ReducerCostFeatures>
ExtractReducerCost(const ReducerPlanInfo &plan, const Map<For, Integer> &widths,
                   const Target &target) {
  ReducerCostFeatures features;
  auto slots = ConstantProduct(plan.storage_layout->OutputShape());
  auto finalizations = ConstantCount(plan.finalize_count);
  bool materializes_finalize =
      !plan.narrow || !plan.steps.empty() || plan.has_seed;
  if (!slots.has_value() || !finalizations.has_value() || plan.batch < 1 ||
      (materializes_finalize && plan.batch > 1 &&
       (plan.batch > slots.value() || slots.value() % plan.batch != 0))) {
    return std::nullopt;
  }
  features.slots = slots.value();
  bool packed_arithmetic = plan.reducer->dtype.is_float16() ||
                           plan.reducer->dtype.is_bfloat16() ||
                           (plan.reducer->dtype == DataType::Float(32) &&
                            TargetHasSMVersionGE(target, 100));
  for (const ReducerUpdatePlanSite &site : plan.updates) {
    auto points = ConstantProduct(site.loop_layout->OutputShape());
    auto repeats = ConstantCount(site.execution_count);
    if (!points.has_value() || !repeats.has_value()) {
      return std::nullopt;
    }
    int width = static_cast<int>(widths.at(site.loop)->value);
    features.vector_widths.push_back(width);
    int arithmetic_width = packed_arithmetic ? std::min(width, 2) : 1;
    int64_t work = MultiplyCost(points.value(), repeats.value());
    features.local_issues =
        AddCost(features.local_issues, CeilDiv(work, arithmetic_width));
  }
  if (plan.packed_layout.defined()) {
    features.local_issues =
        AddCost(features.local_issues,
                MultiplyCost(features.slots, finalizations.value()));
  }
  int warp_size =
      target->GetAttr<Integer>("thread_warp_size").value_or(32)->value;
  for (const auto &[threads, scale] : plan.steps) {
    if (scale <= 0 || threads < scale || threads % scale != 0 ||
        ((threads / scale) & (threads / scale - 1)) != 0) {
      return std::nullopt;
    }
    int64_t batch = threads > warp_size ? plan.batch : 1;
    int64_t groups =
        MultiplyCost(features.slots / batch, finalizations.value());
    int64_t values = MultiplyCost(features.slots, finalizations.value());
    for (int offset = threads / 2; offset >= scale; offset /= 2) {
      if (offset >= warp_size) {
        features.shared_rounds = AddCost(features.shared_rounds, groups);
        features.barriers = AddCost(features.barriers, MultiplyCost(2, groups));
        features.shared_issues =
            AddCost(features.shared_issues, MultiplyCost(2, values));
      } else {
        features.shuffle_issues = AddCost(features.shuffle_issues, values);
      }
      features.combine_issues = AddCost(features.combine_issues, values);
    }
  }
  return features;
}

int64_t CudaReducerIssueCost(const ReducerCostFeatures &features) {
  constexpr int64_t kShuffleWeight = 4;
  constexpr int64_t kBarrierWeight = 32;
  int64_t cost = AddCost(features.local_issues, features.shared_issues);
  cost = AddCost(cost, features.combine_issues);
  cost = AddCost(cost, MultiplyCost(kShuffleWeight, features.shuffle_issues));
  return AddCost(cost, MultiplyCost(kBarrierWeight, features.barriers));
}

class ReductionAwareCostModel final : public LayoutCostModel {
public:
  ReductionAwareCostModel(Target target, const PrimFunc &function,
                          std::vector<int64_t> execution_counts)
      : target_(std::move(target)), plans_(function), io_model_(target_),
        execution_counts_(std::move(execution_counts)) {}

  AttemptCost Score(const std::vector<int> &members,
                    const std::vector<TileOperator> &infer_list,
                    const LayoutMap &layouts) const final {
    Array<Buffer> reducers;
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> seen;
    Map<For, Fragment> loop_layouts;
    for (int member : members) {
      if (auto finalize = infer_list[member].as<FinalizeReducerV2Op>()) {
        if (seen.insert(finalize.value()->reducer).second) {
          reducers.push_back(finalize.value()->reducer);
        }
      }
      if (auto loop = infer_list[member].as<ParallelOp>()) {
        loop_layouts.Set(loop.value()->GetRoot(),
                         loop.value()->GetLoopLayout());
      }
    }
    if (reducers.empty()) {
      return RegisterCountCostModel().Score(members, infer_list, layouts);
    }
    AttemptCost cost;
    cost.mem = CountSpilledBytes(members, infer_list, layouts);
    cost.regs = CountRegisterSlots(layouts);
    cost.execution = kUnknownReductionCost;
    cost.known = false;
    try {
      auto plans = plans_.Analyze(layouts, loop_layouts, reducers);
      if (!plans.has_value()) {
        return cost;
      }
      LayoutMap physical_layouts = layouts;
      for (const ReducerPlanInfo &plan : plans.value()) {
        physical_layouts.Set(plan.reducer, plan.storage_layout);
        for (const auto &[buffer, layout] : plan.layout_overrides) {
          physical_layouts.Set(buffer, layout);
        }
      }
      cost.regs = CountRegisterSlots(physical_layouts);
      Map<For, Integer> widths =
          ReducerVectorWidths(plans.value(), physical_layouts);
      int64_t execution = 0;
      int64_t register_threads = 1;
      bool bank_conflict_free = true;
      for (const ReducerPlanInfo &plan : plans.value()) {
        auto features = ExtractReducerCost(plan, widths, target_);
        if (!features.has_value()) {
          return cost;
        }
        int64_t participants = *as_const_int(plan.thread_bounds->extent);
        register_threads = std::max(register_threads, participants);
        int64_t issue_bytes =
            MultiplyCost(participants, MaxVectorLoadBits(target_, false) / 8);
        execution = AddCost(
            execution,
            MultiplyCost(CudaReducerIssueCost(features.value()), issue_bytes));
        if (plan.packed_layout.defined()) {
          auto packed_slots =
              ConstantProduct(plan.packed_layout.value()->OutputShape());
          if (!packed_slots.has_value()) {
            return cost;
          }
          cost.regs = AddCost(cost.regs, packed_slots.value());
        }
        if (tl_config::ReducerPlanVerboseEnabled()) {
          LOG(INFO) << "[ReducerCost] reducer=" << plan.reducer->name
                    << " narrow=" << plan.narrow << " reason=" << plan.reason
                    << " local_issues=" << features->local_issues
                    << " shared_rounds=" << features->shared_rounds
                    << " barriers=" << features->barriers
                    << " shuffle_issues=" << features->shuffle_issues
                    << " vector_widths="
                    << FormatVector(features->vector_widths);
        }
      }
      int64_t spilled = 0;
      for (int member : members) {
        int64_t repeats = execution_counts_.at(member);
        int64_t spill =
            CountSpilledBytes({member}, infer_list, physical_layouts);
        spilled = AddCost(spilled, MultiplyCost(spill, repeats));
        int64_t memory = io_model_.GlobalMemoryCost(member, infer_list[member],
                                                    physical_layouts);
        Optional<ParallelOp> loop = infer_list[member].as<ParallelOp>();
        if (auto copy = infer_list[member].as<Copy>()) {
          if (copy.value()->par_op_.defined()) {
            loop = copy.value()->par_op_;
          }
        }
        if (loop.defined()) {
          auto loop_memory =
              LoopMemoryCost(loop.value(), physical_layouts, widths, memory);
          if (!loop_memory.has_value()) {
            return cost;
          }
          bank_conflict_free &= repeats == 0 || loop_memory->bank_conflict_free;
          memory = loop_memory->execution;
        }
        execution = AddCost(execution, MultiplyCost(memory, repeats));
      }
      cost.mem = spilled;
      cost.execution = execution;
      constexpr int64_t kRegisterSlotBytes = 4;
      int64_t register_cost = MultiplyCost(
          cost.regs, MultiplyCost(register_threads, kRegisterSlotBytes));
      int64_t total_cost = AddCost(AddCost(spilled, execution), register_cost);
      cost.known = total_cost < kUnknownReductionCost;
      if (cost.known) {
        cost.total_cost = total_cost;
      }
      cost.bank_conflict_free = cost.known && bank_conflict_free;
    } catch (const std::exception &error) {
      DLOG(INFO) << "[ReducerCost] unmeasurable attempt: " << error.what();
      cost.execution = kUnknownReductionCost;
    }
    return cost;
  }

  const char *Name() const final { return "reduction-aware"; }

  ReducerVectorSearch GetReducerVectorSearch() const final {
    return ReducerVectorSearch::kAll;
  }

private:
  struct MemoryIssueCost {
    int64_t execution{0};
    bool bank_conflict_free{true};
  };

  std::optional<int64_t>
  SharedBankConflictFactor(const LoopMemoryAccessCollector::RawAccess &access,
                           const Var &thread, const Range &bounds,
                           int64_t threads, int width, const LayoutMap &layouts,
                           arith::Analyzer *analyzer) const {
    constexpr int64_t kBankCount = 32;
    constexpr int64_t kBankBytes = 4;
    int64_t element_bits =
        access.buffer->dtype.bits() * access.buffer->dtype.lanes();
    if (element_bits < 8 || element_bits % 8 != 0) {
      return std::nullopt;
    }
    int64_t element_bytes = element_bits / 8;
    PrimExpr address = make_zero(DataType::Int(32));
    if (auto shared_layout = layouts.Get(access.buffer)) {
      Array<PrimExpr> indices = shared_layout.value()->Forward(access.indices);
      Array<PrimExpr> shape = shared_layout.value()->OutputShape();
      for (size_t axis = 0; axis < indices.size(); ++axis) {
        address = address * shape[axis] + indices[axis];
      }
    } else {
      auto strides = RowMajorStrides(access.buffer);
      if (!strides.has_value() || strides->size() != access.indices.size()) {
        return std::nullopt;
      }
      for (size_t axis = 0; axis < access.indices.size(); ++axis) {
        address = address +
                  access.indices[axis] *
                      IntImm(access.indices[axis].dtype(), (*strides)[axis]);
      }
    }
    address =
        analyzer->Simplify(address * IntImm(address.dtype(), element_bytes));
    PrimExpr origin = Substitute(address, {{thread, bounds->min}});
    PrimExpr next = Substitute(address, {{thread, bounds->min + 1}});
    PrimExpr stride = analyzer->Simplify(next - origin);
    const int64_t *stride_bytes = as_const_int(stride);
    if (stride_bytes == nullptr || *stride_bytes < 0 ||
        !analyzer->CanProveEqual(address,
                                 origin + (thread - bounds->min) * stride)) {
      return std::nullopt;
    }
    int64_t vector_bytes = std::min<int64_t>(
        width * element_bytes, MaxVectorLoadBits(target_, false) / 8);
    int64_t warp_size =
        target_->GetAttr<Integer>("thread_warp_size").value_or(32)->value;
    int64_t lanes = std::min(
        {threads, warp_size,
         kBankCount * kBankBytes / std::max(kBankBytes, vector_bytes)});
    int64_t alignments = vector_bytes >= kBankBytes ? 1 : kBankBytes;
    int64_t conflicts = 1;
    for (int64_t alignment = 0; alignment < alignments; ++alignment) {
      std::array<std::unordered_set<int64_t>, kBankCount> bank_words;
      for (int64_t lane = 0; lane < lanes; ++lane) {
        int64_t first = (alignment + lane * *stride_bytes) / kBankBytes;
        int64_t last =
            (alignment + lane * *stride_bytes + vector_bytes - 1) / kBankBytes;
        for (int64_t word = first; word <= last; ++word) {
          auto &words = bank_words[word % kBankCount];
          words.insert(word);
          conflicts = std::max(conflicts, static_cast<int64_t>(words.size()));
        }
      }
    }
    return conflicts;
  }

  std::optional<MemoryIssueCost> LoopMemoryCost(const ParallelOp &loop,
                                                const LayoutMap &layouts,
                                                const Map<For, Integer> &widths,
                                                int64_t global) const {
    MemoryIssueCost cost{global};
    LoopMemoryAccessCollector collector;
    collector.Collect(loop->GetRoot());
    auto solved_width = widths.Get(loop->GetRoot());
    if (collector.shared_accesses.empty() &&
        (!solved_width.has_value() || collector.accesses.empty())) {
      return cost;
    }
    Fragment layout = loop->GetLoopLayout();
    if (!layout.defined()) {
      return std::nullopt;
    }
    auto slots = ConstantProduct(layout->OutputShape());
    auto threads = ConstantCount(layout->ThreadExtent());
    if (!slots.has_value() || !threads.has_value()) {
      return std::nullopt;
    }
    int width = solved_width.has_value()
                    ? static_cast<int>(solved_width.value()->value)
                    : 0;
    int64_t shared_accesses = 0;
    if (!collector.shared_accesses.empty()) {
      arith::Analyzer analyzer;
      Var thread("shared_cost_thread", DataType::Int(32));
      Range bounds = layout->ThreadRange();
      if (!bounds.defined()) {
        bounds = Range::FromMinExtent(0, layout->ThreadExtent());
      }
      analyzer.Bind(thread, bounds);
      For partitioned =
          PartitionLoop(loop->GetRoot(), thread, &analyzer, layout);
      if (!solved_width.has_value()) {
        auto vector_analyzer = analyzer.Clone();
        width = GetVectorizeSize(partitioned, vector_analyzer.get(), layouts);
      }
      PostOrderVisit(partitioned, [&](const ObjectRef &object) {
        if (auto serial = object.as<For>()) {
          analyzer.Bind(serial.value()->loop_var,
                        Range::FromMinExtent(serial.value()->min,
                                             serial.value()->extent));
        }
      });
      LoopMemoryAccessCollector physical;
      physical.Collect(partitioned);
      if (physical.shared_accesses.size() != collector.shared_accesses.size()) {
        return std::nullopt;
      }
      for (size_t index = 0; index < physical.shared_accesses.size(); ++index) {
        const auto &access = collector.shared_accesses[index];
        if (access.symbolic_repeat || access.repeat < 0) {
          return std::nullopt;
        }
        auto conflicts = SharedBankConflictFactor(
            physical.shared_accesses[index], thread, bounds, threads.value(),
            width, layouts, &analyzer);
        if (!conflicts.has_value()) {
          return std::nullopt;
        }
        cost.bank_conflict_free &= access.repeat == 0 || conflicts.value() == 1;
        shared_accesses = AddCost(
            shared_accesses, MultiplyCost(access.repeat, conflicts.value()));
      }
    }
    auto issue_cost = [&](int64_t accesses, bool global_only) {
      return MultiplyCost(
          MultiplyCost(CeilDiv(slots.value(), width), accesses),
          MultiplyCost(threads.value(),
                       MaxVectorLoadBits(target_, global_only) / 8));
    };
    if (solved_width.has_value()) {
      int64_t global_accesses = 0;
      for (const auto &access : collector.accesses) {
        if (access.symbolic_repeat || access.repeat < 0) {
          return std::nullopt;
        }
        global_accesses = AddCost(global_accesses, access.repeat);
      }
      cost.execution = std::max(global, issue_cost(global_accesses, true));
    }
    cost.execution =
        AddCost(cost.execution, issue_cost(shared_accesses, false));
    return cost;
  }

  Target target_;
  ReducerPlanAnalyzer plans_;
  IOAwareCostModel io_model_;
  std::vector<int64_t> execution_counts_;
};

Map<Buffer, Map<String, Any>> ReducerCostSummary(const PrimFunc &function) {
  Target target = function->GetAttr<Target>(tvm::attr::kTarget).value();
  ICHECK(TargetIsCuda(target)) << "ReducerCost currently supports CUDA targets";
  With<Target> target_scope(target);
  LayoutMap layouts;
  Map<For, Fragment> loops;
  Array<Buffer> reducers;
  PostOrderVisit(function->body, [&](const ObjectRef &object) {
    if (auto block = object.as<SBlock>()) {
      if (auto annotation = block.value()->annotations.Get(attr::kLayoutMap)) {
        for (const auto &[buffer, layout] :
             Downcast<LayoutMap>(annotation.value())) {
          layouts.Set(buffer, layout);
        }
      }
      for (const Buffer &buffer : block.value()->alloc_buffers) {
        if (IsReducerV2Buffer(buffer)) {
          reducers.push_back(buffer);
        }
      }
    } else if (auto loop = object.as<For>()) {
      if (auto layout =
              loop.value()->annotations.Get(attr::kParallelLoopLayout)) {
        loops.Set(loop.value(), Downcast<Fragment>(layout.value()));
      }
    }
  });
  auto plans = ReducerPlanAnalyzer(function).Analyze(layouts, loops, reducers);
  ICHECK(plans.has_value())
      << "ReducerCost requires completed layout inference";
  for (const ReducerPlanInfo &plan : plans.value()) {
    for (const auto &[buffer, layout] : plan.layout_overrides) {
      layouts.Set(buffer, layout);
    }
  }
  Map<For, Integer> widths = ReducerVectorWidths(plans.value(), layouts);
  Map<Buffer, Map<String, Any>> result;
  for (const ReducerPlanInfo &plan : plans.value()) {
    Map<String, Any> summary;
    summary.Set("narrow", plan.narrow);
    summary.Set("reason", String(plan.reason));
    summary.Set("batch", plan.batch);
    Array<Array<Integer>> steps;
    for (const auto &[threads, scale] : plan.steps) {
      steps.push_back({Integer(threads), Integer(scale)});
    }
    summary.Set("steps", steps);
    auto features = ExtractReducerCost(plan, widths, target);
    summary.Set("known", features.has_value());
    if (features.has_value()) {
      Array<Integer> widths;
      for (int width : features->vector_widths) {
        widths.push_back(Integer(width));
      }
      summary.Set("vector_widths", widths);
      summary.Set("local_issues", features->local_issues);
      summary.Set("shared_issues", features->shared_issues);
      summary.Set("shuffle_issues", features->shuffle_issues);
      summary.Set("shared_rounds", features->shared_rounds);
      summary.Set("barriers", features->barriers);
      summary.Set("combine_issues", features->combine_issues);
      summary.Set("slots", features->slots);
      summary.Set("issue_cost", CudaReducerIssueCost(features.value()));
    }
    result.Set(plan.reducer, summary);
  }
  return result;
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.analysis.ReducerCost",
                                   ReducerCostSummary);
}

std::unique_ptr<LayoutCostModel>
LayoutCostModel::Create(const std::string &name, Target target,
                        const PrimFunc &function,
                        const std::vector<ObjectRef> &statements) {
  if (name == "io-aware") {
    return std::make_unique<IOAwareCostModel>(std::move(target));
  }
  if (name != "register-count") {
    LOG(FATAL) << "Unknown layout cost model \"" << name
               << "\" for pass config `tl.layout_cost_model`; valid values "
                  "are \"register-count\" (default) and \"io-aware\".";
  }
  bool has_reducer = std::any_of(
      statements.begin(), statements.end(), [](const ObjectRef &statement) {
        const auto *call = statement.as<CallNode>();
        return call != nullptr && call->op.same_as(FinalizeReducerV2Op::Get());
      });
  if (!TargetIsCuda(target) || !has_reducer) {
    return std::make_unique<RegisterCountCostModel>();
  }
  ICHECK(function.defined()) << "reduction-aware scoring requires a PrimFunc";
  ExecutionCountCollector collector;
  collector.Collect(function);
  if (!collector.known) {
    return std::make_unique<RegisterCountCostModel>();
  }
  std::vector<int64_t> counts;
  for (const ObjectRef &statement : statements) {
    auto found = collector.counts.find(statement);
    if (found == collector.counts.end() || found->second < 0 ||
        found->second >= kUnknownReductionCost) {
      DLOG(INFO)
          << "[ReducerCost] unknown execution count; using register-count";
      return std::make_unique<RegisterCountCostModel>();
    }
    counts.push_back(found->second);
  }
  return std::make_unique<ReductionAwareCostModel>(std::move(target), function,
                                                   std::move(counts));
}

} // namespace tl
} // namespace tvm

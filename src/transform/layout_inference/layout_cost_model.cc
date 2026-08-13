/*!
 * \file layout_cost_model.cc
 * \brief Statement-level bottleneck traffic model for free-mode layout
 *        attempts (layout RFC, design B2). See layout_cost_model.h.
 */

#include "layout_cost_model.h"

#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <optional>
#include <set>
#include <utility>

#include "../../layout/layout.h"
#include "../../layout/utils.h"
#include "../../op/copy.h"
#include "../../op/parallel.h"
#include "../../op/utils.h"
#include "../../span_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

// ---------------------------------------------------------------------------
// Statement-level bottleneck traffic model (layout RFC, design B2)
//
// The unit of account is one STATEMENT that touches global memory — a
// fragment<->global tl.copy or a parallel loop with direct global accesses —
// described uniformly as
//     iteration:  per-thread serial index `local` in [0, local_size)
//     threads:    `thread` in [0, threads)          (replication included)
//     accesses:   flat element addresses addr(local, thread) per buffer
// Execution time is bounded by two resources, both measured in BYTES so a
// fully busy block streaming 16B lanes scores exactly its logical byte
// count on either:
//     bw    = 128B transactions touched by the active lanes (counted per
//             warp step; intra-warp broadcast merges into one segment,
//             cross-warp repeats are conservatively recounted)
//     issue = per-thread instruction depth x what a fully busy block would
//             stream at 16B/lane over that many steps: steps x threads x 16.
//             Idle lanes (replication guards) do not shorten the depth, so
//             thread-collapse pathologies surface here (issue #1729).
//     time(S) ~ max(bw, issue);      cost = sum over statements.
// ---------------------------------------------------------------------------

/*! \brief Evaluate a quasi-affine index expression at a concrete
 *  (local, thread) point. Foreign vars (block indices inside region
 *  offsets) evaluate to zero: they shift every address of the statement
 *  equally and cancel out of both contiguity and segment geometry. Returns
 *  nullopt outside the Add/Sub/Mul/FloorDiv/FloorMod/Min/Max/Cast family.
 *  A cost model needs an estimate, not a proof; this deliberately stays
 *  off the Analyzer stack (deterministic, microseconds, no prover). */
std::optional<int64_t> EvalIndexExpr(const PrimExpr &e,
                                     const VarNode *local_var,
                                     const VarNode *thread_var,
                                     int64_t local_value,
                                     int64_t thread_value) {
  auto eval = [&](const PrimExpr &sub) {
    return EvalIndexExpr(sub, local_var, thread_var, local_value, thread_value);
  };
  if (const auto *imm = e.as<IntImmNode>()) {
    return imm->value;
  }
  if (const auto *var = e.as<VarNode>()) {
    if (var == local_var) {
      return local_value;
    }
    if (var == thread_var) {
      return thread_value;
    }
    return 0; // foreign additive offset
  }
  auto binary =
      [&](const PrimExpr &a,
          const PrimExpr &b) -> std::optional<std::pair<int64_t, int64_t>> {
    auto va = eval(a);
    auto vb = eval(b);
    if (!va || !vb) {
      return std::nullopt;
    }
    return std::make_pair(*va, *vb);
  };
  if (const auto *op = e.as<AddNode>()) {
    auto ab = binary(op->a, op->b);
    return ab ? std::optional<int64_t>(ab->first + ab->second) : std::nullopt;
  }
  if (const auto *op = e.as<SubNode>()) {
    auto ab = binary(op->a, op->b);
    return ab ? std::optional<int64_t>(ab->first - ab->second) : std::nullopt;
  }
  if (const auto *op = e.as<MulNode>()) {
    auto ab = binary(op->a, op->b);
    return ab ? std::optional<int64_t>(ab->first * ab->second) : std::nullopt;
  }
  if (const auto *op = e.as<FloorDivNode>()) {
    auto ab = binary(op->a, op->b);
    if (!ab || ab->second == 0) {
      return std::nullopt;
    }
    int64_t q = ab->first / ab->second;
    if ((ab->first % ab->second != 0) &&
        ((ab->first < 0) != (ab->second < 0))) {
      --q;
    }
    return q;
  }
  if (const auto *op = e.as<FloorModNode>()) {
    auto ab = binary(op->a, op->b);
    if (!ab || ab->second == 0) {
      return std::nullopt;
    }
    int64_t r = ab->first % ab->second;
    if (r != 0 && ((r < 0) != (ab->second < 0))) {
      r += ab->second;
    }
    return r;
  }
  if (const auto *op = e.as<CastNode>()) {
    return eval(op->value);
  }
  if (const auto *op = e.as<MinNode>()) {
    auto ab = binary(op->a, op->b);
    return ab ? std::optional<int64_t>(std::min(ab->first, ab->second))
              : std::nullopt;
  }
  if (const auto *op = e.as<MaxNode>()) {
    auto ab = binary(op->a, op->b);
    return ab ? std::optional<int64_t>(std::max(ab->first, ab->second))
              : std::nullopt;
  }
  return std::nullopt;
}

/*! \brief One global-memory access stream of a statement. */
struct GlobalAccessProbe {
  PrimExpr addr; // flat element index into the global buffer
  int64_t elem_bytes{4};
  bool is_store{false};
  int64_t repeat{1}; // enclosing serial trip count: the address pattern
                     // replays (shifted) that many times per local step
};

/*! \brief A statement prepared for scoring. `worst_elements` sizes the
 *  conservative fallback when the geometry cannot be evaluated. */
struct StatementProbe {
  std::vector<GlobalAccessProbe> accesses;
  std::optional<PrimExpr> replication_index; // gates STORE lanes (rep != 0)
  int64_t local_size{1};
  int64_t threads{1};
  int64_t worst_elements{0};
};

struct StatementTraffic {
  int64_t bw{0};
  int64_t issue{0};
  int64_t Time() const { return std::max(bw, issue); }
};

/*! \brief The conservative charge for a statement outside the model:
 *  every element its own 128B transaction. An attempt must never profit
 *  from opacity — evaluability depends on the layout under test. */
int64_t WorstCaseBytes(const StatementProbe &probe) {
  int64_t total = 0;
  for (const auto &access : probe.accesses) {
    total += probe.worst_elements * access.repeat * 128;
  }
  return total;
}

std::optional<StatementTraffic> ScoreStatementImpl(const StatementProbe &probe,
                                                   const VarNode *local_var,
                                                   const VarNode *thread_var) {
  if (probe.accesses.empty()) {
    return StatementTraffic{};
  }
  int64_t local_size = probe.local_size;
  int64_t threads = probe.threads;
  if (local_size <= 0 || threads <= 0) {
    return std::nullopt;
  }
  auto eval = [&](const PrimExpr &e, int64_t l,
                  int64_t t) -> std::optional<int64_t> {
    return EvalIndexExpr(e, local_var, thread_var, l, t);
  };
  for (const auto &access : probe.accesses) {
    if (!eval(access.addr, 0, 0).has_value()) {
      return std::nullopt;
    }
  }
  if (probe.replication_index.has_value() &&
      !eval(*probe.replication_index, 0, 0).has_value()) {
    return std::nullopt;
  }

  std::vector<int64_t> thread_samples{0};
  if (threads > 17) {
    thread_samples.push_back(17);
  }
  if (threads > 1) {
    thread_samples.push_back(threads - 1);
  }

  // One vector width for the whole statement: every access must stay
  // unit-strided inside the block (this is how the vectorizer decides).
  int64_t max_vector = local_size;
  for (const auto &access : probe.accesses) {
    max_vector = std::min<int64_t>(
        max_vector, 16 / std::max<int64_t>(1, access.elem_bytes));
  }
  int64_t vector = 1;
  for (int64_t cand = 16; cand >= 2; cand /= 2) {
    if (cand > max_vector || local_size % cand != 0) {
      continue;
    }
    std::vector<int64_t> block_samples{0};
    int64_t num_blocks = local_size / cand;
    if (num_blocks > 2) {
      block_samples.push_back(num_blocks / 2);
    }
    if (num_blocks > 1) {
      block_samples.push_back(num_blocks - 1);
    }
    bool contiguous = true;
    for (const auto &access : probe.accesses) {
      for (int64_t t : thread_samples) {
        for (int64_t q : block_samples) {
          auto base = eval(access.addr, q * cand, t);
          if (!base) {
            contiguous = false;
            break;
          }
          for (int64_t r = 1; r < cand && contiguous; ++r) {
            auto at_r = eval(access.addr, q * cand + r, t);
            if (!at_r || *at_r != *base + r) {
              contiguous = false;
            }
          }
          if (!contiguous) {
            break;
          }
        }
        if (!contiguous) {
          break;
        }
      }
      if (!contiguous) {
        break;
      }
    }
    if (contiguous) {
      vector = cand;
      break;
    }
  }
  int64_t steps = local_size / vector;

  std::vector<int64_t> step_samples{0};
  if (steps > 2) {
    step_samples.push_back(steps / 2);
  }
  if (steps > 1) {
    step_samples.push_back(steps - 1);
  }
  int64_t num_warps = (threads + 31) / 32;

  StatementTraffic traffic;
  for (const auto &access : probe.accesses) {
    traffic.issue += steps * access.repeat * threads * 16;

    int64_t segments_total = 0;
    for (int64_t q : step_samples) {
      int64_t local0 = q * vector;
      for (int64_t w = 0; w < num_warps; ++w) {
        std::set<int64_t> segments;
        for (int64_t lane = 0; lane < 32; ++lane) {
          int64_t t = w * 32 + lane;
          if (t >= threads) {
            break;
          }
          if (access.is_store && probe.replication_index.has_value()) {
            auto rep = eval(*probe.replication_index, local0, t);
            if (!rep) {
              return std::nullopt;
            }
            if (*rep != 0) {
              continue; // guarded replica: this lane is idle for stores
            }
          }
          auto addr = eval(access.addr, local0, t);
          if (!addr) {
            return std::nullopt;
          }
          int64_t first_byte = *addr * access.elem_bytes;
          int64_t last_byte = first_byte + vector * access.elem_bytes - 1;
          for (int64_t seg = first_byte / 128; seg <= last_byte / 128; ++seg) {
            segments.insert(seg);
          }
        }
        segments_total += static_cast<int64_t>(segments.size());
      }
    }
    int64_t sampled = static_cast<int64_t>(step_samples.size());
    int64_t segments_per_step = (segments_total + sampled - 1) / sampled;
    traffic.bw += steps * access.repeat * segments_per_step * 128;
  }
  return traffic;
}

/*! \brief Exception-safe wrapper: inversion/printing of pathological
 *  candidate layouts can throw deep inside the layout stack; every such
 *  case is simply outside the model. */
std::optional<StatementTraffic> ScoreStatement(const StatementProbe &probe,
                                               const VarNode *local_var,
                                               const VarNode *thread_var) {
  try {
    return ScoreStatementImpl(probe, local_var, thread_var);
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

/*! \brief Prepare a fragment<->global tl.copy for scoring: the local walk
 *  is the fragment's per-thread slot space, the thread map is the fragment
 *  layout, and the single access is the global side's affine address
 *  composed with the fragment's inverse layout. */
std::optional<StatementProbe>
BuildCopyProbe(const CopyNode *copy, const Fragment &frag_layout,
               bool frag_is_src, const Var &local_var, const Var &thread_var) {
  const Buffer &global = frag_is_src ? copy->dst : copy->src;
  const Array<Range> &frag_range =
      frag_is_src ? copy->src_range : copy->dst_range;
  const Array<Range> &global_range =
      frag_is_src ? copy->dst_range : copy->src_range;

  if (frag_layout->OutputDim() != 1) {
    return std::nullopt;
  }
  size_t ndim = frag_layout->InputShape().size();
  if (frag_range.size() != ndim || global_range.size() != ndim) {
    return std::nullopt;
  }
  // Whole-fragment copies only: the iteration space is the fragment's full
  // logical shape.
  int64_t logical_elements = 1;
  for (size_t d = 0; d < ndim; ++d) {
    if (!is_zero(frag_range[d]->min)) {
      return std::nullopt;
    }
    const int64_t *frag_extent = as_const_int(frag_range[d]->extent);
    const int64_t *global_extent = as_const_int(global_range[d]->extent);
    const int64_t *shape_extent = as_const_int(frag_layout->InputShape()[d]);
    if (!frag_extent || !global_extent || !shape_extent ||
        *frag_extent != *shape_extent || *global_extent != *shape_extent) {
      return std::nullopt;
    }
    logical_elements *= *shape_extent;
  }

  // Keep the expressions alive: as_const_int returns a pointer into the
  // node, and OutputShape()/ThreadExtent() build fresh temporaries.
  PrimExpr slots_expr = frag_layout->OutputShape()[0];
  PrimExpr threads_expr = frag_layout->ThreadExtent();
  PrimExpr rep_expr = frag_layout->ReplicateExtent();
  const int64_t *slots_ptr = as_const_int(slots_expr);
  const int64_t *threads_ptr = as_const_int(threads_expr);
  const int64_t *rep_ptr = as_const_int(rep_expr);
  if (!slots_ptr || !threads_ptr || !rep_ptr || *slots_ptr <= 0 ||
      *threads_ptr <= 0 || *rep_ptr <= 0) {
    return std::nullopt;
  }

  int64_t elem_bits = global->dtype.bits() * global->dtype.lanes();
  if (elem_bits < 8) {
    return std::nullopt;
  }
  auto strides = RowMajorStrides(global);
  if (!strides.has_value()) {
    return std::nullopt;
  }

  Array<PrimExpr> logical;
  try {
    Layout inverse = frag_layout->InverseWithLevel(false).first;
    logical = inverse->Forward({local_var, thread_var});
  } catch (const std::exception &e) {
    return std::nullopt;
  }
  if (logical.size() < ndim) {
    return std::nullopt;
  }

  PrimExpr addr = make_zero(DataType::Int(32));
  for (size_t d = 0; d < ndim; ++d) {
    addr = addr + (global_range[d]->min + logical[d]) *
                      IntImm(DataType::Int(32), (*strides)[d]);
  }

  StatementProbe probe;
  probe.local_size = *slots_ptr;
  probe.threads = *threads_ptr;
  probe.worst_elements = logical_elements;
  if (*rep_ptr > 1 && logical.size() > ndim) {
    probe.replication_index = logical[ndim];
  }
  GlobalAccessProbe access;
  access.addr = addr;
  access.elem_bytes = elem_bits / 8;
  access.is_store = frag_is_src;
  probe.accesses.push_back(std::move(access));
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
 *  (issue #1729, extension E1): the local walk is the loop layout's
 *  per-thread output space, the thread map is the loop layout itself, and
 *  each access composes its index expressions with the loop's inverse.
 *  Returns: probe with empty accesses (charge zero) when the loop touches
 *  no global memory; a full probe otherwise; nullopt when the loop cannot
 *  even be sized (skip — nothing sensible to charge). */
std::optional<StatementProbe> BuildLoopProbe(const ParallelOpNode *loop,
                                             const Var &local_var,
                                             const Var &thread_var) {
  LoopGlobalAccessCollector collector;
  collector.Collect(loop->GetRoot());
  StatementProbe probe;
  if (collector.accesses.empty()) {
    return probe; // pure fragment/shared loop: no global traffic to model
  }

  Fragment layout = loop->GetLoopLayout();
  if (!layout.defined()) {
    DLOG(INFO) << "[LayoutCost] loop layout undefined";
    return std::nullopt;
  }

  // Domain size for the worst-case fallback.
  int64_t domain = 1;
  for (const PrimExpr &extent : layout->InputShape()) {
    const int64_t *value = as_const_int(extent);
    if (!value) {
      return std::nullopt;
    }
    domain *= *value;
  }
  probe.worst_elements = domain;

  // Mark the probe un-scoreable but sized (worst-case) via empty
  // local_size when any structural requirement fails.
  auto worst_only = [&]() -> std::optional<StatementProbe> {
    for (const auto &raw : collector.accesses) {
      GlobalAccessProbe access;
      access.addr = PrimExpr(); // never evaluated
      access.elem_bytes = std::max<int64_t>(
          1, raw.buffer->dtype.bits() * raw.buffer->dtype.lanes() / 8);
      access.is_store = raw.is_store;
      access.repeat = raw.symbolic_repeat ? 1 : raw.repeat;
      probe.accesses.push_back(std::move(access));
    }
    probe.local_size = 0; // signals: charge WorstCaseBytes
    return probe;
  };

  if (layout->OutputDim() != 1) {
    return worst_only();
  }
  PrimExpr local_expr = layout->OutputShape()[0];
  PrimExpr threads_expr = layout->ThreadExtent();
  PrimExpr rep_expr = layout->ReplicateExtent();
  const int64_t *local_ptr = as_const_int(local_expr);
  const int64_t *threads_ptr = as_const_int(threads_expr);
  const int64_t *rep_ptr = as_const_int(rep_expr);
  if (!local_ptr || !threads_ptr || !rep_ptr || *local_ptr <= 0 ||
      *threads_ptr <= 0 || *rep_ptr <= 0) {
    return worst_only();
  }

  // Nest loop vars, outermost first (the inverse outputs use this order).
  Array<Var> nest_vars;
  {
    const ForNode *cur = loop->GetRoot().get();
    while (cur != nullptr && cur->kind == ForKind::kParallel) {
      nest_vars.push_back(cur->loop_var);
      cur = cur->body.as<ForNode>();
    }
  }
  if (nest_vars.size() != layout->InputShape().size()) {
    return worst_only();
  }

  Array<PrimExpr> logical;
  try {
    Layout inverse = layout->InverseWithLevel(false).first;
    logical = inverse->Forward({local_var, thread_var});
  } catch (const std::exception &e) {
    return worst_only();
  }
  if (logical.size() < nest_vars.size()) {
    return worst_only();
  }
  Map<Var, PrimExpr> substitution;
  for (size_t i = 0; i < nest_vars.size(); ++i) {
    substitution.Set(nest_vars[i], logical[i]);
  }
  probe.local_size = *local_ptr;
  probe.threads = *threads_ptr;
  if (*rep_ptr > 1 && logical.size() > nest_vars.size()) {
    probe.replication_index = logical[nest_vars.size()];
  }

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
      addr = addr + Substitute(raw.indices[d], substitution) *
                        IntImm(DataType::Int(32), (*strides)[d]);
    }
    GlobalAccessProbe access;
    access.addr = std::move(addr);
    access.elem_bytes = elem_bits / 8;
    access.is_store = raw.is_store;
    access.repeat = raw.repeat;
    probe.accesses.push_back(std::move(access));
  }
  return probe;
}

} // namespace

/*! \brief Score one complete attempt (layout RFC, design B2). The
 *  memory term walks the component's global-touching statements —
 *  fragment<->global copies and parallel loops with direct global
 *  accesses — and charges each one max(bw, issue) bytes under the
 *  tentative layouts; statements outside the model are charged the
 *  conservative worst case (an attempt must never profit from opacity).
 *  The register term is the legacy total slot count and doubles as the
 *  tiebreak (and as the entire score when the cost model is disabled,
 *  reproducing the legacy ordering exactly). */
AttemptCost ComputeAttemptCost(const std::vector<int> &members,
                               const std::vector<TileOperator> &infer_list,
                               const LayoutMap &tmp_layout_map,
                               bool cost_model_enabled) {
  AttemptCost cost;
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
      cost.regs += frag_reg_num;
    }
  }
  if (!cost_model_enabled) {
    return cost;
  }

  Var local_var("_cost_local", DataType::Int(32));
  Var thread_var("_cost_thread", DataType::Int(32));

  auto charge = [&](const std::optional<StatementProbe> &probe,
                    const char *what) {
    if (!probe.has_value() || probe->accesses.empty()) {
      return; // no global traffic to model / nothing sensible to charge
    }
    std::optional<StatementTraffic> traffic;
    if (probe->local_size > 0) {
      traffic = ScoreStatement(*probe, local_var.get(), thread_var.get());
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
        probe = BuildCopyProbe(copy, frag_layout.value(), frag_is_src,
                               local_var, thread_var);
      } catch (const std::exception &e) {
        probe = std::nullopt;
      }
      if (!probe.has_value()) {
        // Sized fallback: logical elements, one access, worst geometry.
        StatementProbe worst;
        worst.worst_elements = 1;
        const Array<Range> &range =
            frag_is_src ? copy->src_range : copy->dst_range;
        for (const Range &r : range) {
          const int64_t *extent = as_const_int(r->extent);
          worst.worst_elements *= extent ? *extent : 1;
        }
        worst.local_size = 0;
        worst.accesses.push_back(GlobalAccessProbe{});
        probe = std::move(worst);
      }
      charge(probe, "copy");
    } else if (const auto *loop = infer_list[idx].as<ParallelOpNode>()) {
      std::optional<StatementProbe> probe;
      try {
        probe = BuildLoopProbe(loop, local_var, thread_var);
      } catch (const std::exception &e) {
        probe = std::nullopt;
      }
      charge(probe, "parallel loop");
    }
  }
  return cost;
}

} // namespace tl
} // namespace tvm

/*!
 * \file planner_solver.cc
 * \brief Exact-search planner implementation for Sunmmio tile loop fusion.
 */

#include "planner_internal.h"

#include <tvm/node/structural_equal.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <tuple>
#include <unordered_set>

namespace tvm {
namespace tl {
namespace planner_internal {

namespace {

// Serialization helpers keep memo keys deterministic and human-readable in the
// same ordering used by the resident sorting logic below.
std::string JoinExtents(const Array<PrimExpr> &extents) {
  std::ostringstream os;
  for (size_t i = 0; i < extents.size(); ++i) {
    if (i != 0) {
      os << '/';
    }
    os << PrimExprToString(extents[i]);
  }
  return os.str();
}

std::string JoinAxes(const std::vector<std::string> &axes) {
  std::ostringstream os;
  for (size_t i = 0; i < axes.size(); ++i) {
    if (i != 0) {
      os << '/';
    }
    os << axes[i];
  }
  return os.str();
}

bool HasFuturePotentialConsumers(const WindowPlannerInput &input,
                                 const PlannerState &state,
                                 const ResidentValueState &resident) {
  if (resident.kind == ResidentValueKind::kRead) {
    auto it =
        input.read_consumer_masks_by_region_id.find(resident.buffer_region_id);
    return it != input.read_consumer_masks_by_region_id.end() &&
           HasAnyFutureBits(it->second, state.scheduled_mask);
  }
  RawConsumerKey key{resident.origin_region_local_index,
                     resident.buffer_region_id};
  auto it = input.raw_consumer_masks_by_key.find(key);
  return it != input.raw_consumer_masks_by_key.end() &&
         HasAnyFutureBits(it->second, state.scheduled_mask);
}

bool ResidentValueLess(const ResidentValueState &lhs,
                       const ResidentValueState &rhs) {
  return std::tie(lhs.kind, lhs.origin_region_local_index, lhs.buffer_region_id,
                  lhs.home_depth, lhs.payload_bytes, lhs.instance_count) <
         std::tie(rhs.kind, rhs.origin_region_local_index, rhs.buffer_region_id,
                  rhs.home_depth, rhs.payload_bytes, rhs.instance_count);
}

bool RetainsPendingIncomingEdge(const WindowPlannerInput &input,
                                const PlannerState &state, int close_to_depth,
                                int region_local_index) {
  const std::vector<int> &incoming_edges =
      input.incoming_edges_by_dst[region_local_index];
  for (int depth = 0; depth < close_to_depth; ++depth) {
    const std::vector<int> &pending_edges =
        state.open_scopes[depth].pending_edge_indices;
    for (int edge_index : incoming_edges) {
      if (std::binary_search(pending_edges.begin(), pending_edges.end(),
                             edge_index)) {
        return true;
      }
    }
  }
  return false;
}

void InstallPendingEdgeIfMissing(std::vector<OpenScopeFrame> *open_scopes,
                                 int frame_index, int edge_index) {
  ICHECK_GE(frame_index, 0);
  ICHECK_LT(frame_index, static_cast<int>(open_scopes->size()));
  std::vector<int> &pending_edges =
      (*open_scopes)[frame_index].pending_edge_indices;
  auto it =
      std::lower_bound(pending_edges.begin(), pending_edges.end(), edge_index);
  if (it == pending_edges.end() || *it != edge_index) {
    pending_edges.insert(it, edge_index);
  }
}

int FindPendingFrameIndex(const WindowPlannerEdgeInfo &edge,
                          const TileScopeRegion &source_region,
                          int open_to_depth) {
  ICHECK_GE(edge.max_shared_execution_depth, 0);
  ICHECK_LE(open_to_depth,
            static_cast<int>(source_region.execution_loop_extents.size()));
  for (int depth = edge.max_shared_execution_depth + 1; depth <= open_to_depth;
       ++depth) {
    const auto *extent =
        source_region.execution_loop_extents[depth - 1].as<IntImmNode>();
    if (extent == nullptr || extent->value != 1) {
      return depth - 1;
    }
  }
  return -1;
}

struct MutablePlannerTreeNode {
  bool is_scope{false};
  int region_index{-1};
  std::vector<std::string> shell_axes;
  Array<PrimExpr> shell_extents;
  std::vector<std::shared_ptr<MutablePlannerTreeNode>> children;
};

SunmmioTileLoopFusionPlannerTreeNode
FreezeTree(const std::shared_ptr<MutablePlannerTreeNode> &node) {
  SunmmioTileLoopFusionPlannerTreeNode frozen;
  frozen.is_scope = node->is_scope;
  frozen.region_index = node->region_index;
  frozen.shell_axes = node->shell_axes;
  frozen.shell_extents = node->shell_extents;
  for (const std::shared_ptr<MutablePlannerTreeNode> &child : node->children) {
    frozen.children.push_back(FreezeTree(child));
  }
  return frozen;
}

} // namespace

void SetBit(DynamicBitset *bitset, int index) {
  ICHECK_GE(index, 0);
  ICHECK_LT(index, bitset->num_bits) << "DynamicBitset index out of bounds";
  bitset->words[index / 64] |= (uint64_t{1} << (index % 64));
}

bool TestBit(const DynamicBitset &bitset, int index) {
  ICHECK_GE(index, 0);
  ICHECK_LT(index, bitset.num_bits) << "DynamicBitset index out of bounds";
  return ((bitset.words[index / 64] >> (index % 64)) & uint64_t{1}) != 0;
}

bool ContainsAll(const DynamicBitset &required, const DynamicBitset &present) {
  ICHECK_EQ(required.num_bits, present.num_bits);
  for (size_t i = 0; i < required.words.size(); ++i) {
    if ((required.words[i] & ~present.words[i]) != 0) {
      return false;
    }
  }
  return true;
}

bool HasAnyFutureBits(const DynamicBitset &candidate,
                      const DynamicBitset &scheduled) {
  ICHECK_EQ(candidate.num_bits, scheduled.num_bits);
  for (size_t i = 0; i < candidate.words.size(); ++i) {
    if ((candidate.words[i] & ~scheduled.words[i]) != 0) {
      return true;
    }
  }
  return false;
}

int CountBits(const DynamicBitset &bitset) {
  int count = 0;
  int remaining_bits = bitset.num_bits;
  for (uint64_t word : bitset.words) {
    uint64_t masked = word;
    if (remaining_bits < 64) {
      masked &=
          (remaining_bits <= 0) ? 0 : ((uint64_t{1} << remaining_bits) - 1);
    }
    count += __builtin_popcountll(masked);
    remaining_bits -= 64;
  }
  return count;
}

int CountMissingBits(const DynamicBitset &candidate,
                     const DynamicBitset &present) {
  ICHECK_EQ(candidate.num_bits, present.num_bits);
  DynamicBitset missing(candidate.num_bits);
  for (size_t i = 0; i < candidate.words.size(); ++i) {
    missing.words[i] = candidate.words[i] & ~present.words[i];
  }
  return CountBits(missing);
}

std::string SerializeDynamicBitset(const DynamicBitset &bitset) {
  std::ostringstream os;
  os << bitset.num_bits << ':';
  for (uint64_t word : bitset.words) {
    os << word << ';';
  }
  return os.str();
}

bool SameResidentValue(const ResidentValueState &lhs,
                       const ResidentValueState &rhs) {
  return lhs.kind == rhs.kind &&
         lhs.origin_region_local_index == rhs.origin_region_local_index &&
         lhs.buffer_region_id == rhs.buffer_region_id &&
         lhs.home_depth == rhs.home_depth &&
         lhs.payload_bytes == rhs.payload_bytes &&
         lhs.instance_count == rhs.instance_count;
}

std::string SerializeResident(const ResidentValueState &resident) {
  std::ostringstream os;
  os << static_cast<int>(resident.kind) << ':'
     << resident.origin_region_local_index << ':' << resident.buffer_region_id
     << ':' << resident.buffer_name << ':' << resident.home_depth << ':'
     << resident.payload_bytes << ':' << resident.instance_count;
  return os.str();
}

std::string SerializePlannerState(const PlannerState &state) {
  std::ostringstream os;
  os << SerializeDynamicBitset(state.scheduled_mask);
  for (const OpenScopeFrame &frame : state.open_scopes) {
    os << '[' << JoinAxes(frame.shell_axes) << '@'
       << JoinExtents(frame.shell_extents) << '#' << frame.execution_rank
       << "|P:";
    for (int edge_index : frame.pending_edge_indices) {
      os << edge_index << ',';
    }
    os << "|R:";
    for (const ResidentValueState &resident : frame.residents) {
      os << SerializeResident(resident) << ',';
    }
    os << ']';
  }
  return os.str();
}

std::vector<std::string>
TakeExecutionAxisPrefix(const std::vector<std::string> &axes, int depth) {
  return std::vector<std::string>(axes.begin(), axes.begin() + depth);
}

Array<PrimExpr> TakeExecutionExtentPrefix(const Array<PrimExpr> &extents,
                                          int depth) {
  Array<PrimExpr> prefix;
  for (int i = 0; i < depth; ++i) {
    prefix.push_back(extents[i]);
  }
  return prefix;
}

bool PathMatchesExecutionPrefix(
    const std::vector<OpenScopeFrame> &open_scopes, int close_to_depth,
    const std::vector<std::string> &region_execution_axes,
    const Array<PrimExpr> &region_execution_extents) {
  if (close_to_depth > static_cast<int>(open_scopes.size()) ||
      close_to_depth > static_cast<int>(region_execution_axes.size()) ||
      close_to_depth > static_cast<int>(region_execution_extents.size())) {
    return false;
  }
  StructuralEqual equal;
  int region_execution_rank = static_cast<int>(region_execution_axes.size());
  for (int depth = 1; depth <= close_to_depth; ++depth) {
    const OpenScopeFrame &frame = open_scopes[depth - 1];
    if (frame.shell_axes.size() != static_cast<size_t>(depth) ||
        frame.shell_extents.size() != static_cast<size_t>(depth) ||
        frame.execution_rank != region_execution_rank) {
      return false;
    }
    if (frame.shell_axes[depth - 1] != region_execution_axes[depth - 1]) {
      return false;
    }
    if (!equal(frame.shell_extents[depth - 1],
               region_execution_extents[depth - 1])) {
      return false;
    }
  }
  return true;
}

bool HasAccessibleResident(const std::vector<OpenScopeFrame> &open_scopes,
                           int attach_depth, int buffer_region_id,
                           int required_depth) {
  int visible_depth =
      std::min(attach_depth, static_cast<int>(open_scopes.size()));
  for (int depth = visible_depth; depth >= 1; --depth) {
    const OpenScopeFrame &frame = open_scopes[depth - 1];
    for (const ResidentValueState &resident : frame.residents) {
      if (resident.buffer_region_id == buffer_region_id &&
          resident.home_depth >= required_depth) {
        return true;
      }
    }
  }
  return false;
}

bool HasAccessibleDefinitionResident(
    const std::vector<OpenScopeFrame> &open_scopes, int attach_depth,
    int origin_region_local_index, int buffer_region_id, int required_rho) {
  int visible_depth =
      std::min(attach_depth, static_cast<int>(open_scopes.size()));
  for (int depth = visible_depth; depth >= 1; --depth) {
    const OpenScopeFrame &frame = open_scopes[depth - 1];
    for (const ResidentValueState &resident : frame.residents) {
      if (resident.kind == ResidentValueKind::kDefinition &&
          resident.origin_region_local_index == origin_region_local_index &&
          resident.buffer_region_id == buffer_region_id &&
          resident.home_depth >= required_rho) {
        return true;
      }
    }
  }
  return false;
}

void InstallResidentIfMissing(std::vector<OpenScopeFrame> *open_scopes,
                              const ResidentValueState &resident) {
  if (resident.home_depth <= 0 ||
      resident.home_depth > static_cast<int>(open_scopes->size())) {
    return;
  }
  OpenScopeFrame &frame = (*open_scopes)[resident.home_depth - 1];
  auto it = std::lower_bound(frame.residents.begin(), frame.residents.end(),
                             resident, ResidentValueLess);
  if (it != frame.residents.end() && SameResidentValue(*it, resident)) {
    return;
  }
  frame.residents.insert(it, resident);
}

void KillResidentsForBuffer(std::vector<OpenScopeFrame> *open_scopes,
                            const std::string &buffer_name) {
  for (OpenScopeFrame &frame : *open_scopes) {
    auto &residents = frame.residents;
    residents.erase(std::remove_if(residents.begin(), residents.end(),
                                   [&](const ResidentValueState &resident) {
                                     return resident.buffer_name == buffer_name;
                                   }),
                    residents.end());
  }
}

void PruneDeadResidents(const WindowPlannerInput &input, PlannerState *state) {
  for (OpenScopeFrame &frame : state->open_scopes) {
    auto &residents = frame.residents;
    residents.erase(std::remove_if(residents.begin(), residents.end(),
                                   [&](const ResidentValueState &resident) {
                                     return !HasFuturePotentialConsumers(
                                         input, *state, resident);
                                   }),
                    residents.end());
  }
}

void AccumulatePlannerScoreTerm(int64_t *field, int64_t delta) {
  *field = SaturatingAddSunmmioTileLoopFusionPlannerCost(*field, delta);
}

int64_t ComputeLiveRangeDelta(const PlannerState &state) {
  // Live-range penalty is measured in resident bytes kept alive across the
  // currently open scopes, scaled by the number of execution instances each
  // resident represents.
  int64_t live_range_delta = 0;
  for (const OpenScopeFrame &frame : state.open_scopes) {
    for (const ResidentValueState &resident : frame.residents) {
      AccumulatePlannerScoreTerm(
          &live_range_delta,
          SaturatingMulSunmmioTileLoopFusionPlannerCost(
              resident.payload_bytes, resident.instance_count));
    }
  }
  return live_range_delta;
}

/*!
 * \brief Apply one candidate scheduling action to the current planner state.
 *
 * \details
 * This function is the one-step state transition used by
 * \ref SolveWindowPlan. It does not choose the best action; it evaluates
 * one concrete choice `(region_local_index, close_to_depth, open_to_depth)`
 * and returns both the immediate planner-cost delta and the resulting
 * planner state.
 *
 * The transition has five jobs:
 * 1. Close the existing shell stack to the requested depth and reopen the
 *    candidate region's shell prefix until the attach depth.
 * 2. Inspect incoming RAW edges to determine which dependences remain
 *    internal and which ones are cut, charging write-cut cost for the
 *    latter.
 * 3. Materialize uncovered reads that are not already resident, charging
 *    shared-read cost when they must be fetched.
 * 4. Kill overwritten residents, install new definitions, and record outgoing
 *    dependences that remain incomplete under a deeper multi-trip shell.
 * 5. Mark the region scheduled, prune dead residents, and recompute the
 *    live-range and reorder score terms for the next state.
 *
 * \param input Immutable window-local planning problem.
 * \param state Planner state before applying the action.
 * \param region_local_index Window-local region to schedule.
 * \param close_to_depth Number of currently open shells to keep.
 * \param open_to_depth Depth where the candidate region attaches after
 * reopening any needed shells.
 * \return The next planner state plus the immediate cost delta charged
 * by this action.
 */
TransitionResult ApplyAction(const WindowPlannerInput &input,
                             const PlannerState &state, int region_local_index,
                             int close_to_depth, int open_to_depth) {
  ICHECK_GE(region_local_index, 0);
  ICHECK_LT(region_local_index, static_cast<int>(input.regions.size()));
  ICHECK(input.problem != nullptr);
  ICHECK_GE(close_to_depth, 0);
  ICHECK_LE(close_to_depth, static_cast<int>(state.open_scopes.size()));
  ICHECK_GE(open_to_depth, close_to_depth);

  const WindowPlannerRegionInfo &region_view =
      input.regions[region_local_index];
  const TileScopeRegion &region = input.problem->regions[region_local_index];
  ICHECK_LE(open_to_depth,
            static_cast<int>(region.logical_execution_axis_keys.size()));
  ICHECK_LE(open_to_depth,
            static_cast<int>(region.execution_loop_extents.size()));

  TransitionResult result{state, {0, 0, 0, 0}};

  // Rebuild the shell stack seen by the candidate region: keep the first
  // `close_to_depth` shells from the current state, then open any deeper
  // prefixes needed to attach this region at `open_to_depth`.
  result.next_state.open_scopes.resize(close_to_depth);
  int execution_rank =
      static_cast<int>(region.logical_execution_axis_keys.size());
  for (int depth = close_to_depth + 1; depth <= open_to_depth; ++depth) {
    result.next_state.open_scopes.push_back(
        {TakeExecutionAxisPrefix(region.logical_execution_axis_keys, depth),
         TakeExecutionExtentPrefix(region.execution_loop_extents, depth),
         execution_rank,
         {},
         {}});
  }

  // RAW dependences both constrain legality/profitability and tell us which
  // destination uses are already covered by a producer definition. If the
  // required producer definition is not visible at this attach depth, the edge
  // is cut and contributes write-cut cost.
  std::unordered_set<int> raw_covered_use_indices;
  for (int edge_index : input.incoming_edges_by_dst[region_local_index]) {
    const WindowPlannerEdgeInfo &edge = input.edges[edge_index];
    if (edge.kind != TileScopeDependenceKind::kRAW) {
      continue;
    }
    if (edge.covered_use_index >= 0) {
      raw_covered_use_indices.insert(edge.covered_use_index);
    }
    if (!HasAccessibleDefinitionResident(result.next_state.open_scopes,
                                         open_to_depth, edge.src_local_index,
                                         edge.buffer_region_id, edge.rho)) {
      AccumulatePlannerScoreTerm(&result.delta.write_cut_cost,
                                 SaturatingMulSunmmioTileLoopFusionPlannerCost(
                                     edge.weight, edge.instance_count));
    }
    if (edge.rho <= open_to_depth) {
      InstallResidentIfMissing(&result.next_state.open_scopes,
                               {ResidentValueKind::kDefinition,
                                edge.src_local_index, edge.buffer_region_id,
                                edge.buffer_name, edge.rho, edge.weight,
                                edge.instance_count});
    }
  }

  // Any input use not already satisfied by a covered RAW edge must either find
  // a resident value in the visible shell prefix or pay shared-read cost to be
  // materialized for this action.
  for (size_t use_index = 0; use_index < region_view.use_in.size();
       ++use_index) {
    const PlannerBufferValueInfo &use_info = region_view.use_in[use_index];
    if (raw_covered_use_indices.count(static_cast<int>(use_index)) != 0) {
      continue;
    }
    if (HasAccessibleResident(result.next_state.open_scopes, open_to_depth,
                              use_info.buffer_region_id, use_info.home_depth)) {
      continue;
    }
    AccumulatePlannerScoreTerm(
        &result.delta.shared_read_cost,
        SaturatingMulSunmmioTileLoopFusionPlannerCost(use_info.payload_bytes,
                                                      use_info.instance_count));
    if (use_info.home_depth <= open_to_depth) {
      InstallResidentIfMissing(&result.next_state.open_scopes,
                               {ResidentValueKind::kRead, -1,
                                use_info.buffer_region_id, use_info.buffer_name,
                                use_info.home_depth, use_info.payload_bytes,
                                use_info.instance_count});
    }
  }

  // Definitions produced by the current region overwrite older residents for
  // the same buffer, then seed the new visible definition residents that later
  // actions may reuse. The outgoing RAW walk ensures we also install any
  // producer definitions that become reusable exactly at their dependence rho.
  for (const PlannerBufferValueInfo &def_info : region_view.def_out) {
    KillResidentsForBuffer(&result.next_state.open_scopes,
                           def_info.buffer_name);
  }
  for (const PlannerBufferValueInfo &def_info : region_view.def_out) {
    if (def_info.home_depth <= open_to_depth) {
      InstallResidentIfMissing(
          &result.next_state.open_scopes,
          {ResidentValueKind::kDefinition, region_local_index,
           def_info.buffer_region_id, def_info.buffer_name, def_info.home_depth,
           def_info.payload_bytes, def_info.instance_count});
    }
  }
  for (int edge_index : input.outgoing_edges_by_src[region_local_index]) {
    const WindowPlannerEdgeInfo &edge = input.edges[edge_index];
    if (edge.kind != TileScopeDependenceKind::kRAW ||
        edge.rho > open_to_depth) {
      continue;
    }
    InstallResidentIfMissing(&result.next_state.open_scopes,
                             {ResidentValueKind::kDefinition,
                              region_local_index, edge.buffer_region_id,
                              edge.buffer_name, edge.rho, edge.weight,
                              edge.instance_count});
  }

  // An edge whose source is emitted below its maximum legal shared depth stays
  // pending until the first deeper multi-trip frame closes. Discovery may
  // raise that depth for a WAR edge when every destination write is proven to
  // occur on the final iteration, which preserves safe read-before-write
  // fusion without weakening the default barrier. Static unit-trip shells
  // cannot interleave executions and therefore remain legally fused.
  for (int edge_index : input.outgoing_edges_by_src[region_local_index]) {
    const WindowPlannerEdgeInfo &edge = input.edges[edge_index];
    int pending_frame_index =
        FindPendingFrameIndex(edge, region, open_to_depth);
    if (pending_frame_index < 0) {
      continue;
    }
    InstallPendingEdgeIfMissing(&result.next_state.open_scopes,
                                pending_frame_index, edge_index);
  }

  // Finalize the new planner state and score terms after scheduling this
  // region.
  result.next_state.scheduled_mask = state.scheduled_mask;
  SetBit(&result.next_state.scheduled_mask, region_local_index);
  PruneDeadResidents(input, &result.next_state);
  result.delta.live_range_penalty = ComputeLiveRangeDelta(result.next_state);
  result.delta.reorder_penalty = CountMissingBits(
      input.earlier_source_masks[region_local_index], state.scheduled_mask);
  return result;
}

MemoResult BuildSourceOrderFallbackPlan(const WindowPlannerInput &input) {
  // Fallback scheduling simply emits the next legal region in source order and
  // attaches it at the root. This is intentionally conservative but bounded.
  MemoResult fallback;
  fallback.score = {0, 0, 0, 0};
  PlannerState state{DynamicBitset(static_cast<int>(input.regions.size())), {}};

  int scheduled_count = 0;
  while (scheduled_count < static_cast<int>(input.regions.size())) {
    bool progressed = false;
    for (int region_local_index = 0;
         region_local_index < static_cast<int>(input.regions.size());
         ++region_local_index) {
      if (TestBit(state.scheduled_mask, region_local_index)) {
        continue;
      }
      if (!ContainsAll(input.predecessor_masks[region_local_index],
                       state.scheduled_mask)) {
        continue;
      }

      TransitionResult transition =
          ApplyAction(input, state, region_local_index, 0, 0);
      fallback.score = AddSunmmioTileLoopFusionPlannerScores(fallback.score,
                                                             transition.delta);

      SunmmioTileLoopFusionPlannerAction action;
      action.region_index =
          input.regions[region_local_index].global_region_index;
      action.close_to_depth = 0;
      action.open_to_depth = 0;
      fallback.actions.push_back(std::move(action));

      state = std::move(transition.next_state);
      ++scheduled_count;
      progressed = true;
      break;
    }
    ICHECK(progressed)
        << "Expected a legal next region while building fallback plan";
  }

  return fallback;
}

/*!
 * \brief Solve the optimal remaining schedule suffix from the current planner
 * state.
 *
 * \details
 * This is the exact memoized search for one planning window. The memo key is
 * the full \c PlannerState rather than just the scheduled-region mask because
 * future costs depend on both the currently open shared shells and the
 * resident values attached to them.
 *
 * The recurrence is:
 * \code
 *   best(state) = min_a delta(state, a) + best(next_state(state, a))
 * \endcode
 * over all legal actions \c a = (region_local_index, close_to_depth,
 * open_to_depth).
 *
 * A legal action chooses an unscheduled region whose predecessors are already
 * scheduled, closes the current shell stack to \p close_to_depth, and then
 * reopens shells for the candidate region until \p open_to_depth. Prefix reuse
 * is only legal when the kept shells match the candidate region's execution
 * prefix axes and extents, and when no incoming dependence remains pending in
 * that retained prefix.
 *
 * Because every planner score term is nonnegative, the search can prune any
 * branch whose immediate transition delta is already no better than the best
 * complete plan found so far.
 *
 * \param input Immutable window-local planning problem.
 * \param state Current scheduled mask, open-shell stack, and resident values.
 * \param context Memo table and exhaustion flag shared across recursive calls.
 * \return The minimum remaining planner score and the action trace that
 * achieves it.
 */
MemoResult SolveWindowPlan(const WindowPlannerInput &input,
                           const PlannerState &state,
                           PlannerSearchContext *context) {
  if (context->exhausted) {
    return {MakeInfiniteSunmmioTileLoopFusionPlannerScore(), {}};
  }

  // Memoize by the full planner state so repeated subproblems are solved once.
  std::string key = SerializePlannerState(state);
  auto it = context->memo.find(key);
  if (it != context->memo.end()) {
    return it->second;
  }
  // Cap exact search once the memo grows too large; the caller will fall back
  // to the simpler source-order plan if this happens.
  if (context->memo.size() >= kMaxPlannerMemoEntries) {
    context->exhausted = true;
    return {MakeInfiniteSunmmioTileLoopFusionPlannerScore(), {}};
  }

  // Base case: once every region in the window has been scheduled, the
  // remaining suffix contributes zero cost and no further actions.
  bool all_scheduled = true;
  ICHECK(input.problem != nullptr);
  for (int region_local_index = 0;
       region_local_index < static_cast<int>(input.regions.size());
       ++region_local_index) {
    if (!TestBit(state.scheduled_mask, region_local_index)) {
      all_scheduled = false;
      break;
    }
  }
  if (all_scheduled) {
    MemoResult done;
    done.score = {0, 0, 0, 0};
    context->memo.emplace(key, done);
    return done;
  }

  MemoResult best;
  best.score = MakeInfiniteSunmmioTileLoopFusionPlannerScore();

  // Enumerate every dependence-legal next region and every legal way to attach
  // it under the current shell stack.
  for (int region_local_index = 0;
       region_local_index < static_cast<int>(input.regions.size());
       ++region_local_index) {
    if (TestBit(state.scheduled_mask, region_local_index)) {
      continue;
    }
    if (!ContainsAll(input.predecessor_masks[region_local_index],
                     state.scheduled_mask)) {
      continue;
    }

    const WindowPlannerRegionInfo &region_view =
        input.regions[region_local_index];
    const TileScopeRegion &region = input.problem->regions[region_local_index];
    for (int close_to_depth = 0;
         close_to_depth <= static_cast<int>(state.open_scopes.size());
         ++close_to_depth) {
      // `close_to_depth` keeps the first N currently open shells and closes
      // anything deeper. Only prefixes that match the region's execution
      // prefix are legal to keep.
      if (!PathMatchesExecutionPrefix(state.open_scopes, close_to_depth,
                                      region.logical_execution_axis_keys,
                                      region.execution_loop_extents)) {
        continue;
      }
      if (RetainsPendingIncomingEdge(input, state, close_to_depth,
                                     region_local_index)) {
        continue;
      }
      for (int open_to_depth = close_to_depth;
           open_to_depth <=
           static_cast<int>(region.logical_execution_axis_keys.size());
           ++open_to_depth) {
        // `open_to_depth` extends the kept prefix to the depth where this
        // region attaches. ApplyAction computes the one-step cost and the next
        // planner state induced by that decision.
        TransitionResult transition = ApplyAction(
            input, state, region_local_index, close_to_depth, open_to_depth);
        // Planner score terms are nonnegative, so any suffix can only increase
        // the current transition cost. Once a complete incumbent exists, skip
        // branches whose immediate delta is already no better than that bound.
        if (CompareSunmmioTileLoopFusionPlannerScores(transition.delta,
                                                      best.score) >= 0) {
          continue;
        }
        MemoResult suffix =
            SolveWindowPlan(input, transition.next_state, context);
        if (context->exhausted) {
          return {MakeInfiniteSunmmioTileLoopFusionPlannerScore(), {}};
        }
        SunmmioTileLoopFusionPlannerScore total =
            AddSunmmioTileLoopFusionPlannerScores(transition.delta,
                                                  suffix.score);
        if (CompareSunmmioTileLoopFusionPlannerScores(total, best.score) >= 0) {
          continue;
        }

        SunmmioTileLoopFusionPlannerAction action;
        action.region_index = region_view.global_region_index;
        action.close_to_depth = close_to_depth;
        action.open_to_depth = open_to_depth;
        // Record the shells opened by this step so the final action trace can
        // be turned back into the rewrite-facing tree later.
        for (int depth = close_to_depth + 1; depth <= open_to_depth; ++depth) {
          action.opened_shells.push_back(TakeExecutionAxisPrefix(
              region.logical_execution_axis_keys, depth));
          action.opened_shell_extents.push_back(
              TakeExecutionExtentPrefix(region.execution_loop_extents, depth));
        }

        // This action plus its optimal suffix is the best complete plan seen
        // from the current state so far.
        best.score = total;
        best.actions.clear();
        best.actions.push_back(std::move(action));
        best.actions.insert(best.actions.end(), suffix.actions.begin(),
                            suffix.actions.end());
      }
    }
  }

  context->memo.emplace(key, best);
  return best;
}

std::vector<SunmmioTileLoopFusionPlannerTreeNode>
BuildPlanTree(const std::vector<SunmmioTileLoopFusionPlannerAction> &actions) {
  // Reconstruct the nested rewrite-facing tree by replaying the linear action
  // trace's close/open shell operations against a mutable stack of scope nodes.
  auto root = std::make_shared<MutablePlannerTreeNode>();
  root->is_scope = true;

  std::vector<std::shared_ptr<MutablePlannerTreeNode>> open_path;
  open_path.push_back(root);
  for (const SunmmioTileLoopFusionPlannerAction &action : actions) {
    while (static_cast<int>(open_path.size()) - 1 > action.close_to_depth) {
      open_path.pop_back();
    }
    for (size_t shell_index = 0; shell_index < action.opened_shells.size();
         ++shell_index) {
      auto scope_node = std::make_shared<MutablePlannerTreeNode>();
      scope_node->is_scope = true;
      scope_node->shell_axes = action.opened_shells[shell_index];
      if (shell_index < action.opened_shell_extents.size()) {
        scope_node->shell_extents = action.opened_shell_extents[shell_index];
      }
      open_path.back()->children.push_back(scope_node);
      open_path.push_back(scope_node);
    }

    auto region_node = std::make_shared<MutablePlannerTreeNode>();
    region_node->is_scope = false;
    region_node->region_index = action.region_index;
    open_path.back()->children.push_back(region_node);
  }

  std::vector<SunmmioTileLoopFusionPlannerTreeNode> tree;
  for (const std::shared_ptr<MutablePlannerTreeNode> &child : root->children) {
    tree.push_back(FreezeTree(child));
  }
  return tree;
}

} // namespace planner_internal
} // namespace tl
} // namespace tvm

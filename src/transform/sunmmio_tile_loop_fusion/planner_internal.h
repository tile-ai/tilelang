/*!
 * \file planner_internal.h
 * \brief Private planner preprocessing types and exact-search state for
 * Sunmmio tile loop fusion.
 *
 * Callers only interact with \ref planner.h. This header collects the planner's
 * internal working IR: compact bitsets, window-local edge/value summaries,
 * resident-value state, and the memo/search records used by the exact solver.
 */
#pragma once

#include "planner.h"
#include "utils.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tl {
namespace planner_internal {

// Exact search is intentionally capped. Larger windows fall back to the simpler
// source-order plan rather than exploding the memoized state space.
inline constexpr int kMaxExactPlannerRegions = 15;
inline constexpr size_t kMaxPlannerMemoEntries = 200000;

/*! \brief Minimal dynamic bitset used for scheduled/pending region masks. */
struct DynamicBitset {
  int num_bits{0};
  std::vector<uint64_t> words;

  DynamicBitset() = default;
  explicit DynamicBitset(int num_bits)
      : num_bits(num_bits), words((num_bits + 63) / 64, 0) {}
};

/*! \brief Mark one region id as present in a dynamic bitset. */
void SetBit(DynamicBitset *bitset, int index);
/*! \brief Test whether one region id is present in a dynamic bitset. */
bool TestBit(const DynamicBitset &bitset, int index);
/*! \brief Return true when every bit in \p required is set in \p present. */
bool ContainsAll(const DynamicBitset &required, const DynamicBitset &present);
/*! \brief Return true when \p candidate contains an unscheduled region bit. */
bool HasAnyFutureBits(const DynamicBitset &candidate,
                      const DynamicBitset &scheduled);
/*! \brief Count the number of set bits. */
int CountBits(const DynamicBitset &bitset);
/*! \brief Count bits set in \p candidate but missing from \p present. */
int CountMissingBits(const DynamicBitset &candidate,
                     const DynamicBitset &present);
/*! \brief Serialize one bitset for memo keys and debugging. */
std::string SerializeDynamicBitset(const DynamicBitset &bitset);

/*!
 * \brief Planner-local summary of one normalized external use or definition.
 *
 * Discovery has already resolved the buffer region identity, its reusable home
 * depth, and the payload size. The solver only needs these compact facts.
 */
struct PlannerBufferValueInfo {
  int buffer_region_id{-1};
  std::string buffer_name;
  int home_depth{0};
  int64_t payload_bytes{0};
  int64_t instance_count{1};
};

/*!
 * \brief Planner-local view of one region.
 *
 * This strips the full TIR region record down to the use/def facts needed by
 * the exact search and fallback scheduler.
 */
struct WindowPlannerRegionInfo {
  int global_region_index{-1};
  std::vector<PlannerBufferValueInfo> use_in;
  std::vector<PlannerBufferValueInfo> def_out;
};

/*!
 * \brief Planner-local dependence edge with precomputed multiplicity data.
 *
 * Each edge carries the legality kind, reuse depth, maximum ordering-safe
 * shared depth, and the access/region ids needed during scheduling.
 */
struct WindowPlannerEdgeInfo {
  int src_local_index{-1};
  int dst_local_index{-1};
  TileScopeDependenceKind kind{TileScopeDependenceKind::kRAW};
  int buffer_region_id{-1};
  std::string buffer_name;
  int rho{0};
  int max_shared_execution_depth{0};
  int64_t weight{0};
  int64_t instance_count{1};
  int covered_use_index{-1};
};

/*!
 * \brief Key for future RAW consumers satisfied by one resident definition.
 */
struct RawConsumerKey {
  int origin_region_local_index{-1};
  int buffer_region_id{-1};

  bool operator==(const RawConsumerKey &other) const {
    return origin_region_local_index == other.origin_region_local_index &&
           buffer_region_id == other.buffer_region_id;
  }
};

struct RawConsumerKeyHash {
  std::size_t operator()(const RawConsumerKey &key) const {
    std::size_t seed = std::hash<int>{}(key.origin_region_local_index);
    seed ^= std::hash<int>{}(key.buffer_region_id) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2);
    return seed;
  }
};

/*!
 * \brief Immutable planner input for one window.
 *
 * This is the exact solver's working IR: discovery/preprocessing has already
 * normalized accesses, localized region ids, built predecessor masks, and
 * precomputed the consumer masks needed for resident-value pruning.
 */
struct WindowPlannerInput {
  const SunmmioTileLoopFusionWindowProblem *problem{nullptr};
  std::vector<WindowPlannerRegionInfo> regions;
  std::vector<WindowPlannerEdgeInfo> edges;
  std::vector<std::vector<int>> incoming_edges_by_dst;
  std::vector<std::vector<int>> outgoing_edges_by_src;
  std::vector<DynamicBitset> predecessor_masks;
  std::vector<DynamicBitset> earlier_source_masks;
  std::unordered_map<int, DynamicBitset> read_consumer_masks_by_region_id;
  std::unordered_map<RawConsumerKey, DynamicBitset, RawConsumerKeyHash>
      raw_consumer_masks_by_key;
};

/*! \brief Resident values are tracked either as reusable definitions or reads.
 */
enum class ResidentValueKind : int {
  kDefinition = 0,
  kRead = 1,
};

/*!
 * \brief One currently visible value attached to an open shared shell.
 *
 * Residents model the information the planner may keep alive across actions:
 * reusable producer definitions and already materialized reads.
 */
struct ResidentValueState {
  ResidentValueKind kind{ResidentValueKind::kDefinition};
  int origin_region_local_index{-1};
  int buffer_region_id{-1};
  std::string buffer_name;
  int home_depth{0};
  int64_t payload_bytes{0};
  int64_t instance_count{1};
};

/*!
 * \brief One open shared shell frame, its residents, and incomplete edges.
 */
struct OpenScopeFrame {
  std::vector<std::string> shell_axes;
  Array<PrimExpr> shell_extents;
  // Complete execution rank of the region that opened this shell. A shared
  // prefix is only codegen-compatible when attached regions have equal rank.
  int execution_rank{0};
  std::vector<ResidentValueState> residents;
  std::vector<int> pending_edge_indices;
};

/*! \brief Full memoized solver state for one partial schedule. */
struct PlannerState {
  DynamicBitset scheduled_mask;
  std::vector<OpenScopeFrame> open_scopes;
};

/*! \brief Compare two resident values for exact identity. */
bool SameResidentValue(const ResidentValueState &lhs,
                       const ResidentValueState &rhs);
/*! \brief Serialize one resident for memo keys and debugging. */
std::string SerializeResident(const ResidentValueState &resident);
/*! \brief Serialize the full planner state for memoization. */
std::string SerializePlannerState(const PlannerState &state);
/*! \brief Take the first \p depth logical execution axes. */
std::vector<std::string>
TakeExecutionAxisPrefix(const std::vector<std::string> &axes, int depth);
/*! \brief Take the first \p depth execution extents. */
Array<PrimExpr> TakeExecutionExtentPrefix(const Array<PrimExpr> &extents,
                                          int depth);
/*! \brief Check whether the kept open-scope prefix matches a region prefix. */
bool PathMatchesExecutionPrefix(
    const std::vector<OpenScopeFrame> &open_scopes, int close_to_depth,
    const std::vector<std::string> &region_execution_axes,
    const Array<PrimExpr> &region_execution_extents);
/*! \brief Check whether any resident value can satisfy a needed access. */
bool HasAccessibleResident(const std::vector<OpenScopeFrame> &open_scopes,
                           int attach_depth, int buffer_region_id,
                           int required_depth);
/*! \brief Check whether a matching producer definition resident is visible. */
bool HasAccessibleDefinitionResident(
    const std::vector<OpenScopeFrame> &open_scopes, int attach_depth,
    int origin_region_local_index, int buffer_region_id, int required_rho);
/*! \brief Attach a resident value to its home scope when not already present.
 */
void InstallResidentIfMissing(std::vector<OpenScopeFrame> *open_scopes,
                              const ResidentValueState &resident);
/*! \brief Remove all residents for one buffer after a new overwrite. */
void KillResidentsForBuffer(std::vector<OpenScopeFrame> *open_scopes,
                            const std::string &buffer_name);
/*! \brief Drop residents that no longer have any future unscheduled consumer.
 */
void PruneDeadResidents(const WindowPlannerInput &input, PlannerState *state);
/*! \brief Saturating helper for one planner score field. */
void AccumulatePlannerScoreTerm(int64_t *field, int64_t delta);
/*! \brief Compute the live resident footprint of a planner state. */
int64_t ComputeLiveRangeDelta(const PlannerState &state);

/*! \brief Result of applying one candidate action to a planner state. */
struct TransitionResult {
  PlannerState next_state;
  SunmmioTileLoopFusionPlannerScore delta;
};

TransitionResult ApplyAction(const WindowPlannerInput &input,
                             const PlannerState &state, int region_local_index,
                             int close_to_depth, int open_to_depth);

/*! \brief Memoized best suffix score plus the action trace that realizes it. */
struct MemoResult {
  SunmmioTileLoopFusionPlannerScore score;
  std::vector<SunmmioTileLoopFusionPlannerAction> actions;
};

/*! \brief Shared memo table and exhaustion flag for the recursive exact search.
 */
struct PlannerSearchContext {
  std::unordered_map<std::string, MemoResult> memo;
  bool exhausted{false};
};

/*! \brief Build the conservative source-order fallback plan for one window. */
MemoResult BuildSourceOrderFallbackPlan(const WindowPlannerInput &input);
/*! \brief Solve the optimal remaining suffix from one planner state. */
MemoResult SolveWindowPlan(const WindowPlannerInput &input,
                           const PlannerState &state,
                           PlannerSearchContext *context);
/*! \brief Convert the planner action trace back into the rewrite-facing tree.
 */
std::vector<SunmmioTileLoopFusionPlannerTreeNode>
BuildPlanTree(const std::vector<SunmmioTileLoopFusionPlannerAction> &actions);

} // namespace planner_internal
} // namespace tl
} // namespace tvm

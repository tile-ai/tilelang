/*!
 * \file ascend/costmodel/simtvf/model/ub_access.h
 * \brief DAV-3510 SIMT UB request organization and ideal pipelined service.
 *
 * Input is one machine LDS/STS: byte addresses for 32 lanes, an active mask
 * encoded by optional addresses, and 1/2/4/8 bytes per lane. Each <=4B slice
 * is packed independently into entries. An entry accepts one key per virtual
 * group, requires matching full 8B lines for a shared key, and holds at most
 * 64 charged 2B fragments, including repeated accesses. Store byte coverage
 * identifies entries that require partial-word read/modify/write service.
 *
 * For an active load or full-word store, ideal latency in cycles is
 *   intrinsic base + (entry count - 1) * entry publishing interval.
 * The base includes the first entry. Later entries publish one per cycle and
 * their response stages overlap. A b64 instruction has two independently
 * packed requests but pays its instruction base once. Different full lines
 * sharing a key affect packing directly; no separate conflict penalty is used.
 *
 * This assumes timely request supply, one-cycle entry grants, no interference
 * or resource/response backpressure, and ideal downstream optimization. It
 * does not reproduce avoidable compiler scheduling stalls or infer queueing
 * from resident warp count. Independent instructions share the publishing
 * pipeline: their entry service times add, while completion paths overlap.
 * The caller must separate dependent stages and serial invocations. This is
 * an ideal UB component, not total VF time; instruction supply, arithmetic
 * and register capacity are outside its scope. It is independent of TIR.
 *
 * Inactive accesses cost zero. Invalid geometry is rejected; partial stores
 * retain their geometry but have unknown latency until RMW is modeled.
 */
#ifndef TILELANG_ASCEND_COSTMODEL_SIMTVF_MODEL_UB_ACCESS_H_
#define TILELANG_ASCEND_COSTMODEL_SIMTVF_MODEL_UB_ACCESS_H_

#include "../intrinsic/cost.h"
#include <array>
#include <vector>

namespace tvm::tl::ascend::costmodel::simtvf {
// Ideal entry publishing cadence with no resource stalls. Consecutive
// requests can feed consecutive cycles; b64 does not restart the pipeline.
inline constexpr int kDav3510UbEntryIssueCycles = 1;

// Request geometry shared by IR access extraction and physical request packing.
struct SimtUbGeometry {
  int warp_size;
  int line_bytes;
  int bank_keys;
  int virtual_groups;
  int element_bytes;
  int entry_elements;
  int request_bytes;
  int store_word_bytes;
};

inline constexpr SimtUbGeometry kDav3510UbGeometry{
    /*warp_size=*/32,      /*line_bytes=*/8,
    /*bank_keys=*/64,
    /*virtual_groups=*/16, /*element_bytes=*/2,
    /*entry_elements=*/64,
    /*request_bytes=*/4,   /*store_word_bytes=*/4};

using SimtUbLaneAddresses =
    std::array<std::optional<int64_t>, kDav3510UbGeometry.warp_size>;

// Structural counts, NOT cycles. Independent warp requests are never merged.
struct SimtUbRequestFeatures {
  int64_t entries{0};
  int64_t partial{0};  // Store entries with at least one partial 4B word.
  int64_t requests{0}; // Nonempty <=4B/lane coalescer invocations.
};

// Byte addresses must be nonnegative and not overflow int64. nullopt lanes
// are inactive; an entirely inactive warp has zero request cost. bytes is 1..4.
// Unknown/invalid geometry returns nullopt, never a zero-cost estimate.
std::optional<SimtUbRequestFeatures>
AnalyzeSimtUbRequest(const SimtUbLaneAddresses &addresses, int bytes,
                     bool store);

struct SimtUbInstructionFeatures : SimtUbRequestFeatures {
  // One supplied warp LDS/STS, including an all-inactive instruction.
  int64_t instructions{0};
};

// The caller supplies one normal machine LDS/STS with 1, 2, 4 or 8 bytes/lane.
// b64 creates TWO independent coalescer invocations, but one instruction.
// This is not a prediction of downstream machine code from arbitrary TIR.
std::optional<SimtUbInstructionFeatures>
AnalyzeSimtUbInstruction(const SimtUbLaneAddresses &addresses, int bytes,
                         bool store);

// One estimated machine LDS/STS, not a <=4B coalescer slice. b64 retains
// a single instruction base cost while its two requests are packed separately.
struct SimtUbInstructionAccess {
  SimtUbLaneAddresses addresses;
  int bytes;
  // Independent instances within one batch, not dependent loop iterations.
  int64_t repetitions;
};
struct SimtUbInstructionCost : SimtUbInstructionFeatures {
  CycleEstimate estimated_latency;
  // Entry publishing occupancy, distinct from issue-to-completion latency.
  CycleEstimate service_cycles;
};
// Ideal issue-to-completion latency: reference instruction path plus serial
// entry publishing. Assumes one-cycle grants and no interference, queue stalls
// or response backpressure. Total resident warps are not a concurrency input.
// Invalid geometry returns nullopt. Partial stores retain their features but
// have unknown latency until the full RMW path is modeled. Inactive cost is 0.
std::optional<SimtUbInstructionCost>
EstimateSimtUbInstructionLatency(const SimtUbLaneAddresses &addresses,
                                 int bytes, bool store);

struct SimtUbAccessCost : SimtUbInstructionFeatures {
  // Completion of an independent batch under ideal request supply.
  CycleEstimate estimated_latency;
  CycleEstimate service_cycles;
};
// All instructions (including repetitions) are independent and available to
// feed the same UB interface. Each occupies the publisher for service_cycles;
// its completion can overlap later instructions. All have the same direction,
// so packing-period compression does not change the completion tail. The
// caller, not this model, establishes independence and dependency boundaries.
std::optional<SimtUbAccessCost> EstimateSimtUbAccessLatency(
    const std::vector<SimtUbInstructionAccess> &instructions, bool store);

} // namespace tvm::tl::ascend::costmodel::simtvf
#endif // TILELANG_ASCEND_COSTMODEL_SIMTVF_MODEL_UB_ACCESS_H_

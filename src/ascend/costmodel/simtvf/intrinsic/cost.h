/*!\file ascend/costmodel/simtvf/intrinsic/cost.h */
#ifndef TILELANG_ASCEND_COSTMODEL_SIMTVF_INTRINSIC_COST_H_
#define TILELANG_ASCEND_COSTMODEL_SIMTVF_INTRINSIC_COST_H_

#include <cstdint>
#include <optional>

namespace tvm::tl::ascend::costmodel::simtvf {

// Elapsed cycles. Missing stages and overflow produce unknown, never zero.
using CycleEstimate = std::optional<double>;
CycleEstimate AddCycles(CycleEstimate first, CycleEstimate second);

// Base completion latency of one machine instruction, excluding additional
// delays charged by hardware models. This is not an initiation interval or
// resource occupancy. Unknown costs are never treated as free.
struct InstructionCost {
  CycleEstimate cycles;
};

// Reference single-entry issue-to-completion latency under ideal service.
// Includes the first entry's path; hardware models add only extra work.
// These references do not guarantee a hardware lower bound for every opcode
// variant. Queueing and cross-instruction interference are outside this cost.
inline constexpr InstructionCost kDav3510LdsCost{27.0};
inline constexpr InstructionCost kDav3510StsCost{20.0};
inline constexpr InstructionCost kDav3510XorCost{std::nullopt};

// Serial cost of count instances. Zero instances cost zero even when the
// instruction cost is unknown. Invalid counts and overflow return unknown.
CycleEstimate EstimateInstructionCycles(const InstructionCost &instruction,
                                        int64_t count);

} // namespace tvm::tl::ascend::costmodel::simtvf
#endif // TILELANG_ASCEND_COSTMODEL_SIMTVF_INTRINSIC_COST_H_

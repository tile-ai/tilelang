/*!\file ascend/costmodel/simtvf/intrinsic/cost.cc */
#include "cost.h"
#include <cmath>

namespace tvm::tl::ascend::costmodel::simtvf {
CycleEstimate AddCycles(CycleEstimate first, CycleEstimate second) {
  if (!first || !second || *first < 0 || *second < 0 ||
      !std::isfinite(*first) || !std::isfinite(*second))
    return std::nullopt;
  double sum = *first + *second;
  return std::isfinite(sum) ? CycleEstimate(sum) : std::nullopt;
}
CycleEstimate EstimateInstructionCycles(const InstructionCost &instruction,
                                        int64_t count) {
  if (count < 0)
    return std::nullopt;
  if (count == 0)
    return 0;
  if (!instruction.cycles || *instruction.cycles < 0 ||
      !std::isfinite(*instruction.cycles))
    return std::nullopt;
  double cycles = *instruction.cycles * count;
  return std::isfinite(cycles) ? CycleEstimate(cycles) : std::nullopt;
}
} // namespace tvm::tl::ascend::costmodel::simtvf

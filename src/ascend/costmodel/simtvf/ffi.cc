/*!\file ascend/costmodel/simtvf/ffi.cc
 * \brief Python adapters for Ascend cost analysis.
 */
#include "model/ub_access.h"
#include "support/check.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/expr.h>

namespace tvm::tl::ascend::costmodel::simtvf {
using namespace ffi;
using namespace tirx;

// Python analysis helpers. -1 denotes an inactive lane in this FFI adapter.
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def(
      "tl.ascend.EstimateSimtUbAccessLatency",
      [](Array<Array<Integer>> addresses, Array<Integer> widths,
         Array<Integer> repetitions, bool store) {
        ICHECK_EQ(addresses.size(), widths.size());
        ICHECK_EQ(addresses.size(), repetitions.size());
        std::vector<SimtUbInstructionAccess> instructions;
        for (size_t i = 0; i < addresses.size(); ++i) {
          ICHECK_EQ(addresses[i].size(), 32);
          ICHECK(widths[i]->value >= 1 && widths[i]->value <= 8)
              << "Unsupported SIMT UB batch width";
          SimtUbLaneAddresses lanes;
          for (int lane = 0; lane < 32; ++lane) {
            ICHECK_GE(addresses[i][lane]->value, -1);
            if (addresses[i][lane]->value != -1)
              lanes[lane] = addresses[i][lane]->value;
          }
          instructions.push_back(
              {lanes, int(widths[i]->value), repetitions[i]->value});
        }
        auto result = EstimateSimtUbAccessLatency(instructions, store);
        ICHECK(result) << "Unsupported SIMT UB batch";
        return Map<String, PrimExpr>{
            {"estimated_latency",
             FloatImm(DataType::Float(64), *result->estimated_latency)},
            {"service_cycles",
             FloatImm(DataType::Float(64), *result->service_cycles)},
            {"instructions", Integer(result->instructions)},
            {"entries", Integer(result->entries)}};
      });
  reflection::GlobalDef().def(
      "tl.ascend.EstimateSimtUbInstructionLatency",
      [](Array<Integer> addresses, int bytes, bool store) {
        ICHECK_EQ(addresses.size(), 32);
        SimtUbLaneAddresses lanes;
        for (int i = 0; i < 32; ++i) {
          ICHECK_GE(addresses[i]->value, -1);
          if (addresses[i]->value != -1)
            lanes[i] = addresses[i]->value;
        }
        auto result = EstimateSimtUbInstructionLatency(lanes, bytes, store);
        ICHECK(result) << "Unsupported SIMT UB width or address";
        return result->estimated_latency.value_or(-1.0);
      });
  reflection::GlobalDef().def(
      "tl.ascend.AnalyzeSimtUbInstruction",
      [](Array<Integer> addresses, int bytes, bool store) {
        ICHECK_EQ(addresses.size(), 32);
        SimtUbLaneAddresses lanes;
        for (int i = 0; i < 32; ++i) {
          ICHECK_GE(addresses[i]->value, -1);
          if (addresses[i]->value != -1)
            lanes[i] = addresses[i]->value;
        }
        auto result = AnalyzeSimtUbInstruction(lanes, bytes, store);
        ICHECK(result) << "Unsupported SIMT UB width or address";
        return Map<String, Integer>{
            {"instructions", Integer(result->instructions)},
            {"requests", Integer(result->requests)},
            {"entries", Integer(result->entries)},
            {"partial", Integer(result->partial)}};
      });
}

} // namespace tvm::tl::ascend::costmodel::simtvf

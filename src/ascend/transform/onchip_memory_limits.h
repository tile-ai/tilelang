// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file onchip_memory_limits.h
 * \brief Physical on-chip memory capacities (UB / L1 / L0A / L0B / L0C).
 */

#pragma once

#include <tvm/target/target.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <cstdint>

#include "ascend/target_utils.h"

namespace tvm {
namespace tl {
namespace ascend {

// SIMT_VF (asc_vf_call) reserves 32 KB of the physical 248 KB Unified Buffer
// for thread context, so its presence lowers the usable UB.
// NOTE: vf_regions.h's IsVFRegion also matches SIMD_VF, which does NOT reserve
// the thread context, so it is not a substitute for this detector.
class SimtVFDetector : public tirx::StmtVisitor {
public:
  bool found = false;

  void VisitStmt_(const tirx::SBlockNode *op) override {
    if (op->name_hint == "SIMT_VF") {
      found = true;
    }
    tirx::StmtVisitor::VisitStmt_(op);
  }
};

inline bool HasSimtVF(const tirx::Stmt &stmt) {
  SimtVFDetector detector;
  detector(stmt);
  return detector.found;
}

inline int64_t GetSharedMemoryLimit(Target target,
                                    const tirx::Stmt &kernel_body) {
  if (!TargetIsAscend(target))
    return 0;
  return HasSimtVF(kernel_body) ? kAscendUBMemoryBytes - kAscendSimtContextBytes
                                : kAscendUBMemoryBytes;
}

inline int64_t GetL1MemoryLimit(Target target) {
  return TargetIsAscend(target) ? kAscendL1MemoryBytes : 0;
}

inline int64_t GetL0CMemoryLimit(Target target) {
  return TargetIsAscend(target) ? kAscendL0CMemoryBytes : 0;
}

inline int64_t GetL0AMemoryLimit(Target target) {
  return TargetIsAscend(target) ? kAscendL0ABMemoryBytes : 0;
}

inline int64_t GetL0BMemoryLimit(Target target) {
  return TargetIsAscend(target) ? kAscendL0ABMemoryBytes : 0;
}

} // namespace ascend
} // namespace tl
} // namespace tvm

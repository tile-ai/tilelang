/*!
 * \file tl/backend/common/op/atomic_simt.h
 * \brief Shared iteration and index mapping for SIMT atomic operations.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_ATOMIC_SIMT_H_
#define TVM_TL_BACKEND_COMMON_OP_ATOMIC_SIMT_H_

#include <cstddef>
#include <string>

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/optional.h>
#include <tvm/ir/op.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/op.h>

#include "support/check.h"

namespace tvm {
namespace tl {
namespace backend {

/*! \brief Validated loop variables and operand indices for a SIMT atomic. */
struct AtomicSIMTIndexMap {
  ffi::Array<tirx::IterVar> loop_vars;
  ffi::Array<PrimExpr> src_indices;
  ffi::Array<PrimExpr> dst_indices;
};

/*!
 * \brief Build destination-driven loops and validate operand index mappings.
 *
 * Each non-unit extent consumes one loop variable. A single-element region
 * consumes none and may share the sole loop variable, including the dummy
 * extent-1 loop used when the destination is a single element. An absent source
 * region denotes a scalar expression whose indices do not need mapping.
 */
inline AtomicSIMTIndexMap
MakeAtomicSIMTIndexMap(const Op &operation, const tirx::BufferRegion &dst,
                       const ffi::Optional<tirx::BufferRegion> &src) {
  AtomicSIMTIndexMap result;
  for (const Range &range : dst->region) {
    if (tirx::is_one(range->extent)) {
      continue;
    }
    tirx::Var var(std::string{char('i' + result.loop_vars.size())},
                  range->extent.dtype());
    result.loop_vars.push_back(
        {Range(0, range->extent), var, tirx::IterVarType::kDataPar});
  }
  if (result.loop_vars.empty()) {
    result.loop_vars.push_back(
        {Range(0, 1), tirx::Var("i"), tirx::IterVarType::kDataPar});
  }

  auto make_indices = [&](const tirx::BufferRegion &region) {
    const ffi::Array<Range> &ranges = region->region;
    size_t non_unit_dims = 0;
    for (const Range &range : ranges) {
      non_unit_dims += !tirx::is_one(range->extent);
    }

    const size_t num_loop_vars = result.loop_vars.size();
    // Validate before indexing, preserving the existing single-element case.
    CHECK(num_loop_vars <= ranges.size() &&
              (non_unit_dims == num_loop_vars ||
               (non_unit_dims == 0 && num_loop_vars == 1)),
          ValueError)
        << "Cannot map atomic region to SIMT loop variables for "
        << operation->name << ": buffer " << region->buffer->name << " has "
        << non_unit_dims << " non-unit dimensions, but there are "
        << num_loop_vars << " loop variables; src="
        << (src.defined() ? src.value()->buffer->name
                          : ffi::String("<scalar expression>"))
        << ", dst=" << dst->buffer->name << ", src_range="
        << (src.defined() ? src.value()->region : ffi::Array<Range>{})
        << ", dst_range=" << dst->region;

    ffi::Array<PrimExpr> indices;
    size_t idx = 0;
    for (const Range &range : ranges) {
      indices.push_back(tirx::is_one(range->extent)
                            ? range->min
                            : range->min + result.loop_vars[idx++]->var);
    }
    return indices;
  };

  result.dst_indices = make_indices(dst);
  if (src.defined()) {
    result.src_indices = make_indices(src.value());
  }
  return result;
}

} // namespace backend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_OP_ATOMIC_SIMT_H_

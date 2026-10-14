/*!
 * \file tl/ascend/target_utils.h
 * \brief Ascend target attribute helpers.
 */

#ifndef TVM_TL_ASCEND_TARGET_UTILS_H_
#define TVM_TL_ASCEND_TARGET_UTILS_H_

#include <tvm/runtime/data_type.h>
#include <tvm/target/target.h>

namespace tvm {
namespace tl {

bool TargetIsAscend(Target target);
bool IsAscendVectorizableFP8(DataType dtype);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TARGET_UTILS_H_

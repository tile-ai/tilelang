// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/target/utils.h
 * \brief helper functions for target attributes.
 *
 */

#ifndef TVM_TL_TARGET_UTILS_H_
#define TVM_TL_TARGET_UTILS_H_

#include <tvm/target/target.h>

namespace tvm {
namespace tl {

bool TargetIsCuda(Target target);
bool TargetIsRocm(Target target);

bool TargetIsVolta(Target target);
bool TargetIsTuring(Target target);
bool TargetIsAmpere(Target target);
bool TargetIsHopper(Target target);
bool TargetIsCDNA(Target target);

bool TargetHasAsyncCopy(Target target);
bool TargetHasLdmatrix(Target target);
bool TargetHasStmatrix(Target target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TARGET_UTILS_H_

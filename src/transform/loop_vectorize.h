/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file loop_vectorize.h
 * \brief A tool to automatically vectorize a for loop
 */

#ifndef TVM_TL_LOOP_VECTORIZE_H_
#define TVM_TL_LOOP_VECTORIZE_H_

#include "../op/operator.h"
#include <tvm/arith/analyzer.h>
#include <tvm/target/target.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tl {

using namespace tirx;

// Widest vector memory access, in bits, that the vectorize planner will
// consider for code whose memory-access mix is described by
// `global_only_access` (touches global memory and no shared memory).
// Single source of truth for the width-cap policy: the layout-inference
// cost model calls this too, so a candidate layout is scored under exactly
// the cap the vectorizer will later enforce on it.
int MaxVectorLoadBits(const Target &target, bool global_only_access);

// Extract an existing BufferLoad, optionally wrapped in address_of/access_ptr.
// tvm_access_ptr carries a flat offset instead and must be handled separately.
ffi::Optional<BufferLoad> ExtractBufferLoadForAtomic(const PrimExpr &expr);

// Target/dtype/address-space capability; access legality is checked separately.
int GetMaxAtomicVectorSize(DataType dtype, const ffi::String &storage_scope,
                           const Target &target);

int GetVectorizeSize(const For &loop, const LayoutMap &layout_map = {});

int GetVectorizeSize(const For &loop, arith::Analyzer *analyzer,
                     const LayoutMap &layout_map = {});

For VectorizeLoop(const For &loop, const LayoutMap &layout_map = {},
                  int vectorize_hint = -1);

For VectorizeLoop(const For &loop, arith::Analyzer *analyzer,
                  const LayoutMap &layout_map = {}, int vectorize_hint = -1);

// Can prove expr is independent with var, i.e. the value of expr doesn't change
// when var changes
bool CanProveIndependent(const PrimExpr &expr, Var var,
                         arith::Analyzer *analyzer);

// Check if expr is invariant within vector boundaries
bool IsExprInvariantInVectorBoundary(const PrimExpr &expr, Var var,
                                     int target_vectorized_size,
                                     arith::Analyzer *analyzer);

/*!
 * \brief Check whether an element-offset expression supports a vector width.
 *
 * With allow_broadcast=true, offsets invariant within a vector group are
 * accepted. For example, loading A[i // 4] at width 4 can load one element and
 * broadcast its value. This preserves the existing access analysis; callers
 * must separately check write-side ordering constraints.
 *
 * For widths greater than one, allow_broadcast=false requires every group to
 * simplify to a unit-stride Ramp whose base is divisible by the vector width.
 * Atomic destinations use this mode: four atomic additions to B[i // 4] update
 * the same element, whereas AtomicAddx4 would update four consecutive elements.
 * Alignment must hold for every group, not just at var == 0.
 *
 * The flag applies only to the analyzed address; an atomic's source value may
 * still broadcast. If the check fails, the planner can try a smaller width.
 */
bool IndicesCanVectorize(const PrimExpr &expr, Var var,
                         const PrimExpr &iter_var_size,
                         int target_vectorized_size, arith::Analyzer *analyzer,
                         bool allow_broadcast = true);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_LOOP_VECTORIZE_H_

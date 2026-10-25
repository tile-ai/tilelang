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
 * \file loop_partition.h
 * \brief Partition parallel loops onto threads
 */

#ifndef TVM_TL_LOOP_PARTITION_H_
#define TVM_TL_LOOP_PARTITION_H_

#include "support/check.h"
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/stmt.h>

#include "../layout/layout.h"
#include "../op/operator.h"

namespace tvm {
namespace tl {

using namespace tirx;

TILELANG_DLL For PartitionLoop(For op, PrimExpr thread_index,
                               sym::Analyzer analyzer,
                               const Fragment &loop_layout,
                               bool require_padding_guard = false);

TILELANG_DLL Fragment PlanLoopPartition(const For &op, size_t num_thread,
                                        int vectorize_size);

TILELANG_DLL Fragment PlanLoopPartition(const For &op, int vectorize_size,
                                        const Range &thread_range);

TILELANG_DLL For PragmaUnrollLoop(For stmt);

/*!
 * \brief Lower a parallel loop by partitioning and vectorizing it.
 *
 * This function combines PartitionLoop and VectorizeLoop into a single
 * operation, and optionally wraps the result with an IfThenElse if a
 * predicate is provided.
 *
 * \param loop The parallel For loop to lower.
 * \param loop_layout The Fragment layout for partitioning.
 * \param thread_index The logical thread index expression for partitioning
 *        (the real threadIdx.x Var on GPU, constant 0 without thread
 *        bindings).
 * \param analyzer The arithmetic analyzer.
 * \param predicate ffi::Optional predicate to wrap the loop with IfThenElse.
 * \param parallel_loop Whether this is a true parallel loop requiring thread
 *        partitioning. False for loops that only operate on local/register
 *        buffers. (default true)
 * \return The lowered statement.
 */
TILELANG_DLL Stmt
LowerParallelLoop(For loop, const Fragment &loop_layout, PrimExpr thread_index,
                  sym::Analyzer analyzer, const LayoutMap &layout_map = {},
                  ffi::Optional<PrimExpr> predicate = ffi::Optional<PrimExpr>(),
                  bool parallel_loop = true, bool require_padding_guard = false,
                  bool enable_vectorize = true);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_LOOP_PARTITION_H_

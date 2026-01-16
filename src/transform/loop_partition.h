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

#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include "../layout/layout.h"
#include "../op/operator.h"

namespace tvm {
namespace tl {

using namespace tir;

For PartitionLoop(For op, Var thread_var, arith::Analyzer *analyzer,
                  const Fragment &loop_layout);

Fragment PlanLoopPartition(const For &op, size_t num_thread,
                           int vectorize_size);

Fragment PlanLoopPartition(const For &op, int vectorize_size,
                           const Range &thread_range);

For LoopPragmaUnroll(For stmt);

/*!
 * \brief Lower a parallel loop by partitioning and vectorizing it.
 *
 * This function combines PartitionLoop and VectorizeLoop into a single
 * operation, and optionally wraps the result with an IfThenElse if a
 * predicate is provided.
 *
 * \param loop The parallel For loop to lower.
 * \param loop_layout The Fragment layout for partitioning.
 * \param thread_var The thread variable for partitioning.
 * \param analyzer The arithmetic analyzer.
 * \param predicate Optional predicate to wrap the loop with IfThenElse.
 * \param parallel_loop Whether this is a true parallel loop requiring thread
 *        partitioning. False for loops that only operate on local/register
 *        buffers. (default true)
 * \param should_vectorize Whether to vectorize the loop. False when reducers
 *        are present or when there are no non-local buffer accesses.
 *        (default true)
 * \return The lowered statement.
 */
Stmt LowerParallelLoop(For loop, const Fragment &loop_layout, Var thread_var,
                       arith::Analyzer *analyzer,
                       const LayoutMap &layout_map = {},
                       Optional<PrimExpr> predicate = Optional<PrimExpr>(),
                       bool parallel_loop = true, bool should_vectorize = true);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_LOOP_PARTITION_H_

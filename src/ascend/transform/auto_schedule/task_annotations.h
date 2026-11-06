/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
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

#pragma once
#include "ascend/ir/task.h"
namespace tvm::tl::ascend {
struct TaskCost {
  int64_t latency{0};
  int64_t ii{0};
};

// Apply any latency/II overrides present on T.Task to an estimated cost.
inline TaskCost ResolveTaskCost(const TaskCost &estimated,
                                const TaskMetadata &metadata) {
  TaskCost result = estimated;
  if (metadata.latency.has_value()) {
    result.latency = metadata.latency.value();
  } else if (result.latency == 0) {
    // Keep unmodeled compiler-estimated tasks schedulable. An explicit zero
    // override remains subject to the latency >= II validation below.
    result.latency = 1;
  }
  if (metadata.ii.has_value())
    result.ii = metadata.ii.value();
  ICHECK_GE(result.latency, 0);
  ICHECK_GT(result.ii, 0);
  ICHECK_GE(result.latency, result.ii)
      << "T.Task latency must be greater than or equal to ii, got latency="
      << result.latency << " and ii=" << result.ii;
  return result;
}

} // namespace tvm::tl::ascend

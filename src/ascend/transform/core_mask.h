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

#include <array>
#include <cstdint>

namespace tvm {
namespace tl {

using CoreMask = uint8_t;

constexpr CoreMask kCoreUnassigned = 0;
constexpr CoreMask kCoreVector0 = 1U << 0;
constexpr CoreMask kCoreCube = 1U << 1;
constexpr CoreMask kCoreVector1 = 1U << 2;
constexpr CoreMask kCoreVector = kCoreVector0 | kCoreVector1;
constexpr CoreMask kCoreAll = kCoreVector | kCoreCube;
constexpr std::array<CoreMask, 3> kConcreteCores = {kCoreVector0, kCoreCube,
                                                    kCoreVector1};

constexpr CoreMask VectorCore(int sid) {
  return sid == 0 ? kCoreVector0 : kCoreVector1;
}

constexpr CoreMask VectorCoreMask(int num_aiv_subcores) {
  return num_aiv_subcores == 1 ? kCoreVector0 : kCoreVector;
}

constexpr bool IsVectorCore(CoreMask core) {
  return core == kCoreVector0 || core == kCoreVector1;
}

constexpr int VectorSid(CoreMask core) { return core == kCoreVector1 ? 1 : 0; }

constexpr bool IsValidCoreMask(CoreMask mask) {
  return (mask & static_cast<CoreMask>(~kCoreAll)) == 0;
}

constexpr bool IsConcreteCore(CoreMask mask) {
  return IsVectorCore(mask) || mask == kCoreCube;
}

// A vector task may execute the same body on every participating AIV.
constexpr bool IsCoreLocalMask(CoreMask mask) {
  return IsConcreteCore(mask) || mask == kCoreVector;
}

constexpr bool HasCore(CoreMask mask, CoreMask core) {
  return core != kCoreUnassigned && (mask & core) == core;
}

} // namespace tl
} // namespace tvm

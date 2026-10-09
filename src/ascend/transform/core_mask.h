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
constexpr CoreMask kCoreVector = 1U << 0;
constexpr CoreMask kCoreCube = 1U << 1;
constexpr CoreMask kCoreBroadcast = kCoreVector | kCoreCube;
constexpr std::array<CoreMask, 2> kConcreteCores = {kCoreVector, kCoreCube};

constexpr bool IsValidCoreMask(CoreMask mask) {
  return (mask & static_cast<CoreMask>(~kCoreBroadcast)) == 0;
}

constexpr bool IsConcreteCore(CoreMask mask) {
  return mask == kCoreVector || mask == kCoreCube;
}

constexpr bool HasCore(CoreMask mask, CoreMask core) {
  return IsConcreteCore(core) && (mask & core) != 0;
}

} // namespace tl
} // namespace tvm

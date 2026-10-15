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

/*!
 * \file tl/ascend/transform/buffer_version.h
 * \brief Annotation contract for buffer version metadata.
 */

#pragma once

#include <tvm/ffi/container/map.h>
#include <tvm/tirx/expr.h>

#include "ascend/op/utils.h"
#include "transform/common/attr.h"

namespace tvm {
namespace tl {

/*! \brief Frontend overrides before AutoSchedule; selected versions afterward.
 */
constexpr const char *kBufferVersionsMap = "tl.buffer_versions_map";

/*! \brief Manual-version annotation and physical-version staging key. */
constexpr const char *kManualMultiBuffer = "tl.manual_multi_buffer";

/*! \brief Loop-local list of storage Vars eligible for automatic versioning. */
constexpr const char *kMultiBufferEligible = "multi_buffer_eligible";

/*! \brief Storage data Var -> positive compile-time version count. */
using BufferVersionMap = ffi::Map<tirx::Var, int>;

// Logical reads/writes remain separate. Ownership, physical versions and
// address reuse use the same allocation group, represented by the data Var.
class L0StorageGroups {
public:
  explicit L0StorageGroups(L0SFBindings bindings = {})
      : bindings_(std::move(bindings)) {
    for (const auto &[sf, data] : bindings_)
      scales_.Set(data, sf);
  }
  tirx::Var Representative(const tirx::Var &storage) const {
    return bindings_.Get(storage).value_or(storage);
  }
  ffi::Array<tirx::Var> Members(const tirx::Var &storage) const {
    tirx::Var data = Representative(storage);
    if (auto sf = scales_.Get(data))
      return {data, sf.value()};
    return {data};
  }
  const L0SFBindings &Bindings() const { return bindings_; }

private:
  L0SFBindings bindings_;
  L0SFBindings scales_;
};

// User overrides and prepared metadata must describe the entire group.
// Contradictory member requests are errors, rather than last-writer wins.
template <typename T>
ffi::Map<tirx::Var, T> ExpandL0StorageGroupValues(ffi::Map<tirx::Var, T> values,
                                                  const L0StorageGroups &groups,
                                                  const char *description) {
  auto original = values;
  for (const auto &[storage, value] : original) {
    for (const tirx::Var &member : groups.Members(storage)) {
      if (auto existing = values.Get(member)) {
        ICHECK(existing.value() == value)
            << "Conflicting " << description << " for bound L0 data/SF group "
            << groups.Representative(storage)->name_hint;
      }
      values.Set(member, value);
    }
  }
  return values;
}

} // namespace tl
} // namespace tvm

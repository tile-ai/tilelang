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
 * \file tl/ascend/transform/buffer_alias.h
 * \brief Positive storage-alias contract shared by Ascend scheduling paths.
 */

#pragma once

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/expr.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "support/check.h"

namespace tvm {
namespace tl {

/*! \brief Kernel annotation and long-lived AttrStmt key. */
constexpr const char *kBufferAliasMap = "tl.buffer_alias_map";

/*! \brief Storage data Var -> storages proven safe to alias with it. */
using BufferAliasMap = ffi::Map<tirx::Var, ffi::Array<tirx::Var>>;

inline void AddBufferAliasStorage(BufferAliasMap *aliases,
                                  const tirx::Var &storage) {
  if (!aliases->count(storage))
    aliases->Set(storage, {});
}

inline bool HasBufferAlias(const BufferAliasMap &aliases, const tirx::Var &lhs,
                           const tirx::Var &rhs) {
  auto lhs_it = aliases.find(lhs);
  if (lhs_it == aliases.end())
    return false;
  for (const tirx::Var &candidate : (*lhs_it).second) {
    if (candidate.same_as(rhs))
      return true;
  }
  return false;
}

inline void AddBufferAlias(BufferAliasMap *aliases, const tirx::Var &lhs,
                           const tirx::Var &rhs) {
  ICHECK(!lhs.same_as(rhs)) << "A storage cannot alias itself in the "
                               "buffer-alias contract";
  AddBufferAliasStorage(aliases, lhs);
  AddBufferAliasStorage(aliases, rhs);
  auto append = [&](const tirx::Var &from, const tirx::Var &to) {
    ffi::Array<tirx::Var> values = aliases->at(from);
    for (const tirx::Var &existing : values) {
      if (existing.same_as(to))
        return;
    }
    values.push_back(to);
    aliases->Set(from, std::move(values));
  };
  append(lhs, rhs);
  append(rhs, lhs);
}

// A contract need not list every allocation: unlisted pairs remain conflicts.
// An empty map is valid and permits no reuse. The annotation's presence, not
// the map's size, distinguishes this from an absent contract and still enables
// sequential allocation downstream.
inline void ValidateBufferAliasMap(const BufferAliasMap &aliases) {
  for (const auto &[storage, compatible] : aliases) {
    for (const tirx::Var &other : compatible) {
      ICHECK(!storage.same_as(other))
          << "Buffer-alias contract contains a self edge for " << storage;
      ICHECK(aliases.count(other))
          << "Buffer-alias contract references undeclared "
             "storage "
          << other;
      ICHECK(HasBufferAlias(aliases, other, storage))
          << "Buffer-alias contract must be symmetric: " << storage << " lists "
          << other << " but not vice versa";
    }
  }
}

inline BufferAliasMap
RemapBufferAliasMap(const BufferAliasMap &aliases,
                    const ffi::Map<tirx::Var, tirx::Var> &storage_remap) {
  using VarVectorMap =
      std::unordered_map<tirx::Var, std::vector<tirx::Var>, ffi::ObjectPtrHash,
                         ffi::ObjectPtrEqual>;
  using VarSet =
      std::unordered_set<tirx::Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

  ValidateBufferAliasMap(aliases);
  VarVectorMap preimages;
  std::vector<tirx::Var> remapped_order;
  VarSet seen;
  for (const auto &[storage, _] : aliases) {
    tirx::Var remapped = storage;
    auto remap = storage_remap.find(storage);
    if (remap != storage_remap.end())
      remapped = (*remap).second;
    preimages[remapped].push_back(storage);
    if (seen.insert(remapped).second)
      remapped_order.push_back(remapped);
  }

  BufferAliasMap result;
  for (const tirx::Var &storage : remapped_order)
    AddBufferAliasStorage(&result, storage);
  for (size_t i = 0; i < remapped_order.size(); ++i) {
    for (size_t j = i + 1; j < remapped_order.size(); ++j) {
      const tirx::Var &lhs = remapped_order[i];
      const tirx::Var &rhs = remapped_order[j];
      bool compatible = true;
      for (const tirx::Var &old_lhs : preimages.at(lhs)) {
        for (const tirx::Var &old_rhs : preimages.at(rhs)) {
          if (!HasBufferAlias(aliases, old_lhs, old_rhs)) {
            compatible = false;
            break;
          }
        }
        if (!compatible)
          break;
      }
      if (compatible)
        AddBufferAlias(&result, lhs, rhs);
    }
  }
  ValidateBufferAliasMap(result);
  return result;
}

} // namespace tl
} // namespace tvm

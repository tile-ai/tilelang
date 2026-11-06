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

#include "ascend/analysis/storage.h"
#include "ascend/ir/task.h"
#include "ir/analyzer.h"
#include "ir/tensor_utils.h"
#include "transform/common/storage_size.h"
#include <tvm/ffi/extra/structural_visit.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <tvm/tirx/layout.h>
#include <unordered_map>
#include <unordered_set>

namespace tvm::tl::ascend {

// Keep scheduling costs and materialized version strides on the same native
// allocation/alias model. Layout coordinates and pitches are in dtype bits;
// allocation byte sizes additionally follow the native packed-storage rules.
class StorageFootprints {
public:
  explicit StorageFootprints(const Stmt &body) {
    storage_.Collect(body);
    std::unordered_set<TensorVar, ObjectPtrHash, ObjectPtrEqual> seen;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        body,
        [&](const ffi::ObjectRef &object) -> ffi::Expected<ffi::WalkResult> {
          if (auto variable = object.as<Var>()) {
            if (auto tensor = variable->as<TensorVar>())
              if (seen.insert(*tensor).second)
                views_[storage_.Identity(*tensor)].push_back(*tensor);
          }
          const auto *bind = object.as<BindObj>();
          if (!bind)
            return ffi::WalkResult::Advance();
          auto tensor = bind->var.as<TensorVar>();
          Expr value = bind->value;
          if (const auto *call = value.as<CallObj>(); IsScheduledBind(call))
            value = ReadScheduledBinding(call).value;
          const auto *call = value.as<CallObj>();
          if (tensor && call && call->op.same_as(tirx::alloc_tensor_op()))
            owners_.emplace(storage_.Identity(*tensor), *tensor);
          return ffi::WalkResult::Advance();
        });
  }

  const StorageProvenance &Storage() const { return storage_; }

  const TensorVar &Allocation(const Var &storage) const {
    auto found = owners_.find(storage);
    TVM_FFI_CHECK(found != owners_.end(), ValueError)
        << "Cannot determine the physical allocation for on-chip storage "
        << storage->name;
    return found->second;
  }

  static int64_t ElementBits(const TensorVar &tensor) {
    TVM_FFI_CHECK(!tensor.dtype().IsScalableVector(), ValueError)
        << "Physical storage requires fixed vector lanes";
    int64_t bits = int64_t{tensor.dtype().bits()} * tensor.dtype().lanes();
    TVM_FFI_CHECK_GT(bits, 0, ValueError);
    return bits;
  }

  static tirx::TileLayout MemoryTile(const TensorVar &tensor) {
    auto compact = [&]() {
      // Extents are the tensor's native shape expressions. Widen only the
      // stride arithmetic: shape/iterator matching preserves their exact
      // types and identities, while physical pitches must not overflow the
      // logical index type.
      ffi::Array<PrimExpr> shape;
      for (const PrimExpr &extent : tensor.shape())
        shape.push_back(Wide(extent));
      auto wide = tirx::TileLayout::DefaultLayout(shape);
      ffi::Array<tirx::Iter> shard;
      for (size_t i = 0; i < tensor.shape().size(); ++i)
        shard.push_back(tirx::Iter(tensor.shape()[i], wide->shard[i]->stride,
                                   tirx::Axis::Get("m")));
      return tirx::TileLayout(shard, wide->replica, wide->offset);
    };
    if (!tensor.ty()->layout)
      return compact();
    if (auto tile = tensor.ty()->layout.value().as<tirx::TileLayout>())
      return *tile;
    auto tile = tensor.ty()
                    ->layout.value()
                    .as_or_throw<tirx::ComposeLayout>()
                    ->tile_layout;
    // A bare compose layout swizzles the complete tensor, rather than a
    // single-element tile. LowerTensorLayouts uses the same default shape.
    return tile.IsTrivial() ? compact() : tile;
  }

  PrimExpr PitchBits(const Var &storage) const {
    if (auto found = pitches_.find(storage); found != pitches_.end())
      return found->second;
    const auto &owner = Allocation(storage);
    int64_t alignment = 8;
    for (const TensorVar &tensor : views_.at(storage)) {
      auto tile = MemoryTile(tensor);
      const auto memory = tirx::Axis::Get("m");
      for (const auto &iter : tile->shard)
        TVM_FFI_CHECK(iter->axis == memory, ValueError)
            << "Automatic multi-buffering requires a memory layout";
      for (const auto &iter : tile->replica)
        TVM_FFI_CHECK(iter->axis == memory, ValueError);
      int64_t required = ElementBits(tensor);
      if (auto layout = tensor.ty()->layout)
        if (const auto *swizzle = layout.value().as<tirx::ComposeLayoutObj>()) {
          int bits =
              swizzle->per_element + swizzle->atom_len + swizzle->swizzle_len;
          TVM_FFI_CHECK(bits >= 0 && bits < 55, ValueError);
          required = Checked(ffi::BigInt(required) << bits);
        }
      alignment = Checked(
          ffi::BigInt(alignment / std::gcd(alignment, required)) * required);
    }
    PrimExpr span = MemorySpan(owner);
    PrimExpr pitch = Multiply(span, ElementBits(owner));
    PrimExpr result = Align(pitch, alignment);
    pitches_.emplace(storage, result);
    return result;
  }

  tirx::TensorType VersionedType(const TensorVar &tensor, int versions) const {
    TVM_FFI_CHECK_GT(versions, 0, ValueError);
    auto storage = storage_.Identity(tensor);
    auto analyzer = CreateTileLangAnalyzer();
    PrimExpr pitch = analyzer->Simplify(
        FloorDiv(PitchBits(storage), IntImm::Int64(ElementBits(tensor))));
    PrimExpr count = IntImm(tensor.ty()->DefaultIndexType(), versions);
    ffi::Array<PrimExpr> shape{count};
    for (const auto &extent : tensor.shape())
      shape.push_back(extent);
    // Native compact indexing handles an empty dimension without grouping a
    // zero-sized layout tile, which would require division by zero.
    if (Empty(tensor))
      return tirx::TensorType(shape, tensor.dtype(), tensor.scope(),
                              tensor.ty()->data_alignment, std::nullopt,
                              tensor.ty()->loc);
    auto original = MemoryTile(tensor);
    ffi::Array<tirx::Iter> shard{
        tirx::Iter(count, pitch, tirx::Axis::Get("m"))};
    for (const auto &iter : original->shard)
      shard.push_back(iter);
    tirx::Layout layout =
        tirx::TileLayout(shard, original->replica, original->offset);
    if (auto previous = tensor.ty()->layout)
      if (const auto *swizzle = previous.value().as<tirx::ComposeLayoutObj>())
        layout = tirx::ComposeLayout(
            swizzle->per_element, swizzle->swizzle_len, swizzle->atom_len,
            layout.as_or_throw<tirx::TileLayout>(), swizzle->swizzle_inner);
    return tirx::TensorType(shape, tensor.dtype(), tensor.scope(),
                            tensor.ty()->data_alignment, layout,
                            tensor.ty()->loc);
  }

  // `versions` adds the automatic version dimension. A manual version count
  // instead describes a dimension already present in the allocation/views.
  int64_t AllocationBytes(const Var &storage, int versions,
                          int manual_versions = 1) const {
    TVM_FFI_CHECK(versions > 0 && manual_versions > 0, ValueError);
    const TensorVar &owner = Allocation(storage);
    if (Empty(owner))
      return 0;
    TensorVar physical =
        versions == 1
            ? owner
            : TensorVar(owner.CopyWithType(VersionedType(owner, versions)));
    int count = versions > 1 ? versions : manual_versions;
    PrimExpr elements = MemorySpan(physical);
    bool rewritten = false;
    const auto scope = owner.scope().name();
    if (count > 1 && (scope == "shared" || scope == "shared.dyn")) {
      auto analyzer = CreateTileLangAnalyzer();
      auto carries_versions = [&](const TensorVar &tensor) {
        return !tensor.shape().empty() &&
               analyzer->CanProveEqual(tensor.shape()[0],
                                       IntImm(tensor.shape()[0].ty(), count));
      };
      std::optional<TensorVar> carrier;
      if (carries_versions(physical))
        carrier = physical;
      else if (versions == 1)
        for (const TensorVar &view : views_.at(storage))
          if (carries_versions(view)) {
            carrier = view;
            break;
          }
      if (carrier) {
        const TensorVar &logical = *carrier;
        int64_t bits = ElementBits(logical);
        TVM_FFI_CHECK(256 % bits == 0, ValueError)
            << "UB version padding is not representable in " << logical.dtype();
        ffi::Array<PrimExpr> zero, next;
        for (size_t i = 0; i < logical.shape().size(); ++i) {
          zero.push_back(IntImm::Int64(0));
          next.push_back(IntImm::Int64(i == 0 ? 1 : 0));
        }
        PrimExpr pointer =
            versions > 1 ? IntImm::Int64(0)
                         : Multiply(storage_.Address(logical).byte_offset, 8);
        PrimExpr base = analyzer->Simplify(
            Add(pointer, Multiply(TensorElementOffset(logical, zero), bits)));
        PrimExpr slot = analyzer->Simplify(
            pointer + Multiply(TensorElementOffset(logical, next), bits) -
            base);
        bool whole_allocation =
            logical.same_as(physical) ||
            (physical.ty()->HasDefaultLayout() &&
             analyzer->CanProveEqual(GetBufferStorageSizeBytes(physical) * 8,
                                     base + slot * count));
        if (whole_allocation && !IsZero(slot)) {
          PrimExpr aligned = Align(slot, 256);
          if (!analyzer->CanProveEqual(slot, aligned)) {
            // RewriteAscendBufferVersionLayout retains the allocation prefix
            // before a nonzero origin and allocates every padded version slot.
            PrimExpr total =
                analyzer->Simplify(Add(base, Multiply(aligned, count)));
            elements = CeilDiv(total, ElementBits(owner));
            rewritten = true;
          }
        }
      }
    }
    if (!rewritten)
      if (auto layout = physical.ty()->layout)
        if (const auto *swizzle = layout.value().as<tirx::ComposeLayoutObj>();
            swizzle && swizzle->swizzle_len != 0) {
          int bits = swizzle->per_element + swizzle->swizzle_len;
          if (!swizzle->swizzle_inner)
            bits += swizzle->atom_len;
          TVM_FFI_CHECK(bits >= 0 && bits < 63, ValueError);
          elements = Align(elements, int64_t{1} << bits);
        }
    auto size = Constant(elements);
    TVM_FFI_CHECK(size.has_value(), ValueError)
        << "AutoSchedule cannot determine a static physical memory footprint "
           "for "
        << storage->name << ": " << elements;
    return GetBufferStorageSizeBytes(Checked(*size), owner.dtype());
  }

private:
  StorageProvenance storage_ = MakeAscendStorageProvenance();
  std::unordered_map<Var, TensorVar, ObjectPtrHash, ObjectPtrEqual> owners_;
  std::unordered_map<Var, std::vector<TensorVar>, ObjectPtrHash, ObjectPtrEqual>
      views_;
  mutable std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual>
      pitches_;

  static PrimExpr Wide(const PrimExpr &value) {
    return cast(value, PrimType::Int(64));
  }
  static std::optional<ffi::BigInt> Constant(const PrimExpr &value) {
    auto analyzer = CreateTileLangAnalyzer();
    auto result = analyzer->Simplify(value);
    if (auto integer = result.as<IntImmObj>())
      return integer->value;
    return std::nullopt;
  }
  static int64_t Checked(const ffi::BigInt &value) {
    auto result = value.as<int64_t>();
    TVM_FFI_CHECK(result && *result >= 0, ValueError)
        << "Native physical storage size is negative or overflows int64";
    return *result;
  }
  static int64_t SignedChecked(const ffi::BigInt &value) {
    auto result = value.as<int64_t>();
    TVM_FFI_CHECK(result.has_value(), ValueError)
        << "Native physical storage arithmetic overflows int64";
    return *result;
  }
  static PrimExpr Add(const PrimExpr &left, const PrimExpr &right) {
    auto lhs = Constant(left), rhs = Constant(right);
    if (lhs && rhs)
      return IntImm::Int64(SignedChecked(*lhs + *rhs));
    return Wide(left) + Wide(right);
  }
  static PrimExpr Multiply(const PrimExpr &value, int64_t factor) {
    if (auto constant = Constant(value))
      return IntImm::Int64(SignedChecked(*constant * factor));
    return Wide(value) * IntImm::Int64(factor);
  }
  static PrimExpr CeilDiv(const PrimExpr &value, int64_t divisor) {
    TVM_FFI_CHECK_GT(divisor, 0, ValueError);
    if (auto constant = Constant(value))
      return IntImm::Int64(Checked((*constant + divisor - 1) / divisor));
    PrimExpr wide = Wide(value), denominator = IntImm::Int64(divisor);
    return FloorDiv(wide, denominator) +
           Cast(PrimType::Int(64), FloorMod(wide, denominator) != 0);
  }
  static PrimExpr Align(const PrimExpr &value, int64_t alignment) {
    return Multiply(CeilDiv(value, alignment), alignment);
  }
  static bool Empty(const TensorVar &tensor) {
    for (const PrimExpr &extent : tensor.shape())
      if (auto value = Constant(extent); value && *value == 0)
        return true;
    return false;
  }
  static PrimExpr MemorySpan(const TensorVar &tensor) {
    if (Empty(tensor))
      return IntImm::Int64(0);
    auto tile = MemoryTile(tensor);
    PrimExpr span = IntImm::Int64(1);
    ffi::BigInt constant = 1;
    bool all_constant = true;
    auto add = [&](const PrimExpr &extent, const PrimExpr &stride) {
      auto e = Constant(extent), s = Constant(stride);
      if (e && s) {
        TVM_FFI_CHECK(*e >= 0, ValueError) << "Negative native layout extent";
        // Include the maximum address even for a negative memory stride.
        constant += std::max(ffi::BigInt(0), (*e - 1) * *s);
      } else {
        all_constant = false;
      }
      span = span + max(IntImm::Int64(0), (Wide(extent) - 1) * Wide(stride));
    };
    for (const auto &iter : tile->shard)
      if (iter->axis->IsMemoryAxis())
        add(iter->extent, iter->stride);
    for (const auto &iter : tile->replica)
      if (iter->axis->IsMemoryAxis())
        add(iter->extent, iter->stride);
    for (const auto &[axis, offset] : tile->offset)
      if (axis->IsMemoryAxis()) {
        if (auto value = Constant(offset))
          constant += *value;
        else
          all_constant = false;
        span = span + Wide(offset);
      }
    if (all_constant)
      return IntImm::Int64(Checked(constant));
    auto analyzer = CreateTileLangAnalyzer();
    return analyzer->Simplify(span);
  }
};

} // namespace tvm::tl::ascend

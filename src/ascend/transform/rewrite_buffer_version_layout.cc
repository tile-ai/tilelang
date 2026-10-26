#include "compat/analyzer.h"
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

#include "buffer_version.h"

#include <tvm/tirx/transform.h>

#include "ascend/transform/native_storage.h"
#include "op/builtin.h"

namespace tvm::tl {
namespace {

struct VersionLayout {
  PrimExpr base_bits;
  PrimExpr compact_slot_bits;
  PrimExpr aligned_slot_bits;
  PrimExpr physical_bits;
};

class VersionedStorageCollector : public ascend::StorageAliasAnalysis {
public:
  BufferVersionMap versions;

private:
  void VisitStmt_(const RegionStmtNode *node) final {
    ascend::StorageAliasAnalysis::VisitStmt_(node);
    if (auto annotation = node->attrs->dict.Get(attr::kBufferVersion)) {
      for (const auto &[variable, count] :
           annotation->cast<BufferVersionMap>()) {
        TVM_FFI_CHECK(count > 0, ValueError)
            << "Buffer version counts must be positive";
        auto storage = Resolve(variable);
        TVM_FFI_CHECK(storage, ValueError)
            << "Buffer version metadata must refer to a storage variable";
        auto previous = versions.Get(storage->root);
        TVM_FFI_CHECK(!previous || previous.value() == count, ValueError)
            << "Conflicting buffer version counts for storage "
            << storage->root;
        versions.Set(storage->root, count);
      }
    }
  }
};

class BufferVersionLayoutRewriter : public ascend::AnalyzerMutator {
public:
  BufferVersionLayoutRewriter(sym::Analyzer analyzer,
                              VersionedStorageCollector *storage)
      : ascend::AnalyzerMutator(analyzer), storage_(storage) {
    for (const auto &[root, versions] : storage_->versions) {
      runtime::MemoryScope scope =
          root->ty.as<tirx::TensorTypeNode>()
              ? root->ty.as<tirx::TensorTypeNode>()->scope
              : root->ty.as<PtrTypeNode>()->scope;
      if (versions == 1 ||
          (scope.name() != "shared" && scope.name() != "shared.dyn")) {
        continue;
      }
      auto owner = storage_->allocations.Get(root);
      TVM_FFI_CHECK(owner, ValueError)
          << "No allocation owner found for versioned storage " << root;
      auto tensor = owner.value();
      if (tensor.shape().empty() ||
          !tl::arith::CanProve(analyzer_, tensor.shape()[0] == versions)) {
        continue;
      }
      auto geometry = tirx::MatchShapeStridePattern(tensor.ty());
      TVM_FFI_CHECK(geometry, ValueError)
          << "Versioned UB allocations require a strided storage layout";
      int element_bits = ascend::ElementBits(tensor.dtype());
      TVM_FFI_CHECK(element_bits > 0 && 256 % element_bits == 0, ValueError)
          << "32-byte version alignment is not representable in "
          << tensor.dtype();
      PrimExpr slot_bits = tl::arith::Simplify(
          analyzer_,
          ascend::StorageIndex(geometry.value().get<1>()[0]) * element_bits);
      TVM_FFI_CHECK(!tl::arith::CanProve(analyzer_, slot_bits <= 0), ValueError)
          << "Versioned UB slots must have positive size";
      PrimExpr alignment = IntImm(PrimType::Int(64), 256);
      PrimExpr aligned_bits = tl::arith::Simplify(
          analyzer_, FloorDiv(slot_bits + 255, alignment) * alignment);
      if (tl::arith::CanProve(analyzer_, aligned_bits == slot_bits))
        continue;
      PrimExpr base = tl::arith::Simplify(
          analyzer_,
          ascend::StorageIndex(geometry.value().get<2>()) * element_bits);
      layouts_.emplace(
          root.get(),
          VersionLayout{
              base, slot_bits, aligned_bits,
              tl::arith::Simplify(analyzer_, base + aligned_bits * versions)});
    }
  }

private:
  const VersionLayout *Layout(const ascend::StorageAccess &storage) const {
    auto found = layouts_.find(storage.root.get());
    return found == layouts_.end() ? nullptr : &found->second;
  }

  PrimExpr RewriteBits(const VersionLayout &layout, PrimExpr offset) {
    PrimExpr relative = offset - layout.base_bits;
    return tl::arith::Simplify(
        analyzer_, layout.base_bits +
                       FloorDiv(relative, layout.compact_slot_bits) *
                           layout.aligned_slot_bits +
                       FloorMod(relative, layout.compact_slot_bits));
  }

  PrimExpr Elements(const PrimExpr &bits, int element_bits) {
    PrimExpr divisor = IntImm(bits.ty().WithLanes(1), element_bits);
    return tl::arith::Simplify(analyzer_, floordiv(bits, divisor));
  }

  tirx::TensorType PhysicalType(const tirx::TensorVar &tensor,
                                const VersionLayout &layout) {
    return tirx::TensorType(
        {Elements(layout.physical_bits, ascend::ElementBits(tensor.dtype()))},
        tensor.dtype(), tensor.ty()->scope, tensor.ty()->data_alignment,
        std::nullopt, tensor->loc);
  }

  Expr RootPointer(const Var &root) {
    Var mapped = VisitExpr(root).as_or_throw<Var>();
    if (auto tensor = mapped.as<tirx::TensorVar>())
      return tensor.value().data();
    return mapped;
  }

  void ValidateSpan(const VersionLayout &layout, const PrimExpr &offset,
                    const PrimExpr &size) {
    PrimExpr inner =
        floormod(offset - layout.base_bits, layout.compact_slot_bits);
    TVM_FFI_CHECK(!tl::arith::CanProve(analyzer_,
                                       inner + size > layout.compact_slot_bits),
                  ValueError)
        << "Access crosses a 32-byte-aligned version slot: offset=" << offset
        << ", span_bits=" << size;
  }

  ffi::Array<PrimExpr> PhysicalIndices(const tirx::TensorVar &tensor,
                                       const ffi::Array<PrimExpr> &indices,
                                       const ascend::StorageAccess &storage,
                                       const VersionLayout &layout) {
    auto offsets = tensor.ty()->ElemOffset(indices);
    TVM_FFI_CHECK(offsets.size() == 1, ValueError)
        << "Versioned UB aliases must have one memory offset";
    PrimExpr offset = VisitExpr(offsets[0]).as_or_throw<PrimExpr>();
    int element_bits = ascend::ElementBits(tensor.dtype());
    if (const auto *ramp = offset.as<RampNode>()) {
      PrimExpr begin =
          storage.bit_offset + ascend::StorageIndex(ramp->base) * element_bits;
      PrimExpr size = (ascend::StorageIndex(ramp->lanes) - 1) *
                          ascend::StorageIndex(ramp->stride) * element_bits +
                      element_bits;
      ValidateSpan(layout, begin, size);
      TVM_FFI_CHECK(
          tl::arith::CanProve(analyzer_, floormod(begin - layout.base_bits,
                                                  layout.compact_slot_bits) +
                                                 size <=
                                             layout.compact_slot_bits),
          ValueError)
          << "A versioned vector access must stay within one physical slot";
      return {Ramp(Elements(RewriteBits(layout, begin), element_bits),
                   ascend::StorageIndex(ramp->stride), ramp->lanes)};
    }
    PrimExpr bits =
        storage.bit_offset + ascend::StorageIndex(offset) * element_bits;
    return {Elements(RewriteBits(layout, bits), element_bits)};
  }

  Stmt VisitStmt_(const RegionStmtNode *node) final {
    auto region =
        ascend::AnalyzerMutator::VisitStmt_(node).as_or_throw<RegionStmt>();
    if (region->attrs->dict.count(attr::kBufferVersion)) {
      auto attrs = region->attrs->dict;
      attrs.erase(attr::kBufferVersion);
      region.CopyOnWrite()->attrs = DictAttrs(attrs);
    }
    return region;
  }

  Stmt VisitStmt_(const BindNode *node) final {
    auto tensor = node->var.as<tirx::TensorVar>();
    auto storage = storage_->Resolve(node->var);
    const VersionLayout *layout = storage ? Layout(storage.value()) : nullptr;
    if (!tensor || !layout)
      return ascend::AnalyzerMutator::VisitStmt_(node);
    auto type = PhysicalType(tensor.value(), *layout);
    Var rewritten = node->var.CopyWithType(type);
    Expr value = node->value;
    if (const auto *call = value.as<CallNode>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      value = Call(type, call->op, call->args, call->attrs, {type}, call->loc);
    } else {
      value = Call(type, tirx::decl_tensor_op(), {RootPointer(storage->root)},
                   {}, {type}, node->loc);
    }
    VarRemapSet(node->var, rewritten);
    return Bind(rewritten, value, node->loc);
  }

  PrimExpr VisitExpr_(const TensorLoadNode *node) final {
    auto storage = storage_->Resolve(node->source);
    const VersionLayout *layout = storage ? Layout(storage.value()) : nullptr;
    if (!layout)
      return ascend::AnalyzerMutator::VisitExpr_(node);
    auto indices = PhysicalIndices(node->source.as_or_throw<tirx::TensorVar>(),
                                   node->indices, storage.value(), *layout);
    return TensorLoad(VisitExpr(node->source), indices, node->loc);
  }

  Stmt VisitStmt_(const TensorStoreNode *node) final {
    auto storage = storage_->Resolve(node->dest);
    const VersionLayout *layout = storage ? Layout(storage.value()) : nullptr;
    if (!layout)
      return ascend::AnalyzerMutator::VisitStmt_(node);
    auto indices = PhysicalIndices(node->dest.as_or_throw<tirx::TensorVar>(),
                                   node->indices, storage.value(), *layout);
    return TensorStore(VisitExpr(node->dest), indices,
                       VisitExpr(node->value).as_or_throw<PrimExpr>(),
                       node->loc);
  }

  Expr VisitExpr_(const CallNode *node) final {
    if (node->op.same_as(tirx::ptr_byte_offset_op()) ||
        node->op.same_as(tirx::tensor_data_ptr_op()) ||
        node->op.same_as(tirx::address_of_op())) {
      auto storage = storage_->Resolve(ffi::GetRef<Expr>(node));
      const VersionLayout *layout = storage ? Layout(storage.value()) : nullptr;
      if (layout) {
        PrimExpr offset =
            VisitExpr(storage->bit_offset).as_or_throw<PrimExpr>();
        auto attrs = node->attrs;
        if (auto dict = attrs.as<DictAttrs>()) {
          if (auto count = dict.value()->dict.Get(attr::kAccessExtent)) {
            PrimExpr extent = count->as<PrimExpr>().value_or(
                IntImm(PrimType::Int(64),
                       GetIntAnnotation(count.value()).value_or(0)));
            auto pointer = node->ty.as_or_throw<PtrType>();
            int element_bits = ascend::ElementBits(
                pointer->element_type.as_or_throw<PrimType>());
            ValidateSpan(*layout, offset,
                         ascend::StorageIndex(extent) * element_bits);
          }
        }
        PrimExpr bytes = Elements(RewriteBits(*layout, offset), 8);
        return Call(node->ty, tirx::ptr_byte_offset_op(),
                    {RootPointer(storage->root), bytes}, attrs, node->ty_args,
                    node->loc);
      }
    }
    return ascend::AnalyzerMutator::VisitExpr_(node);
  }

  VersionedStorageCollector *storage_;
  std::unordered_map<const VarNode *, VersionLayout> layouts_;
};

} // namespace

namespace transform {

tvm::transform::Pass RewriteAscendBufferVersionLayout() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (!function->body)
      return function;
    auto storage = ffi::make_object<VersionedStorageCollector>();
    storage->Collect(function);
    if (storage->versions.empty())
      return function;
    sym::Analyzer analyzer;
    auto rewriter =
        ffi::make_object<BufferVersionLayoutRewriter>(analyzer, storage.get());
    auto original = function->body.value();
    function.CopyOnWrite()->body =
        rewriter->Mutate(original).ValueOrUnchanged(original);
    return function;
  };
  return tirx::transform::CreateFunctionPass(
      pass, 0, "tl.RewriteAscendBufferVersionLayout");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.transform.RewriteAscendBufferVersionLayout",
      RewriteAscendBufferVersionLayout);
}

} // namespace transform
} // namespace tvm::tl

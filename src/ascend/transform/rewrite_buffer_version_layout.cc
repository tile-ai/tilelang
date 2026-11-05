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
 * \file rewrite_buffer_version_layout.cc
 * \brief Align physical Ascend UB slots while rebuilding every typed alias.
 */
#include "auto_schedule/native_storage.h"
#include "buffer_version.h"
#include "ir/analyzer.h"
#include "ir/stmt_functor.h"
#include "op/native_intrinsic.h"
#include "op/utils.h"
#include "transform/common/storage_size.h"
#include <limits>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>
#include <unordered_map>
#include <vector>

namespace tvm::tl {
namespace {
using namespace tirx;
using namespace ffi;
using namespace prim;
using namespace ascend;
constexpr const char *kPhysicalVersion = "tl.buffer_version";
constexpr int kAlignmentBits = 256;
using Fields = Map<String, Any>;

struct StorageLayout {
  TensorVar owner;
  PrimExpr base_bits;
  PrimExpr slot_bits;
  PrimExpr aligned_bits;
  PrimExpr total_bits;
};
using Layouts =
    std::unordered_map<Var, StorageLayout, ObjectPtrHash, ObjectPtrEqual>;

class VersionMetadata {
public:
  BufferVersionMap versions;
  NativeStorageContext storage;
  std::unordered_map<Var, TensorVar, ObjectPtrHash, ObjectPtrEqual> owners;
  std::unordered_map<Var, std::vector<TensorVar>, ObjectPtrHash, ObjectPtrEqual>
      views;
  explicit VersionMetadata(const Stmt &stmt) {
    storage.Collect(stmt);
    VisitNativePostOrder(stmt, [&](const ObjectRef &object) {
      if (const auto *bind = object.as<BindObj>()) {
        Expr value = bind->value;
        if (const auto *c = value.as<CallObj>(); IsScheduledBind(c))
          value = ReadScheduledBinding(c).value;
        const auto *call = value.as<CallObj>();
        if (auto tensor = bind->var.as<TensorVar>()) {
          Var root = storage.Identity(*tensor);
          views[root].push_back(*tensor);
          if (call && call->op.same_as(alloc_tensor_op()))
            owners.emplace(root, *tensor);
        }
      }
      if (const auto *call = object.as<CallObj>();
          loop_annotation_detail::IsAnnotation(call)) {
        auto fields = ReadNativeAnnotations(call);
        if (auto value = fields.Get(kPhysicalVersion))
          Add(*value);
      }
      if (const auto *region = object.as<RegionStmtObj>()) {
        if (auto attr = ReadTaskAttribute(region)) {
          if (attr->key == kPhysicalVersion)
            Add(attr->node);
          else if (attr->key == "tl.sblock" ||
                   attr->key == "tl.loop.annotations") {
            auto fields = attr->value.cast<Fields>();
            if (auto value = fields.Get(kPhysicalVersion))
              Add(*value);
            if (attr->key == "tl.sblock")
              if (auto annotations = fields.Get("annotations"))
                if (auto value =
                        annotations->cast<Fields>().Get(kPhysicalVersion))
                  Add(*value);
          }
        }
      }
    });
  }

private:
  void Add(const Any &value) {
    for (const auto &[key, count_any] : value.cast<Map<Any, Any>>()) {
      Var variable = key.cast<Var>();
      auto address = storage.Address(variable);
      TVM_FFI_CHECK(address.exact && address.root.as<VarObj>(), ValueError)
          << "Versioned storage requires exact native allocation provenance: "
          << variable;
      Var root = address.root.as_or_throw<Var>();
      auto count = count_any.try_cast<int64_t>();
      if (!count)
        count = TryConstInt(count_any.cast<PrimExpr>());
      TVM_FFI_CHECK(count && *count > 0 &&
                        *count <= std::numeric_limits<int>::max(),
                    ValueError)
          << "Version count must be a positive int32 constant";
      if (auto previous = versions.Get(root))
        TVM_FFI_CHECK_EQ(*previous, *count, ValueError)
            << "Conflicting buffer version counts for " << root;
      versions.Set(root, static_cast<int>(*count));
    }
  }
};

class BufferVersionLayoutRewriter : public StmtExprMutatorWithAnalyzer {
public:
  using StmtExprMutatorWithAnalyzer::Mutate_;
  BufferVersionLayoutRewriter(const tirx::Function &function,
                              NativeStorageContext storage, Layouts layouts,
                              sym::Analyzer analyzer)
      : StmtExprMutatorWithAnalyzer(analyzer), storage_(std::move(storage)),
        layouts_(std::move(layouts)), analyzer_owner_(std::move(analyzer)) {
    MarkTensorParamShapes(function);
  }

private:
  NativeStorageContext storage_;
  Layouts layouts_;
  Map<TensorVar, TensorVar> tensors_;
  // The native analyzer mutator retains a pointer, so own its lifetime here.
  sym::Analyzer analyzer_owner_;

  const StorageLayout *Layout(const Expr &value) const {
    auto address = storage_.Address(value);
    if (!address.root.as<VarObj>())
      return nullptr;
    auto found = layouts_.find(address.root.as_or_throw<Var>());
    if (found == layouts_.end())
      return nullptr;
    // A transparent pointer chain can still identify this owner while its
    // missing type, changed scope, or effectful offset prevents an exact
    // address proof. Do not pad the owner and silently leave that alias in
    // the old coordinates. This pass consumes fully typed storage addresses;
    // it does not infer types for unfinished pointer expressions.
    TVM_FFI_CHECK(address.exact, ValueError)
        << "Versioned storage requires exact, fully typed pointer provenance "
           "with a preserved storage scope and pure offset: "
        << value;
    return &found->second;
  }
  PrimExpr Wide(PrimExpr value) { return cast(value, PrimType::Int(64)); }
  PrimExpr RewriteBits(const StorageLayout &layout, PrimExpr original) {
    original = Wide(original);
    PrimExpr relative = original - layout.base_bits;
    PrimExpr mapped =
        layout.base_bits +
        FloorDiv(relative, layout.slot_bits) * layout.aligned_bits +
        FloorMod(relative, layout.slot_bits);
    // Prefix bytes before a nonzero layout origin are unchanged.
    // tensor_data_ptr denotes the backing allocation, so it must continue to
    // point at byte 0.
    if (!IsZero(layout.base_bits))
      mapped = Select(original < layout.base_bits, original, mapped);
    return GetSyncedAnalyzer()->Simplify(mapped);
  }
  PrimExpr RewriteUnits(const StorageLayout &layout, PrimExpr original,
                        int unit_bits) {
    auto unit = IntImm::Int64(unit_bits);
    auto aligned = [&](const PrimExpr &value) {
      return GetSyncedAnalyzer()->CanProve(FloorMod(value, unit) == 0);
    };
    original = Wide(original);
    // Construct the quotient in the access type's units when every boundary
    // has that alignment. In particular, a dynamic compact stride n * bits
    // remains an exact element stride without requiring the analyzer to
    // distribute a second modulo through a symbolic FloorMod expression.
    // A cross-dtype view can straddle differently sized source elements, so
    // retain the general bit-coordinate mapping when this factoring fails.
    if (aligned(original) && aligned(layout.base_bits) &&
        aligned(layout.slot_bits) && aligned(layout.aligned_bits)) {
      PrimExpr position =
          GetSyncedAnalyzer()->Simplify(FloorDiv(original, unit));
      PrimExpr base =
          GetSyncedAnalyzer()->Simplify(FloorDiv(layout.base_bits, unit));
      PrimExpr slot =
          GetSyncedAnalyzer()->Simplify(FloorDiv(layout.slot_bits, unit));
      PrimExpr stride =
          GetSyncedAnalyzer()->Simplify(FloorDiv(layout.aligned_bits, unit));
      PrimExpr relative = position - base;
      PrimExpr mapped =
          base + FloorDiv(relative, slot) * stride + FloorMod(relative, slot);
      if (!IsZero(base))
        mapped = Select(position < base, position, mapped);
      return GetSyncedAnalyzer()->Simplify(mapped);
    }
    PrimExpr mapped = RewriteBits(layout, original);
    TVM_FFI_CHECK(aligned(mapped), ValueError)
        << "Versioned view is not aligned to its access width of " << unit_bits
        << " bits";
    return GetSyncedAnalyzer()->Simplify(FloorDiv(mapped, unit));
  }
  TensorVar Physical(const TensorVar &tensor) {
    if (auto known = tensors_.Get(tensor))
      return *known;
    const auto *layout = Layout(tensor);
    if (!layout)
      return tensor;
    int bits = tensor.dtype().bits() * tensor.dtype().lanes();
    TVM_FFI_CHECK(bits > 0 && kAlignmentBits % bits == 0, ValueError)
        << "UB version padding is not representable in " << tensor.dtype()
        << " elements";
    PrimExpr extent = GetSyncedAnalyzer()->Simplify(
        FloorDiv(layout->total_bits + (bits - 1), IntImm::Int64(bits)));
    TensorType type({extent}, tensor.dtype(), tensor.scope(),
                    tensor.ty()->data_alignment, {}, tensor->loc);
    TensorVar result(tensor->name, type, tensor->loc);
    tensors_.Set(tensor, result);
    VarRemapSetExpected(tensor, result).value();
    return result;
  }
  PrimExpr OriginalBits(const TensorVar &tensor, Array<PrimExpr> indices) {
    for (size_t i = 0; i < indices.size(); ++i)
      indices.Set(i, Wide(indices[i]));
    auto address = storage_.Address(tensor);
    return GetSyncedAnalyzer()->Simplify(
        Wide(address.byte_offset) * 8 +
        Wide(TensorElementOffset(tensor, indices)) *
            (tensor.dtype().bits() * tensor.dtype().lanes()));
  }
  PrimExpr Index(const TensorVar &tensor, Array<PrimExpr> indices) {
    const auto *layout = Layout(tensor);
    TVM_FFI_ICHECK(layout);
    int bits = tensor.dtype().bits() * tensor.dtype().lanes();
    PrimExpr original = OriginalBits(tensor, std::move(indices));
    // A wider dtype can place one element across two compact versions.
    // Mapping only its first bit would then read or write padding instead
    // of the bytes in the next physical slot. Such an element needs the
    // same contiguous-span check as an explicit access pointer.
    ValidateSpan(*layout, original, IntImm::Int64(1), bits);
    return RewriteUnits(*layout, original, bits);
  }
  void ValidateSpan(const StorageLayout &layout, PrimExpr start_bits,
                    PrimExpr extent, int bits) {
    PrimExpr span = Wide(extent) * bits;
    PrimExpr inner =
        FloorMod(Wide(start_bits) - layout.base_bits, layout.slot_bits);
    // A pointer describes a contiguous physical range. Preserve the existing
    // contract: reject a span proved to cross padding, using available lexical
    // facts, while an unproved dynamic span remains the caller's
    // responsibility. Requiring a positive proof would reject valid legacy
    // calls whose bounds are guaranteed by the caller but are not encoded in
    // this Function.
    PrimExpr contiguous = inner + span <= layout.slot_bits;
    if (!IsZero(layout.base_bits)) {
      contiguous =
          Select(Wide(start_bits) < layout.base_bits,
                 span <= layout.base_bits - Wide(start_bits) + layout.slot_bits,
                 contiguous);
    }
    TVM_FFI_CHECK(!GetSyncedAnalyzer()->CanProve(Not(contiguous)), ValueError)
        << "Access pointer crosses a 32-byte-aligned version slot: offset="
        << start_bits << ", extent=" << extent
        << ", compact_slot_bits=" << layout.slot_bits;
  }
  Expr BasePointer(const StorageLayout &layout, Type type, Location loc) {
    return tirx::reinterpret(type, Physical(layout.owner).data(), loc);
  }
  Fields Strip(Fields fields) {
    fields.erase(kPhysicalVersion);
    return fields;
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    const auto *call = op->value.as<CallObj>();
    if (!loop_annotation_detail::IsAnnotation(call))
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    auto fields = ReadNativeAnnotations(call);
    if (!fields.count(kPhysicalVersion))
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    fields = Strip(fields);
    Stmt result = fields.empty() ? Stmt(SeqStmt({}))
                                 : MakeNativeAnnotation(fields, op->loc);
    return Mutate(result, mode).ValueOrUnchanged(result);
  }
  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    if (auto attr = ReadTaskAttribute(op)) {
      if (attr->key == kPhysicalVersion)
        return Mutate(op->body, mode).ValueOrUnchanged(op->body);
      if (attr->key == "tl.sblock" || attr->key == "tl.loop.annotations") {
        auto fields = attr->value.cast<Fields>();
        auto updated = Strip(fields);
        if (attr->key == "tl.sblock")
          if (auto annotations = updated.Get("annotations"))
            updated.Set("annotations", Strip(annotations->cast<Fields>()));
        if (!StructuralEqual()(fields, updated)) {
          RegionStmt next = MakeNativeAttribute(attr->node, attr->key, updated,
                                                op->body, op->loc);
          return Mutate(next, mode).ValueOrUnchanged(next);
        }
      }
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }
  UnchangedOr<Stmt> Mutate_(const BindObj *op, InplaceMode mode) final {
    auto tensor = op->var.as<TensorVar>();
    if (!tensor || !Layout(*tensor))
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    const auto &layout = *Layout(*tensor);
    Expr value = op->value;
    const auto *marker = value.as<CallObj>();
    bool scheduled = IsScheduledBind(marker);
    if (scheduled)
      value = ReadScheduledBinding(marker).value;
    const auto *call = value.as<CallObj>();
    TensorVar replacement = Physical(*tensor);
    if (call && call->op.same_as(alloc_tensor_op())) {
      value = Call(replacement.ty(), alloc_tensor_op(), call->args, call->attrs,
                   {replacement.ty()}, call->loc);
    } else {
      // All typed aliases of this owner share the raw base after rewriting;
      // each load/store now carries its complete remapped physical offset.
      value =
          Call(replacement.ty(), decl_tensor_op(),
               {BasePointer(layout, replacement.ty()->DataPtrType(), op->loc)},
               {}, {replacement.ty()}, op->loc);
    }
    Bind result(replacement, value, op->loc);
    if (scheduled) {
      auto metadata = ReadScheduledBinding(marker);
      return MakeScheduledBinding(result, metadata.stage, metadata.task);
    }
    return result;
  }
  UnchangedOr<PrimExpr> Mutate_(const TensorLoadObj *op,
                                InplaceMode mode) final {
    auto tensor = op->source.as_or_throw<TensorVar>();
    if (!Layout(tensor))
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    PrimExpr index = Index(tensor, op->indices);
    index = Mutate(index, mode).ValueOrUnchanged(index).as_or_throw<PrimExpr>();
    return TensorLoad(Physical(tensor), {index}, op->loc);
  }
  UnchangedOr<Stmt> Mutate_(const TensorStoreObj *op, InplaceMode mode) final {
    auto tensor = op->dest.as_or_throw<TensorVar>();
    if (!Layout(tensor))
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    PrimExpr index = Index(tensor, op->indices);
    index = Mutate(index, mode).ValueOrUnchanged(index).as_or_throw<PrimExpr>();
    PrimExpr value = Mutate(op->value, mode)
                         .ValueOrUnchanged(op->value)
                         .as_or_throw<PrimExpr>();
    return TensorStore(Physical(tensor), {index}, value, op->loc);
  }
  UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    if (op->op.same_as(Op::Get("tl.access_ptr"))) {
      TVM_FFI_CHECK_EQ(op->args.size(), 3U, ValueError);
      const auto *load = op->args[0].as<TensorLoadObj>();
      TVM_FFI_CHECK(load, ValueError);
      auto tensor = load->source.as_or_throw<TensorVar>();
      if (const auto *layout = Layout(tensor)) {
        ValidateSpan(*layout, OriginalBits(tensor, load->indices),
                     op->args[1].as_or_throw<PrimExpr>(),
                     tensor.dtype().bits() * tensor.dtype().lanes());
      }
      return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
    }
    if (op->op.same_as(Op::Get("tl.ptr_access"))) {
      TVM_FFI_CHECK_EQ(op->args.size(), 4U, ValueError);
      if (const auto *layout = Layout(op->args[0])) {
        auto address = storage_.Address(op->args[0]);
        auto ptr_type = op->args[0]->ty.as_or_throw<PtrType>();
        auto dtype = ptr_type->element_type.as_or_throw<PrimType>();
        int bits = dtype.bits() * dtype.lanes();
        PrimExpr start = Wide(address.byte_offset) * 8 +
                         Wide(op->args[1].as_or_throw<PrimExpr>()) * bits;
        ValidateSpan(*layout, start, op->args[2].as_or_throw<PrimExpr>(), bits);
        PrimExpr index = RewriteUnits(*layout, start, bits);
        Array<Expr> args = {
            BasePointer(*layout, ptr_type, op->loc), index,
            Mutate(op->args[2], mode).ValueOrUnchanged(op->args[2]),
            op->args[3]};
        return Call(op->ty, op->op, args, op->attrs, op->ty_args, op->loc);
      }
    }
    if (op->ty.as<PtrTypeObj>() && (op->op.same_as(tensor_data_ptr_op()) ||
                                    op->op.same_as(ptr_byte_offset_op()) ||
                                    op->op.same_as(reinterpret_op()))) {
      if (const auto *layout = Layout(GetRef<Call>(op))) {
        auto address = storage_.Address(GetRef<Call>(op));
        PrimExpr offset =
            RewriteUnits(*layout, Wide(address.byte_offset) * 8, 8);
        Expr pointer = BasePointer(*layout, op->ty, op->loc);
        return Call(op->ty, ptr_byte_offset_op(), {pointer, offset}, {}, {},
                    op->loc);
      }
    }
    return StmtExprMutatorWithAnalyzer::Mutate_(op, mode);
  }
};

tirx::Function RewriteBufferVersionLayout(tirx::Function function) {
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return function;
  VersionMetadata metadata(function->body);
  if (metadata.versions.empty())
    return function;
  auto analyzer = CreateTileLangAnalyzer();
  Layouts layouts;
  for (const auto &[root, count] : metadata.versions) {
    String scope;
    if (auto tensor = root.as<TensorVar>())
      scope = tensor->scope().name();
    else if (auto pointer = root->ty.as<PtrType>())
      scope = pointer.value()->scope.name();
    if (count == 1 || (scope != "shared" && scope != "shared.dyn"))
      continue;
    auto found = metadata.owners.find(root);
    TVM_FFI_CHECK(found != metadata.owners.end(), ValueError)
        << "No allocation owner found for versioned storage " << root;
    TensorVar owner = found->second;
    auto carries_versions = [&](const TensorVar &tensor) {
      return !tensor.shape().empty() &&
             analyzer->CanProveEqual(tensor.shape()[0],
                                     IntImm(tensor.shape()[0].ty(), count));
    };
    Optional<TensorVar> carrier;
    if (carries_versions(owner))
      carrier = owner;
    else
      for (const TensorVar &view : metadata.views[root])
        if (carries_versions(view)) {
          carrier = view;
          break;
        }
    if (!carrier)
      continue;
    const TensorVar &logical = *carrier;
    int bits = logical.dtype().bits() * logical.dtype().lanes();
    TVM_FFI_CHECK(bits > 0 && kAlignmentBits % bits == 0, ValueError);
    Array<PrimExpr> zero, next;
    for (size_t i = 0; i < logical.shape().size(); ++i) {
      zero.push_back(IntImm::Int64(0));
      next.push_back(IntImm::Int64(i == 0 ? 1 : 0));
    }
    PrimExpr pointer_bits =
        cast(metadata.storage.Address(logical).byte_offset, PrimType::Int(64)) *
        8;
    PrimExpr base = analyzer->Simplify(
        pointer_bits +
        cast(TensorElementOffset(logical, zero), PrimType::Int(64)) * bits);
    PrimExpr slot = analyzer->Simplify(
        pointer_bits +
        cast(TensorElementOffset(logical, next), PrimType::Int(64)) * bits -
        base);
    // A shaped alias can describe a flat ring allocation, but a small view of
    // a larger workspace does not turn the whole workspace into version slots.
    if (!logical.same_as(owner) &&
        (!owner.ty()->HasDefaultLayout() ||
         !analyzer->CanProveEqual(GetBufferStorageSizeBytes(owner) * 8,
                                  base + slot * count)))
      continue;
    if (auto size = TryConstInt(slot)) {
      TVM_FFI_CHECK(*size >= 0 && *size <= std::numeric_limits<int64_t>::max() -
                                               kAlignmentBits + 1,
                    ValueError)
          << "Version slot size is invalid or overflows int64";
      if (*size == 0)
        continue;
    }
    PrimExpr aligned = analyzer->Simplify(
        FloorDiv(slot + (kAlignmentBits - 1), IntImm::Int64(kAlignmentBits)) *
        kAlignmentBits);
    if (analyzer->CanProveEqual(slot, aligned))
      continue;
    PrimExpr total = analyzer->Simplify(base + aligned * count);
    layouts.emplace(root, StorageLayout{owner, base, slot, aligned, total});
  }
  auto rewriter = make_object<BufferVersionLayoutRewriter>(
      function, metadata.storage, layouts, CreateTileLangAnalyzer());
  auto body = function->body;
  function.CopyOnWrite()->body =
      rewriter->Mutate(body, InplaceMode::kDisallow).ValueOrUnchanged(body);
  return function;
}
} // namespace
namespace transform {
tvm::transform::Pass RewriteAscendBufferVersionLayout() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    return RewriteBufferVersionLayout(std::move(function));
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

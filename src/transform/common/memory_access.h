#ifndef TILELANG_TRANSFORM_COMMON_MEMORY_ACCESS_H_
#define TILELANG_TRANSFORM_COMMON_MEMORY_ACCESS_H_

#include "compat/visit.h"
#include "constr_visitor.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "storage_size.h"

#include <tvm/tirx/op/memory.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {

struct MemoryLocation {
  ffi::Optional<Var> allocation;
  ffi::String scope;
  PrimExpr offset = IntImm::Int64(0);
  bool precise = true;
};

struct MemoryAccess {
  MemoryLocation location;
  PrimExpr bytes = IntImm::Int64(0);
  int mask = kAccessReadWrite;
  bool atomic = false;
  bool asynchronous = false;
  ConstrSet constraints;
};

class MemoryAliases {
public:
  std::unordered_map<const VarNode *, MemoryLocation> aliases;
  std::unordered_set<const VarNode *> allocations;

  MemoryLocation Resolve(const Expr &expression) const {
    if (auto variable = expression.as<Var>()) {
      if (auto found = aliases.find(variable.value().get());
          found != aliases.end())
        return found->second;
      if (auto tensor = variable.value().as<tirx::TensorVar>())
        return {*variable, tensor.value().scope().name()};
      if (const auto *pointer = variable.value()->ty.as<PtrTypeNode>())
        return {*variable, pointer->scope.name()};
      return {std::nullopt, "", IntImm::Int64(0), false};
    }
    const auto *call = expression.as<CallNode>();
    if (!call || call->args.empty())
      return {std::nullopt, "", IntImm::Int64(0), false};
    if (call->op.same_as(tirx::tensor_data_ptr_op()) ||
        call->op.same_as(tirx::reinterpret_op()) ||
        call->op.same_as(tirx::decl_tensor_op()))
      return Resolve(call->args[0]);
    if (call->op.same_as(tirx::ptr_byte_offset_op()) ||
        call->op.same_as(tirx::handle_add_byte_offset_op())) {
      MemoryLocation result = Resolve(call->args[0]);
      result.offset =
          result.offset +
          Cast(PrimType::Int(64), call->args[1].as_or_throw<PrimExpr>());
      return result;
    }
    if (call->op.same_as(tirx::address_of_op()) ||
        call->op.same_as(access_ptr())) {
      if (const auto *load = call->args[0].as<TensorLoadNode>())
        return Element(load->source, load->indices);
      if (const auto *region = call->args[0].as<TensorRegionNode>())
        return Element(region->source,
                       region->region.Map(
                           [](const Range &range) { return range->begin; }));
      return Resolve(call->args[0]);
    }
    return {std::nullopt, "", IntImm::Int64(0), false};
  }

  MemoryLocation Element(const Expr &storage,
                         const ffi::Array<PrimExpr> &indices) const {
    MemoryLocation result = Resolve(storage);
    auto tensor = storage.as<tirx::TensorVar>();
    if (!tensor || indices.size() != tensor.value().shape().size()) {
      result.precise = false;
      return result;
    }
    for (const PrimExpr &index : indices) {
      if (!index.ty().IsScalar()) {
        result.precise = false;
        return result;
      }
    }
    PrimExpr offset = IntImm::Int64(0);
    if (auto layout = tensor.value().ty()->layout) {
      auto mapped = layout.value().Apply(indices, tensor.value().shape());
      if (mapped.size() != 1 || !mapped.count("m")) {
        result.precise = false;
        return result;
      }
      offset = Cast(PrimType::Int(64), mapped.at("m"));
    } else {
      for (size_t axis = 0; axis < indices.size(); ++axis)
        offset =
            offset * Cast(PrimType::Int(64), tensor.value().shape()[axis]) +
            Cast(PrimType::Int(64), indices[axis]);
    }
    PrimType type = tensor.value().dtype();
    int bits = IsPacked4BitStorage(type) ? type.bits() * type.lanes()
                                         : type.StorageBytes() * 8;
    result.offset = result.offset +
                    FloorDiv(offset * IntImm::Int64(bits), IntImm::Int64(8));
    return result;
  }

  void Bind(const BindNode *node) {
    if (!node->var->ty.as<tirx::TensorTypeNode>() &&
        !node->var->ty.as<PtrTypeNode>())
      return;
    const auto *call = node->value.as<CallNode>();
    if (call && call->op.same_as(tirx::alloc_tensor_op())) {
      auto tensor = node->var.as_or_throw<tirx::TensorVar>();
      aliases.insert_or_assign(
          node->var.get(), MemoryLocation{node->var, tensor.scope().name()});
      allocations.insert(node->var.get());
    } else {
      MemoryLocation location = Resolve(node->value);
      if (location.scope.empty()) {
        if (auto tensor = node->var.as<tirx::TensorVar>())
          location.scope = tensor.value().scope().name();
        else
          location.scope = node->var->ty.as_or_throw<PtrType>()->scope.name();
      }
      aliases.insert_or_assign(node->var.get(), location);
    }
  }

  bool Disjoint(const MemoryLocation &left, const MemoryLocation &right) const {
    if (!left.scope.empty() && !right.scope.empty() &&
        left.scope != right.scope) {
      bool both_shared =
          left.scope.find("shared") == 0 && right.scope.find("shared") == 0;
      if (!both_shared)
        return true;
    }
    return left.allocation && right.allocation &&
           !left.allocation.value().same_as(*right.allocation) &&
           (allocations.count(left.allocation.value().get()) ||
            allocations.count(right.allocation.value().get()));
  }
};

class MemoryAccessCollector : public ConstrVisitor {
public:
  explicit MemoryAccessCollector(MemoryAliases aliases = {})
      : aliases(std::move(aliases)) {}

  MemoryAliases aliases;
  std::vector<MemoryAccess> accesses;
  bool exits = false;

private:
  using Result = ffi::Optional<ffi::VisitInterrupt>;

  static bool IsStorageReference(const CallNode *node) {
    return node->op.same_as(tirx::tensor_data_ptr_op()) ||
           node->op.same_as(tirx::reinterpret_op()) ||
           node->op.same_as(tirx::decl_tensor_op()) ||
           node->op.same_as(tirx::alloc_tensor_op()) ||
           node->op.same_as(tirx::address_of_op()) ||
           node->op.same_as(tirx::ptr_byte_offset_op()) ||
           node->op.same_as(tirx::handle_add_byte_offset_op()) ||
           node->op.same_as(access_ptr());
  }

  ConstrSet Snapshot() const {
    ConstrSet constraints = GetConstrSet();
    for (Constr &constraint : constraints.constrs_)
      constraint = constraint.FreshenReads();
    return constraints;
  }

  void Append(MemoryLocation location, PrimExpr bytes, int mask,
              ConstrSet constraints) {
    sym::Analyzer analyzer;
    auto guard = constraints.Populate(analyzer);
    auto check_bounds = [&](const ffi::ObjectRef &node) {
      if (!node.as<AddNode>() && !node.as<SubNode>() && !node.as<MulNode>())
        return;
      PrimExpr expression = node.as_or_throw<PrimExpr>();
      PrimType type = expression.ty();
      if (!type.IsScalar() || !type.MatchesCode(kDLInt, kDLUInt)) {
        location.precise = false;
        return;
      }
      auto bound = analyzer->const_int_bound(expression);
      location.precise &= bound->min_value > sym::ConstIntBound::kNegInf &&
                          bound->max_value < sym::ConstIntBound::kPosInf &&
                          ffi::BigInt(bound->min_value) >=
                              min_value(type).as_or_throw<IntImm>()->value &&
                          ffi::BigInt(bound->max_value) <=
                              max_value(type).as_or_throw<IntImm>()->value;
    };
    PostOrderVisit(location.offset, check_bounds);
    PostOrderVisit(bytes, check_bounds);
    FreshenMutableReads freshen(FreshenMutableReads::Mode::kSnapshot);
    location.offset = freshen(location.offset);
    bytes = freshen(bytes);
    accesses.push_back({location, bytes, mask, atomic_, asynchronous_,
                        std::move(constraints)});
  }

  void Record(const Expr &storage, const ffi::Array<PrimExpr> &indices,
              int mask, ffi::Optional<PrimExpr> extent = std::nullopt) {
    auto tensor = storage.as<tirx::TensorVar>();
    if (!tensor)
      return;
    ConstrSet constraints = Snapshot();
    ffi::Array<PrimExpr> coordinates;
    ffi::Optional<PrimVar> lane;
    ffi::Optional<PrimExpr> lanes;
    bool precise = true;
    for (const PrimExpr &index : indices) {
      if (const auto *ramp = index.as<RampNode>()) {
        if (!lane) {
          lane = PrimVar("memory_lane", ramp->base.ty());
          lanes = ramp->lanes;
          constraints.AddConstr(*lane, Range(0, ramp->lanes));
        } else if (!ffi::StructuralEqual()(*lanes, ramp->lanes)) {
          precise = false;
        }
        coordinates.push_back(ramp->base + *lane * ramp->stride);
      } else if (const auto *broadcast = index.as<BroadcastNode>()) {
        coordinates.push_back(broadcast->value);
      } else if (index.ty().IsScalar()) {
        coordinates.push_back(index);
      } else {
        precise = false;
      }
    }
    MemoryLocation location = aliases.Element(storage, coordinates);
    location.precise &= precise;
    PrimType type = tensor.value().dtype();
    int bits = IsPacked4BitStorage(type) ? type.bits() * type.lanes()
                                         : type.StorageBytes() * 8;
    PrimExpr count = extent.value_or(PrimExpr(1));
    PrimExpr bytes =
        FloorDiv(Cast(PrimType::Int(64), count) * IntImm::Int64(bits) + 7,
                 IntImm::Int64(8));
    Append(location, bytes, mask, std::move(constraints));
  }

  void RecordRegion(const TensorRegion &region, int mask) {
    size_t saved = constr_stack_.size();
    ffi::Array<PrimExpr> indices;
    for (const Range &range : region->region) {
      Visit(range->begin);
      Visit(range->extent);
      PrimVar coordinate("memory_coordinate", range->begin.ty());
      constr_stack_.emplace_back(coordinate, range);
      indices.push_back(coordinate);
    }
    Record(region->source, indices, mask);
    constr_stack_.resize(saved);
  }

  bool RecordAnnotated(const CallNode *node) {
    if (!node->attrs)
      return false;
    auto attributes = node->attrs.value().as<DictAttrs>();
    if (!attributes)
      return false;
    auto region = attributes.value().GetAttr<TensorRegion>(attr::kAccessRegion);
    if (!region)
      return false;
    auto mask = attributes.value().GetAttr<PrimExpr>(attr::kAccessMask);
    int access = mask ? GetConservativeAccessMask(*mask) : kAccessReadWrite;
    auto extent = attributes.value().GetAttr<PrimExpr>(attr::kAccessExtent);
    if (extent) {
      Record(region.value()->source,
             region.value()->region.Map(
                 [](const Range &range) { return range->begin; }),
             access, extent);
      for (const Range &range : region.value()->region) {
        Visit(range->begin);
        Visit(range->extent);
      }
      Visit(*extent);
    } else {
      RecordRegion(*region, access);
    }
    return true;
  }

  Result Visit_(const BindNode *node) final {
    aliases.Bind(node);
    if (node->var->ty.as<tirx::TensorTypeNode>() ||
        node->var->ty.as<PtrTypeNode>()) {
      bool saved = pointer_only_;
      const auto *call = node->value.as<CallNode>();
      pointer_only_ = !call || IsStorageReference(call);
      Visit(node->value);
      pointer_only_ = saved;
      return std::nullopt;
    }
    return ConstrVisitor::Visit_(node);
  }

  Result Visit_(const TensorLoadNode *node) final {
    if (!pointer_only_)
      Record(node->source, node->indices, kAccessRead);
    bool saved = pointer_only_;
    pointer_only_ = false;
    for (const PrimExpr &index : node->indices)
      Visit(index);
    pointer_only_ = saved;
    return std::nullopt;
  }

  Result Visit_(const TensorStoreNode *node) final {
    Record(node->dest, node->indices, kAccessWrite);
    Visit(node->value);
    for (const PrimExpr &index : node->indices)
      Visit(index);
    return std::nullopt;
  }

  Result Visit_(const VarNode *node) final {
    if (!pointer_only_ &&
        (node->ty.as<tirx::TensorTypeNode>() || node->ty.as<PtrTypeNode>())) {
      MemoryLocation location = aliases.Resolve(ffi::GetRef<Var>(node));
      location.precise = false;
      Append(location, 0, kAccessReadWrite, Snapshot());
    }
    return std::nullopt;
  }

  Result Visit_(const CallNode *node) final {
    if (node->op.same_as(tirx::masked_load_op()) ||
        node->op.same_as(tirx::masked_store_op())) {
      bool store = node->op.same_as(tirx::masked_store_op());
      ffi::Array<PrimExpr> indices;
      for (size_t index = store ? 2 : 1; index + 1 < node->args.size(); ++index)
        indices.push_back(node->args[index].as_or_throw<PrimExpr>());
      Record(node->args[0], indices, store ? kAccessWrite : kAccessRead);
      if (store)
        Visit(node->args[1]);
      for (const PrimExpr &index : indices)
        Visit(index);
      Visit(node->args.back());
      return std::nullopt;
    }
    if (!pointer_only_ && RecordAnnotated(node))
      return std::nullopt;
    if (!pointer_only_) {
      auto operation = ParseOperator(ffi::GetRef<Call>(node));
      if (operation.defined()) {
        auto regions = operation->GetAccessRegions();
        for (const TensorRegion &region : regions.reads)
          RecordRegion(region, kAccessRead);
        for (const TensorRegion &region : regions.writes)
          RecordRegion(region, kAccessWrite);
        if (node->attrs) {
          auto attributes = node->attrs.value().as<DictAttrs>();
          if (attributes) {
            if (auto barrier = attributes.value()->dict.Get("barrier")) {
              bool saved = atomic_;
              atomic_ = true;
              if (auto load = barrier.value().as<TensorLoad>())
                Record(load.value()->source, load.value()->indices,
                       kAccessReadWrite);
              else if (auto region = barrier.value().as<TensorRegion>())
                RecordRegion(*region, kAccessReadWrite);
              atomic_ = saved;
            }
          }
        }
        return std::nullopt;
      }
    }
    if (node->op.same_as(access_ptr())) {
      const auto *load = node->args[0].as<TensorLoadNode>();
      if (!pointer_only_) {
        if (load)
          Record(
              load->source, load->indices,
              GetConservativeAccessMask(node->args[2].as_or_throw<PrimExpr>()),
              node->args[1].as_or_throw<PrimExpr>());
        else
          RecordRegion(NormalizeToBufferRegion(node->args[0]),
                       kAccessReadWrite);
      }
      bool saved = pointer_only_;
      pointer_only_ = false;
      if (load) {
        for (const PrimExpr &index : load->indices)
          Visit(index);
      } else {
        for (const Range &range :
             NormalizeToBufferRegion(node->args[0])->region) {
          Visit(range->begin);
          Visit(range->extent);
        }
      }
      Visit(node->args[1]);
      Visit(node->args[2]);
      pointer_only_ = saved;
      return std::nullopt;
    }
    if (auto operation = node->op.as<Op>()) {
      const auto &name = operation.value()->name;
      if (name == "tl.tma_load" || name == "tl.tma_load_im2col" ||
          name == "tl.tma_load_multicast" || name == "tl.tma_load_gather4") {
        bool saved_atomic = atomic_;
        bool saved_asynchronous = asynchronous_;
        asynchronous_ = true;
        for (size_t index = 0; index < node->args.size(); ++index) {
          atomic_ = saved_atomic || index == 1;
          Visit(node->args[index]);
        }
        atomic_ = saved_atomic;
        asynchronous_ = saved_asynchronous;
        return std::nullopt;
      }
    }
    bool pointer = IsStorageReference(node);
    if (pointer && !pointer_only_) {
      MemoryLocation location = aliases.Resolve(ffi::GetRef<Call>(node));
      location.precise = false;
      Append(location, 0, kAccessReadWrite, Snapshot());
    }
    bool saved_pointer = pointer_only_;
    bool saved_atomic = atomic_;
    bool saved_asynchronous = asynchronous_;
    if (auto operation = node->op.as<Op>()) {
      const auto &name = operation.value()->name;
      atomic_ |= name.find("atomic") != std::string::npos ||
                 name.find("mbarrier") != std::string::npos;
      asynchronous_ |=
          name.find("tl.tma_") == 0 || name.find("tirx.ptx.cp_async") == 0;
    }
    pointer_only_ |= pointer;
    auto result = ConstrVisitor::Visit_(node);
    atomic_ = saved_atomic;
    asynchronous_ = saved_asynchronous;
    pointer_only_ = saved_pointer;
    return result;
  }

  Result Visit_(const BreakNode *) final {
    exits = true;
    return std::nullopt;
  }
  Result Visit_(const ContinueNode *) final {
    exits = true;
    return std::nullopt;
  }
  Result Visit_(const ReturnNode *) final {
    exits = true;
    return std::nullopt;
  }

  bool pointer_only_ = false;
  bool atomic_ = false;
  bool asynchronous_ = false;
};

} // namespace tvm::tl

#endif

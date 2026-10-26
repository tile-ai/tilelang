#include "compat/analyzer.h"
#include "compat/metadata.h"
#include "op/prim.h"
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/type.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <list>
#include <map>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#include "transform/common/pointer_value_type.h"
#include <tvm/runtime/device_api.h>
#include <tvm/sym/int_set.h>

#include "transform/tirx_passes.h"

namespace tvm::tl::tirx_passes {
using namespace tvm::tirx;
using tvm::tirx::transform::CreateFunctionPass;
using tvm::transform::Pass;
using tvm::transform::PassContext;
namespace {

using namespace tvm::prim;

using runtime::MemoryAllocProperty;
using runtime::MemoryScope;

namespace {

MemoryAllocProperty GetMemoryAllocProperty(const MemoryScope &scope) {
  static ffi::reflection::TypeAttrColumn state_column(
      ffi::type_attr::kEnumState);
  auto state = state_column[runtime::MemoryScopeObj::RuntimeTypeIndex()]
                   .cast<ffi::EnumState>();
  if (auto column =
          state->attrs.Get(runtime::attr::kMemoryAllocProperty.name)) {
    if (auto value = column->Get(scope)) {
      int property = value->cast<int>();
      TVM_FFI_CHECK(
          property >= static_cast<int>(MemoryAllocProperty::kNormal) &&
              property <=
                  static_cast<int>(MemoryAllocProperty::kSeparateAllocations),
          ValueError)
          << "Invalid memory allocation property for " << scope.name();
      return static_cast<MemoryAllocProperty>(property);
    }
  }
  return MemoryAllocProperty::kNormal;
}

ffi::Optional<Var> GetBufferDataVar(const ffi::Any &data) {
  if (auto var = data.as<Var>()) {
    return var;
  }
  if (const auto *call = data.as<CallNode>();
      call && call->op.same_as(tirx::tensor_data_ptr_op()) &&
      call->args.size() == 1) {
    return call->args[0].as<Var>();
  }
  return std::nullopt;
}

struct PrimTypeHash {
  size_t operator()(const PrimType &ty) const {
    DLDataType dtype = ty->dtype;
    return (static_cast<size_t>(dtype.code) << 24) ^
           (static_cast<size_t>(dtype.bits) << 16) ^
           static_cast<size_t>(dtype.lanes);
  }
};

struct PrimTypeEqual {
  bool operator()(const PrimType &lhs, const PrimType &rhs) const {
    return lhs == rhs;
  }
};

} // namespace

class LinearAccessPatternFinder final : public tvm::tl::StmtExprVisitor {
public:
  explicit LinearAccessPatternFinder(const ffi::Array<Var> &params) {
    for (const Var &param : params) {
      if (auto tensor = param.as<TensorVar>()) {
        tensor_aliases_.Set(tensor.value(), tensor.value());
      }
    }
  }

  struct StmtEntry {

    const ffi::Object *stmt;

    int64_t scope_pair_offset{0};

    std::vector<const VarNode *> touched;
  };

  struct AllocEntry {

    size_t num_physical_dimensions{0};

    size_t level{0};

    const BindNode *alloc{nullptr};
  };

  ffi::Optional<VisitInterrupt> DispatchAllocTensor(const BindNode *op,
                                                    const CallNode *call) {
    tvm::Tuple shape =
        tvm::Tuple(call->ty_args[0].as_or_throw<TensorType>()->shape);
    size_t level = scope_.size();
    const VarNode *buf = op->var.get();
    tensor_aliases_.Set(op->var, op->var);

    AllocEntry entry;
    entry.alloc = op;
    entry.level = level;
    entry.num_physical_dimensions = shape->fields.size();
    alloc_info_[buf] = entry;

    return VisitBindingValue(op);
  }

  ffi::Optional<VisitInterrupt>
  DispatchDeclTensor(const BindNode *op, const CallNode *tensor_call) {
    RegisterTensorAlias(op->var.as_or_throw<TensorVar>(), tensor_call->args[0]);
    return VisitBindingValue(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreNode *op) final {
    scope_.push_back(StmtEntry());

    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(op->value));
    for (const auto &index : op->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
    }
    RecordAccess(op->dest.as_or_throw<TensorVar>());
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (e.touched.size() != 0) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadNode *op) final {
    for (const auto &index : op->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
    }
    RecordAccess(op->source.as_or_throw<tvm::tirx::TensorVar>());
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const EvaluateNode *op) final {
    scope_.push_back(StmtEntry());

    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(tvm::tl::StmtExprVisitor::Visit_(op));
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (e.touched.size() != 0) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ReturnNode *op) final {
    scope_.push_back(StmtEntry());
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(tvm::tl::StmtExprVisitor::Visit_(op));
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (e.touched.size() != 0) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const VarNode *buf) final {
    if (def_region_kind() != kTVMFFIDefRegionKindNone)
      return tvm::tl::StmtExprVisitor::Visit_(buf);

    if (buf->ty.as<TensorTypeNode>()) {
      Var var = ffi::GetRef<Var>(buf);
      buf = tensor_aliases_.Get(var).value_or(var).get();
    }
    auto it = alloc_info_.find(buf);
    if (it != alloc_info_.end() && it->second.alloc) {
      TVM_FFI_ICHECK_LT(it->second.level, scope_.size())
          << " buf=" << buf->name;
      scope_[it->second.level].touched.push_back(buf);
    }
    return std::nullopt;
  }

  template <typename T>
  ffi::Optional<VisitInterrupt> VisitNewScope(const T *op) {
    scope_.push_back(StmtEntry());
    StmtEntry e;
    e.stmt = op;
    int64_t begin_index = static_cast<int64_t>(linear_seq_.size());

    linear_seq_.push_back(e);
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(tvm::tl::StmtExprVisitor::Visit_(op));

    e.touched = std::move(scope_.back().touched);
    scope_.pop_back();
    int64_t end_index = static_cast<int64_t>(linear_seq_.size());
    TVM_FFI_ICHECK_GT(end_index, begin_index);
    e.scope_pair_offset = begin_index - end_index;
    linear_seq_.push_back(e);

    TVM_FFI_ICHECK_NE(end_index, 0U);
    linear_seq_[begin_index].scope_pair_offset = end_index - begin_index;
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const RegionStmtNode *op) final {

    if (op->op.same_as(tirx::launch_thread_op()) &&
        std::string(op->args[0].as_or_throw<StringImm>()->value)
                .rfind("vthread", 0) != 0) {
      if (in_thread_env_)
        return tvm::tl::StmtExprVisitor::Visit_(op);
      in_thread_env_ = true;
      auto result = VisitNewScope(op);
      in_thread_env_ = false;
      return result;
    }
    return VisitNewScope(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const IfNode *op) final {
    return VisitNewScope(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForNode *op) final {
    return VisitNewScope(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const WhileNode *op) final {
    return VisitNewScope(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const AssertStmtNode *op) final {
    return VisitNewScope(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const BindNode *op) final {
    if (const auto *call = op->value.as<CallNode>(); call) {
      if (call->op.same_as(tirx::alloc_tensor_op()))
        return DispatchAllocTensor(op, call);
      if (call->op.same_as(tirx::decl_tensor_op()))
        return DispatchDeclTensor(op, call);
    }
    return VisitBindingValue(op);
  }

  ffi::Optional<VisitInterrupt> VisitBindingValue(const BindNode *op) {
    scope_.push_back(StmtEntry());

    if (const auto *call = op->value.as<CallNode>();
        call && (call->op.same_as(tirx::alloc_tensor_op()) ||
                 call->op.same_as(tirx::decl_tensor_op()))) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(call->attrs));
    }

    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(tvm::tl::StmtExprVisitor::Visit_(op));
    StmtEntry e = scope_.back();
    scope_.pop_back();
    if (e.touched.size() != 0) {
      e.stmt = op;
      linear_seq_.push_back(e);
    }
    return std::nullopt;
  }

  void RecordAccess(const TensorVar &tensor) {
    const VarNode *tensor_var =
        tensor_aliases_.Get(tensor).value_or(tensor).get();
    auto it = alloc_info_.find(tensor_var);
    if (it == alloc_info_.end() || !it->second.alloc)
      return;
    TVM_FFI_ICHECK_LT(it->second.level, scope_.size())
        << "Tensor access occurs outside a statement scope.";
    scope_[it->second.level].touched.push_back(tensor_var);
    TVM_FFI_ICHECK_EQ(1, it->second.num_physical_dimensions)
        << "TensorVar " << tensor->name << " is allocated with "
        << it->second.num_physical_dimensions
        << " physical dimensions, but is accessed as having "
        << "1 physical dimension" << std::endl;
  }

  std::vector<StmtEntry> linear_seq_;

  std::unordered_map<const VarNode *, AllocEntry> alloc_info_;

  ffi::Map<Var, Var> tensor_aliases_;

private:
  void RegisterTensorAlias(TensorVar tensor, const Expr &data) {
    Var root = tensor;
    if (auto source = GetBufferDataVar(data);
        source.has_value() && source.value()->ty.as<TensorTypeNode>()) {
      auto source_root = tensor_aliases_.Get(source.value());
      TVM_FFI_ICHECK(source_root.has_value())
          << "Tensor alias source " << source.value()->name
          << " must be registered before its DeclTensor alias";
      root = source_root.value();
    }
    tensor_aliases_.Set(tensor, root);
  }

  bool in_thread_env_{false};

  std::vector<StmtEntry> scope_;
};

class InplaceOpVerifier : public tvm::tl::StmtExprVisitor {
public:
  bool Check(const ffi::Object *stmt, const VarNode *dst, const VarNode *src) {
    dst_ = dst;
    src_ = src;
    result_ = true;
    if (stmt->IsInstance<ForNode>()) {
      Visit_(static_cast<const ForNode *>(stmt));
    } else if (stmt->IsInstance<IfNode>()) {
      Visit_(static_cast<const IfNode *>(stmt));
    } else if (stmt->IsInstance<WhileNode>()) {
      Visit_(static_cast<const WhileNode *>(stmt));
    } else if (stmt->IsInstance<TensorStoreNode>()) {
      Visit_(static_cast<const TensorStoreNode *>(stmt));
    } else {
      return false;
    }
    return result_;
  }

  using tvm::tl::StmtExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView n) final {
    if (!result_)
      return std::nullopt;
    return tvm::tl::StmtExprVisitor::Visit(n);
  }

  ffi::Optional<VisitInterrupt> Visit_(const VarNode *op) final {
    if (def_region_kind() != kTVMFFIDefRegionKindNone)
      return tvm::tl::StmtExprVisitor::Visit_(op);

    if (op == dst_ || op == src_) {
      result_ = false;
      return std::nullopt;
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorStoreNode *op) final {
    ++mem_nest_;
    for (const auto &index : op->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(index));
    }
    --mem_nest_;
    if (op->dest.as_or_throw<TensorVar>().get() == dst_) {
      store_ = op;
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->value));
      store_ = nullptr;
    } else {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->value));
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const BindNode *op) final {
    if (const auto *call = op->value.as<CallNode>();
        call && call->op.same_as(tirx::alloc_tensor_op()))
      return DispatchAllocTensor(op, call);
    return tvm::tl::StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt>
  DispatchAllocTensor(const BindNode *op, const CallNode *tensor_call) {

    if ((tensor_call->attrs.has_value()
             ? tensor_call->attrs.value().as<DictAttrsNode>()
             : nullptr)
            ->dict.count(tvm::tirx::attr::kVolatile.name)) {
      result_ = false;
      return std::nullopt;
    }
    return tvm::tl::StmtExprVisitor::Visit_(op);
  }

  ffi::Optional<VisitInterrupt> Visit_(const TensorLoadNode *op) final {
    const VarNode *buf = op->source.as_or_throw<tvm::tirx::TensorVar>().get();

    if (buf == dst_) {
      result_ = false;
      return std::nullopt;
    }

    if (mem_nest_ != 0) {
      result_ = false;
      return std::nullopt;
    }
    if (src_ == buf) {
      if (store_ == nullptr ||
          store_->value.ty() != op->ty.as_or_throw<PrimType>()) {
        result_ = false;
        return std::nullopt;
      }
      TVM_FFI_ICHECK_EQ(store_->indices.size(), op->indices.size())
          << "Store/Load occur to the same buffer " << buf->name
          << " with differing number of indices";
      for (size_t i = 0; i < store_->indices.size(); i++) {
        if (!prim::ExprDeepEqual()(store_->indices[i], op->indices[i])) {
          result_ = false;
          return std::nullopt;
        }
      }
    }
    ++mem_nest_;
    for (const auto &index : op->indices) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(Visit(index));
    }
    --mem_nest_;
    return std::nullopt;
  }

private:
  bool result_{true};

  const VarNode *dst_;

  const VarNode *src_;

  int mem_nest_{0};

  const TensorStoreNode *store_{nullptr};
};

class StoragePlanRewriter : public tvm::tl::StmtExprMutator {
public:
  using tvm::tl::StmtExprMutator::Mutate;
  using tvm::tl::StmtExprMutator::Mutate_;
  using StmtEntry = LinearAccessPatternFinder::StmtEntry;
  using AllocEntry = LinearAccessPatternFinder::AllocEntry;

  Stmt Rewrite(Stmt stmt, const ffi::Array<Var> &params, bool detect_inplace,
               bool enable_reuse, bool reuse_require_exact_matched_dtype,
               bool preserve_shared_allocations, bool allow_local_inplace) {
    detect_inplace_ = detect_inplace;
    preserve_shared_allocations_ = preserve_shared_allocations;
    allow_local_inplace_ = allow_local_inplace && detect_inplace;

    auto finder = ffi::make_object<LinearAccessPatternFinder>(params);
    finder->Visit(stmt);
    this->LivenessAnalysis(finder->linear_seq_);
    this->PlanMemory(finder->linear_seq_, finder->alloc_info_, enable_reuse,
                     reuse_require_exact_matched_dtype);
    tensor_aliases_ = std::move(finder->tensor_aliases_);
    this->PrepareNewAlloc();
    for (const auto &[var, entry] : alloc_map_) {
      RemapTensor(ffi::GetRef<Var>(var).as_or_throw<TensorVar>(),
                  entry->alloc_var);
    }
    for (const auto &[var, root] : tensor_aliases_) {
      if (auto it = alloc_map_.find(root.get()); it != alloc_map_.end()) {
        RemapTensor(var.as_or_throw<TensorVar>(), it->second->alloc_var);
      }
    }

    stmt = Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt);
    if (attach_map_.count(nullptr)) {
      return MakeAttach(attach_map_.at(nullptr), stmt);
    }
    return stmt;
  }

  template <typename Node> Node VisitTensorAccess(Node node) {
    const VarNode *root =
        tensor_aliases_.Get(node->dest.template as_or_throw<TensorVar>())
            .value_or(node->dest.template as_or_throw<TensorVar>())
            .get();
    auto it = alloc_map_.find(root);
    if (it != alloc_map_.end()) {
      TensorVar buf = RemapTensor(node->dest.template as_or_throw<TensorVar>(),
                                  it->second->alloc_var);

      ffi::Array<PrimExpr> indices = node->indices;
      indices.Set(
          indices.size() - 1,
          RemapIndex(node->dest.template as_or_throw<TensorVar>().dtype(),
                     indices[indices.size() - 1], it->second));

      auto writer = node.CopyOnWrite();
      writer->dest = buf;
      writer->indices = indices;
    }
    return node;
  }

  TensorLoad VisitTensorAccess(TensorLoad node) {
    TensorVar tensor = node->source.as_or_throw<TensorVar>();
    const VarNode *root = tensor_aliases_.Get(tensor).value_or(tensor).get();
    auto it = alloc_map_.find(root);
    if (it == alloc_map_.end()) {
      return node;
    }

    TensorVar remapped = RemapTensor(tensor, it->second->alloc_var);
    ffi::Array<PrimExpr> indices = node->indices;
    indices.Set(indices.size() - 1,
                RemapIndex(tensor.dtype(), indices.back(), it->second));
    return TensorLoad(remapped, indices, node->loc);
  }

  TensorVar RemapTensor(TensorVar buf, Var new_backing_array) {
    auto mapped = VarRemapGet(buf);
    if (mapped != nullptr) {
      TensorVar remapped = mapped.as_or_throw<TensorVar>();
      TVM_FFI_ICHECK_EQ(remapped_backing_.at(remapped.get()).get(),
                        new_backing_array.get())
          << "Cannot remap tensor " << buf->name << " to use backing array "
          << new_backing_array->name << ", previously used backing array "
          << remapped_backing_.at(remapped.get())->name;
      return remapped;
    }

    TensorVar backing = new_backing_array.as_or_throw<TensorVar>();
    TensorVar remapped =
        buf.same_as(backing)
            ? buf
            : TensorVar(
                  new_backing_array->name,
                  TensorType(ffi::make_object<TensorTypeNode>(*buf.ty().get())),
                  buf->loc);
    VarRemapSet(buf, remapped);
    remapped_backing_.insert_or_assign(remapped.get(), backing);
    return remapped;
  }

  UnchangedOr<Stmt> Mutate_(const TensorStoreNode *op,
                            InplaceMode inplace_mode) final {

    auto value = Mutate(op->value, inplace_mode);
    auto indices = Mutate(op->indices, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorStore node =
        value.UnchangedOrSameAs(op->value) &&
                indices.UnchangedOrSameAs(op->indices)
            ? ffi::GetRef<TensorStore>(op)
            : TensorStore(op->dest.as_or_throw<TensorVar>(),
                          std::move(indices).ValueOrUnchanged(op->indices),
                          std::move(value).ValueOrUnchanged(op->value),
                          op->loc);
    return VisitTensorAccess(std::move(node));
  }

  UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode *op,
                                InplaceMode inplace_mode) final {
    auto indices = Mutate(op->indices, inplace_mode)
                       .as_or_throw<UnchangedOr<ffi::Array<PrimExpr>>>();
    TensorLoad node =
        indices.UnchangedOrSameAs(op->indices)
            ? ffi::GetRef<TensorLoad>(op)
            : TensorLoad(op->source.as_or_throw<TensorVar>(),
                         std::move(indices).ValueUnchecked(), op->loc);
    return VisitTensorAccess(std::move(node));
  }

  UnchangedOr<Expr> Mutate_(const TensorRegionNode *op,
                            InplaceMode inplace_mode) final {
    auto tensor = op->source.as<TensorVar>();
    if (!tensor)
      return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);
    auto ranges = Mutate(op->region, inplace_mode)
                      .as_or_throw<UnchangedOr<ffi::Array<Range>>>()
                      .ValueOrUnchanged(op->region);
    const VarNode *root = tensor_aliases_.Get(*tensor).value_or(*tensor).get();
    auto entry = alloc_map_.find(root);
    if (entry == alloc_map_.end()) {
      return TensorRegion(*tensor, ranges, op->ty, op->loc);
    }
    TVM_FFI_CHECK(!ranges.empty(), ValueError)
        << "Storage regions must have a memory axis";
    TensorVar remapped = RemapTensor(*tensor, entry->second->alloc_var);
    const Range &last = ranges.back();
    PrimExpr begin =
        RemapIndex(tensor.value().dtype(), last->begin, entry->second);
    ranges.Set(ranges.size() - 1, Range(begin, begin + last->extent));
    return TensorRegion(remapped, ranges, op->ty, op->loc);
  }

  UnchangedOr<Expr> Mutate_(const VarNode *op, InplaceMode inplace_mode) final {
    if (def_region_kind() == kTVMFFIDefRegionKindNone) {
      const VarNode *root = op;
      if (op->ty.as<TensorTypeNode>()) {
        Var var = ffi::GetRef<Var>(op);
        root = tensor_aliases_.Get(var).value_or(var).get();
      }
      auto it = alloc_map_.find(root);
      if (it != alloc_map_.end() && it->second->bits_offset != 0) {
        LOG(WARNING)
            << "Use a merged buffer variable address, could cause error";
      }
    }
    return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);
  }
  UnchangedOr<Expr> Mutate_(const CallNode *op,
                            InplaceMode inplace_mode) final {
    if (op->op.same_as(tirx::alloc_tensor_op()) ||
        op->op.same_as(tirx::decl_tensor_op())) {

      auto attrs = Mutate(op->attrs, InplaceMode::kDisallow)
                       .as_or_throw<UnchangedOr<ffi::Optional<Attrs>>>();
      auto call = tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode)
                      .ValueOrUnchanged(ffi::GetRef<Expr>(op))
                      .as_or_throw<Call>();
      if (!attrs.UnchangedOrSameAs(op->attrs)) {
        call.CopyOnWrite()->attrs = std::move(attrs).ValueUnchecked();
      }
      return call;
    }
    if (op->op.same_as(tirx::tensor_data_ptr_op()) && op->args.size() == 1) {
      if (auto var = op->args[0].as<Var>()) {
        Var root = tensor_aliases_.Get(var.value()).value_or(var.value());
        if (auto it = alloc_map_.find(root.get()); it != alloc_map_.end()) {
          const StorageEntry *entry = it->second;
          TVM_FFI_ICHECK_EQ(entry->bits_offset % 8, 0U);
          Expr data = entry->alloc_var.as_or_throw<TensorVar>().data();
          if (entry->bits_offset != 0 ||
              !tl::StructuralEqual()(data->ty, op->ty)) {
            return Call(
                op->ty, tirx::ptr_byte_offset_op(),
                {data, IntImm(PrimType::Int(64),
                              static_cast<int64_t>(entry->bits_offset / 8))},
                {}, {}, op->loc);
          }
          return data;
        }
      }
    }
    if (op->op.same_as(tirx::masked_load_op()) ||
        op->op.same_as(tirx::masked_store_op())) {
      bool is_load = op->op.same_as(tirx::masked_load_op());
      TensorVar tensor = op->args[0].as_or_throw<TensorVar>();
      ffi::Optional<PrimExpr> value;
      if (!is_load)
        value = this->Mutate(op->args[1])
                    .ValueOrUnchanged(op->args[1])
                    .as_or_throw<PrimExpr>();
      ffi::Array<PrimExpr> indices;
      for (size_t i = is_load ? 1 : 2; i + 1 < op->args.size(); ++i) {
        indices.push_back(this->Mutate(op->args[i])
                              .ValueOrUnchanged(op->args[i])
                              .as_or_throw<PrimExpr>());
      }
      if (is_load) {
        TensorLoad access = TensorLoad(tensor, indices, op->loc);
        access = VisitTensorAccess(std::move(access));
        ffi::Array<Expr> args{access->source.as_or_throw<TensorVar>()};
        for (const PrimExpr &index : access->indices)
          args.push_back(index);
        args.push_back(this->Mutate(op->args[op->args.size() - 1])
                           .ValueOrUnchanged(op->args[op->args.size() - 1])
                           .as_or_throw<Expr>());
        return Call(access->ty, op->op, args, op->attrs, op->ty_args, op->loc);
      } else {
        TensorStore access(tensor, indices, value.value(), op->loc);
        access = VisitTensorAccess(std::move(access));
        ffi::Array<Expr> args{access->dest.as_or_throw<TensorVar>(),
                              access->value};
        for (const PrimExpr &index : access->indices)
          args.push_back(index);
        args.push_back(this->Mutate(op->args[op->args.size() - 1])
                           .ValueOrUnchanged(op->args[op->args.size() - 1])
                           .as_or_throw<Expr>());
        return Call(PrimType::Void(), op->op, args, op->attrs, op->ty_args,
                    op->loc);
      }
    } else {
      return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);
    }
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtNode *op,
                            InplaceMode inplace_mode) final {
    auto it = attach_map_.find(op);
    RegionStmt region = tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode)
                            .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                            .as_or_throw<RegionStmt>();
    if (it != attach_map_.end()) {
      region.CopyOnWrite()->body = MakeAttach(it->second, region->body);
    }
    return region;
  }

  UnchangedOr<Stmt> Mutate_(const ForNode *op, InplaceMode inplace_mode) final {
    TVM_FFI_ICHECK(op->kind != ForKind::kVectorized)
        << "VectorizeLoop before LiftStorageAlloc";

    if (attach_map_.count(op)) {
      auto &svec = attach_map_[op];
      Stmt stmt = tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode)
                      .ValueOrUnchanged(ffi::GetRef<Stmt>(op));
      op = stmt.as<ForNode>();
      return For(op->var, op->begin, op->end, op->step, op->kind,
                 MakeAttach(svec, op->body), op->annotations);
    } else {
      return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);
    }
  }

  UnchangedOr<Stmt> Mutate_(const BindNode *op,
                            InplaceMode inplace_mode) final {
    if (const auto *call = op->value.as<CallNode>(); call) {
      if (call->op.same_as(tirx::alloc_tensor_op()))
        return MutateAllocTensor(op, call, inplace_mode);
      if (call->op.same_as(tirx::decl_tensor_op()))
        return MutateDeclTensor(op, call, inplace_mode);
    }
    return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);
  }

  UnchangedOr<Stmt> MutateAllocTensor(const BindNode *op,
                                      const CallNode *tensor_call,
                                      InplaceMode inplace_mode) {

    if (RequiresExactAllocation(tensor_call)) {
      return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);
    }

    if (auto it = alloc_map_.find(op->var.get()); it != alloc_map_.end()) {
      if (it->second->alloc_var.get() == op->var.get()) {

        return Evaluate(0);
      }

      TensorVar buf =
          RemapTensor(op->var.as_or_throw<TensorVar>(), it->second->alloc_var);
      return Bind(buf,
                  Call(buf.ty(), tirx::decl_tensor_op(),
                       {it->second->alloc_var.as_or_throw<TensorVar>().data()},
                       {}, {buf.ty()}, tensor_call->loc),
                  op->loc);
    }

    return Evaluate(0);
  }

  UnchangedOr<Stmt> MutateDeclTensor(const BindNode *op,
                                     const CallNode *tensor_call,
                                     InplaceMode inplace_mode) {
    const VarNode *root = tensor_aliases_.Get(op->var).value_or(op->var).get();
    auto it = alloc_map_.find(root);
    if (it != alloc_map_.end()) {
      TensorVar tensor =
          RemapTensor(op->var.as_or_throw<TensorVar>(), it->second->alloc_var);
      auto attrs = Mutate(tensor_call->attrs, InplaceMode::kDisallow)
                       .as_or_throw<UnchangedOr<ffi::Optional<Attrs>>>()
                       .ValueOrUnchanged(tensor_call->attrs);
      return Bind(tensor,
                  Call(tensor.ty(), tirx::decl_tensor_op(),
                       {it->second->alloc_var.as_or_throw<TensorVar>().data()},
                       attrs, {tensor.ty()}, tensor_call->loc),
                  op->loc);
    }
    return tvm::tl::StmtExprMutator::Mutate_(op, inplace_mode);
  }

private:
  struct StorageEntry {

    const ffi::Object *attach_scope_{nullptr};

    uint64_t const_nbits{0};

    MemoryScope scope;

    size_t ndim;

    std::vector<const BindNode *> allocs;

    std::vector<StorageEntry *> merged_children;

    std::vector<Stmt> alloc_nest;

    Var alloc_var{ffi::UnsafeInit{}};

    PrimType elem_type = PrimType::Void();

    bool is_volatile{false};

    bool requires_exact_allocation{false};

    uint64_t bits_offset{0};
    uint64_t packing_alignment_bits{32};
  };

  bool RequiresExactAllocation(const CallNode *call) const {
    auto type = call->ty_args[0].as_or_throw<TensorType>();
    if (preserve_shared_allocations_ && (type->scope.name() == "shared" ||
                                         type->scope.name() == "shared.dyn")) {
      return true;
    }
    if (GetMemoryAllocProperty(type->scope) ==
            MemoryAllocProperty::kSeparateAllocations &&
        !(allow_local_inplace_ && type->scope.name() == "local")) {
      return true;
    }

    if (type->shape.size() != 1 || type->layout.has_value()) {
      return true;
    }
    if (!call->args.empty())
      return true;
    if (!call->attrs.has_value())
      return false;
    const auto &annotations =
        call->attrs.value().as_or_throw<DictAttrs>()->dict;

    for (const auto &[key, value] : annotations) {
      if (key != tvm::tirx::attr::kVolatile.name)
        return true;
    }
    return false;
  }

  static bool HasCompatibleAlignment(const StorageEntry *entry,
                                     const CallNode *call) {
    auto first = entry->allocs[0]
                     ->value.as<CallNode>()
                     ->ty_args[0]
                     .as_or_throw<TensorType>();
    auto type = call->ty_args[0].as_or_throw<TensorType>();
    return first->data_alignment == type->data_alignment;
  }

  static bool CanPackAllocation(const StorageEntry *entry) {
    if (entry->requires_exact_allocation || entry->const_nbits == 0 ||
        entry->const_nbits >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return false;
    for (const auto *allocation : entry->allocs) {
      const auto *call = allocation->value.as<CallNode>();
      auto type = call->ty_args[0].as_or_throw<TensorType>();
      if (type->shape.size() != 1 || type->dtype.IsScalableVector() ||
          type->layout.has_value())
        return false;
    }
    return true;
  }

  static ffi::Optional<uint64_t>
  PackingAlignment(const std::vector<StorageEntry *> &entries,
                   const MemoryScope &scope) {
    uint64_t alignment = 32;
    auto include_alignment = [&](uint64_t value) {
      uint64_t factor = value / std::gcd(alignment, value);

      constexpr uint64_t limit =
          static_cast<uint64_t>(std::numeric_limits<int>::max()) * 8;
      if (factor > limit / alignment)
        return false;
      alignment *= factor;
      return true;
    };
    for (const auto *entry : entries) {
      if (entry->scope != scope)
        continue;
      if (!CanPackAllocation(entry))
        return std::nullopt;
      for (const auto *allocation : entry->allocs) {
        auto type = allocation->value.as<CallNode>()
                        ->ty_args[0]
                        .as_or_throw<TensorType>();
        uint64_t element_bits = type->dtype.bits() * type->dtype.lanes();
        if (!include_alignment(element_bits) ||
            !include_alignment(
                static_cast<uint64_t>(std::max(type->data_alignment, 1)) * 8)) {
          return std::nullopt;
        }
      }
    }
    uint64_t total_bits = 0;
    constexpr uint64_t limit = std::numeric_limits<int64_t>::max();
    for (const auto *entry : entries) {
      if (entry->scope != scope)
        continue;
      uint64_t padding =
          (alignment - entry->const_nbits % alignment) % alignment;
      if (entry->const_nbits > limit - padding ||
          total_bits > limit - entry->const_nbits - padding)
        return std::nullopt;
      total_bits += entry->const_nbits + padding;
    }
    return alignment;
  }

  struct EventEntry {

    std::vector<const VarNode *> gen;

    std::vector<const VarNode *> kill;
  };

  Stmt MakeAttach(const std::vector<StorageEntry *> &svec, Stmt body) {
    for (auto it = svec.rbegin(); it != svec.rend(); it++) {
      if (!(*it)->alloc_nest.empty()) {
        ffi::Array<Stmt> prefix((*it)->alloc_nest);
        prefix.push_back(body);
        body = SeqStmt(prefix);
      }
    }
    return body;
  }

  PrimExpr RemapIndex(PrimType dtype, PrimExpr index, StorageEntry *e) {
    if (e->bits_offset == 0)
      return index;
    uint64_t elem_bits = dtype.bits();
    TVM_FFI_ICHECK_EQ(e->bits_offset % elem_bits, 0U);
    return tl::MakeConst(index.ty(), e->bits_offset / elem_bits) + index;
  }

  void PrepareNewAlloc() {
    for (size_t i = 0; i < alloc_vec_.size(); ++i) {
      StorageEntry *e = alloc_vec_[i].get();
      attach_map_[e->attach_scope_].push_back(e);
    }

    for (auto &kv : attach_map_) {

      std::vector<StorageEntry *> &vec = kv.second;
      for (size_t i = 0; i < vec.size(); ++i) {
        StorageEntry *e = vec[i];
        if (GetMemoryAllocProperty(e->scope) !=
            MemoryAllocProperty::kSingleAllocation)
          continue;
        auto alignment = PackingAlignment(vec, e->scope);
        if (!alignment)
          continue;
        e->packing_alignment_bits = *alignment;
        for (size_t j = 0; j < i; ++j) {
          if (e->scope == vec[j]->scope) {
            vec[j]->merged_children.push_back(e);
            break;
          }
        }
      }

      for (size_t i = 0; i < vec.size(); ++i) {
        StorageEntry *e = vec[i];
        if (e->requires_exact_allocation) {
          TVM_FFI_ICHECK_EQ(e->allocs.size(), 1U);
          TVM_FFI_ICHECK(e->merged_children.empty());
          e->alloc_var = e->allocs[0]->var;

          continue;
        }

        if (e->bits_offset != 0)
          continue;
        if (e->merged_children.size() != 0) {
          NewAllocMerged(e);
          continue;
        }
        const auto *first_call = e->allocs[0]->value.as<CallNode>();
        DLDataType first_dtype =
            first_call->ty_args[0].as_or_throw<TensorType>()->dtype->dtype;
        PrimType first_type(first_dtype);
        if (e->allocs.size() == 1 && first_type.IsScalableVector()) {

          e->alloc_var = e->allocs[0]->var;
          TensorVar buf = RemapTensor(
              e->allocs[0]->var.as_or_throw<TensorVar>(), e->alloc_var);
          ffi::Map<ffi::String, ffi::Any> annotations;
          if (e->is_volatile) {
            annotations.Set(tvm::tirx::attr::kVolatile.name, true);
          }
          e->alloc_nest.push_back(
              Bind(buf, Call(buf.ty(), tirx::alloc_tensor_op(), {},
                             DictAttrs(annotations), {buf.ty()})));
          continue;
        }

        e->alloc_var = e->allocs[0]->var;
        PrimType alloc_type = first_type;
        for (const BindNode *op : e->allocs) {
          const auto *call = op->value.as<CallNode>();
          DLDataType dtype =
              call->ty_args[0].as_or_throw<TensorType>()->dtype->dtype;
          PrimType element_type(dtype);
          if (element_type.lanes() > alloc_type.lanes()) {
            alloc_type = element_type;
          }
        }

        bool all_allocs_identical = std::all_of(
            e->allocs.begin() + 1, e->allocs.end(),
            [&](const BindNode *op) -> bool {
              const auto *call = op->value.as<CallNode>();
              DLDataType dtype =
                  call->ty_args[0].as_or_throw<TensorType>()->dtype->dtype;
              if (dtype != first_dtype) {
                return false;
              }
              tvm::Tuple shape =
                  tvm::Tuple(call->ty_args[0].as_or_throw<TensorType>()->shape);
              tvm::Tuple first_shape = tvm::Tuple(
                  first_call->ty_args[0].as_or_throw<TensorType>()->shape);
              if (shape->fields.size() != first_shape->fields.size()) {
                return false;
              }
              prim::ExprDeepEqual expr_equal;
              for (size_t i = 0; i < shape->fields.size(); i++) {
                if (!expr_equal(
                        shape->fields[i].as_or_throw<PrimExpr>(),
                        first_shape->fields[i].as_or_throw<PrimExpr>())) {
                  return false;
                }
              }
              return true;
            });

        if (all_allocs_identical) {

          TensorVar buf = RemapTensor(
              e->allocs[0]->var.as_or_throw<TensorVar>(), e->alloc_var);
          ffi::Map<ffi::String, ffi::Any> annotations;
          if (e->is_volatile) {
            annotations.Set(tvm::tirx::attr::kVolatile.name, true);
          }
          e->alloc_nest.push_back(
              Bind(buf, Call(buf.ty(), tirx::alloc_tensor_op(), {},
                             DictAttrs(annotations), {buf.ty()})));
        } else {

          ffi::Optional<PrimExpr> max_size;
          for (const BindNode *op : e->allocs) {
            const auto *call = op->value.as<CallNode>();
            tvm::Tuple shape =
                tvm::Tuple(call->ty_args[0].as_or_throw<TensorType>()->shape);
            DLDataType dtype =
                call->ty_args[0].as_or_throw<TensorType>()->dtype->dtype;
            PrimType element_type(dtype);
            TVM_FFI_ICHECK_EQ(shape->fields.size(), 1)
                << "TensorVar var " << op->var.as_or_throw<TensorVar>()->name
                << " was identified as a re-usable allocation, but has "
                << shape->fields.size() << " physical dimensions.  "
                << "Currently, only flat 1-d memory spaces should be "
                   "identified as re-usable "
                   "allocations.";
            PrimExpr sz = shape->fields[0].as_or_throw<PrimExpr>();
            auto nbits = element_type.bits() * element_type.lanes();
            if (const auto *imm = sz.as<IntImmNode>()) {
              if (imm->value > std::numeric_limits<int>::max() / nbits) {
                LOG(WARNING) << "The allocation requires : " << imm->value
                             << " * " << nbits
                             << " bits, which is greater than the maximum of"
                                " int32. The size is cast to int64."
                             << "\n";
                sz = IntImm::Int64(imm->value);
              }
            }

            auto sz_nbits = sz * nbits;
            if (max_size.has_value()) {
              max_size = max(max_size.value(), sz_nbits);
            } else {
              max_size = sz_nbits;
            }
          }
          PrimExpr combo_size = max_size.value();

          auto type_bits = alloc_type.bits() * alloc_type.lanes();
          bool divided = tl::arith::CanProve(
              analyzer_, indexmod(combo_size, type_bits) == 0);
          combo_size = indexdiv(combo_size, type_bits);

          if (!divided) {
            combo_size = combo_size + IntImm::Int32(1);
          }
          combo_size = tl::arith::Simplify(analyzer_, combo_size);
          TensorVar buf(e->alloc_var->name,
                        TensorType({combo_size}, alloc_type, e->scope,
                                   first_call->ty_args[0]
                                       .as_or_throw<TensorType>()
                                       ->data_alignment));
          e->alloc_var = buf;
          ffi::Map<ffi::String, ffi::Any> annotations;
          if (e->is_volatile) {
            annotations.Set(tvm::tirx::attr::kVolatile.name, true);
          }
          e->alloc_nest.push_back(
              Bind(buf, Call(buf.ty(), tirx::alloc_tensor_op(), {},
                             DictAttrs(annotations), {buf.ty()})));
        }
      }
    }
  }

  void NewAllocMerged(StorageEntry *e) {
    TVM_FFI_ICHECK(GetMemoryAllocProperty(e->scope) ==
                   MemoryAllocProperty::kSingleAllocation);

    TVM_FFI_ICHECK_NE(e->const_nbits, 0U);
    uint64_t total_bits = e->const_nbits;

    uint64_t align = e->packing_alignment_bits;
    if (total_bits % align != 0) {
      total_bits += align - (total_bits % align);
    }
    e->alloc_var = e->allocs[0]->var;
    for (StorageEntry *child : e->merged_children) {
      TVM_FFI_ICHECK_NE(child->const_nbits, 0U);
      TVM_FFI_ICHECK_NE(total_bits, 0U);
      child->bits_offset = total_bits;
      child->alloc_var = e->alloc_var;
      total_bits += child->const_nbits;
      if (total_bits % align != 0) {
        total_bits += align - (total_bits % align);
      }
    }
    uint64_t type_bits = e->elem_type.bits() * e->elem_type.lanes();
    const auto *call = e->allocs[0]->value.as<CallNode>();
    tvm::Tuple shape =
        tvm::Tuple(call->ty_args[0].as_or_throw<TensorType>()->shape);
    uint64_t alloc_elements = total_bits / type_bits;
    PrimType shape_type = shape->fields[0].as_or_throw<PrimExpr>().ty();
    if (shape_type.bits() < 64 &&
        alloc_elements > std::numeric_limits<int32_t>::max()) {
      shape_type = PrimType::Int(64);
    }
    PrimExpr alloc_size = tl::MakeConst(shape_type, alloc_elements);
    TensorVar buf(e->alloc_var->name,
                  TensorType({alloc_size}, e->elem_type, e->scope,
                             static_cast<int>(align / 8)));
    e->alloc_var = buf;
    for (StorageEntry *child : e->merged_children) {
      child->alloc_var = e->alloc_var;
    }
    bool any_volatile = e->is_volatile;
    for (StorageEntry *child : e->merged_children) {
      if (child->is_volatile)
        any_volatile = true;
    }
    ffi::Map<ffi::String, ffi::Any> annotations;
    if (any_volatile) {
      annotations.Set(tvm::tirx::attr::kVolatile.name, true);
    }
    e->alloc_nest.push_back(
        Bind(buf, Call(buf.ty(), tirx::alloc_tensor_op(), {},
                       DictAttrs(annotations), {buf.ty()})));
  }

  void LivenessAnalysis(const std::vector<StmtEntry> &seq) {

    std::unordered_set<const VarNode *> touched;
    for (size_t i = seq.size(); i != 0; --i) {
      const StmtEntry &s = seq[i - 1];
      for (const VarNode *buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          event_map_[s.stmt].kill.push_back(buffer);
        }
      }
    }

    touched.clear();
    for (size_t i = 0; i < seq.size(); ++i) {
      int64_t offset = seq[i].scope_pair_offset;
      if (offset < 0)
        continue;
      const StmtEntry &s = seq[i + offset];
      for (const VarNode *buffer : s.touched) {
        if (!touched.count(buffer)) {
          touched.insert(buffer);
          event_map_[s.stmt].gen.push_back(buffer);
        }
      }
    }
  }
  void PlanNewScope(const ffi::Object *op) {
    if (thread_scope_ == op) {

      for (auto it = const_free_map_.begin(); it != const_free_map_.end();) {
        if (it->second->attach_scope_ == op) {
          it = const_free_map_.erase(it);
        } else {
          ++it;
        }
      }
      for (auto it = sym_free_list_.begin(); it != sym_free_list_.end();) {
        if ((*it)->attach_scope_ == op) {
          it = sym_free_list_.erase(it);
        } else {
          ++it;
        }
      }
      TVM_FFI_ICHECK(!thread_scope_stack_.empty());
      thread_scope_ = thread_scope_stack_.back();
      thread_scope_stack_.pop_back();
    } else {
      thread_scope_stack_.push_back(thread_scope_);
      thread_scope_ = op;
    }
  }

  void
  PlanMemory(const std::vector<StmtEntry> &seq,
             const std::unordered_map<const VarNode *, AllocEntry> &alloc_info,
             bool enable_reuse, bool reuse_require_exact_matched_dtype) {
    std::unordered_set<const VarNode *> inplace_flag;

    for (size_t i = 0; i < seq.size(); ++i) {
      const StmtEntry &s = seq[i];
      auto it = event_map_.find(seq[i].stmt);

      if (it != event_map_.end() && seq[i].scope_pair_offset >= 0) {

        bool detect_inplace = detect_inplace_ && (it->second.gen.size() <= 2);

        for (const VarNode *var : it->second.gen) {
          TVM_FFI_ICHECK(alloc_info.count(var));
          const AllocEntry &entry = alloc_info.at(var);
          const BindNode *alloc = entry.alloc;
          const auto *call = alloc->value.as<CallNode>();
          runtime::MemoryScope scope =
              call->ty_args[0].as_or_throw<TensorType>()->scope;
          auto storage_scope = scope;
          StorageEntry *dst_entry = nullptr;

          if (detect_inplace && !RequiresExactAllocation(call) &&
              (GetMemoryAllocProperty(scope) !=
                   MemoryAllocProperty::kSingleAllocation ||
               alloc->var->ty.as_or_throw<TensorType>()
                       ->ConstantAllocationSize()
                       .value_or(0) > 0)) {
            DLDataType dtype =
                call->ty_args[0].as_or_throw<TensorType>()->dtype->dtype;
            PrimType element_type(dtype);

            bool inplace_found = false;
            for (const VarNode *src : it->second.kill) {
              if (!inplace_flag.count(src) && alloc_map_.count(src)) {
                auto visitor = ffi::make_object<InplaceOpVerifier>();
                StorageEntry *src_entry = alloc_map_.at(src);
                if (!src_entry->requires_exact_allocation &&
                    HasCompatibleAlignment(src_entry, call) &&
                    src_entry->scope == storage_scope &&
                    src_entry->attach_scope_ == thread_scope_ &&
                    !element_type.IsScalableVector() &&
                    src_entry->elem_type == element_type.WithLanes(1) &&
                    visitor->Check(s.stmt, var, src)) {
                  int64_t const_size = alloc->var->ty.as_or_throw<TensorType>()
                                           ->ConstantAllocationSize()
                                           .value_or(0);
                  uint64_t const_nbits = static_cast<uint64_t>(const_size) *
                                         element_type.bits() *
                                         element_type.lanes();
                  if (src_entry->const_nbits == const_nbits && !inplace_found) {

                    dst_entry = src_entry;
                    inplace_flag.insert(src);
                    inplace_found = true;
                  }
                }
              }
            }
          }
          if (dst_entry == nullptr) {
            dst_entry = FindAlloc(alloc, thread_scope_, storage_scope,
                                  entry.num_physical_dimensions, enable_reuse,
                                  reuse_require_exact_matched_dtype);
          }
          dst_entry->allocs.emplace_back(alloc);
          if (call->attrs.has_value() &&
              call->attrs.value().as_or_throw<DictAttrs>()->dict.count(
                  tvm::tirx::attr::kVolatile.name)) {
            dst_entry->is_volatile = true;
          }
          alloc_map_[var] = dst_entry;
        }
      }

      if (s.stmt->IsInstance<RegionStmtNode>()) {
        PlanNewScope(s.stmt);
      } else if (s.stmt->IsInstance<ForNode>()) {
        const auto *op = static_cast<const ForNode *>(s.stmt);
        if (op->kind == ForKind::kParallel) {
          if (thread_scope_ == nullptr || thread_scope_ == op) {
            PlanNewScope(op);
          }
        }
      }

      if (it != event_map_.end() && seq[i].scope_pair_offset <= 0) {
        for (const VarNode *var : it->second.kill) {

          if (!inplace_flag.count(var)) {
            this->Free(var);
          }
        }
      }
    }
  }

  StorageEntry *NewAlloc(const BindNode *op, const ffi::Object *attach_scope,
                         const MemoryScope &scope, size_t const_nbits) {
    TVM_FFI_ICHECK(op != nullptr);
    const auto *call = op->value.as<CallNode>();
    DLDataType dtype = call->ty_args[0].as_or_throw<TensorType>()->dtype->dtype;
    PrimType element_type(dtype);

    auto entry = std::make_unique<StorageEntry>();
    entry->attach_scope_ = attach_scope;
    entry->scope = scope;
    entry->elem_type = element_type.WithLanes(1);
    entry->const_nbits = const_nbits;
    entry->requires_exact_allocation = RequiresExactAllocation(call);
    StorageEntry *e = entry.get();
    alloc_vec_.emplace_back(std::move(entry));
    return e;
  }

  StorageEntry *FindAlloc(const BindNode *op, const ffi::Object *attach_scope,
                          const MemoryScope &scope,
                          size_t num_physical_dimensions, bool enable_reuse,
                          bool reuse_require_exact_matched_dtype) {
    TVM_FFI_ICHECK(op != nullptr);
    const auto *call = op->value.as<CallNode>();
    DLDataType dtype = call->ty_args[0].as_or_throw<TensorType>()->dtype->dtype;
    PrimType element_type(dtype);

    const uint64_t match_range = 16;
    bool is_scalable_vector = element_type.IsScalableVector();
    uint64_t op_elem_bits =
        is_scalable_vector ? 0 : element_type.bits() * element_type.lanes();
    int64_t const_size = op->var->ty.as_or_throw<TensorType>()
                             ->ConstantAllocationSize()
                             .value_or(0);
    uint64_t const_nbits =
        is_scalable_vector ? 0
                           : static_cast<uint64_t>(const_size * op_elem_bits);

    bool is_known_size = (const_nbits != 0);

    bool is_flat_memory_space = (num_physical_dimensions == 1);

    bool is_small_array =
        GetMemoryAllocProperty(scope) == MemoryAllocProperty::kNormal &&
        is_known_size && const_nbits <= 32;

    if (RequiresExactAllocation(call) || is_scalable_vector || !enable_reuse ||
        is_small_array ||
        GetMemoryAllocProperty(scope) ==
            MemoryAllocProperty::kSeparateAllocations ||
        !is_flat_memory_space ||
        (GetMemoryAllocProperty(scope) ==
             MemoryAllocProperty::kSingleAllocation &&
         !is_known_size)) {
      return NewAlloc(op, attach_scope, scope, const_nbits);
    }

    if (is_known_size) {

      auto begin = const_free_map_.lower_bound(const_nbits / match_range);
      auto mid = const_free_map_.lower_bound(const_nbits);
      auto end = const_free_map_.upper_bound(const_nbits * match_range);

      for (auto it = mid; it != end; ++it) {
        StorageEntry *e = it->second;
        if (e->attach_scope_ != attach_scope)
          continue;
        if (e->scope != scope || !HasCompatibleAlignment(e, call))
          continue;

        if (e->bits_offset % op_elem_bits != 0)
          continue;
        if (reuse_require_exact_matched_dtype && e->elem_type != element_type) {
          continue;
        }
        e->const_nbits = std::max(const_nbits, e->const_nbits);
        const_free_map_.erase(it);
        return e;
      }

      for (auto it = mid; it != begin;) {
        --it;
        StorageEntry *e = it->second;
        if (e->attach_scope_ != attach_scope)
          continue;
        if (e->scope != scope || !HasCompatibleAlignment(e, call))
          continue;
        if (e->elem_type != element_type.WithLanes(1))
          continue;
        if (reuse_require_exact_matched_dtype && e->elem_type != element_type) {
          continue;
        }
        e->const_nbits = std::max(const_nbits, e->const_nbits);
        const_free_map_.erase(it);
        return e;
      }
    } else {

      for (auto it = sym_free_list_.begin(); it != sym_free_list_.end(); ++it) {
        StorageEntry *e = *it;
        if (e->attach_scope_ != attach_scope)
          continue;
        if (e->scope != scope || !HasCompatibleAlignment(e, call))
          continue;
        if (e->elem_type != element_type.WithLanes(1))
          continue;
        sym_free_list_.erase(it);
        return e;
      }
    }
    return NewAlloc(op, attach_scope, scope, const_nbits);
  }

  void Free(const VarNode *var) {
    auto it = alloc_map_.find(var);
    TVM_FFI_ICHECK(it != alloc_map_.end());
    StorageEntry *e = it->second;
    TVM_FFI_ICHECK_NE(e->allocs.size(), 0U);
    if (e->requires_exact_allocation)
      return;

    const auto *call = e->allocs[0]->value.as<CallNode>();
    PrimType element_type(
        call->ty_args[0].as_or_throw<TensorType>()->dtype->dtype);
    if (element_type.IsScalableVector())
      return;
    if (GetMemoryAllocProperty(e->scope) == MemoryAllocProperty::kNormal &&
        e->const_nbits > 0 && e->const_nbits <= 32)
      return;
    if (GetMemoryAllocProperty(e->scope) ==
            MemoryAllocProperty::kSingleAllocation &&
        e->const_nbits == 0)
      return;

    if (e->const_nbits != 0) {
      const_free_map_.insert({e->const_nbits, e});
    } else {
      sym_free_list_.push_back(e);
    }
  }

  const ffi::Object *thread_scope_{nullptr};
  std::vector<const ffi::Object *> thread_scope_stack_;

  bool detect_inplace_{false};
  bool preserve_shared_allocations_{false};
  bool allow_local_inplace_{false};

  std::unordered_map<const ffi::Object *, EventEntry> event_map_;

  std::multimap<uint64_t, StorageEntry *> const_free_map_;

  std::list<StorageEntry *> sym_free_list_;

  std::unordered_map<const ffi::Object *, std::vector<StorageEntry *>>
      attach_map_;

  std::unordered_map<const VarNode *, StorageEntry *> alloc_map_;

  std::vector<std::unique_ptr<StorageEntry>> alloc_vec_;

  std::unordered_map<const VarNode *, TensorVar> remapped_backing_;

  ffi::Map<Var, Var> tensor_aliases_;

  sym::Analyzer analyzer_;
};

} // namespace

Pass StorageRewrite(bool enable_reuse, bool detect_inplace,
                    bool preserve_shared_allocations,
                    bool allow_local_inplace) {
  auto pass_func = [=](Function f, IRModule m, PassContext ctx) {
    if (!f->body.has_value())
      return f;
    bool reuse = enable_reuse;
    bool reuse_require_exact_matched_dtype = false;
    bool merge_static_smem =
        ctx->GetConfig<bool>("tirx.merge_static_smem").value_or(false);
    if (merge_static_smem) {

      reuse = false;
    }

    ffi::Optional<Target> target = f->GetAttr(tvm::attr::kTarget);
    if (target.has_value() && (target.value()->kind->name == "vulkan" ||
                               target.value()->kind->name == "webgpu")) {

      reuse_require_exact_matched_dtype = true;
    }
    auto *n = f.CopyOnWrite();
    n->body = ffi::make_object<StoragePlanRewriter>()->Rewrite(
        std::move(n->body).value(), n->params, detect_inplace, reuse,
        reuse_require_exact_matched_dtype, preserve_shared_allocations,
        allow_local_inplace);

    return tl::RewritePointerValueTypes(std::move(f), true);
  };
  return CreateFunctionPass(pass_func, 0, "tl.StorageRewrite");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl._tirx_passes.StorageRewrite",
                                   StorageRewrite);
}

} // namespace tvm::tl::tirx_passes

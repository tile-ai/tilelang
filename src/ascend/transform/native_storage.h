#ifndef TILELANG_ASCEND_TRANSFORM_NATIVE_STORAGE_H_
#define TILELANG_ASCEND_TRANSFORM_NATIVE_STORAGE_H_
#include "op/prim.h"

#include <tvm/tirx/op/memory.h>

#include <optional>
#include <unordered_map>

#include "ascend/transform/native_visitor.h"
#include "compat/expr.h"

namespace tvm::tl::ascend {

struct StorageAccess {
  Var root;
  PrimExpr bit_offset;
};

inline int ElementBits(PrimType dtype) { return dtype.bits() * dtype.lanes(); }

inline PrimExpr StorageIndex(const PrimExpr &index) {
  return tl::CastValue(index, PrimType::Int(64).WithLanes(index.ty().lanes()));
}

class StorageAliasAnalysis : public AnalysisVisitor {
public:
  void Collect(const tirx::Function &function) {
    for (const Var &parameter : function->params)
      RecordRoot(parameter, false);
    Visit(function->body);
  }

  std::optional<StorageAccess> Resolve(const Expr &expression) const {
    if (auto variable = expression.as<Var>()) {
      auto found = accesses_.find(variable.value().get());
      if (found != accesses_.end())
        return found->second;
      if (variable.value()->ty.as<tirx::TensorTypeNode>() ||
          variable.value()->ty.as<PtrTypeNode>()) {
        return StorageAccess{variable.value(), IntImm(PrimType::Int(64), 0)};
      }
    }
    if (const auto *call = expression.as<CallNode>()) {
      if (call->op.same_as(tirx::tensor_data_ptr_op()) ||
          call->op.same_as(tirx::decl_tensor_op()) ||
          call->op.same_as(tirx::reinterpret_op())) {
        if (!call->args.empty())
          return Resolve(call->args[0]);
      }
      if (call->op.same_as(tirx::ptr_byte_offset_op())) {
        if (auto base = Resolve(call->args[0])) {
          base.value().bit_offset =
              base.value().bit_offset +
              StorageIndex(call->args[1].as_or_throw<PrimExpr>()) * 8;
          return base;
        }
      }
      if (call->op.same_as(tirx::address_of_op())) {
        if (const auto *load = call->args[0].as<TensorLoadNode>()) {
          auto source = load->source.as_or_throw<tirx::TensorVar>();
          if (auto base = Resolve(source)) {
            auto offsets = source.ty()->ElemOffset(load->indices);
            TVM_FFI_CHECK(offsets.size() == 1, ValueError)
                << "Ascend storage must have one memory axis";
            base.value().bit_offset =
                base.value().bit_offset +
                StorageIndex(offsets[0]) * ElementBits(source.dtype());
            return base;
          }
        }
      }
    }
    return std::nullopt;
  }

  ffi::Map<Var, tirx::TensorVar> allocations;
  std::vector<tirx::TensorVar> tensors;

protected:
  using AnalysisVisitor::VisitStmt_;

  void VisitStmt_(const RegionStmtNode *node) override {
    Visit(node->body);
    auto values = node->attrs.GetAttr<ffi::Array<Expr>>("tl.schedule_yield");
    if (!values)
      return;
    TVM_FFI_CHECK_EQ(values->size(), node->result_vars.size(), ValueError)
        << "Scheduled storage results must match their yields";
    for (size_t index = 0; index < values->size(); ++index) {
      auto access = Resolve((*values)[index]);
      if (!access)
        continue;
      const Var &result = node->result_vars[index];
      Var previous = access->root;
      if ((*values)[index].same_as(previous) && allocations.count(previous)) {
        for (auto &[variable, existing] : accesses_)
          if (existing.root.same_as(previous))
            existing.root = result;
        allocations.erase(previous);
        allocations.Set(result, result.as_or_throw<tirx::TensorVar>());
        access->root = result;
      }
      accesses_.insert_or_assign(result.get(), access.value());
      if (auto tensor = result.as<tirx::TensorVar>())
        tensors.push_back(tensor.value());
    }
  }

  void VisitStmt_(const BindNode *node) override {
    Visit(node->value);
    if (const auto *call = node->value.as<CallNode>();
        call && call->op.same_as(tirx::alloc_tensor_op())) {
      RecordRoot(node->var, true);
      return;
    }
    if (auto access = Resolve(node->value)) {
      accesses_.insert_or_assign(node->var.get(), access.value());
      if (auto tensor = node->var.as<tirx::TensorVar>()) {
        tensors.push_back(tensor.value());
      }
    }
  }

private:
  void RecordRoot(const Var &variable, bool allocated) {
    if (!variable->ty.as<tirx::TensorTypeNode>() &&
        !variable->ty.as<PtrTypeNode>()) {
      return;
    }
    accesses_.insert_or_assign(
        variable.get(), StorageAccess{variable, IntImm(PrimType::Int(64), 0)});
    if (auto tensor = variable.as<tirx::TensorVar>()) {
      tensors.push_back(tensor.value());
      if (allocated)
        allocations.Set(variable, tensor.value());
    }
  }

  std::unordered_map<const VarNode *, StorageAccess> accesses_;
};

} // namespace tvm::tl::ascend

#endif

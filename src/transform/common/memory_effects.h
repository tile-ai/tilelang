#ifndef TILELANG_TRANSFORM_COMMON_MEMORY_EFFECTS_H_
#define TILELANG_TRANSFORM_COMMON_MEMORY_EFFECTS_H_

#include "compat/expr.h"
#include "compat/visit.h"
#include "op/builtin.h"
#include "op/operator.h"
#include "op/parallel.h"

#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/memory.h>

#include <unordered_map>
#include <unordered_set>

namespace tvm::tl::analysis {

using Roots = std::unordered_set<const VarNode *>;

inline bool IsAddress(const Expr &expression) {
  return expression->ty.as<PtrTypeNode>() ||
         expression->ty.as<tirx::TensorTypeNode>();
}

class AliasOrigins {
public:
  explicit AliasOrigins(const tirx::Function &function)
      : no_alias_(function->GetAttr(tirx::attr::kNoAlias).value_or(false)) {
    for (const Var &parameter : function->params) {
      if (IsAddress(parameter)) {
        roots_[parameter.get()] = {parameter.get()};
        parameters_.insert(parameter.get());
      }
    }
    auto nonrestrict = function->GetAttr<ffi::Array<ffi::Any>>(
        "tl.non_restrict_param_indices");
    if (nonrestrict) {
      for (const auto &index : nonrestrict.value()) {
        auto integer = GetIntAnnotation(index);
        if (integer && *integer >= 0 && *integer < function->params.size()) {
          nonrestrict_.insert(function->params[*integer].get());
        }
      }
    }
    PostOrderVisit(function->body.value(), [&](const ffi::ObjectRef &object) {
      if (const auto *binding = object.as<BindNode>();
          binding && IsAddress(binding->var)) {
        const auto *call = binding->value.as<CallNode>();
        if (call && (call->op.same_as(tirx::alloc_tensor_op()) ||
                     call->op.same_as(tirx::stack_alloca_op()))) {
          roots_[binding->var.get()] = {binding->var.get()};
          allocations_.insert(binding->var.get());
        } else {
          roots_[binding->var.get()] = Get(binding->value);
        }
      }
      if (const auto *region = object.as<RegionStmtNode>();
          region && region->attrs->dict.count("tl.non_restrict_params")) {
        no_alias_ = false;
      }
    });
  }

  Roots Get(const Expr &expression) const {
    if (auto variable = expression.as<Var>()) {
      auto found = roots_.find(variable.value().get());
      if (found != roots_.end())
        return found->second;
      return {nullptr};
    }
    const auto *call = expression.as<CallNode>();
    if (!call)
      return {nullptr};
    if ((call->op.same_as(access_ptr()) ||
         call->op.same_as(tirx::address_of_op())) &&
        !call->args.empty()) {
      if (const auto *load = call->args[0].as<TensorLoadNode>())
        return Get(load->source);
      return Get(call->args[0]);
    }
    if (call->op.same_as(prim::if_then_else_op()) && call->args.size() == 3) {
      Roots roots = Get(call->args[1]);
      Roots alternative = Get(call->args[2]);
      roots.insert(alternative.begin(), alternative.end());
      return roots;
    }
    if ((call->op.same_as(tirx::tensor_data_ptr_op()) ||
         call->op.same_as(tirx::decl_tensor_op()) ||
         call->op.same_as(tirx::ptr_byte_offset_op()) ||
         call->op.same_as(tirx::handle_add_byte_offset_op()) ||
         call->op.same_as(tirx::reinterpret_op())) &&
        !call->args.empty()) {
      return Get(call->args[0]);
    }
    return {nullptr};
  }

  bool MayAlias(const Roots &reads, const Roots &writes) const {
    for (const VarNode *read : reads) {
      for (const VarNode *write : writes) {
        if (!read || !write || read == write)
          return true;
        if (allocations_.count(read) || allocations_.count(write))
          continue;
        if (no_alias_ && parameters_.count(read) && parameters_.count(write) &&
            !(nonrestrict_.count(read) && nonrestrict_.count(write)))
          continue;
        return true;
      }
    }
    return false;
  }

private:
  std::unordered_map<const VarNode *, Roots> roots_;
  Roots parameters_;
  Roots allocations_;
  Roots nonrestrict_;
  bool no_alias_;
};

struct MemoryEffects {
  Roots reads;
  Roots writes;
  bool unknown_read{false};
  bool unknown_write{false};
};

class EffectCollector : public tirx::StmtExprVisitor {
public:
  explicit EffectCollector(const AliasOrigins &aliases) : aliases_(aliases) {}

  MemoryEffects Collect(const ffi::ObjectRef &object) {
    Visit(object);
    return effects_;
  }

private:
  void Add(const Expr &tensor, bool write) {
    Roots roots = aliases_.Get(tensor);
    auto &destination = write ? effects_.writes : effects_.reads;
    destination.insert(roots.begin(), roots.end());
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorLoadNode *node) final {
    Add(node->source, false);
    return StmtExprVisitor::Visit_(node);
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const TensorStoreNode *node) final {
    Add(node->dest, true);
    return StmtExprVisitor::Visit_(node);
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const VarNode *) final {
    return std::nullopt;
  }

  ffi::Optional<ffi::VisitInterrupt> Visit_(const CallNode *node) final {
    if (auto operation = ParseOperator(ffi::GetRef<Call>(node));
        operation.defined()) {
      if (const auto *parallel = operation.as<ParallelOpNode>()) {
        Visit(parallel->GetRoot());
      } else {
        auto regions = operation->GetAccessRegions();
        for (const auto &region : regions.reads)
          Add(region->source, false);
        for (const auto &region : regions.writes)
          Add(region->source, true);
      }
      return StmtExprVisitor::Visit_(node);
    }
    if ((node->op.same_as(access_ptr()) ||
         node->op.same_as(tirx::address_of_op())) &&
        !node->args.empty()) {
      if (const auto *load = node->args[0].as<TensorLoadNode>()) {
        int mask = kAccessReadWrite;
        if (node->op.same_as(access_ptr()) && node->args.size() == 3) {
          mask =
              GetConservativeAccessMask(node->args[2].as_or_throw<PrimExpr>());
        } else if (node->attrs) {
          if (auto attrs = node->attrs.value().as<DictAttrs>()) {
            if (auto access =
                    attrs.value().GetAttr<PrimExpr>(attr::kAccessMask)) {
              mask = GetConservativeAccessMask(access.value());
            }
          }
        }
        if (mask & kAccessRead)
          Add(load->source, false);
        if (mask & kAccessWrite)
          Add(load->source, true);
        Visit(load->indices);
        for (size_t index = 1; index < node->args.size(); ++index)
          Visit(node->args[index]);
        return std::nullopt;
      }
    }
    auto effect = Op::GetAttrMap(tvm::op_attr::kCallEffectKind)
                      .get(node->op, CallEffectKind::kOpaque);
    effects_.unknown_read |= effect == CallEffectKind::kReadState;
    effects_.unknown_write |= effect > CallEffectKind::kReadState;
    return StmtExprVisitor::Visit_(node);
  }

  const AliasOrigins &aliases_;
  MemoryEffects effects_;
};

} // namespace tvm::tl::analysis

#endif

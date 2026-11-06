/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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

#ifndef TVM_TL_TRANSFORM_COMMON_UNUSED_TENSOR_ALIAS_H_
#define TVM_TL_TRANSFORM_COMMON_UNUSED_TENSOR_ALIAS_H_

#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm::tl {
namespace detail {

// Flattening introduces tensor views even for source parameters that the
// body never uses. Do not capture those pure, dead aliases as kernel data
// arguments: doing so would turn a nullable shape carrier into a data use.
// Other calls and allocation operations remain untouched, including their
// errors and effects. Dependencies include native types/layouts by identity.
class UnusedTensorAliasRemover : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  explicit UnusedTensorAliasRemover(const Stmt &body) {
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
        body,
        [&](const BindObj *bind) -> ffi::Expected<ffi::WalkResult> {
          if (!IsPureAlias(bind))
            return ffi::WalkResult::Advance();
          auto &dependencies = aliases_[bind->var.get()];
          auto collect =
              [&](const VarObj *variable) -> ffi::Expected<ffi::WalkResult> {
            dependencies.insert(variable);
            return ffi::WalkResult::Advance();
          };
          ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(bind->value, collect);
          ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(bind->var->ty,
                                                          collect);
          return ffi::WalkResult::Skip();
        },
        [&](const VarObj *variable) -> ffi::Expected<ffi::WalkResult> {
          live_.insert(variable);
          return ffi::WalkResult::Advance();
        });
    std::vector<const VarObj *> pending(live_.begin(), live_.end());
    while (!pending.empty()) {
      const auto *variable = pending.back();
      pending.pop_back();
      auto found = aliases_.find(variable);
      if (found == aliases_.end())
        continue;
      for (const auto *dependency : found->second) {
        if (live_.insert(dependency).second)
          pending.push_back(dependency);
      }
    }
  }

  ffi::UnchangedOr<Stmt> Mutate_(const BindObj *bind, InplaceMode mode) final {
    if (aliases_.count(bind->var.get()) && !live_.count(bind->var.get()))
      return SeqStmt({});
    return StmtExprMutator::Mutate_(bind, mode);
  }

private:
  static bool IsPureAlias(const BindObj *bind) {
    if (!bind->var->ty.as<PtrTypeObj>() &&
        !bind->var->ty.as<tirx::TensorTypeObj>())
      return false;
    const auto *call = bind->value.as<CallObj>();
    if (!call || !call->op.as<OpObj>())
      return false;
    bool is_alias = call->op.same_as(tirx::decl_tensor_op()) ||
                    call->op.same_as(tirx::tensor_data_ptr_op()) ||
                    call->op.same_as(tirx::reinterpret_op()) ||
                    call->op.same_as(tirx::ptr_byte_offset_op()) ||
                    call->op.same_as(tirx::handle_add_byte_offset_op());
    return is_alias && tvm::SideEffect(bind->value) == CallEffectKind::kPure;
  }
  std::unordered_map<const VarObj *, std::unordered_set<const VarObj *>>
      aliases_;
  std::unordered_set<const VarObj *> live_;
};

} // namespace detail

// Run before deciding whether an input is a data use or a nullable shape
// carrier, both before outlining and at the direct packed-ABI boundary.
inline SeqStmt RemoveUnusedTensorAliases(const SeqStmt &body) {
  auto remover = ffi::make_object<detail::UnusedTensorAliasRemover>(body);
  return remover->Mutate(body, InplaceMode::kDisallow).ValueOrUnchanged(body);
}

} // namespace tvm::tl

#endif // TVM_TL_TRANSFORM_COMMON_UNUSED_TENSOR_ALIAS_H_

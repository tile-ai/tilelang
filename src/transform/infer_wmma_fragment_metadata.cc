/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file infer_wmma_fragment_metadata.cc
 * \brief Retain WMMA shape/layout checks at the native tensor boundary.
 *
 * The metadata contract follows tensorcore_infer_fragment.cc at old TVM
 * 4211874e9e4b8770fd20ad0d98ab80afd1df5028. This is a whole-function
 * consistency check, not an initialization or dominance analysis.
 */
#include "cuda/op/builtin.h"
#include "ir/alias_analysis.h"
#include <tvm/ffi/extra/structural_visit.h>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <array>
#include <unordered_map>

namespace tvm::tl {
namespace {
using namespace tirx;

struct WmmaInfo {
  std::array<int64_t, 3> shape;
  std::string layout;
};

class WmmaMetadata {
public:
  explicit WmmaMetadata(const tirx::Function &function) : roots_(function) {
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        function->body,
        [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
          Gather(call);
          return ffi::WalkResult::Advance();
        });
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        function->body,
        [&](const CallObj *call) -> ffi::Expected<ffi::WalkResult> {
          CheckMma(call);
          return ffi::WalkResult::Advance();
        });
  }

  ffi::Optional<Var> Subject(const Expr &expression) const {
    // Frontend .data is a native tensor_data_ptr. Known aliases have one
    // storage identity; unresolvable expressions must not invent an owner.
    return roots_.Root(expression);
  }

  std::unordered_map<const VarObj *, WmmaInfo> info;

private:
  static std::string Name(const CallObj *call) {
    const auto *op = call->op.as<OpObj>();
    return op ? std::string(op->name) : std::string();
  }

  static std::string Scope(const Var &subject) {
    if (const auto *tensor = subject->ty.as<TensorTypeObj>())
      return tensor->scope.name();
    if (const auto *pointer = subject->ty.as<PtrTypeObj>())
      return pointer->scope.name();
    TVM_FFI_THROW(InternalError)
        << "WMMA fragment must be a tensor or typed pointer";
    return {};
  }

  void Gather(const CallObj *call) {
    const auto name = Name(call);
    const bool fill = name == "tirx.gpu_fill_fragment";
    const bool memory = call->op.same_as(tl::wmma_load_matrix_sync()) ||
                        name == "tirx.gpu_load_matrix_sync" ||
                        name == "tirx.gpu_store_matrix_sync";
    if (!fill && !memory)
      return;
    TVM_FFI_ICHECK_EQ(call->args.size(), fill ? 6 : 8);
    auto subject = Subject(call->args[0]);
    TVM_FFI_ICHECK(subject)
        << "WMMA intrinsic requires an identifiable fragment";
    WmmaInfo next;
    for (size_t i = 0; i < 3; ++i) {
      const auto *dimension = call->args[i + 1].as<IntImmObj>();
      TVM_FFI_ICHECK(dimension) << "WMMA dimensions must be IntImm";
      next.shape[i] = static_cast<int64_t>(dimension->value);
    }
    const auto scope = Scope(*subject);
    const bool matrix = scope == "wmma.matrix_a" || scope == "wmma.matrix_b";
    if (fill) {
      next.layout = "row_major";
    } else {
      const auto *layout = call->args[7].as<StringImmObj>();
      TVM_FFI_ICHECK(layout) << "WMMA layout must be StringImm";
      next.layout = matrix ? std::string(layout->value) : std::string();
    }
    auto [iterator, inserted] = info.emplace(subject.value().get(), next);
    if (!inserted) {
      TVM_FFI_ICHECK(iterator->second.shape == next.shape)
          << "Inconsistent WMMA fragment shape for " << subject.value()->name;
      if (memory && matrix) {
        TVM_FFI_ICHECK_EQ(iterator->second.layout, next.layout)
            << "Inconsistent WMMA fragment layout for "
            << subject.value()->name;
      }
    }
  }

  void CheckMma(const CallObj *call) {
    const auto name = Name(call);
    if (name != "tirx.gpu_mma_sync" && name != "tirx.cuda.bmma_sync")
      return;
    TVM_FFI_ICHECK_EQ(call->args.size(), 8);
    const WmmaInfo *expected = nullptr;
    for (size_t i = 0; i < 8; i += 2) {
      auto subject = Subject(call->args[i]);
      TVM_FFI_ICHECK(subject)
          << "WMMA intrinsic requires an identifiable fragment";
      auto found = info.find(subject.value().get());
      TVM_FFI_ICHECK(found != info.end())
          << "Tensorcore fragment " << subject.value()->name
          << " must be filled (with tvm_fill_fragment) or loaded (with "
             "tvm_load_matrix_sync) before use.";
      if (expected) {
        TVM_FFI_ICHECK(expected->shape == found->second.shape)
            << "Inconsistent WMMA fragment shape for " << subject.value()->name;
      } else {
        expected = &found->second;
      }
    }
  }

  PhysicalRootAnalysis roots_;
};

class WmmaMetadataWriter : public StmtExprMutator {
public:
  explicit WmmaMetadataWriter(const WmmaMetadata *metadata)
      : metadata_(metadata) {}
  using StmtExprMutator::Mutate_;

  ffi::UnchangedOr<Stmt> Mutate_(const BindObj *bind, InplaceMode mode) final {
    const auto *allocation = bind->value.as<CallObj>();
    auto found = metadata_->info.find(bind->var.get());
    if (!allocation || !allocation->op.same_as(alloc_tensor_op()) ||
        found == metadata_->info.end()) {
      return StmtExprMutator::Mutate_(bind, mode);
    }
    const auto &info = found->second;
    ffi::Map<ffi::String, ffi::Any> fields;
    if (allocation->attrs)
      fields = allocation->attrs.value().as_or_throw<DictAttrs>()->dict;
    fields.Set("tl.wmma_shape",
               ffi::String(std::to_string(info.shape[0]) + ", " +
                           std::to_string(info.shape[1]) + ", " +
                           std::to_string(info.shape[2])));
    if (!info.layout.empty())
      fields.Set("tl.wmma_layout", ffi::String(info.layout));
    return Bind(bind->var,
                Call(allocation->ty, allocation->op, allocation->args,
                     DictAttrs(fields), allocation->ty_args, allocation->loc),
                bind->loc);
  }

private:
  const WmmaMetadata *metadata_;
};
} // namespace

namespace transform {
tvm::transform::Pass InferWmmaFragmentMetadata() {
  return tirx::transform::CreateFunctionPass(
      [](tirx::Function function, IRModule, tvm::transform::PassContext) {
        if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
          return function;
        WmmaMetadata metadata(function);
        auto writer = ffi::make_object<WmmaMetadataWriter>(&metadata);
        auto body = function->body;
        function.CopyOnWrite()->body =
            writer->Mutate(body).ValueOrUnchanged(body);
        return function;
      },
      0, "tl.InferWmmaFragmentMetadata");
}
} // namespace transform

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.InferWmmaFragmentMetadata",
                                   transform::InferWmmaFragmentMetadata);
}
} // namespace tvm::tl

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
 * \file tl/ascend/transform/normalize_buffer_version.cc
 * \brief Normalize staged version metadata at its native definition site.
 */
#include "buffer_version.h"

#include "auto_schedule/native_task_ir.h"
#include <limits>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
namespace {
using namespace tirx;
using namespace ffi;
using namespace ascend;
using Fields = Map<String, Any>;
constexpr const char *kPhysicalBufferVersion = "tl.buffer_version";

class BufferVersionNormalizer : public StmtExprMutator {
public:
  using StmtExprMutator::Mutate_;

  static Fields Normalize(Fields fields) {
    TVM_FFI_CHECK(!fields.count(kPhysicalBufferVersion), ValueError)
        << "'" << kPhysicalBufferVersion
        << "' is compiler-internal and must not be provided by the frontend";
    if (auto staged = fields.Get(kManualMultiBuffer)) {
      BufferVersionMap versions;
      for (const auto &[key, value] : staged->cast<Map<Any, Any>>()) {
        Var storage = key.cast<Var>();
        TVM_FFI_CHECK(storage->ty.as<TensorTypeObj>() ||
                          storage->ty.as<PtrTypeObj>(),
                      ValueError)
            << "Manual versions require a tensor or pointer storage";
        auto count = value.try_cast<int64_t>();
        if (!count) {
          PrimExpr expression = value.cast<PrimExpr>();
          count = TryConstInt(expression);
          TVM_FFI_CHECK(expression.ty().IsScalar() &&
                            !expression.ty().MatchesCode(kDLBool),
                        ValueError)
              << "Manual version count must be a constant scalar integer";
        }
        TVM_FFI_CHECK(count && *count > 0 &&
                          *count <= std::numeric_limits<int>::max(),
                      ValueError)
            << "Manual version count must be a positive int32 value";
        versions.Set(storage, static_cast<int>(*count));
      }
      fields.erase(kManualMultiBuffer);
      if (!versions.empty())
        fields.Set(kPhysicalBufferVersion, versions);
    }
    return fields;
  }

  UnchangedOr<Stmt> Mutate_(const EvaluateObj *op, InplaceMode mode) final {
    const auto *call = op->value.as<CallObj>();
    if (!loop_annotation_detail::IsAnnotation(call))
      return StmtExprMutator::Mutate_(op, mode);
    // Keep the native carrier in place. Wrapping the suffix in a zero-result
    // RegionStmt would hide definitions which can be used after this sequence.
    auto fields = ReadNativeAnnotations(call);
    auto result = Normalize(fields);
    if (StructuralEqual()(fields, result))
      return Unchanged();
    return MakeNativeAnnotation(result, op->loc);
  }

  UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op, InplaceMode mode) final {
    RegionStmt region = StmtExprMutator::Mutate_(op, mode)
                            .ValueOrUnchanged(GetRef<RegionStmt>(op))
                            .as_or_throw<RegionStmt>();
    auto attribute = ReadTaskAttribute(region.get());
    if (!attribute)
      return region;
    TVM_FFI_CHECK(attribute->key != kPhysicalBufferVersion, ValueError)
        << "'" << kPhysicalBufferVersion << "' is compiler-internal";
    if (attribute->key != "tl.sblock" &&
        attribute->key != "tl.loop.annotations")
      return region;
    auto fields = attribute->value.cast<Fields>();
    auto result = Normalize(fields);
    if (attribute->key == "tl.sblock") {
      if (auto annotations = result.Get("annotations"))
        result.Set("annotations", Normalize(annotations->cast<Fields>()));
    }
    if (StructuralEqual()(fields, result))
      return region;
    return MakeNativeAttribute(attribute->node, attribute->key, result,
                               region->body, region->loc);
  }
};
} // namespace

tvm::transform::Pass NormalizeBufferVersion() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    TVM_FFI_CHECK(!function->attrs->dict.count(kPhysicalBufferVersion),
                  ValueError)
        << "'" << kPhysicalBufferVersion << "' is compiler-internal";
    if (!function->GetAttr(tvm::attr::kDeclaration).value_or(false)) {
      auto rewriter = ffi::make_object<BufferVersionNormalizer>();
      auto body = function->body;
      function.CopyOnWrite()->body =
          rewriter->Mutate(body, InplaceMode::kDisallow).ValueOrUnchanged(body);
    }
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.NormalizeBufferVersion");
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.NormalizeBufferVersion",
                                   NormalizeBufferVersion);
}
} // namespace tvm::tl

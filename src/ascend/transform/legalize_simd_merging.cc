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
 * \file legalize_simd_merging.cc
 * \brief Legalize MODE_MERGING SIMD expressions into explicit read-modify-write
 *        statements.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <utility>

#include "op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

namespace {

constexpr const char *kModeMerging = "MODE_MERGING";
constexpr const char *kSimdOpPrefix = "tl.simd.";
constexpr int kReadWriteAccessMask = 3;

bool IsSimdMergingCall(const CallNode *call) {
  if (call->args.empty()) {
    return false;
  }
  const auto *mode = call->args.back().as<StringImmNode>();
  if (mode == nullptr || mode->value != kModeMerging) {
    return false;
  }
  if (auto call_op = call->op.as<Op>()) {
    const std::string &op_name = call_op.value()->name;
    return op_name.rfind(kSimdOpPrefix, 0) == 0;
  }
  return false;
}

class SimdMergingLegalizer : public StmtExprMutator {
public:
  static PrimFunc Rewrite(PrimFunc func) {
    SimdMergingLegalizer legalizer;
    func.CopyOnWrite()->body = legalizer(func->body);
    SimdMergingVerifier::Verify(func->body);
    return func;
  }

private:
  class SimdMergingVerifier : public StmtExprVisitor {
  public:
    static void Verify(const Stmt &stmt) {
      SimdMergingVerifier verifier;
      verifier(stmt);
    }

  private:
    void VisitExpr_(const CallNode *call) final {
      if (IsSimdMergingCall(call)) {
        ICHECK(call->dtype.is_void())
            << "MODE_MERGING is a read-modify-write operation and requires "
               "assignment to a mutable SIMD register, for example "
               "`dst[i] = T.simd.vadds(...)`.";
        ICHECK(!call->args.empty());
        const auto *access_ptr = call->args[0].as<CallNode>();
        ICHECK(access_ptr && access_ptr->op.same_as(tl::access_ptr()) &&
               access_ptr->args.size() == 3)
            << "Legalized MODE_MERGING calls must carry an explicit "
               "tl.access_ptr destination.";
        const auto *rw_mask = access_ptr->args[2].as<IntImmNode>();
        ICHECK(rw_mask && rw_mask->value == kReadWriteAccessMask)
            << "MODE_MERGING destination must be marked read-write.";
      }
      StmtExprVisitor::VisitExpr_(call);
    }
  };

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    const auto *call = store->value.as<CallNode>();
    if (call == nullptr || !IsSimdMergingCall(call)) {
      return store;
    }

    // Inactive lanes read the old value from the assignment destination.
    const BufferLoad destination(store->buffer, store->indices);
    const DataType index_dtype =
        store->indices.empty() ? DataType::Int(32) : store->indices[0].dtype();
    const PrimExpr destination_ptr =
        Call(DataType::Handle(), tl::access_ptr(),
             {destination, IntImm(index_dtype, 1),
              IntImm(DataType::Int(32), kReadWriteAccessMask)});

    ffi::Array<PrimExpr> args;
    args.reserve(call->args.size() + 1);
    args.push_back(destination_ptr);
    for (const PrimExpr &arg : call->args) {
      args.push_back(arg);
    }

    Stmt result = Evaluate(Call(DataType::Void(), call->op, std::move(args),
                                call->annotations, call->span),
                           store->span);
    if (store->predicate.defined()) {
      result = IfThenElse(store->predicate.value(), std::move(result));
    }
    return result;
  }
};

} // namespace

tvm::transform::Pass LegalizeSimdMerging() {
  auto pass_func = [](PrimFunc func, const IRModule &, const PassContext &) {
    return SimdMergingLegalizer::Rewrite(std::move(func));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LegalizeSimdMerging", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeSimdMerging",
                        LegalizeSimdMerging);
}

} // namespace tl
} // namespace tvm

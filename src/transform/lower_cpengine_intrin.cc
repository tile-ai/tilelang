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

/*!
 * \file lower cpengine intrin.cc
 * \brief Lower cpengine intrinsics
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/distributed.h"

namespace tvm {
namespace tl {

using namespace tir;

#if (CUDA_MAJOR_VERSION >= 12)
class LowerCpengineIntrin : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    LowerCpengineIntrin substituter;
    fptr->body = substituter.VisitStmt(f->body);
    for (auto call : substituter.cpengine_calls_) {
      fptr->body = SeqStmt({call, fptr->body});
    }
    return f;
  }

  PrimExpr VisitExpr_(const CallNode *call) final {
    if (call->op.same_as(CpengineCpAsync())) {
      LOG(INFO) << "call CpengineCpAsync";
      cpengine_calls_.push_back(
          Evaluate(Call(DataType::Handle(), builtin::tvm_call_packed(),
                        {StringImm("tvm_cpengine_cp_async")})));
      return 0;
    } else {
      return StmtExprMutator::VisitExpr_(call);
    }
  }

private:
  LowerCpengineIntrin() = default;
  Array<Stmt> cpengine_calls_;
};

using namespace tir::transform;

tvm::transform::Pass LowerCpengineIntrin() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerCpengineIntrin::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerCpengineIntrin", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerCpengineIntrin",
                        LowerCpengineIntrin);
}
#endif // (CUDA_MAJOR_VERSION >= 12)

} // namespace tl
} // namespace tvm

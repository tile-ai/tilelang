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
 * \file lower_stmt_group_block.cc
 */

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Remove Block to ensure that the TIR can not be scheduled again.
 */
class StmtGroupBlockLower : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body) {
    return StmtGroupBlockLower()(std::move(body));
  }

private:
  Stmt VisitStmt_(const BlockNode *op) final {
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    if (block->annotations.count("stmt_group")) {
      return block->body;
    }
    return block;
  }
};

PrimFunc LowerStmtGroupBlock(PrimFunc f) {
  auto fptr = f.CopyOnWrite();
  fptr->body = StmtGroupBlockLower::Rewrite(std::move(fptr->body));
  return f;
}

namespace transform {

using namespace tir::transform;

Pass LowerStmtGroupBlock() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return tl::LowerStmtGroupBlock(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerStmtGroupBlock", {});
}

TVM_FFI_REGISTER_GLOBAL("tl.transform.LowerStmtGroupBlock")
    .set_body_typed(LowerStmtGroupBlock);
} // namespace transform

} // namespace tl
} // namespace tvm

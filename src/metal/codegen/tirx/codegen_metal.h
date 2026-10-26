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
 * \file codegen_metal.h
 * \brief Generate Metal device code.
 */
#ifndef TVM_TARGET_METAL_CODEGEN_METAL_H_
#define TVM_TARGET_METAL_CODEGEN_METAL_H_

#include "backend/common/codegen/attr.h"
#include <tvm/runtime/memory_scope.h>
#include <tvm/sym/analyzer.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "backend/common/codegen/tirx/codegen_c.h"

namespace tvm::tl::codegen_impl {
namespace attr = tvm::tl::codegen_attr;

class CodeGenMetal final : public CodeGenC {
public:
  explicit CodeGenMetal(Target target);
  // override print thread tag.
  void PrintArgUnionDecl();
  void AddFunction(const GlobalVar &gvar, const Function &func) final;
  void InitFuncState(const Function &f) final;
  void PrintStorageScope(const runtime::MemoryScope &scope,
                         std::ostream &os) final;            // NOLINT(*)
  void PrintStorageSync(const CallNode *op) final;           // NOLINT(*)
  void PrintType(const PrimType &t, std::ostream &os) final; // NOLINT(*)
  void BindThreadIndex(const PrimVar &var,
                       const ffi::String &thread_tag) final; // NOLINT(*)
  // print load of single element
  void PrintVecElemLoad(const std::string &vec, const PrimType &t, int i,
                        std::ostream &os) final; // NOLINT(*)
  // print store of single element.
  void PrintVecElemStore(const std::string &vec, const PrimType &t, int i,
                         const std::string &value) final;
  void PrintVecBinaryOp(const std::string &operation,
                        const PrimType &result_type, PrimExpr left,
                        PrimExpr right, std::ostream &output) final;
  // overload visitor
  void Dispatch_(const BindNode *op) final; // NOLINT(*)
  void Dispatch_(const ForNode *node) final;
  void Dispatch_(const RegionStmtNode *node) final;
  void DispatchAllocTensor(const BindNode *op, const CallNode *buffer_call);
  void Dispatch_(const prim::SelectNode *op,
                 std::ostream &os) final; // NOLINT(*)
  void Dispatch_(const prim::BroadcastNode *op,
                 std::ostream &os) final;                         // NOLINT(*)
  void Dispatch_(const CallNode *op, std::ostream &os) final;     // NOLINT(*)
  void Dispatch_(const FloatImmNode *op, std::ostream &os) final; // NOLINT(*)
  void Dispatch_(const prim::CastNode *node, std::ostream &output) final;
  void Dispatch_(const prim::AddNode *node, std::ostream &output) final;
  void Dispatch_(const prim::MulNode *node, std::ostream &output) final;
  void Dispatch_(const prim::DivNode *node, std::ostream &output) final;
  void Dispatch_(const prim::FloorDivNode *node, std::ostream &output) final;
  void Dispatch_(const prim::RShiftNode *node, std::ostream &output) final;
  void Dispatch_(const prim::BitwiseAndNode *node, std::ostream &output) final;

  // reuse parent's function.
  using CodeGenC::PrintType;

private:
  struct CooperativeTensorState {
    std::string dtype;
    std::string operation;
    std::vector<std::string> tiles;
  };
  Var GetMatrixBufferVar(const Expr &data);
  void AnalyzeMetalFunction(const Function &function);
  void AllocateCooperativeTensor(const Var &buffer, size_t element_count);
  CooperativeTensorState &GetCooperativeTensor(const Var &buffer);
  void PrintCooperativeFill(const CallNode *node, std::ostream &output);
  void PrintCooperativeCopy(const CallNode *node, bool store,
                            std::ostream &output);
  void PrintCooperativeMultiply(const CallNode *node, std::ostream &output);
  std::string CooperativeElement(const Var &buffer, const PrimExpr &index,
                                 int elements_per_tile, int element);
  int64_t ConstantIndex(const Expr &value);
  bool TryPrintMlxCoordinate(const PrimExpr &value, std::ostream &output);
  sym::Analyzer analyzer_;
  std::unordered_map<const VarNode *, std::string> simdgroup_dtype_;
  std::unordered_map<const VarNode *, CooperativeTensorState>
      cooperative_tensors_;
  std::unordered_map<const VarNode *, Call> cooperative_accumulators_;
  std::unordered_set<const VarNode *> cooperative_inputs_;
  bool uses_cooperative_tensors_{false};
  bool needs_fragment_lane_vars_{false};
  bool emitted_cooperative_header_{false};
  int64_t total_threads_{0};
  ffi::Optional<PrimVar> block_indices_[2];
  int active_mlx_panel_{0};
  int thread_index_bits_{32};
  int thread_work_dim_{0};
  Target target_;
};
} // namespace tvm::tl::codegen_impl

#endif // TVM_TARGET_METAL_CODEGEN_METAL_H_

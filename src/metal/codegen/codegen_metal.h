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
 * \file metal/codegen/codegen_metal.h
 * \brief Generate Metal device code.
 */
#ifndef TILELANG_METAL_CODEGEN_CODEGEN_METAL_H_
#define TILELANG_METAL_CODEGEN_CODEGEN_METAL_H_

#include <tvm/target/codegen.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

class CodeGenTileLangMetal final : public CodeGenC {
public:
  explicit CodeGenTileLangMetal(Target target);
  // override print thread tag.
  void PrintArgUnionDecl();
  void AddFunction(const GlobalVar &gvar, const PrimFunc &func) final;
  void InitFuncState(const PrimFunc &f) final;
  void PrintStorageScope(const std::string &scope,
                         std::ostream &os) final;     // NOLINT(*)
  void PrintStorageSync(const CallNode *op) final;    // NOLINT(*)
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void BindThreadIndex(const IterVar &iv) final;      // NOLINT(*)
  void VisitExpr_(const BufferLoadNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitStmt_(const BufferStoreNode *op) final;                  // NOLINT(*)
  // print load of single element
  void PrintVecElemLoad(const std::string &vec, DataType t, int i,
                        std::ostream &os) final; // NOLINT(*)
  // print store of single element.
  void PrintVecElemStore(const std::string &vec, DataType t, int i,
                         const std::string &value) final;
  // overload visitor
  void VisitStmt_(const AllocBufferNode *op) final;                 // NOLINT(*)
  void VisitStmt_(const AttrStmtNode *op) final;                    // NOLINT(*)
  void VisitStmt_(const ForNode *op) final;                         // NOLINT(*)
  void VisitExpr_(const SelectNode *op, std::ostream &os) final;    // NOLINT(*)
  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const AddNode *op, std::ostream &os) final;       // NOLINT(*)
  void VisitExpr_(const CastNode *op, std::ostream &os) final;      // NOLINT(*)
  void VisitExpr_(const CallNode *op, std::ostream &os) final;      // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;  // NOLINT(*)

  // reuse parent's function.
  using CodeGenC::PrintType;

private:
  std::string GetAddrSpaceOf(const PrimExpr &ptr_expr) const;
  std::string GetPointeeTypeOf(const PrimExpr &ptr_expr,
                               const std::string &fallback);
  bool IsThreadIdxXExpr(const PrimExpr &expr) const;
  bool IsBlockIdxExpr(const PrimExpr &expr, int dim) const;
  bool IsConstIntExpr(const PrimExpr &expr, int64_t value) const;
  bool IsMlxPanelRemainderExpr(const PrimExpr &expr) const;
  bool IsMlxPanelRowExpr(const PrimExpr &expr) const;
  bool IsMlxLogicalBlockXExpr(const PrimExpr &expr) const;
  bool IsMlxLogicalBlockYExpr(const PrimExpr &expr) const;
  bool TryPrintMlxLogicalYAffineExpr(const PrimExpr &expr,
                                     std::ostream &os);
  bool TryPrintMlxSwizzleExpr(const PrimExpr &expr, std::ostream &os);
  bool TryPrintSimdgroupIndexExpr(const CallNode *op, std::ostream &os);
  void PrintSimdgroupIndexExpr(int64_t group_mask, int64_t group_shift,
                               std::ostream &os) const;
  void EnsureCooperativeTensorBuffer(const Var &var);

  std::unordered_map<Var, std::string, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      simdgroup_dtype_;
  std::unordered_map<Var, std::string, ffi::ObjectPtrHash,
                     ffi::ObjectPtrEqual>
      cooperative_tensor_dtype_;
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      ct_c_inlined_;
  std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
      ct_c_storage_elided_;
  Var thread_idx_x_var_;
  Var block_idx_x_var_;
  Var block_idx_y_var_;
  int active_mlx_swizzle_panel_{0};
  int active_mlx_swizzle_log_{0};
  bool emitted_metal_simdgroup_id_{false};
  bool emitted_frag_lane_vars_{false};
  int thread_index_bits_{32};
  int thread_work_dim_{0};
  Target target_;
};
} // namespace codegen
} // namespace tvm

#endif // TILELANG_METAL_CODEGEN_CODEGEN_METAL_H_

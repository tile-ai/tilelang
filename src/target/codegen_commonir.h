/*!
 * \file target/codegen.h
 * \brief Utility to generate code
 */

#ifndef TVM_TL_TARGET_CODEGEN_COMMONIR_H_
#define TVM_TL_TARGET_CODEGEN_COMMONIR_H_

#include "../op/operator.h"
#include "target/source/codegen_c.h"
#include <tvm/target/codegen.h>
#include <tvm/tir/expr.h>

#include <assert.h>
#include <string>
#include <unordered_map>

namespace tvm {
namespace codegen {

using ffi::String;
using ffi::Array;

class SSAType {
public:
  String type_str = "";
  String var_id = "";

  virtual std::string printType() = 0;
};

class Scalar : public SSAType {
public:
  Scalar(String name, String type) {
    this->var_id = name;
    this->type_str = type;
  }

  std::string printType() { return type_str; }
};

class Memref : public SSAType {
  void GetIntStride();

public:
  Memref(String name, Buffer buffer, bool is_arg = false);
  Memref(String name, Array<PrimExpr> shape_in, DataType dtype_in,
         String address_space, bool var_offset_in,
         Array<PrimExpr> stride_in = Array<PrimExpr>(), int offset_in = 0,
         bool is_arg_in = false);
  std::string printType() { return type_str; }
  int dim;
  Array<PrimExpr> shape;
  Array<PrimExpr> stride;
  std::vector<unsigned long> stride_int;
  unsigned long offset = 0;
  bool var_offset = false;
  bool is_arg = false;
  DataType dtype;
  String address_space = "gm";
};

class Tensor : public SSAType {
  void GetIntStride();

public:
  Tensor(String name, Buffer buffer);
  Tensor(String name, Array<PrimExpr> shape_in, DataType dtype_in,
         String address_space);
  Tensor(String name, Memref memref);
  std::string printType() { return type_str; }
  int dim;
  Array<PrimExpr> shape;
  DataType dtype;
  String address_space = "gm";
};

class CodeGenTileLangCOMMONIR final : public CodeGenC {
public:
  CodeGenTileLangCOMMONIR();
  std::string Finish();

  // override behavior
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &f);
  void PrintStorageScope(const std::string &scope,
                         std::ostream &os) final;     // NOLINT(*)
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void PrintShape(Array<PrimExpr> shape, std::string delimiter,
                  std::ostream &os); // Added function
  void PrintSSAAssign(const std::string &target, const std::string &src,
                      DataType t) final;

  // overload visitor
  void VisitExpr_(const MinNode *op, std::ostream &os) final;
  void VisitExpr_(const MaxNode *op, std::ostream &os) final;
  void VisitExpr_(const AddNode *op, std::ostream &os) final;
  void VisitExpr_(const AndNode *op, std::ostream &os) final;
  void VisitExpr_(const OrNode *op, std::ostream &os) final;
  void VisitExpr_(const SubNode *op, std::ostream &os) final;
  void VisitExpr_(const MulNode *op, std::ostream &os) final;
  void VisitExpr_(const DivNode *op, std::ostream &os) final;
  void VisitExpr_(const LTNode *op, std::ostream &os) final;
  void VisitExpr_(const LENode *op, std::ostream &os) final;
  void VisitExpr_(const NENode *op, std::ostream &os) final;
  void VisitExpr_(const EQNode *op, std::ostream &os) final;
  void VisitExpr_(const GTNode *op, std::ostream &os) final;
  void VisitExpr_(const GENode *op, std::ostream &os) final;
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const IntImmNode *op, std::ostream &os) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void VisitExpr_(const FloorDivNode *op, std::ostream &os);
  void VisitExpr_(const FloorModNode *op, std::ostream &os);
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  void VisitExpr_(const SelectNode *op, std::ostream &os) final;
  void VisitExpr_(const BufferLoadNode *op, std::ostream &os) final;

  void VisitStmt_(const BufferStoreNode *op) final;
  void VisitStmt_(const AllocateNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitStmt_(const LetStmtNode *op) final;
  void VisitStmt_(const ForNode *op) final;
  void VisitStmt_(const tir::IfThenElseNode *op) final;

  void AddFunction(const GlobalVar &gvar, const PrimFunc &f);

private:
  Array<String> GenConvertIndex(Array<PrimExpr> exprs);
  String GenSubviewFromRegion(const CallNode *region_node);
  String GenSubviewFromRegion(Buffer buffer_data, Array<Range> range);
  void GenRecastFromArg(Buffer curr_buffer, String arg_name,
                        String &recast_inst);
  String GetMemrefInfo(String name);
  String GetMemrefInfo(Memref *memrefObj);
  String GetTensorInfo(String name);
  String GetTensorInfo(Tensor *tensorObj);
  String CreateMemrefToTensor(String src_data_name);
  String CastTensorToTensor(String src_data_name, DataType dtype_in);
  String CreateNewTensor(String src_data_name, String input_data_name);

  void FillCodegen(const CallNode *op, std::ostream &os);
  void CopyCodegen(const CallNode *op, std::ostream &os);
  void GemmCodegen(const CallNode *op, std::ostream &os);

  // save memref name and type
  std::map<String, Memref *> type_info;
  std::map<String, Tensor *> type_info_tensor;

  // whether enable fp16
  bool enable_fp16_{false};
  // whether enable bf16
  bool enable_bf16_{false};
  // whether enable fp8
  bool enable_fp8_{false};
  // whether enable int8
  bool enable_int8_{false};

  std::vector<std::string> thread_context_args{6, ""};
  tvm::tl::BufferMap vmap{tvm::tl::BufferMap()};
};
} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_COMMONIR_H_
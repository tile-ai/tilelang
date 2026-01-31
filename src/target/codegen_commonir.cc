/*!
 * \file target/codegen.cc
 */

#include "codegen_commonir.h"
#include "../op/builtin.h"
#include "../op/region.h"
#include "../op/fill.h"
#include "../op/gemm.h"
#include "../op/copy.h"
#include "arith/pattern_match.h"
#include "tvm/ir/expr.h"
#include "tvm/runtime/data_type.h"
#include "tvm/tir/buffer.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/stmt.h"
#include <cassert>
#include <cmath>
#include <elf.h>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/index_map.h>
#include <tvm/ffi/container/array.h>
#include <tvm/tir/op.h>
#include <utility>
#include <vector>

namespace tvm {
namespace codegen {

using ffi::String;
using ffi::Array;

template <typename T>
inline void PrintBinary(const T *op, const char *opstr, std::ostream &os,
                        CodeGenC *CG) {
  auto PrintOp = [op, &os, CG](auto Operand) {
    std::ostringstream tmpos;
    CG->PrintExpr(Operand, tmpos << "%");
    return tmpos.str();
  };

  if (op->dtype.lanes() == 1) {
    // left op
    os << "arith." << opstr << " ";
    os << PrintOp(op->a);
    os << ", ";
    // right op
    os << PrintOp(op->b);
    os << " : ";
    CG->PrintType(op->a->dtype, os);
  } else {
    os << "<<<invalid-op-dtype-lanes-not-one: %" << opstr << ">>>\n";
  }
}

// for future use
String GetAddressSpace(String address_space) {
  if (address_space == "global")
    return "global";
  else if (address_space == "shared")
    return "shared";
  else if (address_space == "shared.dyn")
    return "shared";
  else if (address_space == "local.fragment")
    return "local";
  return "unknown";
}

bool IsEqual(Array<PrimExpr> a, Array<PrimExpr> b) {
  if (a.size() != b.size())
    return false;
  for (int i = 0; i < a.size(); i++) {
    if (!(a[i].same_as(b[i])))
      return false;
  }
  return true;
}

bool AllZero(Array<PrimExpr> a) {
  for (PrimExpr pe : a) {
    if (!is_zero(pe))
      return false;
  }
  return true;
}

std::vector<unsigned long> GetStrideFromShape(Array<tvm::PrimExpr> shape) {
  std::vector<unsigned long> strides;
  unsigned long total_size = 1;
  std::vector<int> shape_int;
  for (PrimExpr s : shape) {
    if (auto s_int = as_const_int(s)) {
      total_size *= *s_int;
      shape_int.push_back(*s_int);
    }
  }
  for (int i = 0; i < shape.size(); i++) {
    total_size /= shape_int[i];
    strides.push_back(total_size);
  }
  return strides;
}

String GetBufferStrides(Buffer buffer) {
  Array<PrimExpr> shape = buffer->shape;
  std::vector<unsigned long> strides;
  int dim = buffer->shape.size();
  if (buffer->strides.empty()) {
    strides = GetStrideFromShape(shape);
  } else {
    for (PrimExpr stride : buffer->strides) {
      if (auto stride_int = as_const_int(stride)) {
        strides.push_back(*stride_int);
      }
    }
  }
  String res = "[";
  for (int i = 0; i < dim; i++) {
    if (i > 0)
      res = res + ", ";
    res = res + std::to_string(strides[i]);
  }
  res = res + "]";
  return res;
}

static std::vector<int> getBroadcastDim(Array<PrimExpr> &buffer_shape0,
                                        Array<PrimExpr> &buffer_shape1) {
  assert(buffer_shape0.size() == buffer_shape1.size());
  std::vector<int> dims;
  for (int i = 0; i < buffer_shape0.size(); i++) {
    if (*as_const_int(buffer_shape0[i]) == 1 &&
        *as_const_int(buffer_shape1[i]) != 1) {
      dims.emplace_back(i);
    }
    if (*as_const_int(buffer_shape0[i]) != 1 &&
        *as_const_int(buffer_shape1[i]) == 1) {
      dims.emplace_back(i);
    }
    assert(*as_const_int(buffer_shape0[i]) == *as_const_int(buffer_shape1[i]));
  }
  return dims;
}

static std::string broadcastAttrCodegen(Array<PrimExpr> &buffer_shape0,
                                        Array<PrimExpr> &buffer_shape1) {
  if (buffer_shape0.empty() || buffer_shape1.empty()) {
    return "";
  }
  std::vector<int> broadcastDims;
  if (buffer_shape0.size() && buffer_shape1.size()) {
    broadcastDims = getBroadcastDim(buffer_shape0, buffer_shape1);
  }
  std::ostringstream temp;
  if (broadcastDims.size()) {
    temp << " = [";
    for (auto dim : broadcastDims) {
      temp << dim;
      if (dim != broadcastDims.back()) {
        temp << ", ";
      }
    }
    temp << "]";
  }
  return temp.str();
}


void PrintBufferMap(const Map<Var, Buffer> &buffer_map) {
  for (const auto &kv : buffer_map) {
    const Var &var = kv.first;
    const Buffer &buffer = kv.second;

    LOG(INFO) << "Var: " << var->name_hint << ", Buffer Name: " << buffer->name
              << ", Buffer Shape: " << buffer->shape
              << ", Buffer Dtype: " << buffer->dtype;
  }
}

std::string GetCastOp(DataType src_type, DataType dst_type) {
  bool srcIsFloat = src_type.is_float() || src_type.is_bfloat16();
  bool srcIsInt = src_type.is_int();
  bool srcIsUInt = src_type.is_uint();
  bool targetIsFloat = dst_type.is_float() || dst_type.is_bfloat16();
  bool targetIsInt = dst_type.is_int();
  bool targetIsUInt = dst_type.is_uint();
  if (srcIsFloat && targetIsInt) {
    return "arith.fptosi";
  } else if (srcIsFloat && targetIsUInt) {
    return "arith.fptoui";
  } else if (srcIsInt && targetIsFloat) {
    return "arith.sitofp";
  } else if (srcIsUInt && targetIsFloat) {
    return "arith.uitofp";
  } else if (targetIsInt) {
    if (dst_type.bits() > src_type.bits()) {
      return "arith.extsi";
    } else {
      return "arith.trunci";
    }
  } else if (targetIsUInt) {
    if (dst_type.bits() > src_type.bits()) {
      return "arith.extui";
    } else {
      return "arith.trunci";
    }
  } else if (targetIsFloat) {
    if (dst_type.bits() > src_type.bits()) {
      return "arith.extf";
    } else {
      return "arith.truncf";
    }
  }
}

CodeGenTileLangCOMMONIR::CodeGenTileLangCOMMONIR() {}

void CodeGenTileLangCOMMONIR::PrintFuncPrefix(std::ostream &os) {}

std::string CodeGenTileLangCOMMONIR::Finish() {
  std::ostringstream code;
  code << decl_stream.str();
  code << stream.str();
  return code.str();
}

void CodeGenTileLangCOMMONIR::VisitStmt_(const tir::ForNode *op) {
  if (op->kind == tir::ForKind::kParallel) {
    assert(op->extent.dtype().is_int() || op->extent.dtype().is_uint());
    assert(op->min.dtype() == op->extent.dtype());
    std::string upperBoundId =
        SSAGetID(PrintExpr(arith::Analyzer().Simplify(op->extent + op->min)),
                 op->extent->dtype);

    std::ostringstream temp;
    temp << "arith.index_cast %" << upperBoundId << ": ";
    PrintType(op->extent.dtype(), temp);
    temp << " to index";
    std::string upperBoundId_index = SSAGetID(temp.str(), op->extent->dtype);

    std::string lowerBoundId = SSAGetID(PrintExpr(op->min), op->min->dtype);
    temp.str("");
    temp.clear();
    temp << "arith.index_cast %" << lowerBoundId << ": ";
    PrintType(op->min.dtype(), temp);
    temp << " to index";
    std::string lowerBoundId_index = SSAGetID(temp.str(), op->min->dtype);

    auto stepNode = std::make_unique<IntImm>(op->min.dtype(), 1);
    auto stepId = SSAGetID(PrintExpr(*stepNode), stepNode->dtype());
    temp.str("");
    temp.clear();
    temp << "arith.index_cast %" << stepId << ": ";
    PrintType(op->min.dtype(), temp);
    temp << " to index";
    std::string stepId_index = SSAGetID(temp.str(), stepNode->dtype());

    PrintIndent();

    std::string vid =
        SSAGetID(AllocVarID(op->loop_var.get()), op->loop_var->dtype);
    stream << "scf.parallel"
           << " (%" << vid << "_index) = (%" << lowerBoundId_index << ") to (%"
           << upperBoundId_index << ") step (%" << stepId_index << ") ";
    stream << " {\n";

    int for_scope = BeginScope();
    PrintIndent();
    stream << "%" << vid << "= arith.index_cast %" << vid
           << "_index: index to ";
    PrintType(op->loop_var->dtype, stream);
    stream << "\n";
    PrintStmt(op->body);
    this->EndScope(for_scope);
    PrintIndent();
    stream << "}\n";
  } else if (op->kind == tir::ForKind::kSerial) {
    std::string upperBoundId =
        SSAGetID(PrintExpr(arith::Analyzer().Simplify(op->extent + op->min)),
                 op->extent->dtype);
    assert(op->extent.dtype().is_int() || op->extent.dtype().is_uint());
    assert(op->min.dtype() == op->extent.dtype());
    std::string vid =
        SSAGetID(AllocVarID(op->loop_var.get()), op->loop_var->dtype);
    std::string lowerBoundId = SSAGetID(PrintExpr(op->min), op->min->dtype);
    std::string extentId = SSAGetID(PrintExpr(op->extent), op->extent->dtype);
    auto stepNode = std::make_unique<IntImm>(op->min.dtype(), 1);
    auto stepId = SSAGetID(PrintExpr(*stepNode), stepNode->dtype());
    PrintIndent();
    stream << "scf.for"
           << " %" << vid << " = %" << lowerBoundId << " to %" << upperBoundId
           << " step %" << stepId << " : ";
    PrintType(op->min.dtype(), stream);
    stream << " {\n";
    int for_scope = BeginScope();
    PrintStmt(op->body);
    this->EndScope(for_scope);
    PrintIndent();
    stream << "}\n";
  } else {
    std::string upperBoundId =
        SSAGetID(PrintExpr(arith::Analyzer().Simplify(op->extent + op->min)),
                 op->extent->dtype);
    assert(op->extent.dtype().is_int() || op->extent.dtype().is_uint());
    assert(op->min.dtype() == op->extent.dtype());
    std::string vid =
        SSAGetID(AllocVarID(op->loop_var.get()), op->loop_var->dtype);
    std::string lowerBoundId = SSAGetID(PrintExpr(op->min), op->min->dtype);
    std::string extentId = SSAGetID(PrintExpr(op->extent), op->extent->dtype);
    auto stepNode = std::make_unique<IntImm>(op->min.dtype(), 1);
    auto stepId = SSAGetID(PrintExpr(*stepNode), stepNode->dtype());
    PrintIndent();
    stream << "scf.<<<invalid-for-type %" << ForKind2String(op->kind) << ">>>"
           << " %" << vid << " = %" << lowerBoundId << " to %" << upperBoundId
           << " step %" << stepId << " : ";
    PrintType(op->min.dtype(), stream);
    stream << " {\n";
    int for_scope = BeginScope();
    PrintStmt(op->body);
    this->EndScope(for_scope);
    PrintIndent();
    stream << "}\n";
  }
}

void CodeGenTileLangCOMMONIR::VisitStmt_(const tir::IfThenElseNode *op) {
  std::string cond = SSAGetID(PrintExpr(op->condition), op->condition->dtype);
  PrintIndent();
  stream << "scf.if %" << cond << " {\n";
  int then_scope = BeginScope();
  PrintStmt(op->then_case);
  this->EndScope(then_scope);
  if (op->else_case) {
    PrintIndent();
    stream << "} else {\n";
    int else_scope = BeginScope();
    PrintStmt(op->else_case.value());
    this->EndScope(else_scope);
  }
  PrintIndent();
  stream << "}\n";
}

void CodeGenTileLangCOMMONIR::PrintSSAAssign(const std::string &target,
                                             const std::string &src,
                                             DataType t) {
  stream << "%" << target << " = " << src << "\n";
}

void CodeGenTileLangCOMMONIR::PrintShape(Array<PrimExpr> shape,
                                         std::string delimiter,
                                         std::ostream &os) {
  for (size_t i = 0; i < shape.size(); i++) {
    if (i != 0)
      os << delimiter;
    os << shape[i];
  }
}

void CodeGenTileLangCOMMONIR::PrintType(DataType t,
                                        std::ostream &os) { // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    // ICHECK(t.is_scalar()) << "do not yet support vector types";
    // os << "void*";
    return;
  }

  if (t.is_void()) {
    //    os << "void";
    return;
  }

  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
    case 16:
      enable_fp16_ = true;
      if (t.is_scalar()) {
        os << "f16";
      } else {
        fail = true;
      }
      break;
    case 32:
      os << "f32";
      break;
    case 64:
      os << "f64";
      break;
    default:
      fail = true;
      break;
    }
    if (!fail && (t.is_scalar() || t.bits() == 16))
      return;
  } else if (t.is_bfloat16()) {
    enable_bf16_ = true;
    if (t.is_scalar()) {
      os << "bf16";
    } else {
      fail = true;
    }
    if (!fail)
      return;
  } else if (t == DataType::Bool()) {
    os << "i1";
    return;
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << "u";
    }
    switch (t.bits()) {
    case 1: {
      if (t.is_scalar()) {
        os << "i1";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t;
      }
    }
    case 4: {
      if (t.is_scalar()) {
        os << "i4";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t;
      }
    }
    case 8: {
      if (t.is_scalar()) {
        os << "i8";
        return;
      } else {
        LOG(FATAL) << "Cannot convert type " << t;
      }
    }
    case 16: {
      if (t.is_scalar()) {
        os << "i16";
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 32: {
      if (t.is_scalar()) {
        os << "i32";
      } else {
        fail = true;
      }
      if (!fail) {
        return;
      }
      break;
    }
    case 64: {
      if (t.is_scalar()) {
        os << "i64";
      }
      return;
    }
    default:
      fail = true;
      break;
    }
    if (!fail) {
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t;
}

void CodeGenTileLangCOMMONIR::PrintStorageScope(const std::string &scope,
                                                std::ostream &os) { // NOLINT(*)
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const FloorDivNode *op,
                                         std::ostream &os) {
  // FIXME: The floor div in python is not the same as arith.divsi in negative
  // scenarios.
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "divsi", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "divf", os, this);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const FloorModNode *op,
                                         std::ostream &os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "remsi", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "remf", os, this);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const LTNode *op, std::ostream &os) {
  if (op->a->dtype.is_int()) {
    PrintBinary(op, "cmpi slt,", os, this);
  } else if (op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi ult,", os, this);
  } else {
    PrintBinary(op, "cmpf olt,", os, this);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const NENode *op, std::ostream &os) {
  if (op->a->dtype.is_int() || op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi ne,", os, this);
  } else {
    PrintBinary(op, "cmpf one,", os, this);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const EQNode *op, std::ostream &os) {
  if (op->a->dtype.is_int() || op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi eq,", os, this);
  } else {
    PrintBinary(op, "cmpf oeq,", os, this);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const LENode *op, std::ostream &os) {
  if (op->a->dtype.is_int()) {
    PrintBinary(op, "cmpi sle,", os, this);
  } else if (op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi ule,", os, this);
  } else {
    PrintBinary(op, "cmpf ole,", os, this);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const GENode *op, std::ostream &os) {
  if (op->a->dtype.is_int()) {
    PrintBinary(op, "cmpi sge,", os, this);
  } else if (op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi uge,", os, this);
  } else {
    PrintBinary(op, "cmpf oge,", os, this);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const GTNode *op, std::ostream &os) {
  if (op->a->dtype.is_int()) {
    PrintBinary(op, "cmpi sgt,", os, this);
  } else if (op->a->dtype.is_uint()) {
    PrintBinary(op, "cmpi ugt,", os, this);
  } else {
    PrintBinary(op, "cmpf ogt,", os, this);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const CastNode *op, std::ostream &os) {
  bool srcIsFloat =
      op->value->dtype.is_float() || op->value->dtype.is_bfloat16();
  bool srcIsInt = op->value->dtype.is_int();
  bool srcIsUInt = op->value->dtype.is_uint();
  bool targetIsFloat = op->dtype.is_float() || op->dtype.is_bfloat16();
  bool targetIsInt = op->dtype.is_int();
  bool targetIsUInt = op->dtype.is_uint();
  auto val = PrintExpr(op->value);
  if (srcIsFloat && targetIsInt) {
    os << "arith.fptosi \%" << val << " : ";
  } else if (srcIsFloat && targetIsUInt) {
    os << "arith.fptoui \%" << val << " : ";
  } else if (srcIsInt && targetIsFloat) {
    os << "arith.sitofp \%" << val << " : ";
  } else if (srcIsUInt && targetIsFloat) {
    os << "arith.uitofp \%" << val << " : ";
  } else if (targetIsInt) {
    if (op->dtype.bits() > op->value->dtype.bits()) {
      os << "arith.extsi \%" << val << " : ";
    } else {
      os << "arith.trunci \%" << val << " : ";
    }
  } else if (targetIsUInt) {
    if (op->dtype.bits() > op->value->dtype.bits()) {
      os << "arith.extui \%" << val << " : ";
    } else {
      os << "arith.trunci \%" << val << " : ";
    }
  } else if (targetIsFloat) {
    if (op->dtype.bits() > op->value->dtype.bits()) {
      os << "arith.extf \%" << val << " : ";
    } else {
      os << "arith.truncf \%" << val << " : ";
    }
  }
  PrintType(op->value->dtype, os);
  os << " to ";
  PrintType(op->dtype, os);
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const BufferLoadNode *op,
                                         std::ostream &os) {

  std::ostringstream temp;
  Buffer buffer_data = op->buffer;

  DataType buffer_type = buffer_data->dtype;
  String buffer_name = buffer_data->name;
  Array<PrimExpr> buffer_shape = buffer_data->shape;
  int dim = buffer_shape.size();

  String buffer_name_val = "";
  if (auto memrefInfo = dynamic_cast<Memref *>(type_info[buffer_name])) {
    if (memrefInfo->is_arg) {
      buffer_name_val = buffer_name + "_Recast";
    } else {
      buffer_name_val = buffer_name;
    }
  } else {
    LOG(FATAL) << buffer_name << " should be a memref";
  }

  Array<String> cast_index_array = GenConvertIndex(op->indices);
  temp << "memref.load  \%" + buffer_name_val;
  temp << "[";
  for (int i = 0; i < dim; i++) {
    if (i > 0) {
      temp << ", ";
    }
    temp << cast_index_array[i];
  }
  temp << "] :";
  String data_info = GetMemrefInfo(buffer_name_val);
  temp << data_info;
  os << SSAGetID(temp.str(), buffer_type);
}

Array<String> CodeGenTileLangCOMMONIR::GenConvertIndex(Array<PrimExpr> exprs) {
  Array<String> cast_array;
  for (PrimExpr curr_expr : exprs) {
    if (auto curr_expr_int = curr_expr.as<IntImmNode>()) {
      cast_array.push_back(std::to_string(curr_expr_int->value));
    } else {
      DataType indice_type = curr_expr->dtype;
      std::ostringstream temp;
      std::string var_name;
      if (!curr_expr.as<VarNode>()) {
        var_name = SSAGetID(PrintExpr(curr_expr), indice_type);
      } else {
        var_name = PrintExpr(curr_expr);
      }
      temp << "arith.index_cast \%" << var_name << " : ";
      PrintType(indice_type, temp);
      temp << " to index";
      String cast_indice_name = "\%" + SSAGetID(temp.str(), indice_type);
      cast_array.push_back(cast_indice_name);
    }
  }
  return cast_array;
}

unsigned long ComputeOffset(Memref *src_buffer, Array<PrimExpr> op_offset) {
  if (src_buffer->var_offset)
    return -1;
  if (src_buffer->stride_int.size() != src_buffer->dim)
    return -1;
  unsigned long offset = src_buffer->offset;
  for (int i = 0; i < src_buffer->dim; i++) {
    const int64_t *op_off = as_const_int(op_offset[i]);
    if (op_off == nullptr)
      return -1;
    offset += (*op_off) * src_buffer->stride_int[i];
  }
  return offset;
}

String
CodeGenTileLangCOMMONIR::GenSubviewFromRegion(const CallNode *region_node) {
  tvm::tl::RegionOp regionop(region_node->args);
  return GenSubviewFromRegion(regionop->GetBuffer(), regionop->GetRanges());
}

String CodeGenTileLangCOMMONIR::GenSubviewFromRegion(Buffer buffer_data,
                                                     Array<Range> range) {
  std::ostringstream temp;
  DataType buffer_type = buffer_data->dtype;
  String buffer_name = buffer_data->name;
  Array<PrimExpr> buffer_shape = buffer_data->shape;
  int dim = buffer_shape.size();
  Array<PrimExpr> region_shape, region_indeces;
  for (Range r : range) {
    region_shape.push_back(r.get()->extent);
    region_indeces.push_back(r.get()->min);
  }
  String buffer_name_val = "";
  if (auto memrefInfo = dynamic_cast<Memref *>(type_info[buffer_name])) {
    if (memrefInfo->is_arg) {
      buffer_name_val = buffer_name + "_Recast";
    } else {
      buffer_name_val = buffer_name;
    }
  } else {
    LOG(FATAL) << buffer_name << " should be a memref";
  }
  String new_buffer_name = buffer_name_val;
  String src_data_info = GetMemrefInfo(buffer_name_val);
  if (!(IsEqual(buffer_shape, region_shape) && AllZero(region_indeces))) {
    Array<String> cast_offset_array = GenConvertIndex(region_indeces);
    Array<String> cast_shape_array = GenConvertIndex(region_shape);
    if (!dynamic_cast<Memref *>(type_info[buffer_name_val])) {
      LOG(FATAL) << buffer_name_val << " should be a memref";
    }
    unsigned long offset = ComputeOffset(
        dynamic_cast<Memref *>(type_info[buffer_name_val]), region_indeces);
    new_buffer_name = buffer_name_val + "_subview";
    auto tempMemref = new Memref(
        new_buffer_name, region_shape, buffer_type,
        dynamic_cast<Memref *>(type_info[buffer_name_val])->address_space,
        offset == -1,
        dynamic_cast<Memref *>(type_info[buffer_name_val])->stride, offset);
    String dst_data_info = GetMemrefInfo(tempMemref);
    temp << "memref.subview \%" + buffer_name_val;
    temp << "[";
    for (int i = 0; i < dim; i++) {
      if (i > 0) {
        temp << ", ";
      }
      temp << cast_offset_array[i];
    }
    temp << "]";
    temp << "[";
    for (int i = 0; i < dim; i++) {
      if (i > 0) {
        temp << ", ";
      }
      temp << cast_shape_array[i];
    }
    temp << "]";
    temp << "[";
    for (int i = 0; i < dim; i++) {
      if (i > 0) {
        temp << ", ";
      }
      temp << "1";
    }
    temp << "]";
    temp << " : ";
    temp << src_data_info;
    temp << " to ";
    temp << dst_data_info;
    delete tempMemref;
    new_buffer_name = SSAGetID(temp.str(), buffer_type);
    this->type_info[new_buffer_name] = new Memref(
        new_buffer_name, region_shape, buffer_type,
        dynamic_cast<Memref *>(type_info[buffer_name_val])->address_space,
        offset == -1,
        dynamic_cast<Memref *>(type_info[buffer_name_val])->stride, offset);
  }
  return new_buffer_name;
}


String CodeGenTileLangCOMMONIR::CreateMemrefToTensor(String src_data_name) {
  if (!dynamic_cast<Memref *>(type_info[src_data_name])) {
    LOG(FATAL) << src_data_name << " should be a memref";
  }
  Memref *src_memref = dynamic_cast<Memref *>(type_info[src_data_name]);
  DataType src_dtype = src_memref->dtype;
  String new_tensor_name = src_data_name + "_buffer";
  auto tempTensor = new Tensor(new_tensor_name, *src_memref);
  std::ostringstream temp;
  temp << "bufferization.to_tensor %" << src_data_name
       << " restrict writable : " << GetMemrefInfo(src_data_name);
  temp <<  " to " << GetTensorInfo(tempTensor);
  new_tensor_name = SSAGetID(temp.str(), src_dtype);
  tempTensor->var_id = new_tensor_name;
  this->type_info_tensor[new_tensor_name] = tempTensor;
  
  return new_tensor_name;
}

String CodeGenTileLangCOMMONIR::CastTensorToTensor(String src_data_name,
                                                   DataType dtype_in) {
  if (!dynamic_cast<Tensor *>(type_info_tensor[src_data_name])) {
    LOG(FATAL) << src_data_name << " should be a tensor";
  }

  Tensor *src_tensor = dynamic_cast<Tensor *>(type_info_tensor[src_data_name]);
  DataType src_dtype = src_tensor->dtype;

  if (src_dtype == dtype_in) {
    return src_data_name;
  }

  String cast_tensor_name = src_data_name + "_cast";
  auto tempTensor = new Tensor(cast_tensor_name, src_tensor->shape, dtype_in,
                               src_tensor->address_space);

  std::ostringstream temp;
  temp << GetCastOp(src_dtype, dtype_in) << " %" << src_data_name << " : ";
  temp << GetTensorInfo(src_data_name) << " to ";
  temp << GetTensorInfo(tempTensor);

  cast_tensor_name = SSAGetID(temp.str(), dtype_in);
  tempTensor->var_id = cast_tensor_name;
  this->type_info_tensor[cast_tensor_name] = tempTensor;

  return cast_tensor_name;
}

String CodeGenTileLangCOMMONIR::CreateNewTensor(String src_data_name,
                                                String input_data_name) {
  if (!dynamic_cast<Memref *>(type_info[src_data_name])) {
    LOG(FATAL) << src_data_name << " should be a memref";
  }
  String new_tensor_name = input_data_name + "_local_tensor_tmp";
  auto tempTensor = new Tensor(
      new_tensor_name, *(dynamic_cast<Memref *>(type_info[src_data_name])));
  std::ostringstream temp;
  temp << "tensor.empty() :" << GetTensorInfo(tempTensor);
  new_tensor_name = SSAGetID(temp.str(), tempTensor->dtype);
  tempTensor->var_id = new_tensor_name;
  this->type_info_tensor[new_tensor_name] = tempTensor;
  return new_tensor_name;
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const CallNode *op, std::ostream &os) {
  if (op->op.same_as(Op::Get("tl.tileop.fill"))) {
    FillCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.tileop.copy"))) {
    CopyCodegen(op, os);
  } else if (op->op.same_as(Op::Get("tl.tileop.gemm")) || op->op.same_as(Op::Get("tl.tileop.gemm_py"))) {
    GemmCodegen(op, os);
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

void CodeGenTileLangCOMMONIR::FillCodegen(const CallNode *op,
                                          std::ostream &os) {
  tvm::tl::Fill fillop(op->args);
  std::string value_name =
      SSAGetID(PrintExpr(fillop->value), fillop->value->dtype);

  this->PrintIndent();
  this->stream << "linalg.fill ins(%" << value_name << " : ";
  PrintType(fillop->value->dtype, this->stream);
  this->stream << ") outs(%" << fillop->dst.get()->name << " : ";
  this->stream << GetMemrefInfo(fillop->dst.get()->name) << ")\n";
}

void CodeGenTileLangCOMMONIR::CopyCodegen(const CallNode *op,
                                          std::ostream &os) {
  tvm::tl::Copy copyop(op->args);

  String src_data_name = GenSubviewFromRegion(copyop->src, copyop->src_range);
  String dst_data_name = GenSubviewFromRegion(copyop->dst, copyop->dst_range);

  if (!dynamic_cast<Memref *>(type_info[src_data_name])) {
    LOG(FATAL) << src_data_name << " should be a memref";
  }
  if (!dynamic_cast<Memref *>(type_info[dst_data_name])) {
    LOG(FATAL) << dst_data_name << " should be a memref";
  }

  DataType src_dtype = dynamic_cast<Memref *>(type_info[src_data_name])->dtype;
  DataType dst_dtype = dynamic_cast<Memref *>(type_info[dst_data_name])->dtype;
  if (src_dtype == dst_dtype) {
    this->PrintIndent();
    this->stream << "memref.copy"
                 << " \%" << src_data_name << ", "
                 << "\%" << dst_data_name << " : ";
    this->stream << GetMemrefInfo(src_data_name) << " to "
                 << GetMemrefInfo(dst_data_name) << "\n";
  } else {
    LOG(INFO) << "memref.copy: src_dtype: " << src_dtype
              << " != dst_dtype: " << dst_dtype;

    std::ostringstream temp;

    String new_tensor_name = CreateMemrefToTensor(src_data_name);
    String cast_tensor_name = CastTensorToTensor(new_tensor_name, dst_dtype);

    this->PrintIndent();
    this->stream << "bufferization.materialize_in_destination \%";
    this->stream << cast_tensor_name << " in writable  \%" << dst_data_name
                 << " : (";
    this->stream << GetTensorInfo(cast_tensor_name) << " , "
                 << GetMemrefInfo(dst_data_name) << ") -> ()";
    this->stream << "\n";
  }
}

void CodeGenTileLangCOMMONIR::GemmCodegen(const CallNode *op,
                                          std::ostream &os) {
  tvm::tl::Gemm gemmop(op->args);
  // todo(dkx): support transpose indexing_maps
  ICHECK(!gemmop->transA_) << "Currently we only support: transA_ must be false";
  ICHECK(!gemmop->transB_) << "Currently we only support: transB_ must be false";
  // todo(dkx): support clearAccum_ = True
  ICHECK(is_zero(gemmop->clearAccum_))
      << "Currently we only support: clearAccum_ must be const_false";
  // todo(dkx): maybe not necessary
  // ICHECK(gemmop->policy_ == tvm::tl::GemmWarpPolicyType::kSquare)
  //     << "Currently we only support: policy_ must be kSquare";
  ICHECK(gemmop->kPack_ == 1) << "Currently we only support: kPack_ must be 1";
  ICHECK(gemmop->wgWait_ == 0) << "Currently we only support: wgWait_ must be 0";

  Buffer a_buffer = gemmop->a_;
  Buffer b_buffer = gemmop->b_;
  Buffer c_buffer = gemmop->c_;
  String a_buffer_name = a_buffer->name;
  String b_buffer_name = b_buffer->name;
  String c_buffer_name = c_buffer->name;
  String A_shared_tensor = CreateMemrefToTensor(a_buffer_name);
  String B_shared_tensor = CreateMemrefToTensor(b_buffer_name);
  String C_shared_tensor = CreateMemrefToTensor(c_buffer_name);
  String new_tensor_name = CreateNewTensor(c_buffer_name, C_shared_tensor);
  std::ostringstream temp;
  DataType dst_dtype = this->type_info_tensor[new_tensor_name]->dtype;
  temp << "linalg.matmul ins(\%" << A_shared_tensor << ", \%" << B_shared_tensor
       << " : " << GetTensorInfo(A_shared_tensor) << ", "
       << GetTensorInfo(B_shared_tensor) << ") ";
  temp << "outs(\%" << new_tensor_name << " : "
       << GetTensorInfo(new_tensor_name) << ") -> "
       << GetTensorInfo(new_tensor_name);
  String C_tensor_out = SSAGetID(temp.str(), dst_dtype);
  temp.str("");
  temp.clear();
  if (dst_dtype.is_int() || dst_dtype.is_uint()) {
    temp << "arith.addi ";
  } else if (dst_dtype.is_float()) {
    temp << "arith.addf ";
  }
  temp << "\%" << C_shared_tensor << ", \%" << C_tensor_out << " : "
       << GetTensorInfo(C_shared_tensor);
  String tmp_out = SSAGetID(temp.str(), dst_dtype);
  this->PrintIndent();
  this->stream << "bufferization.materialize_in_destination %" << tmp_out
               << " in writable %" << c_buffer_name << " : ("
               << GetTensorInfo(new_tensor_name) << ", "
               << GetMemrefInfo(c_buffer_name) << ") -> ()\n";
}

void CodeGenTileLangCOMMONIR::VisitStmt_(const LetStmtNode *op) {
  std::string value = PrintExpr(op->value);
  PrintIndent();
  this->stream << '%' << AllocVarID(op->var.get()) << " = " << value << "\n";
  PrintStmt(op->body);
}

void CodeGenTileLangCOMMONIR::VisitStmt_(const BufferStoreNode *op) {
  std::string value = SSAGetID(PrintExpr(op->value), op->value->dtype);

  PrintIndent();

  Buffer buffer_data = op->buffer;
  DataType buffer_type = buffer_data->dtype;
  String buffer_name = buffer_data->name;
  Array<PrimExpr> buffer_shape = buffer_data->shape;
  int dim = buffer_shape.size();

  String buffer_name_val = "";
  if (auto memrefInfo = dynamic_cast<Memref *>(type_info[buffer_name])) {
    if (memrefInfo->is_arg) {
      buffer_name_val = buffer_name + "_Recast";
    } else {
      buffer_name_val = buffer_name;
    }
  } else {
    LOG(FATAL) << buffer_name << " should be a memref";
  }

  Array<String> cast_index_array = GenConvertIndex(op->indices);
  this->stream << "memref.store  \%" + value + ", \%" + buffer_name_val;
  this->stream << "[";
  for (int i = 0; i < dim; i++) {
    if (i > 0) {
      this->stream << ", ";
    }
    this->stream << cast_index_array[i];
  }
  this->stream << "] :";

  String data_info = GetMemrefInfo(buffer_name_val);
  this->stream << data_info << "\n";
}

void CodeGenTileLangCOMMONIR::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == "thread_extent") {
    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag == "blockIdx.x" && iv->var->name_hint != "_") {

      std::ostringstream temp;
      temp << "arith.constant 0"
           << " : ";
      PrintType(iv->var->dtype, temp);
      std::string vid = SSAGetID(temp.str(), iv->var->dtype);

      auto block_id_ = AllocVarID(iv->var.get());
      this->PrintIndent();
      this->stream << "%" << block_id_ << " = arith.addi  %" << vid << ", "
                   << this->thread_context_args[3] << ": i32\n";
    } else if (iv->thread_tag == "blockIdx.y" && iv->var->name_hint != "_") {
      std::ostringstream temp;
      temp << "arith.constant 0"
           << " : ";
      PrintType(iv->var->dtype, temp);
      std::string vid = SSAGetID(temp.str(), iv->var->dtype);

      auto block_id_ = AllocVarID(iv->var.get());
      this->PrintIndent();
      this->stream << "%" << block_id_ << " = arith.addi  %" << vid << ", "
                   << this->thread_context_args[4] << ": i32\n";
    } else if (iv->thread_tag == "blockIdx.z" && iv->var->name_hint != "_") {
      std::ostringstream temp;
      temp << "arith.constant 0"
           << " : ";
      PrintType(iv->var->dtype, temp);
      std::string vid = SSAGetID(temp.str(), iv->var->dtype);
      auto block_id_ = AllocVarID(iv->var.get());
      this->PrintIndent();
      this->stream << "%" << block_id_ << " = arith.addi  %" << vid << ", "
                   << this->thread_context_args[5] << ": i32\n";
    }
    this->VisitStmt(op->body);
    return;
  }

  CodeGenC::VisitStmt_(op);
}

void CodeGenTileLangCOMMONIR::VisitStmt_(const AllocateNode *op) {
  ICHECK(!is_zero(op->condition));
  std::string scope = GetPtrStorageScope(op->buffer_var);

  std::string vid = AllocVarID(op->buffer_var.get());
  String address_space = GetAddressSpace(scope);
  if (!op->buffer_var->type_annotation.defined()) {
    LOG(FATAL) << "AllocateNode buffer_var must have a type annotation";
  }
  auto ptr_type = op->buffer_var->type_annotation.as<PointerTypeNode>();
  if (!ptr_type) {
    LOG(FATAL) << "AllocateNode buffer_var must be a pointer type";
  }
  auto prim_type = ptr_type->element_type.as<PrimTypeNode>();
  if (!prim_type) {
    LOG(FATAL) << "AllocateNode buffer_var must point to a primitive type";
  }
  Buffer buffer = decl_buffer(op->extents, prim_type->dtype, vid, scope,
                              Array<IntImm>(), Span());
  vmap.Set(op->buffer_var, buffer);

  this->type_info[vid] =
      new Memref(vid, op->extents, op->dtype, address_space, false);
  this->PrintIndent();
  stream << "%" << vid << " = "
         << "memref.alloc() : " << GetMemrefInfo(vid) << "\n";

  this->VisitStmt(op->body);
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const MinNode *op, std::ostream &os) {
  if (op->dtype.is_int()) {
    PrintBinary(op, "minsi", os, this);
  } else if (op->dtype.is_uint()) {
    PrintBinary(op, "minui", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "minnumf", os, this);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const MaxNode *op, std::ostream &os) {
  if (op->dtype.is_int()) {
    PrintBinary(op, "maxsi", os, this);
  } else if (op->dtype.is_uint()) {
    PrintBinary(op, "maxui", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "maxnumf", os, this);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const AddNode *op, std::ostream &os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "addi", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "addf", os, this);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const SubNode *op, std::ostream &os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "subi", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "subf", os, this);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const FloatImmNode *op,
                                         std::ostream &os) { // NOLINT(*)
  std::ostringstream temp;
  if (op->value == -std::numeric_limits<float>::infinity()) {
    temp << "arith.constant 0xFF800000 : ";
  } else if (op->value == std::numeric_limits<float>::infinity()) {
    temp << "arith.constant 0x7F800000 : ";
  } else {
    temp << "arith.constant " << op->value << " : ";
  }
  PrintType(op->dtype, temp);
  os << SSAGetID(temp.str(), op->dtype);
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const IntImmNode *op,
                                         std::ostream &os) {
  std::ostringstream temp;
  temp << "arith.constant ";
  if (op->dtype.is_bool()) {
    temp << (op->value == 1 ? "true" : "false");
  } else {
    temp << op->value << " : ";
    PrintType(op->dtype, temp);
  }
  os << SSAGetID(temp.str(), op->dtype);
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const MulNode *op, std::ostream &os) {
  if (op->dtype.is_int() || op->dtype.is_uint()) {
    PrintBinary(op, "muli", os, this);
  } else if (op->dtype.is_float()) {
    PrintBinary(op, "mulf", os, this);
  }
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const AndNode *op, std::ostream &os) {
  assert(op->a.dtype().is_int() || op->a.dtype().is_uint());
  assert(op->b.dtype().is_int() || op->b.dtype().is_uint());
  PrintBinary(op, "andi", os, this);
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const OrNode *op, std::ostream &os) {
  assert(op->a.dtype().is_int() || op->a.dtype().is_uint());
  assert(op->b.dtype().is_int() || op->b.dtype().is_uint());
  PrintBinary(op, "ori", os, this);
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const DivNode *op, std::ostream &os) {
  PrintBinary(op, "<<<divf>>>", os, this);
}

void CodeGenTileLangCOMMONIR::VisitExpr_(const SelectNode *op,
                                         std::ostream &os) {
  auto condition = PrintExpr(op->condition);
  auto true_value = PrintExpr(op->true_value);
  auto false_value = PrintExpr(op->false_value);

  os << "(" << condition << " ? "
     << "" << true_value << " : " << false_value << ")";
}

void PrintHostFunc(const PrimFunc &f, const std::string &name, std::ostream &os,
                   int core) {
  os << "extern \"C\" void call(";
  std::vector<std::string> arg_names;
  for (size_t i = 0; i < f->params.size(); ++i) {
    auto v = f->params[i];
    if (i != 0) {
      os << ", ";
    }
    arg_names.push_back(v->name_hint);
    os << "uint8_t* " << v->name_hint;
  }
  os << ", aclrtStream stream) {\n  ";

  os << name << "<<<" << core << ", nullptr, stream>>>(";
  for (auto &arg_name : arg_names) {
    os << arg_name;
    if (arg_name != arg_names.back()) {
      os << ", ";
    }
  }
  os << ");\n";
  os << "}\n";
}

void CodeGenTileLangCOMMONIR::GenRecastFromArg(Buffer curr_buffer,
                                               String arg_name,
                                               String &recast_inst) {
  std::ostringstream res;
  String target_strides = GetBufferStrides(curr_buffer);
  String cast_name = arg_name + "_Recast";
  this->type_info[cast_name] = new Memref(cast_name, curr_buffer);
  res << "\%" << cast_name << " = ";
  res << "memref.reinterpret_cast \%";
  res << arg_name;
  res << " to offset: [0], sizes: [";
  PrintShape(curr_buffer->shape, ", ", res);
  res << "], strides: ";
  res << target_strides;
  res << " : ";
  res << GetMemrefInfo(arg_name);
  res << " to ";
  res << GetMemrefInfo(cast_name);
  res << "\n";
  recast_inst = res.str();
}

void CodeGenTileLangCOMMONIR::AddFunction(const GlobalVar &gvar,
                                          const PrimFunc &f) {
  this->stream << "module {\n";

  // If the function has already been forward-declared, this is a
  // no-op.
  CodeGenC::DeclareFunction(gvar, f);
  // clear previous generated state.
  this->InitFuncState(f);

  bool no_alias = f->HasNonzeroAttr(tir::attr::kNoAlias);
  auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.has_value())
      << "CodeGenC: Expect PrimFunc to have the global_symbol attribute";

  int func_scope = this->BeginScope();
  this->PrintIndent();
  auto func_name = static_cast<std::string>(global_symbol.value());

  this->stream << "func.func @" << func_name << "(";

  std::vector<String> recast_need_insert;

  this->type_info.clear();
  size_t n = f->params.size();
  for (size_t i = 0; i < f->params.size(); ++i) {
    tir::Var v = f->params[i];
    std::string vid = AllocVarID(v.get());

    if (i != 0)
      stream << ", ";

    if (v.dtype().is_handle()) {
      this->vmap = f->buffer_map;
      auto real_v = f->buffer_map[v]->data;
      String arg_name = AllocVarID(real_v.get());
      Memref *buffer = new Memref(arg_name, f->buffer_map[v], true);
      this->type_info[arg_name] = buffer;
      stream << "%" << arg_name << ": " << GetMemrefInfo(arg_name);
      String recast_inst = "";
      GenRecastFromArg(f->buffer_map[v], arg_name, recast_inst);
      recast_need_insert.push_back(recast_inst);

      if (auto *ptr = v->type_annotation.as<PointerTypeNode>()) {
        if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
          RegisterHandleType(v.get(), prim->dtype);
        }
      }
    } else {
      stream << "%" << vid << ": ";
      CodeGenC::PrintType(GetType(v), stream);
    }
  }

  for (size_t i = 0; i < 6; ++i) {
    this->thread_context_args[i] = "\%args" + std::to_string(n + i);
    stream << ", ";
    stream << thread_context_args[i] << ": i32";
  }
  stream << ")\n";
  this->PrintIndent();
  stream << "{\n";
  int func_body_scope = this->BeginScope();
  for (String recast_inst : recast_need_insert) {
    this->PrintIndent();
    stream << recast_inst;
  }
  this->PrintStmt(f->body);
  this->EndScope(func_body_scope);
  this->PrintIndent();
  this->stream << "return\n";
  this->PrintIndent();
  this->stream << "}\n";
  this->EndScope(func_scope);
  this->PrintIndent();
  this->stream << "}\n";
}

String CodeGenTileLangCOMMONIR::GetMemrefInfo(String name) {
  if (this->type_info.count(name) == 0)
    LOG(FATAL) << "Can not find memref ssa object: " << name;
  if (!dynamic_cast<Memref *>(type_info[name])) {
    LOG(FATAL) << name << " should be a memref";
  }
  Memref *MemrefObj = dynamic_cast<Memref *>(this->type_info[name]);
  return GetMemrefInfo(MemrefObj);
}

String CodeGenTileLangCOMMONIR::GetMemrefInfo(Memref *memrefObj) {
  if (memrefObj->type_str != "")
    return memrefObj->type_str;
  std::ostringstream memref_type;
  memref_type << "memref<";
  if (memrefObj->is_arg) {
    memref_type << "?x";
  } else {
    for (PrimExpr s : memrefObj->shape) {
      if (auto s_int = as_const_int(s)) {
        memref_type << std::to_string(*s_int) + "x";
      } else {
        // not support ssa var in memref size
        memref_type << "?x";
      }
    }
  }
  PrintType(memrefObj->dtype, memref_type);
  if (!memrefObj->is_arg) {
    memref_type << ", strided<[";
    for (int i = 0; i < memrefObj->dim; i++) {
      if (i > 0)
        memref_type << ", ";
      if (memrefObj->stride.size() > 0) {
        if (auto s_int = as_const_int(memrefObj->stride[i])) {
          memref_type << std::to_string(*s_int);
        } else {
          // not support ssa var in memref size
          memref_type << "?";
        }
      } else {
        memref_type << memrefObj->stride_int[i];
      }
    }
    memref_type << "], offset:";
    if (memrefObj->var_offset)
      memref_type << "?";
    else
      memref_type << memrefObj->offset;
    memref_type << ">";
  }
  memref_type << ">";
  // memref_type << ", #address_space<" << memrefObj->address_space << ">>";
  memrefObj->type_str = memref_type.str();
  return memrefObj->type_str;
}

void Memref::GetIntStride() {
  if (stride.empty()) {
    stride_int = GetStrideFromShape(shape);
    for (unsigned long s : stride_int) {
      stride.push_back(IntImm(DataType::Int(64), s));
    }
  } else {
    for (PrimExpr s : stride) {
      if (auto s_int = as_const_int(s))
        stride_int.push_back(*s_int);
    }
  }
}

Memref::Memref(String name, Array<PrimExpr> shape_in, DataType dtype_in,
               String addr_space_in, bool var_offset_in,
               Array<PrimExpr> stride_in, int offset_in, bool is_arg_in) {
  var_id = name;
  shape = shape_in;
  stride = stride_in;
  dtype = dtype_in;
  offset = offset_in;
  is_arg = is_arg_in;
  address_space = addr_space_in;
  var_offset = var_offset_in;
  dim = shape_in.size();
  GetIntStride();
}

Memref::Memref(String name, Buffer buffer, bool is_arg_in) {
  var_id = name;
  shape = buffer->shape;
  stride = buffer->strides;
  dtype = buffer->dtype;
  offset = 0;
  is_arg = is_arg_in;
  String scope = GetPtrStorageScope(buffer->data);
  address_space = GetAddressSpace(scope);
  var_offset = false;
  dim = shape.size();
  GetIntStride();
}

String CodeGenTileLangCOMMONIR::GetTensorInfo(String name) {
  if (this->type_info_tensor.count(name) == 0)
    LOG(FATAL) << "Can not find tensor ssa object: " << name;
  if (!dynamic_cast<Tensor *>(type_info_tensor[name])) {
    LOG(FATAL) << name << " should be a tensor";
  }
  Tensor *TensorObj = dynamic_cast<Tensor *>(this->type_info_tensor[name]);
  return GetTensorInfo(TensorObj);
}

String CodeGenTileLangCOMMONIR::GetTensorInfo(Tensor *tensorObj) {
  if (tensorObj->type_str != "")
    return tensorObj->type_str;
  std::ostringstream tensor_type;
  tensor_type << "tensor<";
  for (PrimExpr s : tensorObj->shape) {
    if (auto s_int = as_const_int(s)) {
      tensor_type << std::to_string(*s_int) + "x";
    } else {
      // not support ssa var in memref size
      tensor_type << "?x";
    }
  }
  PrintType(tensorObj->dtype, tensor_type);
  tensor_type << ">";
  // tensor_type << ", #address_space<" << tensorObj->address_space << ">>";
  tensorObj->type_str = tensor_type.str();
  return tensorObj->type_str;
}

Tensor::Tensor(String name, Array<PrimExpr> shape_in, DataType dtype_in,
               String addr_space_in) {
  var_id = name;
  shape = shape_in;
  dtype = dtype_in;
  address_space = addr_space_in;
  dim = shape_in.size();
}

Tensor::Tensor(String name, Buffer buffer) {
  var_id = name;
  shape = buffer->shape;
  dtype = buffer->dtype;
  String scope = GetPtrStorageScope(buffer->data);
  address_space = GetAddressSpace(scope);
  dim = shape.size();
}
Tensor::Tensor(String name, Memref memref) {
  var_id = name;
  shape = memref.shape;
  dtype = memref.dtype;
  address_space = memref.address_space;
  dim = shape.size();
}
} // namespace codegen

} // namespace tvm
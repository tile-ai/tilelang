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
 * \file arg_binder.cc
 * \brief Helper utility to match and bind arguments.
 */
#include "arg_binder.h"

#include <tvm/runtime/device_api.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <sstream>

#include "tir/transforms/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tir;

void BinderAddAssert(arith::Analyzer *ana, PrimExpr cond,
                     const std::string &arg_name, std::vector<Stmt> *asserts) {
  PrimExpr scond = ana->Simplify(cond);
  if (is_zero(scond)) {
    LOG(FATAL) << "Bind have an unmet assertion: " << cond << ", "
               << " on argument " << arg_name;
  }
  if (!is_one(scond)) {
    std::ostringstream os;
    os << "Argument " << arg_name << " has an unsatisfied constraint: " << cond;
    asserts->emplace_back(AssertStmt(scond, StringImm(os.str()), Evaluate(0)));
  }
}

bool ArgBinder::BindNullable(const PrimExpr &arg, const PrimExpr &value,
                             const std::string &arg_name, bool with_lets,
                             const PrimExpr &nullable_guard) {
  // Currently only used in BindDLTensor, nullable_guard is already a defined
  // bool, so use it directly.
  auto MakeGuarded = [&](PrimExpr basic) -> PrimExpr {
    // is_null || basic
    return Or(nullable_guard, basic);
  };

  ICHECK_EQ(arg.dtype(), value.dtype()) << "arg " << arg << " value " << value;
  if (const VarNode *v = arg.as<VarNode>()) {
    auto it = def_map_->find(v);
    if (it == def_map_->end()) {
      // First time binding: identical behavior as Bind_
      Var v_arg = Downcast<Var>(arg);
      defs_.emplace_back(v_arg);
      if (with_lets) {
        (*def_map_)[v] = arg;
        init_nest_.emplace_back(LetStmt(v_arg, value, Evaluate(0)));
      } else {
        (*def_map_)[v] = value;
      }
      return true;
    } else {
      // Second or later binding: add is_null short-circuit
      PrimExpr cond = MakeGuarded(it->second == value);
      BinderAddAssert(&analyzer_, cond, arg_name, &asserts_);
    }
  } else {
    // For non-Var expressions, also add is_null short-circuit
    PrimExpr cond = MakeGuarded(arg == value);
    BinderAddAssert(&analyzer_, cond, arg_name, &asserts_);
  }
  return false;
}

bool ArgBinder::Bind_(const PrimExpr &arg, const PrimExpr &value,
                      const std::string &arg_name, bool with_lets) {
  ICHECK_EQ(arg.dtype(), value.dtype()) << "arg " << arg << " value " << value;
  if (const VarNode *v = arg.as<VarNode>()) {
    auto it = def_map_->find(v);
    if (it == def_map_->end()) {
      Var v_arg = Downcast<Var>(arg);
      defs_.emplace_back(v_arg);
      if (with_lets) {
        (*def_map_)[v] = arg;
        init_nest_.emplace_back(LetStmt(v_arg, value, Evaluate(0)));
      } else {
        (*def_map_)[v] = value;
      }
      return true;
    } else {
      BinderAddAssert(&analyzer_, it->second == value, arg_name, &asserts_);
    }
  } else {
    BinderAddAssert(&analyzer_, arg == value, arg_name, &asserts_);
  }
  return false;
}

void ArgBinder::Bind(const PrimExpr &arg, const PrimExpr &value,
                     const std::string &arg_name, bool with_let) {
  Bind_(arg, value, arg_name, with_let);
}

void ArgBinder::BindArray(const ffi::Array<PrimExpr> &arg,
                          const ffi::Array<PrimExpr> &value,
                          const std::string &arg_name) {
  ICHECK_EQ(arg.size(), value.size())
      << "Argument " << arg_name << " array size mismatch";
  for (size_t i = 0; i < arg.size(); ++i) {
    std::ostringstream os;
    os << arg_name << "[" << i << "]";
    this->Bind(arg[i], value[i], os.str());
  }
}

void ArgBinder::BindBuffer(const Buffer &arg, const Buffer &value,
                           const std::string &arg_name, bool fuzzy_match) {
  ICHECK_EQ(arg.scope(), value.scope())
      << "Argument " << arg_name << " Buffer bind scope mismatch";
  // Relax dtype check to allow FP8 E4M3 variants to bind together.
  auto dtype_compatible = [](DataType expected, DataType provided) -> bool {
    if (expected == provided)
      return true;
    // If expected is float8_e4m3, allow float8_e4m3fn/float8_e4m3fnuz as well.
    if (expected.is_float8_e4m3()) {
      return provided.is_float8_e4m3() || provided.is_float8_e4m3fn() ||
             provided.is_float8_e4m3fnuz();
    }
    // If expected is float8_e5m2, allow float8_e5m2fnuz as well.
    if (expected.is_float8_e5m2()) {
      return provided.is_float8_e5m2() || provided.is_float8_e5m2fnuz();
    }
    // If expected is bool, allow binding from int8/uint8 with same lanes.
    if (expected.is_bool()) {
      bool is_i8 = provided.is_int() && provided.bits() == 8;
      bool is_u8 = provided.is_uint() && provided.bits() == 8;
      return (is_i8 || is_u8) && expected.lanes() == provided.lanes();
    }
    return false;
  };
  ICHECK(dtype_compatible(arg->dtype, value->dtype))
      << "Argument " << arg_name << " Buffer bind data type mismatch: expected "
      << arg->dtype << ", got " << value->dtype;
  if (value->data_alignment % arg->data_alignment != 0) {
    LOG(WARNING) << "Trying to bind buffer to another one with lower alignment "
                    "requirement "
                 << " required_alignment=" << arg->data_alignment
                 << ", provided_alignment=" << value->data_alignment;
  }

  if (value->elem_offset.defined()) {
    // bind pointer and offset.
    if (is_zero(arg->elem_offset)) {
      ICHECK(is_zero(value->elem_offset))
          << "Trying to bind a Buffer with offset into one without offset "
          << " required elem_offset=" << arg->elem_offset
          << ", provided elem_offset=" << value->elem_offset;
    }

    this->Bind(arg->data, value->data, arg_name + ".data");
    if (Bind_(arg->elem_offset, value->elem_offset, arg_name + ".elem_offset",
              false)) {
      if (arg->offset_factor > 1) {
        PrimExpr offset = value->elem_offset;
        PrimExpr factor = make_const(offset.dtype(), arg->offset_factor);
        PrimExpr zero = make_zero(offset.dtype());
        BinderAddAssert(&analyzer_, truncmod(offset, factor) == zero,
                        arg_name + ".elem_offset", &asserts_);
      }
    }
  }

  if (arg->shape.size() < value->shape.size()) {
    ICHECK(fuzzy_match) << "Argument " << arg_name << " size mismatch";
    size_t diff = value->shape.size() - arg->shape.size();
    for (size_t i = 0; i < diff; ++i) {
      ICHECK(is_one(analyzer_.Simplify(value->shape[i])))
          << "Argument " << arg_name << " shape mismatch" << arg->shape
          << " vs " << value->shape;
    }
    for (size_t i = 0; i < arg->shape.size(); ++i) {
      std::ostringstream os;
      os << arg_name << ".shape[" << i << "]";
      this->Bind(arg->shape[i], value->shape[i + diff], os.str());
    }
    if (!value->strides.empty()) {
      ICHECK_EQ(arg->strides.size(), arg->shape.size());
      ICHECK_EQ(value->strides.size(), value->shape.size());
      for (size_t i = 0; i < arg->strides.size(); ++i) {
        std::ostringstream os;
        os << arg_name << ".strides[" << i << "]";
        this->Bind(arg->strides[i], value->strides[i + diff], os.str());
      }
    }
  } else {
    this->BindArray(arg->shape, value->shape, arg_name + ".shape");
    this->BindArray(arg->strides, value->strides, arg_name + ".strides");
  }
}

inline PrimExpr TVMArrayGet(DataType t, Var arr,
                            builtin::TVMStructFieldKind kind) {
  return TVMStructGet(t, arr, 0, kind);
}

void ArgBinder::BindDLTensor(const Buffer &buffer, const PrimExpr &device_type,
                             const PrimExpr &device_id, const Var &handle,
                             const std::string &arg_name) {
  const DataType tvm_shape_type = DataType::ShapeIndex();
  const DataType tvm_ndim_type = DataType::Int(32);
  const Stmt nop = Evaluate(0);

  // Allow NULL DLTensor* for optional inputs.  When the handle is NULL,
  // avoid dereferencing it by using expression-level conditionals and
  // short-circuiting guards in asserts. Cache the null check in a Let-bound
  // boolean so codegen does not repeat `(handle == NULL)` everywhere.
  Var is_null_var(arg_name + "_is_null", DataType::Bool());
  init_nest_.emplace_back(
      LetStmt(is_null_var,
              Call(DataType::Bool(), builtin::isnullptr(), {handle}), nop));
  const PrimExpr &is_null = is_null_var;

  // dimension checks
  PrimExpr v_ndim = TVMArrayGet(tvm_ndim_type, handle, builtin::kArrNDim);

  // Helper functions for shape/stride name formatting
  auto shape_handle_name = [&]() { return arg_name + ".shape"; };
  auto stride_handle_name = [&]() { return arg_name + ".strides"; };
  auto array_element_name = [&](const std::string &arr_name, size_t k) {
    std::stringstream ss;
    ss << arr_name << '[' << k << ']';
    return ss.str();
  };
  auto shape_element_name = [&](size_t k) {
    return array_element_name(shape_handle_name(), k);
  };
  auto stride_element_name = [&](size_t k) {
    return array_element_name(stride_handle_name(), k);
  };

  PrimExpr a_ndim =
      make_const(tvm_ndim_type, static_cast<int64_t>(buffer->shape.size()));
  std::ostringstream ndim_err_msg;
  // Note: We cannot embed runtime values into the message string.
  // Keep message human-friendly without printing TIR exprs.
  ndim_err_msg << arg_name << ".ndim is expected to equal "
               << buffer->shape.size() << ", but got mismatched ndim";
  auto msg = StringImm(ndim_err_msg.str());
  // Only check ndim when handle is non-NULL (using short-circuit OR)
  v_ndim = tvm::if_then_else(Not(is_null), v_ndim, make_zero(tvm_ndim_type));
  init_nest_.emplace_back(AssertStmt(Or(is_null, a_ndim == v_ndim), msg, nop));
  // type checks
  std::ostringstream type_err_msg;
  // Avoid dumping TIR expressions in error text; just state mismatch.
  // Include expected dtype triplet for clarity.
  type_err_msg << arg_name << ".dtype is expected to be " << buffer->dtype
               << ", but got incompatible dtype";
  // Guard all dtype field loads by `is_null` using if_then_else
  PrimExpr v_type_code = tvm::if_then_else(
      Not(is_null),
      TVMArrayGet(DataType::UInt(8), handle, builtin::kArrTypeCode),
      IntImm(DataType::UInt(8), buffer->dtype.code()));
  PrimExpr v_type_bits = tvm::if_then_else(
      Not(is_null),
      TVMArrayGet(DataType::UInt(8), handle, builtin::kArrTypeBits),
      IntImm(DataType::UInt(8), buffer->dtype.bits()));
  PrimExpr v_type_lanes = tvm::if_then_else(
      Not(is_null),
      TVMArrayGet(DataType::UInt(16), handle, builtin::kArrTypeLanes),
      IntImm(DataType::UInt(16), buffer->dtype.lanes()));
  PrimExpr expect_code = IntImm(DataType::UInt(8), buffer->dtype.code());
  PrimExpr expect_bits = IntImm(DataType::UInt(8), buffer->dtype.bits());
  PrimExpr expect_lanes = IntImm(DataType::UInt(16), buffer->dtype.lanes());

  PrimExpr cond = (v_type_code == expect_code && v_type_bits == expect_bits &&
                   v_type_lanes == expect_lanes);

  // Allow float8_e4m3 to match float8_e4m3fn/float8_e4m3fnuz at runtime.
  if (buffer->dtype.is_float8_e4m3()) {
    PrimExpr code_e4m3 = IntImm(DataType::UInt(8), DataType::kFloat8_e4m3);
    PrimExpr code_e4m3fn = IntImm(DataType::UInt(8), DataType::kFloat8_e4m3fn);
    PrimExpr code_e4m3fnuz =
        IntImm(DataType::UInt(8), DataType::kFloat8_e4m3fnuz);
    PrimExpr code_match =
        (v_type_code == code_e4m3 || v_type_code == code_e4m3fn ||
         v_type_code == code_e4m3fnuz);
    cond = cond || (code_match && v_type_bits == expect_bits &&
                    v_type_lanes == expect_lanes);
  }
  // Allow float8_e5m2 to match float8_e5m2fnuz at runtime.
  if (buffer->dtype.is_float8_e5m2()) {
    PrimExpr code_e5m2 = IntImm(DataType::UInt(8), DataType::kFloat8_e5m2);
    PrimExpr code_e5m2fnuz =
        IntImm(DataType::UInt(8), DataType::kFloat8_e5m2fnuz);
    PrimExpr code_match =
        (v_type_code == code_e5m2 || v_type_code == code_e5m2fnuz);
    cond = cond || (code_match && v_type_bits == expect_bits &&
                    v_type_lanes == expect_lanes);
  }
  // Allow bool to match int8/uint8 at runtime, and also kDLBool(code=6).
  if (buffer->dtype.is_bool()) {
    PrimExpr code_int = IntImm(DataType::UInt(8), DataType::kInt);
    PrimExpr code_uint = IntImm(DataType::UInt(8), DataType::kUInt);
    PrimExpr code_kdlbool = IntImm(DataType::UInt(8), 6);
    PrimExpr bits8 = IntImm(DataType::UInt(8), 8);
    PrimExpr bits1 = IntImm(DataType::UInt(8), 1);
    PrimExpr lanes_ok = (v_type_lanes == expect_lanes);
    PrimExpr int8_ok =
        (v_type_code == code_int && v_type_bits == bits8 && lanes_ok);
    PrimExpr uint8_ok =
        (v_type_code == code_uint && v_type_bits == bits8 && lanes_ok);
    // Some frontends may tag bool tensors as kDLBool(code=6), commonly with
    // bits=8 or bits=1.
    PrimExpr kdlbool8_ok =
        (v_type_code == code_kdlbool && v_type_bits == bits8 && lanes_ok);
    PrimExpr kdlbool1_ok =
        (v_type_code == code_kdlbool && v_type_bits == bits1 && lanes_ok);
    // Also accept any dtype whose bitwidth=1, regardless of code, to be
    // defensive.
    PrimExpr bit1_ok = (v_type_bits == bits1 && lanes_ok);
    cond = cond || int8_ok || uint8_ok || kdlbool8_ok || kdlbool1_ok || bit1_ok;
  }
  if (!(buffer->dtype == DataType::Int(1) ||
        buffer->dtype == DataType::Int(4) ||
        buffer->dtype == DataType::UInt(4))) {
    auto type_msg = StringImm(type_err_msg.str());
    // Only check dtype when handle is non-NULL (short-circuit)
    asserts_.emplace_back(AssertStmt(Or(is_null, cond), type_msg, nop));
  }

  // shape field
  Buffer buf_shape =
      decl_buffer({IntImm(DataType::Int(32), buffer->shape.size())},
                  tvm_shape_type, shape_handle_name());
  Var v_shape(shape_handle_name(), DataType::Handle());
  def_handle_dtype_.Set(v_shape, make_const(tvm_shape_type, 0));
  // Use if_then_else for NULL guard on the shape pointer itself, avoiding
  // dereferencing TVMStructGet(handle, kArrShape) when handle is NULL.
  init_nest_.emplace_back(
      LetStmt(buf_shape->data,
              tvm::if_then_else(
                  Not(is_null),
                  TVMArrayGet(DataType::Handle(), handle, builtin::kArrShape),
                  make_zero(DataType::Handle())),
              nop));
  init_nest_.emplace_back(DeclBuffer(buf_shape, nop));

  for (size_t k = 0; k < buffer->shape.size(); ++k) {
    // These packed-bit dtype shapes were not bound in the original
    // implementation, so we just use them as is.
    if (buffer->dtype == DataType::Int(4) ||
        buffer->dtype == DataType::UInt(4) ||
        buffer->dtype == DataType::Int(1)) {
      break;
    }

    // The "real" runtime shape value read from DLTensor
    PrimExpr raw_shape_val =
        cast(buffer->shape[k].dtype(),
             BufferLoad(buf_shape,
                        {IntImm(DataType::Int(32), static_cast<int>(k))}));

    // Bind to the value of the symbolic dimension (e.g., m) in TIR, with an
    // is_null guard:
    //   handle is NULL → use 0, placeholder but no dereference
    //   handle non-NULL → actually read from DLTensor's shape array
    PrimExpr bound_shape_val = tvm::if_then_else(
        is_null, make_zero(buffer->shape[k].dtype()), raw_shape_val);

    // When first encountering a Var (e.g., m), this will generate:
    //   Let(m, bound_shape_val, ...)
    // Constant dimensions will only generate consistency assertions.
    BindNullable(buffer->shape[k], bound_shape_val, shape_element_name(k), true,
                 is_null);

    // Keep an explicit "consistency check": when non-NULL, the symbolic
    // dimension must equal the DLTensor's shape.
    Stmt shape_check = AssertStmt(
        Or(is_null, buffer->shape[k] == raw_shape_val),
        StringImm(shape_element_name(k) + " mismatch with DLTensor shape"),
        Evaluate(0));
    asserts_.emplace_back(shape_check);
  }

  // strides field
  Buffer buf_strides =
      decl_buffer({IntImm(DataType::Int(32), buffer->strides.size())},
                  tvm_shape_type, arg_name + ".strides");
  def_handle_dtype_.Set(buf_strides->data, tir::TypeAnnotation(tvm_shape_type));
  init_nest_.emplace_back(
      LetStmt(buf_strides->data,
              tvm::if_then_else(
                  Not(is_null),
                  TVMArrayGet(DataType::Handle(), handle, builtin::kArrStrides),
                  make_zero(DataType::Handle())),
              nop));
  init_nest_.emplace_back(DeclBuffer(buf_strides, nop));
  PrimExpr v_strides_is_null =
      Call(DataType::Bool(1), builtin::isnullptr(), {buf_strides->data});

  if (buffer->strides.empty()) {
    // Assert the buffer is compact
    DataType stype = buffer->DefaultIndexType();
    PrimExpr expect_stride = make_const(stype, 1);
    ffi::Array<PrimExpr> conds;
    for (size_t i = buffer->shape.size(); i != 0; --i) {
      size_t k = i - 1;
      PrimExpr svalue = cast(
          stype, BufferLoad(buf_strides,
                            {IntImm(DataType::Int(32), static_cast<int>(k))}));
      conds.push_back(buffer->shape[k] == 1 || expect_stride == svalue);
      expect_stride = expect_stride * buffer->shape[k];
    }
    std::ostringstream stride_err_msg;
    stride_err_msg
        << stride_handle_name()
        << ": expected to be compact array, but got non-compact strides";
    if (!conds.empty()) {
      auto stride_msg = StringImm(stride_err_msg.str());
      Stmt check =
          AssertStmt(foldl([](PrimExpr a, PrimExpr b,
                              Span span) { return logical_and(a, b, span); },
                           const_true(1), conds),
                     stride_msg, Evaluate(0));
      // Only check when strides array is actually present at runtime
      check = IfThenElse(Not(v_strides_is_null), check);
      asserts_.emplace_back(SeqStmt({check, Evaluate(0)}));
    }
  } else if (buffer->buffer_type == kAutoBroadcast) {
    PrimExpr stride_from_shape = make_const(buffer->DefaultIndexType(), 1);
    for (size_t i = buffer->shape.size(); i != 0; --i) {
      size_t k = i - 1;
      DataType stride_dtype = buffer->strides[k].dtype();
      PrimExpr explicit_stride =
          cast(stride_dtype,
               BufferLoad(buf_strides,
                          {IntImm(DataType::Int(32), static_cast<int>(k))}));
      PrimExpr stride_from_shape_cast = cast(stride_dtype, stride_from_shape);

      PrimExpr core_value = tvm::if_then_else(
          v_strides_is_null, stride_from_shape_cast, explicit_stride);
      core_value = tvm::if_then_else(buffer->shape[k] == 1,
                                     make_zero(stride_dtype), core_value);

      // Bind like shape: define var when needed, and only assert when non-NULL
      PrimExpr bound_stride_val =
          tvm::if_then_else(is_null, make_zero(stride_dtype), core_value);
      BindNullable(buffer->strides[k], bound_stride_val, stride_element_name(k),
                   true, is_null);

      Stmt stride_check = AssertStmt(
          Or(is_null, buffer->strides[k] == core_value),
          StringImm(stride_element_name(k) + " mismatch with DLTensor strides"),
          Evaluate(0));
      asserts_.emplace_back(stride_check);

      PrimExpr shape_extent = cast(stride_dtype, buffer->shape[k]);
      stride_from_shape =
          analyzer_.Simplify(stride_from_shape_cast * shape_extent);
    }
  } else {
    PrimExpr stride_from_shape = make_const(buffer->DefaultIndexType(), 1);

    for (int k = static_cast<int>(buffer->strides.size()) - 1; k >= 0; --k) {
      DataType stride_dtype = buffer->strides[k].dtype();
      PrimExpr explicit_stride =
          cast(stride_dtype,
               BufferLoad(buf_strides, {IntImm(DataType::Int(32), k)}));
      PrimExpr shape_stride = cast(
          stride_dtype, BufferLoad(buf_shape, {IntImm(DataType::Int(32), k)}));
      PrimExpr stride_from_shape_cast = cast(stride_dtype, stride_from_shape);

      PrimExpr core_value = tvm::if_then_else(
          v_strides_is_null, stride_from_shape_cast, explicit_stride);

      PrimExpr bound_stride_val =
          tvm::if_then_else(is_null, make_zero(stride_dtype), core_value);
      BindNullable(buffer->strides[k], bound_stride_val, stride_element_name(k),
                   true, is_null);

      Stmt stride_check = AssertStmt(
          Or(is_null, buffer->strides[k] == core_value),
          StringImm(stride_element_name(k) + " mismatch with DLTensor strides"),
          Evaluate(0));
      asserts_.emplace_back(stride_check);

      stride_from_shape =
          analyzer_.Simplify(stride_from_shape_cast * shape_stride);
    }
  }

  // Byte_offset field.
  int data_bytes = GetVectorBytes(buffer->dtype);

  if (const auto *const_offset = buffer->elem_offset.as<IntImmNode>()) {
    // Constant elem_offset: only need consistency check, no need for additional
    // Var binding.
    PrimExpr actual_byte_offset = tvm::if_then_else(
        Not(is_null),
        TVMArrayGet(DataType::UInt(64), handle, builtin::kArrByteOffset),
        make_const(DataType::UInt(64), 0));
    PrimExpr expect_byte_offset =
        make_const(DataType::UInt(64), const_offset->value * data_bytes);
    Stmt byte_off_check =
        AssertStmt(Or(is_null, expect_byte_offset == actual_byte_offset),
                   StringImm(arg_name + ".byte_offset mismatch"), nop);
    asserts_.emplace_back(byte_off_check);
  } else {
    PrimExpr actual_byte_offset = tvm::if_then_else(
        Not(is_null),
        TVMArrayGet(DataType::UInt(64), handle, builtin::kArrByteOffset),
        make_const(DataType::UInt(64), 0));
    PrimExpr expect_elem_off =
        cast(buffer->elem_offset.dtype(),
             (actual_byte_offset / make_const(DataType::UInt(64), data_bytes)));

    // Like shape/stride, do NULL-safe binding for elem_offset:
    //   handle is NULL → 0
    //   handle non-NULL → actual_byte_offset / data_bytes
    PrimExpr bound_elem_off = tvm::if_then_else(
        is_null, make_zero(buffer->elem_offset.dtype()), expect_elem_off);
    BindNullable(buffer->elem_offset, bound_elem_off, arg_name + ".elem_offset",
                 true, is_null);

    // Strict consistency check for non-NULL case
    Stmt elem_off_check =
        AssertStmt(Or(is_null, buffer->elem_offset == expect_elem_off),
                   StringImm(arg_name + ".elem_offset mismatch"), nop);
    asserts_.emplace_back(elem_off_check);

    if (buffer->offset_factor > 1) {
      PrimExpr offset = buffer->elem_offset;
      PrimExpr factor = make_const(offset.dtype(), buffer->offset_factor);
      PrimExpr zero = make_zero(offset.dtype());
      Stmt off_factor_check =
          AssertStmt(Or(is_null, truncmod(offset, factor) == zero),
                     StringImm(arg_name + ".elem_offset factor mismatch"), nop);
      asserts_.emplace_back(off_factor_check);
    }
  }

  // device info.
  // Define device_id from handle when available (so later passes can use it)
  PrimExpr actual_dev_type = tvm::if_then_else(
      Not(is_null),
      TVMArrayGet(DataType::Int(32), handle, builtin::kArrDeviceType),
      make_zero(DataType::Int(32)));
  PrimExpr actual_dev_id = tvm::if_then_else(
      Not(is_null),
      TVMArrayGet(DataType::Int(32), handle, builtin::kArrDeviceId),
      make_zero(DataType::Int(32)));
  // Bind device_id to a safe expression (0 when NULL handle)
  BindNullable(device_id, actual_dev_id, arg_name + ".device_id", true,
               is_null);
  // Check device_type consistency (device_id equality is implicitly ensured by
  // binding above)
  init_nest_.emplace_back(
      AssertStmt(Or(is_null, device_type == actual_dev_type),
                 StringImm(arg_name + ".device_type mismatch"), nop));

  // Data field.  Because the validation of the data field may depend
  // on a dynamic size defined by the other DLTensor* parameters, this
  // field must be generated last.
  // Bind data pointer using expression-level guard to avoid deref on NULL.
  {
    Var vptr(buffer->data);
    PrimExpr data_ptr = tvm::if_then_else(
        Not(is_null),
        TVMArrayGet(DataType::Handle(), handle, builtin::kArrData),
        make_zero(DataType::Handle()));
    BindNullable(buffer->data, data_ptr, arg_name + ".data", true, is_null);

    // Check if the data pointer is NULL.  This check is skipped for
    // size-0 arrays and also skipped when handle itself is NULL.
    auto alloc_size = [&]() -> PrimExpr {
      PrimExpr product = IntImm(buffer->DefaultIndexType(), 1);
      for (const auto &dim : buffer->shape)
        product *= dim;
      return product;
    }();
    asserts_.emplace_back(AssertStmt(
        Or(is_null, (alloc_size == 0) ||
                        !Call(DataType::Bool(), builtin::isnullptr(), {vptr})),
        StringImm(arg_name +
                  " is expected to have non-NULL data pointer, but got NULL"),
        nop));

    // mark alignment of external bufs
    init_nest_.emplace_back(
        AttrStmt(vptr, tir::attr::storage_alignment,
                 IntImm(DataType::Int(32), buffer->data_alignment), nop));

    def_handle_dtype_.Set(vptr, tir::TypeAnnotation(buffer->dtype));
  }
}

} // namespace tl
} // namespace tvm

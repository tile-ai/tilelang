/*!
 * \file arg_binder.cc
 * \brief Helper utility to match and bind arguments.
 */
#include "arg_binder.h"

#include <tvm/runtime/device_api.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <sstream>
#include <unordered_set>

#include "../runtime/error_helpers.h"
#include "tir/transforms/ir_utils.h"
#include "tvm/arith/int_solver.h"
#include "tvm/ffi/cast.h"
#include "tvm/ffi/container/array.h"
#include "tvm/tir/stmt.h"
#include "tvm/tir/stmt_functor.h"

namespace tvm {
namespace tl {

using namespace tir;

void BinderAddAssert(arith::Analyzer *ana, PrimExpr cond,
                     const std::string &arg_name, std::vector<Stmt> *asserts,
                     PrimExpr nullable_guard = PrimExpr()) {
  PrimExpr scond = ana->Simplify(cond);
  if (is_zero(scond)) {
    LOG(FATAL) << "Bind have an unmet assertion: " << cond << ", "
               << " on argument " << arg_name;
  }

  if (!is_one(scond)) {
    // Extract kernel/buffer/field from arg_name (e.g., "main.A.shape[0]")
    std::string kernel = arg_name;
    std::string buf_and_field = arg_name;
    size_t dot_pos = arg_name.find('.');
    if (dot_pos != std::string::npos) {
      kernel = arg_name.substr(0, dot_pos);
      buf_and_field = arg_name.substr(dot_pos + 1);
    }
    std::string buffer = buf_and_field;
    std::string field;
    size_t dot2 = buf_and_field.find('.');
    if (dot2 != std::string::npos) {
      buffer = buf_and_field.substr(0, dot2);
      field = buf_and_field.substr(dot2 + 1);
    }

    // If cond is an equality, prefer structured packed error with expect/got
    if (const auto *eq = scond.as<tvm::tir::EQNode>()) {
      PrimExpr lhs = eq->a;
      PrimExpr rhs = eq->b;
      // Choose rhs as expected and lhs as got for better semantics in most
      // binding cases
      ffi::Array<PrimExpr> pargs;
      pargs.push_back(StringImm(tvm_error_expect_eq));
      pargs.push_back(StringImm(kernel));
      pargs.push_back(StringImm(buffer));
      pargs.push_back(StringImm(field.empty() ? std::string("value") : field));
      pargs.push_back(cast(DataType::Int(64), rhs)); // expected
      pargs.push_back(cast(DataType::Int(64), lhs)); // got

      Stmt call_err =
          Evaluate(Call(DataType::Int(32), builtin::tvm_call_packed(), pargs));
      // Only emit at runtime when the equality fails
      Stmt inner = IfThenElse(Not(scond), call_err);
      if (nullable_guard.defined()) {
        inner = IfThenElse(Not(nullable_guard), inner);
      }
      asserts->emplace_back(SeqStmt({inner, Evaluate(0)}));
    } else {
      // Fallback: packed generic constraint violation without dumping cond
      ffi::Array<PrimExpr> pargs;
      pargs.push_back(StringImm(tvm_error_constraint_violation));
      pargs.push_back(StringImm(kernel));
      pargs.push_back(StringImm(buffer));
      pargs.push_back(StringImm(field.empty() ? std::string("value") : field));
      Stmt call_err =
          Evaluate(Call(DataType::Int(32), builtin::tvm_call_packed(), pargs));
      Stmt inner = IfThenElse(Not(scond), call_err);
      if (nullable_guard.defined()) {
        inner = IfThenElse(Not(nullable_guard), inner);
      }
      asserts->emplace_back(SeqStmt({inner, Evaluate(0)}));
    }
  }
}

std::vector<Var> ArgBinder::getUndefVars(const std::vector<PrimExpr> &args) {
  std::unordered_set<const VarNode *> visit;
  std::vector<Var> res;
  for (const auto &arg : args) {
    PostOrderVisit(arg, [&](ObjectRef r) {
      if (auto var = r.as<VarNode>()) {
        if (!visit.count(var)) {
          visit.insert(var);
        }
        auto it = def_map_->find(var);
        if (it == def_map_->end()) {
          // res.push_back(var);
          res.push_back(ffi::GetRef<Var>(var));
        }
      }
    });
  }
  return res;
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
  auto BindVar = [&](const VarNode *v, PrimExpr value) {
    auto v_arg = ffi::GetRef<Var>(v);
    defs_.emplace_back(v_arg);
    if (with_lets) {
      (*def_map_)[v] = value;
      init_nest_.emplace_back(LetStmt(v_arg, value, Evaluate(0)));
    } else {
      (*def_map_)[v] = value;
    }
  };
  // 1. simple binding var = value
  if (const VarNode *v = arg.as<VarNode>()) {
    auto it = def_map_->find(v);
    if (it == def_map_->end()) {
      BindVar(v, value);
      // First time binding: identical behavior as Bind_
      return true;
    } else {
      // Second or later binding: add is_null short-circuit
      PrimExpr cond = value == it->second;
      BinderAddAssert(&analyzer_, cond, arg_name, &asserts_, nullable_guard);
    }
  } else {
    // 2. complex binding expr = value
    //  get undefined variables
    auto undefs = ffi::Array<Var>(getUndefVars({arg}));
    if (!undefs.empty()) {
      // if value is not integer, such as float, we are unable to solve it
      if (!value.dtype().is_int() && !value.dtype().is_uint()) {
        LOG(FATAL) << "Unable to solve non-integer variables " << undefs
                   << " from equation `" << value << "`";
      }
      arith::IntConstraints constraints(undefs, {}, {arg == value});
      auto sol = arith::SolveLinearEquations(constraints);
      if (!sol->dst->variables.empty()) {
        LOG(FATAL) << "TVM is unable to solve variables " << undefs
                   << " from equation " << constraints;
      }
      for (const auto &v : undefs) {
        auto value_opt = sol->src_to_dst.Get(v);
        ICHECK(value_opt->defined())
            << "Unable to solve variable `" << v << "` from expression `"
            << (value == arg) << "`";
        auto value = ffi::GetRef<PrimExpr>(sol->src_to_dst.Get(v)->get());
        BindVar(v.as<VarNode>(), value);
      }
    }
    // we must add the assert again
    //    because the solved expression may contain floordiv (e.g. 3 * m == n
    //    ==>   m = n // 3) we re-compute the constraint to verify the solution
    //    is correct
    PrimExpr cond = value == arg;
    BinderAddAssert(&analyzer_, cond, arg_name, &asserts_, nullable_guard);
  }
  // ICHECK(false);
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
      BinderAddAssert(&analyzer_, value == it->second, arg_name, &asserts_);
    }
  } else {
    BinderAddAssert(&analyzer_, value == arg, arg_name, &asserts_);
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
        BinderAddAssert(&analyzer_, zero == truncmod(offset, factor),
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
                             const std::string &arg_name, bool is_used) {
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
  const PrimExpr &is_null = is_used ? const_false() : is_null_var;
  if (is_used) {
    init_nest_.emplace_back(AssertStmt(
        !is_null_var,
        tvm::tir::StringImm(arg_name + " is expected to have non-NULL pointer"),
        nop));
  }

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
  // Build clearer ndim message with kernel/buffer names
  std::string kernel_nm = arg_name;
  std::string buf_nm = arg_name;
  size_t dot_pos = arg_name.find('.');
  if (dot_pos != std::string::npos) {
    kernel_nm = arg_name.substr(0, dot_pos);
    buf_nm = arg_name.substr(dot_pos + 1);
  }
  // Only check ndim when handle is non-NULL: use packed error helper
  PrimExpr ndim_ok = (a_ndim == v_ndim);
  ffi::Array<PrimExpr> ndim_args;
  ndim_args.push_back(StringImm(tvm_error_ndim_mismatch));
  ndim_args.push_back(StringImm(kernel_nm));
  ndim_args.push_back(StringImm(buf_nm));
  ndim_args.push_back(cast(DataType::Int(64), a_ndim));
  ndim_args.push_back(cast(DataType::Int(64), v_ndim));
  Stmt ndim_call =
      Evaluate(Call(DataType::Int(32), builtin::tvm_call_packed(), ndim_args));
  init_nest_.emplace_back(
      SeqStmt({IfThenElse(Not(is_null), IfThenElse(Not(ndim_ok), ndim_call),
                          Evaluate(0)),
               nop}));
  // type checks
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
    // Build FFI packed call to __tvm_error_dtype_mismatch when mismatch occurs.
    // Only issue the call when handle is non-NULL and cond is false.
    ffi::Array<PrimExpr> packed_args;
    packed_args.push_back(StringImm(tvm_error_dtype_mismatch));
    // Split arg_name of the form "<kernel>.<buffer>" into parts for clearer
    // diagnostics
    std::string kernel_name = arg_name;
    std::string buffer_name = arg_name;
    size_t dot_pos = arg_name.find('.');
    if (dot_pos != std::string::npos) {
      kernel_name = arg_name.substr(0, dot_pos);
      buffer_name = arg_name.substr(dot_pos + 1);
    }
    packed_args.push_back(StringImm(kernel_name));
    packed_args.push_back(StringImm(buffer_name));

    auto i64 = DataType::Int(64);
    // Cast to int64 for FFI function signature
    packed_args.push_back(cast(i64, v_type_code));  // actual_code
    packed_args.push_back(cast(i64, v_type_bits));  // actual_bits
    packed_args.push_back(cast(i64, v_type_lanes)); // actual_lanes
    packed_args.push_back(cast(i64, expect_code));  // expect_code
    packed_args.push_back(cast(i64, expect_bits));  // expect_bits
    packed_args.push_back(cast(i64, expect_lanes)); // expect_lanes

    Stmt call_err = Evaluate(
        Call(DataType::Int(32), builtin::tvm_call_packed(), packed_args));
    // Guard the call: only when handle is not null and cond fails
    Stmt guarded = IfThenElse(Not(is_null) && Not(cond), call_err);
    asserts_.emplace_back(SeqStmt({guarded, nop}));
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
    PrimExpr shape_val =
        cast(buffer->shape[k].dtype(),
             BufferLoad(buf_shape,
                        {IntImm(DataType::Int(32), static_cast<int>(k))}));

    // When first encountering a Var (e.g., m), this will generate:
    //   Let(m, bound_shape_val, ...)
    // Constant dimensions will only generate consistency assertions.
    BindNullable(buffer->shape[k], shape_val, shape_element_name(k), true,
                 is_null);
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
      PrimExpr all_ok = foldl([](PrimExpr a, PrimExpr b,
                                 Span span) { return logical_and(a, b, span); },
                              const_true(1), conds);
      // Packed generic violation for non-compact strides
      std::string kernel_nm3 = arg_name;
      std::string buf_nm3 = arg_name;
      size_t dot_pos3 = arg_name.find('.');
      if (dot_pos3 != std::string::npos) {
        kernel_nm3 = arg_name.substr(0, dot_pos3);
        buf_nm3 = arg_name.substr(dot_pos3 + 1);
      }
      ffi::Array<PrimExpr> pargs4;
      pargs4.push_back(StringImm(tvm_error_constraint_violation));
      pargs4.push_back(StringImm(kernel_nm3));
      pargs4.push_back(StringImm(buf_nm3));
      pargs4.push_back(StringImm("strides"));
      Stmt call_err4 =
          Evaluate(Call(DataType::Int(32), builtin::tvm_call_packed(), pargs4));
      // Only check when strides array is present and condition fails
      Stmt check = IfThenElse(Not(v_strides_is_null),
                              IfThenElse(Not(all_ok), call_err4), Evaluate(0));
      asserts_.emplace_back(SeqStmt({check, Evaluate(0)}));
    }
  } else if (buffer->buffer_type == kAutoBroadcast) {
    PrimExpr stride_from_shape = 1;
    for (size_t i = buffer->shape.size(); i != 0; --i) {
      size_t k = i - 1;
      DataType stride_dtype = buffer->strides[k].dtype();
      PrimExpr explicit_stride =
          cast(stride_dtype,
               BufferLoad(buf_strides,
                          {IntImm(DataType::Int(32), static_cast<int>(k))}));

      PrimExpr stride_val = tvm::if_then_else(
          v_strides_is_null, stride_from_shape, explicit_stride);

      BindNullable(buffer->strides[k], stride_val, stride_element_name(k), true,
                   is_null);
    }
  } else {
    PrimExpr stride_from_shape = 1;

    for (int k = static_cast<int>(buffer->strides.size()) - 1; k >= 0; --k) {
      DataType stride_dtype = buffer->strides[k].dtype();
      PrimExpr explicit_stride =
          cast(stride_dtype,
               BufferLoad(buf_strides, {IntImm(DataType::Int(32), k)}));
      PrimExpr shape_stride = cast(
          stride_dtype, BufferLoad(buf_shape, {IntImm(DataType::Int(32), k)}));

      PrimExpr stride_val = tvm::if_then_else(
          v_strides_is_null, stride_from_shape, explicit_stride);

      BindNullable(buffer->strides[k], stride_val, stride_element_name(k), true,
                   is_null);
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
    PrimExpr ok = (expect_byte_offset == actual_byte_offset);
    ffi::Array<PrimExpr> pargs;
    pargs.push_back(StringImm(tvm_error_byte_offset_mismatch));
    pargs.push_back(StringImm(kernel_nm));
    pargs.push_back(StringImm(buf_nm));
    pargs.push_back(cast(DataType::Int(64), expect_byte_offset));
    pargs.push_back(cast(DataType::Int(64), actual_byte_offset));
    Stmt call_err =
        Evaluate(Call(DataType::Int(32), builtin::tvm_call_packed(), pargs));
    asserts_.emplace_back(SeqStmt(
        {IfThenElse(Not(is_null), IfThenElse(Not(ok), call_err), Evaluate(0)),
         nop}));
  } else {
    PrimExpr actual_byte_offset = tvm::if_then_else(
        Not(is_null),
        TVMArrayGet(DataType::UInt(64), handle, builtin::kArrByteOffset),
        make_const(DataType::UInt(64), 0));
    PrimExpr expect_elem_off =
        cast(buffer->elem_offset.dtype(),
             (actual_byte_offset / make_const(DataType::UInt(64), data_bytes)));

    BindNullable(buffer->elem_offset, expect_elem_off,
                 arg_name + ".elem_offset", true, is_null);

    if (buffer->offset_factor > 1) {
      PrimExpr offset = buffer->elem_offset;
      PrimExpr factor = make_const(offset.dtype(), buffer->offset_factor);
      PrimExpr zero = make_zero(offset.dtype());
      BindNullable(offset, truncmod(offset, factor), arg_name + ".elem_offset",
                   true, is_null);
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
  {
    PrimExpr ok = (device_type == actual_dev_type);
    ffi::Array<PrimExpr> pargs2;
    pargs2.push_back(StringImm(tvm_error_device_type_mismatch));
    pargs2.push_back(StringImm(kernel_nm));
    pargs2.push_back(StringImm(buf_nm));
    pargs2.push_back(cast(DataType::Int(64), device_type));
    pargs2.push_back(cast(DataType::Int(64), actual_dev_type));
    Stmt call_err2 =
        Evaluate(Call(DataType::Int(32), builtin::tvm_call_packed(), pargs2));
    asserts_.emplace_back(SeqStmt(
        {IfThenElse(Not(is_null), IfThenElse(Not(ok), call_err2), Evaluate(0)),
         Evaluate(0)}));
  }

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
    // Improve message: kernel/buffer naming for data pointer null check
    std::string kernel_nm2 = arg_name;
    std::string buf_nm2 = arg_name;
    size_t dot_pos2 = arg_name.find('.');
    if (dot_pos2 != std::string::npos) {
      kernel_nm2 = arg_name.substr(0, dot_pos2);
      buf_nm2 = arg_name.substr(dot_pos2 + 1);
    }
    // expand combined condition via nested IfThenElse for portability
    ffi::Array<PrimExpr> pargs3;
    pargs3.push_back(StringImm(tvm_error_null_ptr));
    pargs3.push_back(StringImm(kernel_nm2));
    pargs3.push_back(StringImm(buf_nm2));
    pargs3.push_back(StringImm("data pointer"));
    Stmt call_err3 =
        Evaluate(Call(DataType::Int(32), builtin::tvm_call_packed(), pargs3));
    asserts_.emplace_back(SeqStmt(
        {IfThenElse(Not(is_null),
                    IfThenElse(Not(alloc_size == 0),
                               IfThenElse(Call(DataType::Bool(),
                                               builtin::isnullptr(), {vptr}),
                                          call_err3),
                               Evaluate(0)),
                    Evaluate(0)),
         nop}));

    // mark alignment of external bufs
    init_nest_.emplace_back(
        AttrStmt(vptr, tir::attr::storage_alignment,
                 IntImm(DataType::Int(32), buffer->data_alignment), nop));

    def_handle_dtype_.Set(vptr, tir::TypeAnnotation(buffer->dtype));
  }
}

} // namespace tl
} // namespace tvm

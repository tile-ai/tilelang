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
 * \file legalize_pto_vmi_memory_access.cc
 * \brief LegalizePTOVMIMemoryAccess: build and verify the explicit VMI memory
 * descriptor before scheduling.
 *
 * Ordinary (automatic) VMI accesses carry a scalar-origin BufferLoad address
 * and an instruction scalar offset.  This pass replaces the address with the
 * formal ``tl.access_ptr(BufferLoad, extent, rw_mask)`` descriptor, proving in
 * one shared acceptance exit that the descriptor preserves the source address,
 * contains the complete physical access envelope and covers the declared
 * extent under the existing logical pointer-region analysis.  Indexed
 * gather/scatter accesses, whether automatic or caller-written through
 * one direct handle Bind/Let, are rebuilt at the unique active allocation
 * owner origin after the owner identity, layout, capacity and stable-coverage
 * proofs and the bound-address snapshot check.  An ordinary caller-written
 * explicit descriptor is never rewritten: only its coverage is verified and
 * the instruction offset is kept.  There is no whole-allocation fallback for
 * an ordinary access.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "arith/ir_mutator_with_analyzer.h"
#include "ascend/target_utils.h"
#include "span_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

constexpr const char *kContext = "LegalizePTOVMIMemoryAccess";
constexpr const char *kCoveragePrefix =
    "the existing logical pointer-region analysis does not cover the complete "
    "access";

[[noreturn]] void Fail(const std::string &message, const Span &span) {
  LOG(FATAL) << kContext << ": " << message << SpanHintSuffix(span);
}

[[noreturn]] void Fail(const std::string &message) { Fail(message, Span()); }

const Op &VloadOp() {
  static const Op &op = Op::Get("tl.vmi.vload");
  return op;
}

const Op &VstoreOp() {
  static const Op &op = Op::Get("tl.vmi.vstore");
  return op;
}

const Op &VgatherOp() {
  static const Op &op = Op::Get("tl.vmi.vgather");
  return op;
}

const Op &VscatterOp() {
  static const Op &op = Op::Get("tl.vmi.vscatter");
  return op;
}

const Op &AccessPtrOp() {
  static const Op &op = Op::Get("tl.access_ptr");
  return op;
}

std::optional<int64_t> ConstInt(const PrimExpr &expr) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    if (!imm->dtype.is_bool()) {
      return imm->value;
    }
  }
  return std::nullopt;
}

PrimExpr IntConst(int64_t value) {
  if (value >= std::numeric_limits<int32_t>::min() &&
      value <= std::numeric_limits<int32_t>::max()) {
    return IntImm(DataType::Int(32), value);
  }
  return IntImm(DataType::Int(64), value);
}

// A value can be widened to signed 64-bit without changing its value for every
// signed integer and for unsigned integers narrower than 64 bits.  An unsigned
// 64-bit value is safe only when its range is proven finite; otherwise the
// widening cast could reinterpret values above INT64_MAX.
bool RepresentableInInt64(arith::Analyzer *analyzer, const PrimExpr &value) {
  const DataType dtype = value.dtype();
  if (!dtype.is_uint() || dtype.bits() < 64) {
    return true;
  }
  return analyzer->const_int_bound(value)->max_value !=
         arith::ConstIntBound::kPosInf;
}

PrimExpr WidenToInt64(const PrimExpr &value) {
  if (value.dtype().is_int() && value.dtype().bits() == 64) {
    return value;
  }
  return Cast(DataType::Int(64), value);
}

bool IsIntegerScalar(const PrimExpr &expr) {
  DataType dtype = expr.dtype();
  return dtype.is_scalar() && (dtype.is_int() || dtype.is_uint());
}

int64_t CheckedAdd(int64_t lhs, int64_t rhs, const std::string &op_name,
                   const Span &span) {
  __int128 result = static_cast<__int128>(lhs) + rhs;
  if (result < std::numeric_limits<int64_t>::min() ||
      result > std::numeric_limits<int64_t>::max()) {
    Fail(op_name + " integer arithmetic overflows int64 while sizing the "
                   "access envelope",
         span);
  }
  return static_cast<int64_t>(result);
}

int64_t CheckedMul(int64_t lhs, int64_t rhs, const std::string &op_name,
                   const Span &span) {
  __int128 result = static_cast<__int128>(lhs) * rhs;
  if (result < std::numeric_limits<int64_t>::min() ||
      result > std::numeric_limits<int64_t>::max()) {
    Fail(op_name + " integer arithmetic overflows int64 while sizing the "
                   "access envelope",
         span);
  }
  return static_cast<int64_t>(result);
}

int64_t CheckedNeg(int64_t value, const std::string &op_name,
                   const Span &span) {
  __int128 result = -static_cast<__int128>(value);
  if (result > std::numeric_limits<int64_t>::max()) {
    Fail(op_name + " integer arithmetic overflows int64 while sizing the "
                   "access envelope",
         span);
  }
  return static_cast<int64_t>(result);
}

int64_t CeilDiv(int64_t value, int64_t divisor, const std::string &op_name,
                const Span &span) {
  TVM_FFI_ICHECK_GT(divisor, 0);
  __int128 numerator = static_cast<__int128>(value) + divisor - 1;
  if (numerator > std::numeric_limits<int64_t>::max()) {
    Fail(op_name + " integer arithmetic overflows int64 while sizing the "
                   "access envelope",
         span);
  }
  return static_cast<int64_t>(numerator) / divisor;
}

std::optional<ObjectRef> FindAnnotation(const CallNode *call, const char *key) {
  auto it = call->annotations.find(String(key));
  if (it == call->annotations.end()) {
    return std::nullopt;
  }
  return (*it).second;
}

std::optional<PrimExpr> AnnotationExpr(const CallNode *call, const char *key,
                                       const std::string &op_name,
                                       const Span &span) {
  std::optional<ObjectRef> value = FindAnnotation(call, key);
  if (!value.has_value()) {
    return std::nullopt;
  }
  std::optional<PrimExpr> expr = value->as<PrimExpr>();
  if (!expr.has_value()) {
    Fail(op_name + " annotation `" + key + "` must be a scalar expression",
         span);
  }
  return expr;
}

std::optional<int64_t> AnnotationInt(const CallNode *call, const char *key,
                                     const std::string &op_name,
                                     const Span &span) {
  std::optional<PrimExpr> expr = AnnotationExpr(call, key, op_name, span);
  if (!expr.has_value()) {
    return std::nullopt;
  }
  return ConstInt(expr.value());
}

std::optional<std::string> AnnotationString(const CallNode *call,
                                            const char *key,
                                            const std::string &op_name,
                                            const Span &span) {
  std::optional<ObjectRef> value = FindAnnotation(call, key);
  if (!value.has_value()) {
    return std::nullopt;
  }
  if (const auto *imm = value->as<StringImmNode>()) {
    return std::string(imm->value.c_str(), imm->value.size());
  }
  Fail(op_name + " annotation `" + key + "` must be a string", span);
}

// ---------------------------------------------------------------------------
// Buffer and index helpers
// ---------------------------------------------------------------------------
bool IsUbScope(const String &scope) {
  // These are TileLang TIR Buffer scopes; PTO's ``ub`` pointer address space
  // is emitted later and is not a valid Buffer scope.
  return scope == "shared" || scope == "shared.dyn";
}

void ValidateUbBuffer(const Buffer &buffer, const std::string &op_name,
                      const Span &span) {
  String scope = buffer.scope();
  if (!IsUbScope(scope)) {
    Fail(op_name + " requires a UB pointer, got buffer scope '" +
             std::string(scope.c_str(), scope.size()) + "'",
         span);
  }
  if (!buffer->axis_separators.empty()) {
    Fail(op_name +
             " requires a single scalar linear buffer address, but buffer '" +
             std::string(buffer->name.c_str(), buffer->name.size()) +
             "' declares axis separators",
         span);
  }
}

Array<PrimExpr> BufferZeros(const Buffer &buffer) {
  DataType dtype =
      buffer->shape.empty() ? DataType::Int(32) : buffer->shape[0].dtype();
  Array<PrimExpr> starts;
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    starts.push_back(IntImm(dtype, 0));
  }
  return starts;
}

Array<PrimExpr> LoadStarts(const BufferLoadNode *load, const Buffer &buffer,
                           bool allow_ramp, const std::string &op_name,
                           const Span &span) {
  if (load->indices.size() != buffer->shape.size()) {
    Fail(op_name + " requires a full-rank buffer address, got " +
             std::to_string(load->indices.size()) + " indices for rank " +
             std::to_string(buffer->shape.size()),
         span);
  }
  Array<PrimExpr> starts;
  for (PrimExpr index : load->indices) {
    if (allow_ramp) {
      if (const auto *ramp = index.as<RampNode>()) {
        index = ramp->base;
      }
    }
    if (!IsIntegerScalar(index)) {
      Fail(op_name +
               " requires scalar integer buffer indices; vector indices cannot "
               "be reduced to one address start",
           span);
    }
    starts.push_back(index);
  }
  return starts;
}

void ValidateLinearAddress(const Buffer &buffer, const Array<PrimExpr> &starts,
                           const std::string &op_name, const Span &span) {
  Array<PrimExpr> offsets = buffer.OffsetOf(starts);
  Array<PrimExpr> origin = buffer.OffsetOf(BufferZeros(buffer));
  if (offsets.size() != 1 || origin.size() != 1) {
    Fail(op_name +
             " requires a single scalar linear buffer address; axis separators "
             "and multiple address spaces are not supported",
         span);
  }
}

std::optional<PrimExpr> RelativeOffset(const Buffer &buffer,
                                       const Array<PrimExpr> &starts) {
  if (buffer->shape.empty() || starts.size() != buffer->shape.size()) {
    return std::nullopt;
  }
  Array<PrimExpr> offsets = buffer.OffsetOf(starts);
  Array<PrimExpr> origin = buffer.OffsetOf(BufferZeros(buffer));
  if (offsets.size() != 1 || origin.size() != 1) {
    return std::nullopt;
  }
  return offsets[0] - origin[0];
}

bool IsTightBuffer(const Buffer &buffer) {
  size_t ndim = buffer->shape.size();
  if (ndim == 0) {
    return false;
  }
  if (buffer->strides.empty()) {
    return true;
  }
  if (buffer->strides.size() != ndim) {
    return false;
  }
  int64_t expected = 1;
  for (size_t dim = ndim; dim-- > 0;) {
    std::optional<int64_t> stride = ConstInt(buffer->strides[dim]);
    if (!stride.has_value() || stride.value() != expected) {
      return false;
    }
    if (dim > 0) {
      std::optional<int64_t> length = ConstInt(buffer->shape[dim]);
      if (!length.has_value() || length.value() <= 0) {
        return false;
      }
      __int128 next = static_cast<__int128>(expected) * length.value();
      if (next > std::numeric_limits<int64_t>::max()) {
        return false;
      }
      expected = static_cast<int64_t>(next);
    }
  }
  return true;
}

bool IsUnitStride1d(const Buffer &buffer) {
  if (buffer->shape.size() != 1) {
    return false;
  }
  if (buffer->strides.empty()) {
    return true;
  }
  if (buffer->strides.size() != 1) {
    return false;
  }
  std::optional<int64_t> stride = ConstInt(buffer->strides[0]);
  return stride.has_value() && stride.value() == 1;
}

// ---------------------------------------------------------------------------
// Logical coverage proofs, ported from the frontend's existing consumer
// ---------------------------------------------------------------------------
std::optional<std::string> LocalCoverageReason(arith::Analyzer *analyzer,
                                               const Buffer &buffer,
                                               const Array<PrimExpr> &starts,
                                               const PrimExpr &extent) {
  size_t ndim = buffer->shape.size();
  if (ndim == 0) {
    return std::string(kCoveragePrefix) + ": the buffer has no dimensions";
  }
  if (starts.size() != ndim) {
    return std::string(kCoveragePrefix) +
           ": the descriptor start arity does not match the buffer rank";
  }
  if (!IsTightBuffer(buffer)) {
    return std::string(kCoveragePrefix) +
           ": the address is a local coordinate of a buffer whose layout is "
           "not "
           "proven tight row-major, so no accepted local coverage proof "
           "applies";
  }
  std::optional<int64_t> constant = ConstInt(extent);
  if (!constant.has_value()) {
    if (!analyzer->CanProve(extent > 0)) {
      return std::string(kCoveragePrefix) +
             ": cannot prove the descriptor extent is positive";
    }
    if (ndim > 1) {
      size_t dim = ndim - 1;
      PrimExpr start = starts[dim];
      if (!analyzer->CanProve(start >= 0) ||
          !analyzer->CanProve(start + extent <= buffer->shape[dim])) {
        std::ostringstream oss;
        oss << kCoveragePrefix
            << ": a non-constant extent must stay inside the innermost "
               "dimension "
            << dim << " (length " << buffer->shape[dim] << ", start " << start
            << "), but that inclusion cannot be proven";
        return oss.str();
      }
    }
    return std::nullopt;
  }
  int64_t rem = constant.value();
  if (rem <= 0) {
    return std::string(kCoveragePrefix) +
           ": the descriptor extent must be positive, got " +
           std::to_string(rem);
  }
  // Dim 0 is the outermost axis: there is no further outward carry to declare.
  for (size_t dim = ndim - 1; dim > 0; --dim) {
    if (rem <= 1) {
      return std::nullopt;
    }
    PrimExpr start = starts[dim];
    std::optional<int64_t> length = ConstInt(buffer->shape[dim]);
    if (!length.has_value()) {
      if (!analyzer->CanProve(start >= 0) ||
          !analyzer->CanProve(start + IntConst(rem) <= buffer->shape[dim])) {
        std::ostringstream oss;
        oss << kCoveragePrefix << ": " << rem
            << " remaining elements must stay inside the non-constant "
               "dimension "
            << dim << " (start " << start
            << "), but that inclusion cannot be proven";
        return oss.str();
      }
      return std::nullopt;
    }
    if (length.value() <= 0) {
      return std::string(kCoveragePrefix) + ": dimension " +
             std::to_string(dim) + " has non-positive length " +
             std::to_string(length.value());
    }
    int64_t rnext = CeilDiv(rem, length.value(), kContext, Span());
    __int128 bound = static_cast<__int128>(rnext) * length.value();
    bool fits = true;
    bool carries = false;
    if (bound <= std::numeric_limits<int64_t>::max()) {
      carries = analyzer->CanProve(start + IntConst(rem) <=
                                   IntConst(static_cast<int64_t>(bound)));
    } else {
      std::optional<int64_t> start_const = ConstInt(start);
      if (start_const.has_value()) {
        carries = static_cast<__int128>(start_const.value()) + rem <= bound;
      } else {
        fits = false;
      }
    }
    if (!analyzer->CanProve(start >= 0) ||
        !analyzer->CanProve(start < IntConst(length.value())) || !carries) {
      std::ostringstream oss;
      oss << kCoveragePrefix << ": dimension " << dim << " (length "
          << length.value() << ") starts at " << start << " and carries " << rem
          << " elements, but the existing row-major carry only declares "
          << rnext << " dimension(s) from the start";
      if (!fits) {
        oss << " (the static bound does not fit int64)";
      }
      return oss.str();
    }
    rem = rnext;
  }
  return std::nullopt;
}

std::optional<std::string> OriginPrefixReason(arith::Analyzer *analyzer,
                                              const Buffer &buffer,
                                              const PrimExpr &extent,
                                              bool require_static_shape) {
  if (buffer->shape.empty()) {
    return std::string(kCoveragePrefix) + ": the buffer has no dimensions";
  }
  for (size_t dim = 0; dim < buffer->shape.size(); ++dim) {
    std::optional<int64_t> length = ConstInt(buffer->shape[dim]);
    if (length.has_value() && length.value() <= 0) {
      return std::string(kCoveragePrefix) + ": dimension " +
             std::to_string(dim) + " has non-positive length " +
             std::to_string(length.value());
    }
    if (require_static_shape && !length.has_value()) {
      return std::string(kCoveragePrefix) +
             ": the origin descriptor has a "
             "non-constant dimension " +
             std::to_string(dim) + ", so no stable dense-prefix proof applies";
    }
  }
  std::optional<int64_t> constant = ConstInt(extent);
  if (constant.has_value()) {
    if (constant.value() <= 0) {
      return std::string(kCoveragePrefix) +
             ": the descriptor extent must be positive, got " +
             std::to_string(constant.value());
    }
    return std::nullopt;
  }
  if (!analyzer->CanProve(extent > 0)) {
    return std::string(kCoveragePrefix) +
           ": cannot prove the descriptor extent is positive";
  }
  return std::nullopt;
}

std::optional<std::string>
WholeDensePermutationReason(const Buffer &buffer, const Array<PrimExpr> &starts,
                            const PrimExpr &extent) {
  if (buffer->strides.empty() ||
      buffer->strides.size() != buffer->shape.size()) {
    return std::string("the buffer has no explicit permutation strides");
  }
  for (const PrimExpr &start : starts) {
    std::optional<int64_t> value = ConstInt(start);
    if (!value.has_value() || value.value() != 0) {
      return std::string("the descriptor start is not the origin");
    }
  }
  std::vector<int64_t> dims;
  int64_t total = 1;
  for (const PrimExpr &shape : buffer->shape) {
    std::optional<int64_t> length = ConstInt(shape);
    if (!length.has_value() || length.value() <= 0) {
      return std::string("the shape is not static and positive");
    }
    dims.push_back(length.value());
    __int128 next = static_cast<__int128>(total) * length.value();
    if (next > std::numeric_limits<int64_t>::max()) {
      return std::string("the whole dense span does not fit int64");
    }
    total = static_cast<int64_t>(next);
  }
  std::optional<int64_t> constant = ConstInt(extent);
  if (!constant.has_value() || constant.value() != total) {
    return std::string(
               "the descriptor extent is not exactly the whole dense span ") +
           std::to_string(total);
  }
  std::vector<std::pair<int64_t, int64_t>> axes;
  for (size_t dim = 0; dim < dims.size(); ++dim) {
    std::optional<int64_t> stride = ConstInt(buffer->strides[dim]);
    if (!stride.has_value() || stride.value() < 0) {
      return std::string("a stride is not a static non-negative integer");
    }
    if (dims[dim] == 1) {
      continue;
    }
    axes.emplace_back(stride.value(), dims[dim]);
  }
  std::sort(axes.begin(), axes.end());
  int64_t expected = 1;
  for (const auto &[stride, length] : axes) {
    if (stride != expected) {
      return std::string("stride ") + std::to_string(stride) +
             " does not continue the dense permutation at " +
             std::to_string(expected);
    }
    __int128 next = static_cast<__int128>(expected) * length;
    if (next > std::numeric_limits<int64_t>::max()) {
      return std::string("the whole dense span does not fit int64");
    }
    expected = static_cast<int64_t>(next);
  }
  return std::nullopt;
}

bool PaddedRowMajor(arith::Analyzer *analyzer, const Buffer &buffer) {
  size_t ndim = buffer->shape.size();
  if (ndim == 0 || buffer->strides.empty() || buffer->strides.size() != ndim) {
    return false;
  }
  int64_t dense_stride = 1;
  for (size_t dim = ndim; dim-- > 0;) {
    if (!analyzer->CanProve(buffer->strides[dim] >= IntConst(dense_stride))) {
      return false;
    }
    if (dim > 0) {
      std::optional<int64_t> length = ConstInt(buffer->shape[dim]);
      if (!length.has_value() || length.value() <= 0) {
        return false;
      }
      __int128 next = static_cast<__int128>(dense_stride) * length.value();
      if (next > std::numeric_limits<int64_t>::max()) {
        return false;
      }
      dense_stride = static_cast<int64_t>(next);
    }
  }
  return true;
}

std::optional<std::string> ExplicitCoverageReason(arith::Analyzer *analyzer,
                                                  const Buffer &buffer,
                                                  const Array<PrimExpr> &starts,
                                                  const PrimExpr &extent) {
  if (IsTightBuffer(buffer)) {
    return LocalCoverageReason(analyzer, buffer, starts, extent);
  }
  bool origin = true;
  for (const PrimExpr &start : starts) {
    std::optional<int64_t> value = ConstInt(start);
    if (!value.has_value() || value.value() != 0) {
      origin = false;
      break;
    }
  }
  if (origin && !buffer->strides.empty() &&
      buffer->strides.size() == buffer->shape.size() &&
      PaddedRowMajor(analyzer, buffer)) {
    return OriginPrefixReason(analyzer, buffer, extent,
                              /*require_static_shape=*/true);
  }
  std::optional<std::string> permutation_reason =
      WholeDensePermutationReason(buffer, starts, extent);
  if (!permutation_reason.has_value()) {
    return std::nullopt;
  }
  return std::string(kCoveragePrefix) +
         ": the descriptor starts at a local coordinate of a buffer whose "
         "layout "
         "is not proven tight row-major, and neither an origin-prefix nor an "
         "exact whole dense permutation proof applies (" +
         permutation_reason.value() + ")";
}

std::optional<PrimExpr> WholeBufferSpan(arith::Analyzer *analyzer,
                                        const Buffer &buffer) {
  if (!IsTightBuffer(buffer)) {
    return std::nullopt;
  }
  if (buffer->shape.size() == 1) {
    std::optional<int64_t> length = ConstInt(buffer->shape[0]);
    if (length.has_value()) {
      if (length.value() <= 0) {
        return std::nullopt;
      }
      return IntConst(length.value());
    }
    if (analyzer->CanProve(buffer->shape[0] > 0)) {
      return buffer->shape[0];
    }
    return std::nullopt;
  }
  int64_t span = 1;
  for (const PrimExpr &shape : buffer->shape) {
    std::optional<int64_t> length = ConstInt(shape);
    if (!length.has_value() || length.value() <= 0) {
      return std::nullopt;
    }
    __int128 next = static_cast<__int128>(span) * length.value();
    if (next > std::numeric_limits<int64_t>::max()) {
      return std::nullopt;
    }
    span = static_cast<int64_t>(next);
  }
  return IntConst(span);
}

// ---------------------------------------------------------------------------
// Access envelope
// ---------------------------------------------------------------------------
struct VmiModes {
  std::optional<std::string> dist_mode;
  std::optional<PrimExpr> stride;
  std::optional<PrimExpr> block_stride;
  std::optional<PrimExpr> group;
};

VmiModes ParseModes(const CallNode *call, const std::string &op_name,
                    bool is_load, const Span &span) {
  VmiModes modes;
  modes.dist_mode = AnnotationString(call, "dist_mode", op_name, span);
  modes.stride = AnnotationExpr(call, "stride", op_name, span);
  modes.block_stride = AnnotationExpr(call, "block_stride", op_name, span);
  modes.group = AnnotationExpr(call, "group", op_name, span);

  if (modes.dist_mode.has_value()) {
    const std::string &mode = modes.dist_mode.value();
    bool supported =
        mode == "continuous" || mode == "dintlv" || (is_load && mode == "brc");
    if (!supported) {
      Fail(op_name + " does not support dist_mode=\"" + mode + "\"", span);
    }
  }

  if (modes.group.has_value()) {
    if (modes.dist_mode.has_value() &&
        (!is_load || modes.dist_mode.value() != "brc")) {
      Fail(op_name + " does not allow dist_mode together with group", span);
    }
    if (modes.block_stride.has_value()) {
      Fail(op_name + " does not allow block_stride together with group", span);
    }
    if (!modes.stride.has_value()) {
      Fail(op_name + " with group=... requires stride", span);
    }
    return modes;
  }

  if (modes.block_stride.has_value()) {
    if (modes.dist_mode.has_value()) {
      Fail(op_name + " does not allow dist_mode together with block_stride",
           span);
    }
    if (modes.stride.has_value()) {
      Fail(op_name + " does not allow stride together with block_stride", span);
    }
    return modes;
  }

  if (modes.stride.has_value()) {
    Fail(op_name + " accepts stride only when group is provided", span);
  }
  return modes;
}

int64_t GroupConstant(const std::optional<PrimExpr> &group,
                      const std::string &op_name, const Span &span) {
  std::optional<int64_t> value = ConstInt(group.value());
  if (!value.has_value()) {
    Fail(op_name + " group has no proven bounds for this buffer view", span);
  }
  if (value.value() <= 0) {
    Fail(op_name + " requires group to be a positive integer, got " +
             std::to_string(value.value()),
         span);
  }
  return value.value();
}

std::pair<int64_t, int64_t> GroupEnvelope(const VmiModes &modes, int64_t lanes,
                                          DataType elem_dtype, bool is_load,
                                          const std::string &op_name,
                                          const Span &span) {
  int64_t group = GroupConstant(modes.group, op_name, span);
  bool brc = modes.dist_mode.has_value() && modes.dist_mode.value() == "brc";
  int64_t per_group;
  if (brc || lanes <= group) {
    per_group = 1;
  } else {
    per_group = CeilDiv(lanes, group, op_name, span);
    if (is_load) {
      int bits = elem_dtype.bits();
      if (bits <= 0) {
        Fail(op_name +
                 " the physical element width is unknown, so the complete "
                 "physical vector width cannot be computed",
             span);
      }
      int64_t vector_elements = std::max<int64_t>(1, 2048 / bits);
      per_group = CheckedMul(CeilDiv(per_group, vector_elements, op_name, span),
                             vector_elements, op_name, span);
    }
  }
  std::optional<int64_t> stride = ConstInt(modes.stride.value());
  if (!stride.has_value()) {
    if (group == 1) {
      return {0, per_group};
    }
    Fail(op_name + " group stride has no proven bounds for this buffer view",
         span);
  }
  int64_t displacement = CheckedMul(group - 1, stride.value(), op_name, span);
  int64_t low = std::min<int64_t>(0, displacement);
  int64_t high =
      CheckedAdd(std::max<int64_t>(0, displacement), per_group, op_name, span);
  return {low, high};
}

std::pair<int64_t, int64_t> BlockEnvelope(const std::optional<PrimExpr> &block,
                                          int64_t lanes, DataType elem_dtype,
                                          const std::string &op_name,
                                          const Span &span) {
  int bits = elem_dtype.bits();
  if (bits <= 0) {
    Fail(op_name + " the physical element width is unknown, so the complete "
                   "block-stride footprint cannot be computed",
         span);
  }
  int64_t elements_per_block = std::max<int64_t>(1, 256 / bits);
  int64_t vector_elements = std::max<int64_t>(1, 2048 / bits);
  int64_t chunks = CeilDiv(lanes, vector_elements, op_name, span);
  int64_t blocks_per_chunk =
      std::max<int64_t>(1, vector_elements / elements_per_block);
  int64_t blocks = CheckedMul(chunks, blocks_per_chunk, op_name, span);
  std::optional<int64_t> block_const = ConstInt(block.value());
  if (!block_const.has_value()) {
    Fail(op_name + " block stride has no proven bounds for this buffer view",
         span);
  }
  if (block_const.value() < 0) {
    Fail(op_name +
             " negative block-stride control words have no proven physical "
             "bounds",
         span);
  }
  int64_t high =
      CheckedAdd(CheckedMul(CheckedMul(CheckedAdd(blocks, -1, op_name, span),
                                       block_const.value(), op_name, span),
                            elements_per_block, op_name, span),
                 elements_per_block, op_name, span);
  return {0, high};
}

std::pair<int64_t, int64_t> AccessEnvelope(const VmiModes &modes, int64_t lanes,
                                           DataType elem_dtype, bool is_load,
                                           const std::string &op_name,
                                           const Span &span) {
  if (modes.dist_mode.has_value() && modes.dist_mode.value() == "dintlv") {
    return {0, CheckedMul(2, lanes, op_name, span)};
  }
  if (modes.group.has_value()) {
    return GroupEnvelope(modes, lanes, elem_dtype, is_load, op_name, span);
  }
  if (modes.block_stride.has_value()) {
    return BlockEnvelope(modes.block_stride, lanes, elem_dtype, op_name, span);
  }
  if (elem_dtype.is_float4_e2m1fn()) {
    // Genuine FP4 descriptors stay in logical element units; the later
    // RewriteFp4ToFp4x2 performs the single physical packing.
    return {0, lanes};
  }
  if (modes.dist_mode.has_value() && modes.dist_mode.value() == "brc") {
    return {0, 1};
  }
  if (is_load) {
    int bits = elem_dtype.bits();
    if (bits <= 0) {
      Fail(op_name + " the physical element width is unknown, so the complete "
                     "physical vector width cannot be computed",
           span);
    }
    int64_t vector_elements = std::max<int64_t>(1, 2048 / bits);
    return {0, CheckedMul(CeilDiv(lanes, vector_elements, op_name, span),
                          vector_elements, op_name, span)};
  }
  return {0, lanes};
}

// ---------------------------------------------------------------------------
// Address parsing
// ---------------------------------------------------------------------------
struct AccessAddress {
  bool explicit_address{false};
  // True when the source operand is a handle Var resolved through a direct
  // Bind/Let to tl.access_ptr.  Indexed descriptors copied from such a binding
  // must not re-read memory or execute calls at the use point.
  bool from_binding{false};
  PrimExpr source_operand;
  Buffer buffer;
  Array<PrimExpr> starts;
  PrimExpr extent;
  int rw_mask{0};
};

AccessAddress ParseAccessPtrCall(const CallNode *call,
                                 const PrimExpr &source_operand, bool is_load,
                                 const std::string &op_name, const Span &span) {
  if (call->args.size() != 3) {
    Fail(op_name + " requires a three-argument tl.access_ptr descriptor, got " +
             std::to_string(call->args.size()),
         span);
  }
  PrimExpr base = call->args[0];
  PrimExpr extent = call->args[1];
  PrimExpr mask = call->args[2];
  const auto *base_load = base.as<BufferLoadNode>();
  if (base_load == nullptr) {
    Fail(op_name +
             " requires tl.access_ptr to be constructed from a BufferLoad base",
         span);
  }
  const auto *mask_imm = mask.as<IntImmNode>();
  if (mask_imm == nullptr || mask_imm->dtype.is_bool() ||
      (mask_imm->value != 1 && mask_imm->value != 2 && mask_imm->value != 3)) {
    Fail(op_name +
             " requires a valid tl.access_ptr rw_mask (1, 2, or 3), got wrong "
             "operand",
         span);
  }
  int rw_mask = static_cast<int>(mask_imm->value);
  if (is_load && (rw_mask & 1) == 0) {
    Fail(op_name +
             " requires the explicit descriptor to declare read access, got "
             "rw_mask=" +
             std::to_string(rw_mask),
         span);
  }
  if (!is_load && (rw_mask & 2) == 0) {
    Fail(op_name +
             " requires the explicit descriptor to declare write access, got "
             "rw_mask=" +
             std::to_string(rw_mask),
         span);
  }
  bool stable = true;
  PostOrderVisit(extent, [&](const ObjectRef &node) {
    if (!stable) {
      return;
    }
    if (node.as<VarNode>() || node.as<IntImmNode>() || node.as<CastNode>() ||
        node.as<AddNode>() || node.as<SubNode>() || node.as<MulNode>() ||
        node.as<FloorDivNode>() || node.as<FloorModNode>() ||
        node.as<MinNode>() || node.as<MaxNode>()) {
      return;
    }
    stable = false;
  });
  if (!IsIntegerScalar(extent) || !stable) {
    Fail(op_name +
             " requires a stable scalar integer tl.access_ptr extent, got a "
             "non-stable expression",
         span);
  }
  Buffer buffer = base_load->buffer;
  ValidateUbBuffer(buffer, op_name, span);
  Array<PrimExpr> starts =
      LoadStarts(base_load, buffer, /*allow_ramp=*/false, op_name, span);
  ValidateLinearAddress(buffer, starts, op_name, span);
  AccessAddress address;
  address.explicit_address = true;
  address.source_operand = source_operand;
  address.buffer = buffer;
  address.starts = starts;
  address.extent = extent;
  address.rw_mask = rw_mask;
  return address;
}

AccessAddress ParseAutoAddress(const PrimExpr &operand,
                               const std::string &op_name, const Span &span) {
  if (const auto *load = operand.as<BufferLoadNode>()) {
    Buffer buffer = load->buffer;
    ValidateUbBuffer(buffer, op_name, span);
    Array<PrimExpr> starts =
        LoadStarts(load, buffer, /*allow_ramp=*/true, op_name, span);
    ValidateLinearAddress(buffer, starts, op_name, span);
    AccessAddress address;
    address.explicit_address = false;
    address.source_operand = BufferLoad(buffer, starts);
    address.buffer = buffer;
    address.starts = starts;
    return address;
  }
  if (std::optional<Buffer> buffer = operand.as<Buffer>()) {
    ValidateUbBuffer(buffer.value(), op_name, span);
    Array<PrimExpr> starts = BufferZeros(buffer.value());
    ValidateLinearAddress(buffer.value(), starts, op_name, span);
    AccessAddress address;
    address.explicit_address = false;
    address.source_operand = BufferLoad(buffer.value(), starts);
    address.buffer = buffer.value();
    address.starts = starts;
    return address;
  }
  Fail(op_name +
           " requires a Buffer address or an explicit tl.access_ptr; got an "
           "unsupported operand",
       span);
}

std::optional<PrimExpr>
LookupBind(const std::vector<std::pair<Var, PrimExpr>> &bindings,
           const Var &var) {
  for (auto it = bindings.rbegin(); it != bindings.rend(); ++it) {
    if (it->first.same_as(var)) {
      return it->second;
    }
  }
  return std::nullopt;
}

AccessAddress
ParseAddress(const PrimExpr &operand, bool is_load, const std::string &op_name,
             const Span &span,
             const std::vector<std::pair<Var, PrimExpr>> &bindings) {
  if (const auto *var = operand.as<VarNode>()) {
    Var handle = GetRef<Var>(var);
    if (!handle.dtype().is_handle()) {
      Fail(op_name +
               " requires a Buffer address or an explicit tl.access_ptr; got a "
               "scalar variable instead of a handle",
           span);
    }
    std::optional<PrimExpr> bound = LookupBind(bindings, handle);
    if (!bound.has_value()) {
      Fail(op_name +
               " requires a Buffer address or an explicit tl.access_ptr; got a "
               "handle with no recoverable definition",
           span);
    }
    if (bound->as<VarNode>() != nullptr) {
      Fail(op_name + " does not support multi-level pointer aliases; bind the "
                     "tl.access_ptr result directly",
           span);
    }
    const auto *bound_call = bound->as<CallNode>();
    if (bound_call != nullptr && bound_call->op.same_as(AccessPtrOp())) {
      if (!bound_call->dtype.is_handle()) {
        Fail(op_name + " requires a bound tl.access_ptr to produce a handle",
             span);
      }
      AccessAddress address =
          ParseAccessPtrCall(bound_call, operand, is_load, op_name, span);
      address.from_binding = true;
      return address;
    }
    if (bound->as<BufferLoadNode>() != nullptr ||
        bound->as<Buffer>().has_value() ||
        bound->as<BufferRegion>().has_value()) {
      Fail(op_name +
               " requires a pointer-producing tl.access_ptr; got a value alias "
               "of a buffer address",
           span);
    }
    Fail(op_name +
             " requires a Buffer address or an explicit tl.access_ptr; got an "
             "unsupported handle definition",
         span);
  }
  if (const auto *call = operand.as<CallNode>()) {
    if (!call->op.same_as(AccessPtrOp()) || !call->dtype.is_handle()) {
      Fail(op_name + " requires a Buffer address or an explicit tl.access_ptr",
           span);
    }
    return ParseAccessPtrCall(call, operand, is_load, op_name, span);
  }
  return ParseAutoAddress(operand, op_name, span);
}

// ---------------------------------------------------------------------------
// Descriptor plans and the shared acceptance exit
// ---------------------------------------------------------------------------
enum class RejectionKind { kNone, kExtent, kAddress, kLow, kHigh, kCoverage };

struct Rejection {
  RejectionKind kind{RejectionKind::kNone};
  std::string message;
};

bool Accepted(const Rejection &rejection) {
  return rejection.kind == RejectionKind::kNone;
}

struct DescriptorPlan {
  Array<PrimExpr> starts;
  PrimExpr extent;
  PrimExpr offset;
  std::optional<PrimExpr> relative;
  bool origin{false};
  bool explicit_plan{false};
  bool require_static_shape{false};
};

Rejection AcceptPlan(arith::Analyzer *analyzer, const AccessAddress &address,
                     const DescriptorPlan &plan, int64_t low, int64_t high) {
  std::optional<int64_t> constant = ConstInt(plan.extent);
  if (constant.has_value()) {
    if (constant.value() <= 0) {
      return {RejectionKind::kExtent,
              "the descriptor extent must be positive, got " +
                  std::to_string(constant.value())};
    }
  } else if (!analyzer->CanProve(plan.extent > 0)) {
    return {RejectionKind::kExtent,
            "cannot prove the descriptor extent is positive"};
  }

  if (plan.relative.has_value()) {
    std::optional<PrimExpr> source_relative =
        RelativeOffset(address.buffer, address.starts);
    if (!source_relative.has_value() ||
        !analyzer->CanProveEqual(plan.relative.value() + plan.offset,
                                 source_relative.value())) {
      return {RejectionKind::kAddress,
              "the descriptor construction does not preserve the original "
              "access address"};
    }
  }

  if (!analyzer->CanProve(plan.offset + IntConst(low) >= 0)) {
    return {RejectionKind::kLow,
            "the complete physical access envelope starts below the "
            "descriptor: 0 <= instruction_offset + envelope_low cannot be "
            "proven"};
  }
  if (!analyzer->CanProve(plan.offset + IntConst(high) <= plan.extent)) {
    return {RejectionKind::kHigh,
            "the complete physical access envelope exceeds the descriptor: "
            "instruction_offset + envelope_high <= descriptor_extent cannot be "
            "proven"};
  }

  std::optional<std::string> coverage;
  if (plan.explicit_plan) {
    coverage = ExplicitCoverageReason(analyzer, address.buffer, plan.starts,
                                      plan.extent);
  } else if (plan.origin) {
    coverage = OriginPrefixReason(analyzer, address.buffer, plan.extent,
                                  plan.require_static_shape);
  } else {
    coverage =
        LocalCoverageReason(analyzer, address.buffer, plan.starts, plan.extent);
  }
  if (coverage.has_value()) {
    return {RejectionKind::kCoverage, coverage.value()};
  }
  return {};
}

bool ProvablyNonNegativeStarts(arith::Analyzer *analyzer,
                               const AccessAddress &address) {
  for (const PrimExpr &start : address.starts) {
    if (!IsIntegerScalar(start) || !analyzer->CanProve(start >= 0)) {
      return false;
    }
  }
  return true;
}

std::string BufferContext(const Buffer &buffer) {
  std::ostringstream oss;
  oss << "buffer '" << buffer->name << "': shape=[";
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << buffer->shape[i];
  }
  oss << "]";
  if (!buffer->strides.empty()) {
    oss << ", strides=[";
    for (size_t i = 0; i < buffer->strides.size(); ++i) {
      if (i != 0) {
        oss << ", ";
      }
      oss << buffer->strides[i];
    }
    oss << "]";
  }
  oss << ", elem_offset=" << buffer->elem_offset;
  return oss.str();
}

[[noreturn]] void FailDescriptor(const std::string &op_name,
                                 const std::string &reason,
                                 const Buffer &buffer, const Span &span) {
  Fail(op_name +
           " cannot construct a descriptor covering the complete physical "
           "access: " +
           reason + " (" + BufferContext(buffer) +
           "). No conservative range was generated; use an actual root buffer "
           "with a known finite range, or an explicit pointer whose declared "
           "extent covers this access.",
       span);
}

DescriptorPlan LocalPlan(const AccessAddress &address, int64_t high) {
  DescriptorPlan plan;
  plan.starts = address.starts;
  plan.extent = IntConst(high);
  plan.offset = IntConst(0);
  return plan;
}

DescriptorPlan ChooseNegativePlan(arith::Analyzer *analyzer,
                                  const AccessAddress &address, int64_t low,
                                  int64_t high, bool is_load,
                                  const std::string &op_name,
                                  const Span &span) {
  std::optional<std::string> coverage_reason;
  int64_t neg_low = CheckedNeg(low, op_name, span);
  int64_t width = CheckedAdd(high, neg_low, op_name, span);

  // Candidate 1: one-dimensional unit-stride lowest-address relocation.
  const auto *load = address.source_operand.as<BufferLoadNode>();
  if (load != nullptr && load->indices.size() == 1 &&
      IsUnitStride1d(address.buffer)) {
    PrimExpr shifted = load->indices[0] + IntConst(low);
    if (analyzer->CanProve(shifted >= 0)) {
      DescriptorPlan plan;
      plan.starts = {shifted};
      plan.extent = IntConst(width);
      plan.offset = IntConst(neg_low);
      plan.relative = RelativeOffset(address.buffer, plan.starts);
      if (plan.relative.has_value()) {
        Rejection rejection = AcceptPlan(analyzer, address, plan, low, high);
        if (Accepted(rejection)) {
          return plan;
        }
        if (rejection.kind == RejectionKind::kCoverage) {
          coverage_reason = rejection.message;
        }
      }
    }
  }

  // Candidate 2: padded origin prefix.
  std::optional<PrimExpr> source_relative =
      RelativeOffset(address.buffer, address.starts);
  if (source_relative.has_value() &&
      ProvablyNonNegativeStarts(analyzer, address) &&
      PaddedRowMajor(analyzer, address.buffer)) {
    DescriptorPlan plan;
    plan.starts = BufferZeros(address.buffer);
    plan.extent = analyzer->Simplify(source_relative.value() + IntConst(high));
    plan.offset = source_relative.value();
    plan.relative = IntConst(0);
    plan.origin = true;
    plan.require_static_shape = true;
    Rejection rejection = AcceptPlan(analyzer, address, plan, low, high);
    if (Accepted(rejection)) {
      return plan;
    }
    if (rejection.kind == RejectionKind::kCoverage &&
        !coverage_reason.has_value()) {
      coverage_reason = rejection.message;
    }
  }

  // Candidate 3: whole tight-buffer origin.
  if (ProvablyNonNegativeStarts(analyzer, address)) {
    std::optional<PrimExpr> span = WholeBufferSpan(analyzer, address.buffer);
    if (span.has_value() && source_relative.has_value()) {
      DescriptorPlan plan;
      plan.starts = BufferZeros(address.buffer);
      plan.extent = span.value();
      plan.offset = source_relative.value();
      plan.relative = IntConst(0);
      plan.origin = true;
      plan.require_static_shape = address.buffer->shape.size() > 1;
      Rejection rejection = AcceptPlan(analyzer, address, plan, low, high);
      if (Accepted(rejection)) {
        return plan;
      }
      if (rejection.kind == RejectionKind::kCoverage &&
          !coverage_reason.has_value()) {
        coverage_reason = rejection.message;
      }
    }
  }

  std::string reason;
  if (!source_relative.has_value()) {
    reason = "the access address cannot be reduced to a single linear element "
             "offset";
  } else if (!ProvablyNonNegativeStarts(analyzer, address)) {
    reason =
        "the source index is negative or has no proven sign, so relocating the "
        "descriptor cannot preserve the address selected by negative-index "
        "legalization";
  } else if (analyzer->CanProve(source_relative.value() + IntConst(low) < 0)) {
    reason =
        "the finite access envelope starts below the buffer view origin and no "
        "constructible descriptor covers it";
  } else {
    reason = "cannot prove that any constructible descriptor covers the finite "
             "access envelope";
  }
  FailDescriptor(op_name, coverage_reason.value_or(reason), address.buffer,
                 span);
}

DescriptorPlan ChooseLocalPlan(arith::Analyzer *analyzer,
                               const AccessAddress &address, int64_t low,
                               int64_t high, const std::string &op_name,
                               const Span &span) {
  DescriptorPlan plan = LocalPlan(address, high);
  Rejection rejection = AcceptPlan(analyzer, address, plan, low, high);
  if (!Accepted(rejection)) {
    FailDescriptor(op_name, rejection.message, address.buffer, span);
  }
  return plan;
}

PrimExpr ValidateExplicitDescriptor(arith::Analyzer *analyzer,
                                    const AccessAddress &address, int64_t low,
                                    int64_t high, const PrimExpr &delta,
                                    bool is_load, const std::string &op_name,
                                    const Span &span) {
  if (!IsIntegerScalar(delta)) {
    Fail(op_name + " requires a scalar integer instruction offset", span);
  }
  DescriptorPlan plan;
  plan.starts = address.starts;
  plan.extent = address.extent;
  plan.offset = delta;
  plan.explicit_plan = true;
  Rejection rejection = AcceptPlan(analyzer, address, plan, low, high);
  if (!Accepted(rejection)) {
    std::ostringstream oss;
    oss << op_name << " cannot verify the explicit access_ptr descriptor: "
        << rejection.message << " (" << BufferContext(address.buffer)
        << ", descriptor_extent=" << address.extent
        << ", instruction_offset=" << delta << ", access_envelope=[" << low
        << ", " << high << "))";
    Fail(oss.str(), span);
  }
  return delta;
}

PrimExpr PackFp4Offset(const PrimExpr &offset, const std::string &op_name,
                       const Span &span) {
  std::optional<int64_t> constant = ConstInt(offset);
  if (constant.has_value()) {
    if (constant.value() % 2 != 0) {
      Fail(op_name + " requires an even FP4 element offset, got " +
               std::to_string(constant.value()),
           span);
    }
    return IntConst(constant.value() / 2);
  }
  return FloorDiv(offset, IntConst(2));
}

PrimExpr MakeDescriptor(const Buffer &buffer, const Array<PrimExpr> &starts,
                        const PrimExpr &extent, int rw_mask) {
  return Call(
      DataType::Handle(), AccessPtrOp(),
      {BufferLoad(buffer, starts), extent, IntImm(DataType::Int(32), rw_mask)});
}

// ---------------------------------------------------------------------------
// Indexed owner identity, layout and stable whole-area coverage
// ---------------------------------------------------------------------------
Buffer FindAllocationOwner(const std::vector<Buffer> &active_allocations,
                           const Buffer &source, const std::string &op_name,
                           const Span &span) {
  std::optional<Buffer> owner;
  for (const Buffer &buffer : active_allocations) {
    if (!buffer->data.same_as(source->data)) {
      continue;
    }
    if (!owner.has_value()) {
      owner = buffer;
    } else if (!owner->same_as(buffer)) {
      Fail(op_name +
               " found multiple active allocations sharing the source data; "
               "the owner is ambiguous, and different allocation objects are "
               "never merged by data pointer",
           span);
    }
  }
  if (!owner.has_value()) {
    Fail(op_name +
             " requires a unique local allocation that is active at this "
             "position and owns the source data; no active allocation was "
             "found (a parameter, view, or full-function collection table is "
             "not an owner)",
         span);
  }
  return owner.value();
}

std::optional<std::string> OwnerTightReason(arith::Analyzer *analyzer,
                                            const Buffer &owner) {
  size_t ndim = owner->shape.size();
  if (ndim == 0) {
    return std::string("the owner has no dimensions");
  }
  if (owner->strides.empty()) {
    return std::nullopt;
  }
  if (owner->strides.size() != ndim) {
    return std::string("the owner stride rank ") +
           std::to_string(owner->strides.size()) + " does not match its rank " +
           std::to_string(ndim);
  }
  // Compute the expected tight strides from the inner shape dimensions in a
  // value-preserving int64 domain.  Multiplying in the shape dtype wraps: for
  // shape (2, 67108865, 64) the int32 product 67108865 * 64 becomes 64, which
  // would accept an overlapping stride[0]=64 and reject the mathematically
  // tight stride[0]=4294967360.  The layout contract itself is unchanged.
  PrimExpr expected = IntImm(DataType::Int(64), 1);
  for (size_t dim = ndim; dim-- > 0;) {
    const PrimExpr &stride = owner->strides[dim];
    if (!RepresentableInInt64(analyzer, stride)) {
      std::ostringstream oss;
      oss << "the owner layout is not provably tight row-major: stride[" << dim
          << "]=" << stride
          << " has an unsigned 64-bit range that cannot be compared without "
             "losing integer precision";
      return oss.str();
    }
    if (!analyzer->CanProveEqual(WidenToInt64(stride), expected)) {
      std::ostringstream oss;
      oss << "the owner layout is not provably tight row-major: stride[" << dim
          << "]=" << stride << " does not equal the expected " << expected;
      return oss.str();
    }
    if (dim > 0) {
      const PrimExpr &factor = owner->shape[dim];
      if (!RepresentableInInt64(analyzer, factor)) {
        std::ostringstream oss;
        oss << "the owner layout is not provably tight row-major: dimension "
            << dim << " has an unsigned 64-bit length that cannot be widened "
            << "without losing integer precision";
        return oss.str();
      }
      __int128 product_bound =
          static_cast<__int128>(
              analyzer->const_int_bound(expected)->max_value) *
          static_cast<__int128>(analyzer->const_int_bound(factor)->max_value);
      if (product_bound > std::numeric_limits<int64_t>::max()) {
        std::ostringstream oss;
        oss << "the owner layout is not provably tight row-major within the "
               "signed 64-bit range: the expected stride outside dimension "
            << dim - 1 << " is the product of dimensions " << dim << ".."
            << ndim - 1 << ", whose upper bound exceeds int64";
        return oss.str();
      }
      expected = analyzer->Simplify(expected * WidenToInt64(factor));
    }
  }
  return std::nullopt;
}

std::optional<std::string> OwnerCoverageReason(arith::Analyzer *analyzer,
                                               const Buffer &owner) {
  size_t ndim = owner->shape.size();
  if (ndim == 0) {
    return std::string("the owner has no dimensions");
  }
  // With a constant extent, the existing logical pointer-region consumer
  // distributes the extent innermost-first and treats the first non-constant
  // dimension as the whole dimension (remaining=1).  A later constant extent
  // is therefore stable only when every dimension outside that innermost
  // non-constant dimension is provably length one; inner dimensions are
  // consumed in full because E is exactly their product.  This applies to
  // symbolic axes too: the consumer does not turn an Analyzer constant fact
  // into an IntImm shape node, so a dynamic outer axis that is not provably
  // one would degrade to its origin index after the extent becomes constant.
  size_t innermost_dynamic = ndim;
  for (size_t dim = ndim; dim-- > 0;) {
    if (!ConstInt(owner->shape[dim]).has_value()) {
      innermost_dynamic = dim;
      break;
    }
  }
  if (innermost_dynamic == ndim) {
    return std::nullopt;
  }
  for (size_t dim = 0; dim < innermost_dynamic; ++dim) {
    if (!analyzer->CanProveEqual(owner->shape[dim], IntConst(1))) {
      std::ostringstream oss;
      oss << "the owner does not have a stable whole-area logical coverage "
             "proof: dimension "
          << dim << " (length " << owner->shape[dim]
          << ") is outside the innermost non-constant dimension "
          << innermost_dynamic
          << " and is not provably length one; a later constant descriptor "
             "extent would cover only one index of that dimension";
      return oss.str();
    }
  }
  return std::nullopt;
}

PrimExpr OwnerExtent(arith::Analyzer *analyzer, const Buffer &owner,
                     const std::string &op_name, const Span &span) {
  bool all_static = true;
  int64_t static_extent = 1;
  // Upper bound of the mathematical shape product, computed without ever
  // evaluating the product in the shape dtype.  `unbounded` counts symbolic
  // factors whose ConstIntBound has no finite maximum.
  int64_t max_extent = 1;
  int unbounded = 0;
  bool unbounded_uint = false;

  auto capacity_failure = [&](const std::string &detail) {
    Fail(op_name +
             " cannot represent the whole-owner element capacity as a stable "
             "descriptor extent: " +
             detail,
         span);
  };

  for (size_t dim = 0; dim < owner->shape.size(); ++dim) {
    const PrimExpr &shape = owner->shape[dim];
    std::optional<int64_t> length = ConstInt(shape);
    int64_t factor_bound = 1;
    if (length.has_value()) {
      if (length.value() <= 0) {
        Fail(op_name + " owner dimension " + std::to_string(dim) +
                 " has non-positive length " + std::to_string(length.value()),
             span);
      }
      static_extent = CheckedMul(static_extent, length.value(), op_name, span);
      factor_bound = length.value();
    } else {
      all_static = false;
      if (!analyzer->CanProve(shape > 0)) {
        Fail(op_name + " cannot prove owner dimension " + std::to_string(dim) +
                 " is positive",
             span);
      }
      if (!analyzer->CanProveEqual(shape, IntConst(1))) {
        arith::ConstIntBound bound = analyzer->const_int_bound(shape);
        if (bound->max_value == arith::ConstIntBound::kPosInf) {
          ++unbounded;
          unbounded_uint = unbounded_uint || shape.dtype().is_uint();
        } else {
          if (bound->max_value <= 0) {
            capacity_failure("dimension " + std::to_string(dim) +
                             " has no positive upper bound");
          }
          factor_bound = bound->max_value;
        }
      }
    }
    if (factor_bound > 1) {
      __int128 next = static_cast<__int128>(max_extent) * factor_bound;
      if (next > std::numeric_limits<int64_t>::max()) {
        capacity_failure("the shape product exceeds the int64 capacity range");
      }
      max_extent = static_cast<int64_t>(next);
    }
  }
  if (all_static) {
    return IntImm(DataType::Int(64), static_extent);
  }
  if (unbounded > 0) {
    // A single unbounded factor is representable only when every other factor
    // is one: the capacity then is that factor itself, not a wider product.
    if (unbounded > 1 || max_extent > 1 || unbounded_uint) {
      capacity_failure(
          "a symbolic owner dimension has no finite upper bound, so the shape "
          "product cannot be proven representable in int64");
    }
  }

  // The owner capacity is always emitted in int64, including small static
  // capacities.  The source Buffer shape and stride expressions keep their
  // original dtypes and evaluation semantics; each factor is widened before
  // the new multiplication, so that multiplication does not run in the shape
  // dtype.  Arithmetic already present inside a source shape still evaluates
  // in its own dtype.
  PrimExpr extent = IntImm(DataType::Int(64), 1);
  for (const PrimExpr &shape : owner->shape) {
    PrimExpr factor;
    if (analyzer->CanProveEqual(shape, IntConst(1))) {
      factor = IntImm(DataType::Int(64), 1);
    } else {
      factor = WidenToInt64(shape);
    }
    extent = extent * factor;
  }
  // Build the emitted extent without the outer analyzer's equality facts.  A
  // saved scalar Var is a snapshot leaf: simplifying with the outer analyzer
  // would expand its definition (for example a pure Call) back into the
  // descriptor at the use point.  The outer analyzer is still used above and
  // below for positivity, range, and capacity proofs; only the emitted
  // expression construction uses a local context.
  arith::Analyzer local_analyzer;
  extent = local_analyzer.Simplify(extent);
  if (!analyzer->CanProve(extent > 0)) {
    Fail(op_name +
             " cannot prove the whole-owner element capacity is positive; the "
             "owner shape product is not a stable capacity proof",
         span);
  }
  return extent;
}

bool StableIndexedAddressNode(const ObjectRef &node) {
  return node.as<VarNode>() || node.as<IntImmNode>() || node.as<CastNode>() ||
         node.as<AddNode>() || node.as<SubNode>() || node.as<MulNode>() ||
         node.as<FloorDivNode>() || node.as<FloorModNode>() ||
         node.as<MinNode>() || node.as<MaxNode>();
}

// The address snapshot whitelist above is deliberately narrow because it
// feeds bound-pointer reconstruction.  An owner capacity only needs a stable
// scalar expression that can be evaluated at the use point, so it also admits
// truncating div/mod, comparisons, boolean combinations, and Select.  Keep
// this classification local: extending it must not relax address admission.
bool StableOwnerCapacityNode(const ObjectRef &node) {
  return StableIndexedAddressNode(node) || node.as<DivNode>() ||
         node.as<ModNode>() || node.as<SelectNode>() || node.as<EQNode>() ||
         node.as<NENode>() || node.as<LTNode>() || node.as<LENode>() ||
         node.as<GTNode>() || node.as<GENode>() || node.as<AndNode>() ||
         node.as<OrNode>() || node.as<NotNode>();
}

struct ExpressionStabilityScan {
  bool has_load = false;
  bool has_call = false;
  std::string first_unstable_type;
};

// One traversal for the checks shared by address and capacity snapshots.
// Each caller supplies its own admission set and keeps its own diagnostics;
// sharing the walk must not merge those sets.  Post-order reaches a read or a
// call only after its arguments, so collect the whole expression and let the
// caller give reads and calls priority over an incidental leaf type from
// inside them.  This pass conservatively rejects every direct Call in these
// snapshots, including a pure call; it does not classify whether a particular
// pure call would be evaluable at the use point.
ExpressionStabilityScan
ScanExpressionStability(const PrimExpr &expr,
                        bool (*is_stable)(const ObjectRef &)) {
  ExpressionStabilityScan scan;
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    scan.has_load = scan.has_load || node.as<BufferLoadNode>() != nullptr;
    scan.has_call = scan.has_call || node.as<CallNode>() != nullptr;
    if (scan.first_unstable_type.empty() && !is_stable(node)) {
      scan.first_unstable_type = node->GetTypeKey();
    }
  });
  return scan;
}

// Inspect the allocation shape nodes as written, before OwnerExtent or any
// other Analyzer simplification runs.  A shape such as `read * 0 + 64` folds
// to a constant, but the physical capacity was still fixed by a mutable read
// at allocation time; the use-point extent must not re-evaluate that read.
// A Var is a snapshot leaf: its definition is deliberately not followed, so a
// scalar saved before the allocation remains a stable capacity even when the
// memory behind it changes later.
std::optional<std::string> OwnerCapacityStabilityReason(const Buffer &owner) {
  for (size_t dim = 0; dim < owner->shape.size(); ++dim) {
    const PrimExpr &shape = owner->shape[dim];
    ExpressionStabilityScan scan =
        ScanExpressionStability(shape, StableOwnerCapacityNode);
    std::string reason;
    if (scan.has_load) {
      reason = "the shape re-reads a buffer at the use point; save the shape "
               "value in a scalar Var before allocating the owner";
    } else if (scan.has_call) {
      reason =
          "the shape executes a call at the use point; save the shape value "
          "in a scalar Var before allocating the owner";
    } else if (!scan.first_unstable_type.empty()) {
      std::ostringstream oss;
      oss << "the shape contains a non-stable node of type "
          << scan.first_unstable_type
          << "; save the shape value in a scalar Var before allocating the "
             "owner";
      reason = oss.str();
    }
    if (!reason.empty()) {
      std::ostringstream oss;
      oss << "owner capacity is not a stable snapshot: shape dimension " << dim
          << " (length " << shape << ", " << BufferContext(owner)
          << ") cannot be rebuilt at the use point: " << reason;
      return oss.str();
    }
  }
  return std::nullopt;
}

void CheckIndexedAddressSnapshot(const AccessAddress &address,
                                 const std::string &op_name, const Span &span) {
  std::string reason;
  // The rebuild copies the linear source offset into the new D, not the raw
  // start/stride metadata lists.  OffsetOf folds zero-index dimensions and
  // materializes implicit row-major strides from the source shape exactly
  // when those expressions participate in that address, so scan the actual
  // offset expression.  This catches a shape read that enters an implicit
  // stride without rejecting unrelated shape nodes.
  Array<PrimExpr> source_offsets = address.buffer.OffsetOf(address.starts);
  if (source_offsets.size() != 1) {
    reason =
        "the source address has multiple linear components; axis separators "
        "are not supported";
  } else {
    ExpressionStabilityScan stability =
        ScanExpressionStability(source_offsets[0], StableIndexedAddressNode);
    if (stability.has_load) {
      reason = "the coordinates re-read a buffer at the use point; save the "
               "scalar value in a variable before binding the pointer";
    } else if (stability.has_call) {
      reason =
          "the coordinates execute a call at the use point; save the scalar "
          "value in a variable before binding the pointer";
    } else if (!stability.first_unstable_type.empty()) {
      std::ostringstream oss;
      oss << "the coordinates contain a non-stable node of type "
          << stability.first_unstable_type;
      reason = oss.str();
    }
  }
  if (!reason.empty()) {
    Fail(op_name +
             " cannot rebuild the descriptor from a bound pointer because its "
             "address is not a stable snapshot: " +
             reason,
         span);
  }
}

struct IndexedOwner {
  Buffer buffer;
  PrimExpr origin;
  PrimExpr extent;
};

IndexedOwner ValidateIndexedOwner(arith::Analyzer *analyzer,
                                  const AccessAddress &address,
                                  const std::vector<Buffer> &active_allocations,
                                  const std::string &op_name,
                                  const Span &span) {
  Buffer owner =
      FindAllocationOwner(active_allocations, address.buffer, op_name, span);
  // Capacity stability is an owner property and must be checked before any
  // Analyzer folding, coverage, or extent construction can rewrite the shape.
  if (std::optional<std::string> stability =
          OwnerCapacityStabilityReason(owner)) {
    Fail(op_name + " " + stability.value(), span);
  }
  if (address.explicit_address && address.from_binding) {
    CheckIndexedAddressSnapshot(address, op_name, span);
  }
  ValidateUbBuffer(owner, op_name + " owner", span);
  if (address.buffer->dtype != owner->dtype) {
    std::ostringstream oss;
    oss << op_name << " requires the source element dtype "
        << address.buffer->dtype << " to match the owner element dtype "
        << owner->dtype;
    Fail(oss.str(), span);
  }
  if (std::optional<std::string> tight = OwnerTightReason(analyzer, owner)) {
    Fail(op_name + " owner is not usable: " + tight.value(), span);
  }
  Array<PrimExpr> owner_origin_expr = owner.OffsetOf(BufferZeros(owner));
  if (owner_origin_expr.size() != 1) {
    Fail(op_name +
             " owner requires a single scalar linear origin; axis separators "
             "are not supported",
         span);
  }
  PrimExpr origin = owner_origin_expr[0];
  if (!analyzer->CanProveEqual(origin, IntConst(0))) {
    std::ostringstream oss;
    oss << op_name << " owner origin must be provably zero, but elem_offset "
        << origin << " (buffer " << BufferContext(owner)
        << ") is not provably zero";
    Fail(oss.str(), span);
  }
  // Coverage is an independent owner property: check it before constructing
  // the capacity expression so a shape without a stable whole-area proof is
  // diagnosed as such, not as an arithmetic failure.
  if (std::optional<std::string> coverage =
          OwnerCoverageReason(analyzer, owner)) {
    Fail(op_name + " owner is not usable: " + coverage.value(), span);
  }
  PrimExpr extent = OwnerExtent(analyzer, owner, op_name, span);
  return {owner, origin, extent};
}

// ---------------------------------------------------------------------------
// Mutator
// ---------------------------------------------------------------------------
class VmiMemoryAccessLegalizer : public arith::IRMutatorWithAnalyzer {
public:
  static Stmt Apply(Stmt stmt, arith::Analyzer *analyzer) {
    VmiMemoryAccessLegalizer legalizer(analyzer);
    return legalizer(std::move(stmt));
  }

private:
  using Parent = arith::IRMutatorWithAnalyzer;

  explicit VmiMemoryAccessLegalizer(arith::Analyzer *analyzer)
      : Parent(analyzer) {}

  // Scope every statement reached through the default control-flow traversal.
  // The branches/loops of IRMutatorWithAnalyzer call the virtual VisitStmt for
  // their bodies, so bindings and active allocations declared inside them are
  // restored on exit.  While deliberately gets no condition fact and no extra
  // analyzer scope; only this side environment is restored.
  Stmt VisitStmt(const Stmt &stmt) final {
    size_t bind_mark = bindings_.size();
    size_t allocation_mark = active_allocations_.size();
    Stmt visited = Parent::VisitStmt(stmt);
    bindings_.resize(bind_mark);
    active_allocations_.resize(allocation_mark);
    return visited;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    size_t mark = active_allocations_.size();
    for (const Buffer &buffer : op->alloc_buffers) {
      RegisterAllocation(buffer);
    }
    Stmt visited = Parent::VisitStmt_(op);
    active_allocations_.resize(mark);
    return visited;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    size_t bind_mark = bindings_.size();
    size_t allocation_mark = active_allocations_.size();
    Array<Stmt> seq;
    seq.reserve(op->seq.size());
    bool changed = false;
    for (const Stmt &stmt : op->seq) {
      // A qualified base call bypasses the generic VisitStmt scope wrapper so
      // declarations registered after this child stay visible to its later
      // siblings in this SeqStmt, as the statement order requires.
      Stmt visited = Parent::VisitStmt(stmt);
      // Register a declaration only when the original child is that
      // declaration.  The base visitor folds a literal-true/false If into its
      // selected branch; the folded body must stay lexical to the original If
      // and must not leak an AllocBuffer/Bind into this sibling list.
      if (const auto *original_bind = stmt.as<BindNode>()) {
        const auto *visited_bind = visited.as<BindNode>();
        bindings_.emplace_back(original_bind->var, visited_bind != nullptr
                                                       ? visited_bind->value
                                                       : original_bind->value);
      }
      if (const auto *original_alloc = stmt.as<AllocBufferNode>()) {
        const auto *visited_alloc = visited.as<AllocBufferNode>();
        RegisterAllocation(visited_alloc != nullptr ? visited_alloc->buffer
                                                    : original_alloc->buffer);
      }
      changed = changed || !visited.same_as(stmt);
      seq.push_back(std::move(visited));
    }
    bindings_.resize(bind_mark);
    active_allocations_.resize(allocation_mark);
    if (!changed) {
      return GetRef<Stmt>(op);
    }
    return SeqStmt(seq, op->span);
  }

  PrimExpr VisitExpr_(const LetNode *op) final {
    // Match the default analyzer handling, but also expose the direct
    // definition to address resolution for the lexical body only.  A value
    // that is not pure is still recorded for address lookup (the variable is
    // bound regardless of purity); only the analyzer fact is conditional.
    PrimExpr value = this->VisitExpr(op->value);
    if (SideEffect(value) <= CallEffectKind::kPure) {
      analyzer_->Bind(op->var, value);
    }
    size_t mark = bindings_.size();
    bindings_.emplace_back(op->var, value);
    PrimExpr body = this->VisitExpr(op->body);
    bindings_.resize(mark);
    if (value.same_as(op->value) && body.same_as(op->body)) {
      return GetRef<PrimExpr>(op);
    }
    return Let(op->var, value, body, op->span);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    PrimExpr visited = Parent::VisitExpr_(op);
    const auto *call = visited.as<CallNode>();
    if (call == nullptr) {
      return visited;
    }
    if (call->op.same_as(VloadOp())) {
      return ProcessCall(call, /*is_load=*/true);
    }
    if (call->op.same_as(VstoreOp())) {
      return ProcessCall(call, /*is_load=*/false);
    }
    if (call->op.same_as(VgatherOp())) {
      return ProcessIndexedCall(call, /*is_gather=*/true);
    }
    if (call->op.same_as(VscatterOp())) {
      return ProcessIndexedCall(call, /*is_gather=*/false);
    }
    return visited;
  }

  PrimExpr ProcessCall(const CallNode *call, bool is_load) {
    const std::string op_name = is_load ? "T.vmi.vload" : "T.vmi.vstore";
    const Span span = call->span;
    VmiModes modes = ParseModes(call, op_name, is_load, span);
    const bool dintlv =
        modes.dist_mode.has_value() && modes.dist_mode.value() == "dintlv";

    PrimExpr value0;
    PrimExpr value1;
    PrimExpr mask;
    bool has_mask = false;
    int address_index = 0;
    int64_t lanes = 0;

    if (is_load) {
      if (call->args.size() != 2) {
        Fail(op_name + " requires a pointer address and an instruction offset",
             span);
      }
      std::optional<int64_t> size = AnnotationInt(call, "size", op_name, span);
      if (!size.has_value()) {
        Fail(op_name + " requires a constant positive size annotation", span);
      }
      if (size.value() <= 0) {
        Fail(op_name + " requires a positive size, got " +
                 std::to_string(size.value()),
             span);
      }
      lanes = size.value();
    } else {
      if (dintlv) {
        if (call->args.size() != 4 && call->args.size() != 5) {
          Fail(op_name +
                   " with dist_mode=\"dintlv\" requires an (even, odd) value "
                   "pair, a pointer address and an instruction offset",
               span);
        }
        value0 = call->args[0];
        value1 = call->args[1];
        address_index = 2;
        if (value0.dtype() != value1.dtype()) {
          Fail(op_name + " requires identical VMI vector types", span);
        }
        has_mask = call->args.size() == 5;
        if (has_mask) {
          mask = call->args[4];
        }
      } else {
        if (call->args.size() != 3 && call->args.size() != 4) {
          Fail(op_name +
                   " requires a value, a pointer address and an instruction "
                   "offset",
               span);
        }
        value0 = call->args[0];
        address_index = 1;
        has_mask = call->args.size() == 4;
        if (has_mask) {
          mask = call->args[3];
        }
      }
      // A one-lane VMI value has a scalar TIR dtype, so lane count is not a
      // provenance check.  This pass admits the value exactly as the frontend
      // produced it and only checks the element dtype below.  Ordinary vstore
      // has no consumer-side provenance check in codegen; the PTODSL builder
      // and the PTO MLIR verifier reject a non-VMI value operand.
      lanes = value0.dtype().lanes();
      if (modes.group.has_value() && has_mask) {
        Fail(op_name + " group mode does not take a mask operand", span);
      }
    }

    PrimExpr input_offset = call->args[address_index + 1];
    AccessAddress address = ParseAddress(call->args[address_index], is_load,
                                         op_name, span, bindings_);

    if (!IsIntegerScalar(input_offset)) {
      Fail(op_name + " requires a scalar integer instruction offset", span);
    }
    if (!address.explicit_address) {
      if (!is_zero(analyzer_->Simplify(input_offset))) {
        Fail(
            op_name +
                " buffer addresses do not accept an offset; express the offset "
                "in the buffer index",
            span);
      }
    }

    DataType value_dtype = is_load ? address.buffer->dtype : value0.dtype();
    DataType value_elem = is_load ? value_dtype : value_dtype.with_lanes(1);
    if (!is_load && value_elem != address.buffer->dtype) {
      Fail(op_name + " requires the destination element dtype to match the "
                     "value element dtype; convert the value before storing",
           span);
    }
    bool is_fp4 = value_elem.is_float4_e2m1fn();
    if (is_fp4) {
      if (lanes % 2 != 0) {
        Fail(op_name +
                 " requires an even FP4 lane count for packed FP4 storage",
             span);
      }
      if (dintlv) {
        Fail(op_name + " does not support packed FP4 with dist_mode=\"dintlv\"",
             span);
      }
      if (is_load) {
        if (modes.group.has_value() || modes.block_stride.has_value() ||
            (modes.dist_mode.has_value() && modes.dist_mode.value() == "brc")) {
          Fail(op_name + " only supports contiguous packed FP4 loads", span);
        }
      } else if (modes.group.has_value() || modes.block_stride.has_value()) {
        Fail(op_name + " only supports contiguous packed FP4 stores", span);
      }
      if (!is_load && has_mask) {
        if (!mask.dtype().is_bool()) {
          Fail(op_name + " of FP4 expects a VMI mask operand", span);
        }
        if (mask.dtype().lanes() != lanes / 2) {
          Fail(op_name +
                   " of FP4 requires a physical FP4x2 mask with half as many "
                   "lanes as its logical FP4 value",
               span);
        }
      }
    }

    std::pair<int64_t, int64_t> envelope =
        AccessEnvelope(modes, lanes, value_elem, is_load, op_name, span);
    int64_t low = envelope.first;
    int64_t high = envelope.second;

    PrimExpr pointer;
    PrimExpr new_offset;
    if (address.explicit_address) {
      new_offset = ValidateExplicitDescriptor(
          analyzer_, address, low, high, input_offset, is_load, op_name, span);
      pointer = address.source_operand;
    } else {
      DescriptorPlan plan =
          low >= 0
              ? ChooseLocalPlan(analyzer_, address, low, high, op_name, span)
              : ChooseNegativePlan(analyzer_, address, low, high, is_load,
                                   op_name, span);
      pointer = MakeDescriptor(address.buffer, plan.starts, plan.extent,
                               is_load ? 1 : 2);
      new_offset = plan.offset;
    }

    if (is_fp4) {
      new_offset = PackFp4Offset(new_offset, op_name, span);
    }

    Array<PrimExpr> args;
    if (is_load) {
      args.push_back(pointer);
      args.push_back(new_offset);
    } else {
      if (dintlv) {
        args.push_back(value0);
        args.push_back(value1);
      } else {
        args.push_back(value0);
      }
      args.push_back(pointer);
      args.push_back(new_offset);
      if (has_mask) {
        args.push_back(mask);
      }
    }
    return Call(call->dtype, call->op, args, call->annotations, call->span);
  }

  PrimExpr ProcessIndexedCall(const CallNode *call, bool is_gather) {
    const std::string op_name = is_gather ? "T.vmi.vgather" : "T.vmi.vscatter";
    const Span span = call->span;
    const int address_index = is_gather ? 0 : 1;
    const size_t expected_args = is_gather ? 4 : 5;
    if (call->args.size() != expected_args) {
      Fail(op_name +
               (is_gather
                    ? " requires (pointer, offset, offsets, mask)"
                    : " requires (value, pointer, offset, offsets, mask)"),
           span);
    }

    PrimExpr input_offset = call->args[address_index + 1];
    if (!IsIntegerScalar(input_offset)) {
      Fail(op_name + " requires a scalar integer instruction offset", span);
    }
    AccessAddress address = ParseAddress(call->args[address_index], is_gather,
                                         op_name, span, bindings_);
    IndexedOwner owner = ValidateIndexedOwner(
        analyzer_, address, active_allocations_, op_name, span);

    if (owner.buffer->dtype.is_float4_e2m1fn()) {
      Fail(op_name + " does not support packed FP4 source/destination buffers",
           span);
    }
    if (!is_gather) {
      DataType value_elem = call->args[0].dtype().with_lanes(1);
      if (value_elem != owner.buffer->dtype) {
        Fail(op_name +
                 " requires the destination element dtype to match the value "
                 "element dtype; convert the value before storing",
             span);
      }
    }

    if (!ProvablyNonNegativeStarts(analyzer_, address)) {
      Fail(op_name +
               " source index is negative or has no proven sign; run "
               "negative-index legalization before descriptor construction",
           span);
    }
    Array<PrimExpr> source_offsets = address.buffer.OffsetOf(address.starts);
    if (source_offsets.size() != 1) {
      Fail(op_name + " requires a single scalar linear source address; axis "
                     "separators are not supported",
           span);
    }
    // A saved scalar Var is a snapshot leaf: never let an outer analyzer
    // equality expand it back to a mutable load.  The outer analyzer is used
    // only for the proofs below, not to build the emitted expression.
    arith::Analyzer local_analyzer;
    PrimExpr instruction_offset = local_analyzer.Simplify(
        source_offsets[0] - owner.origin + input_offset);
    if (!analyzer_->CanProveEqual(owner.origin + instruction_offset,
                                  source_offsets[0] + input_offset)) {
      Fail(op_name + " descriptor construction does not preserve the original "
                     "access address",
           span);
    }
    if (!analyzer_->CanProve(instruction_offset >= 0)) {
      Fail(op_name +
               " requires the instruction base offset D = S - O + offset to be "
               "provably non-negative and inside the owner",
           span);
    }
    // Compare D against the int64 owner capacity in a signed domain that
    // preserves D.  D keeps its original expression and dtype in the emitted
    // instruction; only this comparison widens it.  Narrowing a wide D can
    // wrap, or abort while the analyzer builds an out-of-range literal, before
    // any proof is attempted.  Widen an unsigned 64-bit D only after its range
    // is proven to fit the signed 64-bit domain.
    PrimExpr capacity_offset = instruction_offset;
    const bool offset_comparable =
        RepresentableInInt64(analyzer_, capacity_offset);
    if (offset_comparable) {
      capacity_offset = WidenToInt64(capacity_offset);
    }
    if (!offset_comparable ||
        !analyzer_->CanProve(capacity_offset < owner.extent)) {
      std::ostringstream oss;
      oss << op_name
          << " requires the instruction base offset D = S - O + offset to be "
             "provably inside the owner (D < whole-owner element capacity)";
      if (!offset_comparable) {
        oss << "; D has an unsigned 64-bit range that cannot be compared with "
               "the int64 capacity without losing integer precision";
      }
      Fail(oss.str(), span);
    }

    // The owner descriptor replaces any existing caller-declared extent.  All
    // non-address operands, annotations and the result dtype are kept verbatim;
    // gather offset vectors are never converted or folded into D.
    PrimExpr pointer = MakeDescriptor(owner.buffer, BufferZeros(owner.buffer),
                                      owner.extent, is_gather ? 1 : 2);
    Array<PrimExpr> args;
    args.reserve(call->args.size());
    for (size_t index = 0; index < call->args.size(); ++index) {
      if (index == static_cast<size_t>(address_index)) {
        args.push_back(pointer);
      } else if (index == static_cast<size_t>(address_index + 1)) {
        args.push_back(instruction_offset);
      } else {
        args.push_back(call->args[index]);
      }
    }
    return Call(call->dtype, call->op, args, call->annotations, call->span);
  }

  void RegisterAllocation(const Buffer &buffer) {
    for (const Buffer &active : active_allocations_) {
      if (active.same_as(buffer)) {
        return;
      }
    }
    active_allocations_.push_back(buffer);
  }

  std::vector<std::pair<Var, PrimExpr>> bindings_;
  std::vector<Buffer> active_allocations_;
};

void ScanResidualAutomaticAddresses(const Stmt &stmt) {
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    const auto *call = node.as<CallNode>();
    if (call == nullptr) {
      return;
    }
    bool is_load = call->op.same_as(VloadOp());
    bool is_store = call->op.same_as(VstoreOp());
    bool is_gather = call->op.same_as(VgatherOp());
    bool is_scatter = call->op.same_as(VscatterOp());
    if (!is_load && !is_store && !is_gather && !is_scatter) {
      return;
    }
    std::string op_name;
    int address_index = 0;
    if (is_load) {
      op_name = "T.vmi.vload";
    } else if (is_store || is_scatter) {
      op_name = is_store ? "T.vmi.vstore" : "T.vmi.vscatter";
      bool dintlv = false;
      if (is_store) {
        if (std::optional<ObjectRef> mode = FindAnnotation(call, "dist_mode")) {
          if (const auto *imm = mode->as<StringImmNode>()) {
            dintlv =
                std::string(imm->value.c_str(), imm->value.size()) == "dintlv";
          }
        }
      }
      address_index = dintlv ? 2 : 1;
    } else {
      op_name = "T.vmi.vgather";
      address_index = 0;
    }
    if (address_index >= static_cast<int>(call->args.size())) {
      return;
    }
    const PrimExpr &address = call->args[address_index];
    if (address.as<BufferLoadNode>() != nullptr ||
        address.as<Buffer>().has_value() ||
        address.as<BufferRegion>().has_value()) {
      Fail(std::string("internal error: ") + op_name +
               " still carries an unconstructed automatic descriptor address",
           call->span);
    }
  });
}

Stmt LegalizeVmiStmt(Stmt stmt, arith::Analyzer *analyzer) {
  Stmt result = VmiMemoryAccessLegalizer::Apply(std::move(stmt), analyzer);
  ScanResidualAutomaticAddresses(result);
  return result;
}

} // namespace

namespace transform {

using namespace tirx::transform;

Pass LegalizePTOVMIMemoryAccess() {
  auto pass_func = [](PrimFunc func, const IRModule &, const PassContext &) {
    Optional<Target> target = func->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPTO(target.value())) {
      return func;
    }
    arith::Analyzer analyzer;
    Stmt body = LegalizeVmiStmt(std::move(func->body), &analyzer);
    if (body.same_as(func->body)) {
      return func;
    }
    auto *writer = func.CopyOnWrite();
    writer->body = std::move(body);
    return func;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LegalizePTOVMIMemoryAccess", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizePTOVMIMemoryAccess",
                        LegalizePTOVMIMemoryAccess);
}

} // namespace transform
} // namespace tl
} // namespace tvm

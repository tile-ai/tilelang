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
 * \file ascend_pipe.h
 * \brief Shared Ascend hardware-pipe classification before and after tile-op
 * lowering.
 */

#ifndef TVM_TL_ASCEND_TRANSFORM_ASCEND_PIPE_H_
#define TVM_TL_ASCEND_TRANSFORM_ASCEND_PIPE_H_

#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "ascend/op/builtin.h"
#include "ascend/op/copy.h"
#include "ascend/op/utils.h"
#include "op/copy.h"
#include "op/fill.h"
#include "op/utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

// Hardware execution pipe bitmask. Multiple bits describe a composite
// lowered instruction; kAll is used only for a full-core PIPE_ALL fence.
enum class ResourcePipe : uint16_t {
  kNone = 0,
  kUnknown = kNone,
  kMTE1 = 1 << 0,    // L1 -> L0A / L1 -> L0B
  kMTE2 = 1 << 1,    // GM -> UB / GM -> L1
  kMTE3 = 1 << 2,    // UB -> GM / UB -> L1
  kCube = 1 << 3,    // M pipe: MAD matrix multiply
  kVector = 1 << 4,  // V pipe: SimtVF / SimdVF / vector arithmetic
  kFixpipe = 1 << 5, // L0C -> UB / L0C -> GM
  kScalar = 1 << 6,  // S pipe: scalar and configuration operations
  kAll = (1 << 7) - 1,
};

inline constexpr uint16_t PipeMask(ResourcePipe pipe) {
  return static_cast<uint16_t>(pipe);
}

inline constexpr uint16_t kAllResourcePipes = PipeMask(ResourcePipe::kAll);

inline bool IsSingleResourcePipe(uint16_t mask) {
  return mask != 0 && (mask & (mask - 1)) == 0;
}

inline std::string GetResourcePipeName(uint16_t mask) {
  if (mask == kAllResourcePipes)
    return "ALL";
  ICHECK(mask == 0 || IsSingleResourcePipe(mask))
      << "Multiple pipes where one pipe is required: " << mask;
  if (mask & PipeMask(ResourcePipe::kMTE1))
    return "MTE1";
  if (mask & PipeMask(ResourcePipe::kMTE2))
    return "MTE2";
  if (mask & PipeMask(ResourcePipe::kMTE3))
    return "MTE3";
  if (mask & PipeMask(ResourcePipe::kCube))
    return "M";
  if (mask & PipeMask(ResourcePipe::kVector))
    return "V";
  if (mask & PipeMask(ResourcePipe::kFixpipe))
    return "FIX";
  if (mask & PipeMask(ResourcePipe::kScalar))
    return "S";
  return "UNKNOWN";
}

inline ResourcePipe ParseAscendPipe(const std::string &name) {
  if (name.rfind("PIPE_", 0) == 0)
    return ParseAscendPipe(name.substr(5));
  if (name == "MTE1")
    return ResourcePipe::kMTE1;
  if (name == "MTE2")
    return ResourcePipe::kMTE2;
  if (name == "MTE3")
    return ResourcePipe::kMTE3;
  if (name == "M" || name == "MAD")
    return ResourcePipe::kCube;
  if (name == "V" || name == "VEC" || name == "VECTOR")
    return ResourcePipe::kVector;
  if (name == "FIX" || name == "FIXPIPE")
    return ResourcePipe::kFixpipe;
  if (name == "S" || name == "SCALAR")
    return ResourcePipe::kScalar;
  if (name == "ALL")
    return ResourcePipe::kAll;
  return ResourcePipe::kUnknown;
}

enum class FlagEndpoint : uint8_t {
  kSet,
  kWait,
};

inline ResourcePipe GetFlagEndpointPipe(const std::string &hard_event,
                                        FlagEndpoint endpoint) {
  size_t separator = hard_event.find('_');
  if (separator == std::string::npos)
    return ResourcePipe::kUnknown;
  return endpoint == FlagEndpoint::kSet
             ? ParseAscendPipe(hard_event.substr(0, separator))
             : ParseAscendPipe(hard_event.substr(separator + 1));
}

inline uint16_t GetAscendCopyPipeMask(const Buffer &src, const Buffer &dst) {
  if (IsGlobalBuffer(src) && (IsSharedBuffer(dst) || IsL1Buffer(dst))) {
    return PipeMask(ResourcePipe::kMTE2);
  }
  if (IsSharedBuffer(src) && (IsGlobalBuffer(dst) || IsL1Buffer(dst))) {
    return PipeMask(ResourcePipe::kMTE3);
  }
  if (IsL1Buffer(src) &&
      (IsL0ABuffer(dst) || IsL0BBuffer(dst) || IsL0SFBuffer(dst))) {
    return PipeMask(ResourcePipe::kMTE1);
  }
  if (IsL0CBuffer(src) && (IsSharedBuffer(dst) || IsGlobalBuffer(dst))) {
    return PipeMask(ResourcePipe::kFixpipe);
  }
  // All remaining Ascend copies lower to element-wise work on the Vector
  // core rather than an MTE instruction.
  return PipeMask(ResourcePipe::kVector);
}

// A fill on L1 lowers to ascend_fill_l1 (MTE2). Every other fill lowers to
// element-wise BufferStores, and out-of-VF element-wise work is scalar work on
// Ascend: vector stores, vector arithmetic, and the vector constructors they
// need are only available inside a VF body. A fill that does run on the vector
// unit is therefore always inside a VF block, which contributes PIPE_V through
// GetAscendBlockPipe before this classifier is ever consulted. Reporting a fill
// as PIPE_V here would make InsertSync treat its scalar stores as vector work
// and drop the S->V handshake with the consumer of the filled tile.
inline uint16_t GetAscendFillPipeMask(const Buffer &dst) {
  return PipeMask(IsL1Buffer(dst) ? ResourcePipe::kMTE2
                                  : ResourcePipe::kScalar);
}

inline ResourcePipe GetAscendBlockPipe(const String &name_hint) {
  if (name_hint == "VECTOR" || name_hint == "SIMT_VF" ||
      name_hint == "SIMD_VF") {
    return ResourcePipe::kVector;
  }
  return ResourcePipe::kUnknown;
}

inline ResourcePipe GetConstantPipeArgument(const CallNode *call,
                                            size_t argument) {
  if (argument >= call->args.size())
    return ResourcePipe::kUnknown;
  const auto *pipe = call->args[argument].as<StringImmNode>();
  return pipe == nullptr ? ResourcePipe::kUnknown
                         : ParseAscendPipe(pipe->value);
}

inline ResourcePipe GetConstantFlagEndpoint(const CallNode *call,
                                            FlagEndpoint endpoint) {
  if (call->args.empty())
    return ResourcePipe::kUnknown;
  const auto *event = call->args[0].as<StringImmNode>();
  return event == nullptr ? ResourcePipe::kUnknown
                          : GetFlagEndpointPipe(event->value, endpoint);
}

// Classify both frontend tile operations and the intrinsics emitted by
// LowerTileOp. A zero result means the call has no hardware execution or is
// not an Ascend operation; multiple bits describe a composite instruction.
inline uint16_t GetAscendCallPipeMask(const Call &call) {
  const auto callee = call->op.as<Op>();
  if (!callee)
    return 0;
  const Op &op = callee.value();
  const std::string &name = op->name;

  static const Op &fill_op = Op::Get("tl.tileop.fill");
  if (IsAscendCopyCall(call.get())) {
    Copy copy(call->args, call->annotations);
    return GetAscendCopyPipeMask(copy->src, copy->dst);
  }
  if (op.same_as(fill_op)) {
    Fill fill(call->args, call->annotations);
    return GetAscendFillPipeMask(fill->dst);
  }

  if (name == "tl.tileop.gemm" || name == "tl.tileop.gemm_blockscaled") {
    return PipeMask(ResourcePipe::kCube);
  }
  if (name == "tl.tileop.reduce" || name == "tl.tileop.finalize_reducer") {
    return PipeMask(ResourcePipe::kVector);
  }
  // tl.tileop.region and tl.access_ptr only carry access metadata for their
  // enclosing operation; they do not execute on a hardware pipe themselves.
  if (name == "tl.tileop.region" || op.same_as(tl::access_ptr()))
    return 0;

  if (op.same_as(ascend_mad()) || op.same_as(ascend_mad_mx()))
    return PipeMask(ResourcePipe::kCube);
  if (op.same_as(ascend_gemm_l1()) ||
      op.same_as(ascend_blockscaled_gemm_l1())) {
    return PipeMask(ResourcePipe::kMTE1) | PipeMask(ResourcePipe::kCube);
  }
  if (op.same_as(ascend_copy_gm_to_ubuf()) ||
      op.same_as(ascend_copy_gm_to_cbuf()) || op.same_as(ascend_fill_l1())) {
    return PipeMask(ResourcePipe::kMTE2);
  }
  if (op.same_as(ascend_load_cbuf_to_ca()) ||
      op.same_as(ascend_load_cbuf_to_cb()) || op.same_as(ascend_load_ca_sf()) ||
      op.same_as(ascend_load_cb_sf())) {
    return PipeMask(ResourcePipe::kMTE1);
  }
  if (op.same_as(ascend_copy_matrix_cc_to_ub()) ||
      op.same_as(ascend_copy_matrix_cc_to_gm())) {
    return PipeMask(ResourcePipe::kFixpipe);
  }
  if (op.same_as(ascend_copy_ubuf_to_gm()) ||
      op.same_as(ascend_copy_ubuf_to_cbuf()) ||
      op.same_as(ascend_nd2nz_post_copy())) {
    return PipeMask(ResourcePipe::kMTE3);
  }
  if (op.same_as(ascend_nd2nz_scatter()) || name.rfind("tl.simd.", 0) == 0 ||
      op.same_as(sync_warp()) || op.same_as(ballot_sync()) ||
      op.same_as(ballot()) || op.same_as(activemask()) ||
      op.same_as(warp_reduce_sum()) || op.same_as(warp_reduce_max()) ||
      op.same_as(warp_reduce_min()) || op.same_as(warp_reduce_bitand()) ||
      op.same_as(warp_reduce_bitor()) || op.same_as(atomic_add_elem_op()) ||
      op.same_as(atomic_add_ret_elem_op()) ||
      op.same_as(atomic_addx2_elem_op()) ||
      op.same_as(atomic_addx2_ret_elem_op()) ||
      op.same_as(atomic_addx4_elem_op()) ||
      op.same_as(atomic_addx4_ret_elem_op()) ||
      op.same_as(atomic_load_elem_op()) || op.same_as(atomic_store_elem_op()) ||
      op.same_as(atomic_or_elem_op()) || op.same_as(atomic_max_elem_op()) ||
      op.same_as(atomic_max_ret_elem_op()) ||
      op.same_as(atomic_min_elem_op()) ||
      op.same_as(atomic_min_ret_elem_op()) || op.same_as(rng_init()) ||
      op.same_as(rng_rand()) || op.same_as(rng_rand_float())) {
    return PipeMask(ResourcePipe::kVector);
  }

  if (op.same_as(ascend_set_flag()))
    return PipeMask(GetConstantFlagEndpoint(call.get(), FlagEndpoint::kSet));
  if (op.same_as(ascend_wait_flag()))
    return PipeMask(GetConstantFlagEndpoint(call.get(), FlagEndpoint::kWait));
  if (op.same_as(ascend_pipe_barrier()) || op.same_as(ascend_get_buf()) ||
      op.same_as(ascend_rls_buf())) {
    return PipeMask(GetConstantPipeArgument(call.get(), 0));
  }
  if (op.same_as(ascend_cross_core_set_flag()) ||
      op.same_as(ascend_cross_core_wait_flag())) {
    return PipeMask(GetConstantPipeArgument(call.get(), 1));
  }

  if (op.same_as(ascend_set_copy_pad_value()) ||
      op.same_as(ascend_threadfence()) || op.same_as(ascend_set_hf32_mode()) ||
      op.same_as(ascend_set_mmad_direction()) ||
      op.same_as(ascend_set_atomic()) || op.same_as(ascend_set_atomic_none()) ||
      op.same_as(ascend_read_gm_bypass_dcache()) ||
      op.same_as(ascend_write_gm_bypass_dcache()) || op.same_as(loop_break()) ||
      op.same_as(device_assert()) || op.same_as(device_assert_with_msg())) {
    return PipeMask(ResourcePipe::kScalar);
  }

  if (op.same_as(builtin::call_extern()) && !call->args.empty()) {
    if (const auto *symbol = call->args[0].as<StringImmNode>()) {
      const std::string &external_name = symbol->value;
      if (external_name == "debug_print_msg" ||
          external_name == "debug_print_var" ||
          external_name == "debug_print_buffer_value") {
        return PipeMask(ResourcePipe::kScalar);
      }
      if (external_name == "debug_print_buffer_dump" && call->args.size() > 3) {
        if (const auto *scope_call = call->args[3].as<CallNode>()) {
          if (!scope_call->args.empty()) {
            if (const auto *load = scope_call->args[0].as<BufferLoadNode>()) {
              std::string scope = load->buffer.scope();
              return PipeMask(scope == "shared" || scope == "shared.dyn"
                                  ? ResourcePipe::kVector
                                  : ResourcePipe::kCube);
            }
          }
        }
      }
      if (external_name == "debug_print_global_buffer_dump")
        return PipeMask(ResourcePipe::kMTE3);
    }
  }
  return 0;
}

// AutoSchedule models synchronization through special-register dependencies.
// A standalone sync task falls back to Scalar after its otherwise-empty mask
// is resolved, while a sync nested in T.PerCoreTask must not contaminate the
// surrounding data pipe. MergeUB queries GetAscendCallPipeMask directly to
// recover the concrete endpoint pipe instead.
inline uint16_t GetAscendTaskPipeMask(const Call &call) {
  if (call->op.same_as(ascend_set_flag()) ||
      call->op.same_as(ascend_wait_flag()) ||
      call->op.same_as(ascend_pipe_barrier()) ||
      call->op.same_as(ascend_cross_core_set_flag()) ||
      call->op.same_as(ascend_cross_core_wait_flag()) ||
      call->op.same_as(ascend_get_buf()) ||
      call->op.same_as(ascend_rls_buf())) {
    return 0;
  }
  return GetAscendCallPipeMask(call);
}

inline ResourcePipe GetAscendCallArgumentPipe(const Call &call,
                                              size_t argument) {
  if (call->op.same_as(ascend_gemm_l1()) ||
      call->op.same_as(ascend_blockscaled_gemm_l1())) {
    size_t pointer_arguments = call->op.same_as(ascend_gemm_l1()) ? 3 : 5;
    if (argument < pointer_arguments)
      return argument == 0 ? ResourcePipe::kCube : ResourcePipe::kMTE1;
    return ResourcePipe::kScalar;
  }

  uint16_t mask = GetAscendCallPipeMask(call);
  if (mask == kAllResourcePipes || IsSingleResourcePipe(mask))
    return static_cast<ResourcePipe>(mask);
  return ResourcePipe::kUnknown;
}

inline bool IsAscendPipeRelevantCall(const Call &call) {
  const auto callee = call->op.as<Op>();
  if (!callee)
    return false;
  const std::string &name = callee.value()->name;
  return name.rfind("tl.ascend", 0) == 0 || name.rfind("tl.simd.", 0) == 0 ||
         name.rfind("tl.tileop.", 0) == 0;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_ASCEND_TRANSFORM_ASCEND_PIPE_H_

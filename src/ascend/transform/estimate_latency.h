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

#pragma once

#include <cstdint>

namespace tvm {
namespace tl {

struct AscendLatencyParams {
  // Convention: latency is in cycles, bandwidth in bytes/cycle, and compute
  // throughput in operations/cycle (two operations per multiply-add).

  // Cube completion overhead beyond the throughput cost.
  int64_t cube_base_latency{8};
  int64_t cube_fallback_throughput{8192};
  int64_t cube_min_ii{1};
  int64_t cube_pipeline_depth{1};
  int64_t cube_unit_m{16};
  int64_t cube_unit_k{16};
  int64_t cube_unit_n{16};
  int64_t cube_throughput_fp16{8192};
  int64_t cube_throughput_bf16{8192};
  // FP32/HF32 throughput, selected by each GEMM's HF32 register mode.
  int64_t cube_throughput_fp32{512};
  int64_t cube_throughput_hf32{4096};
  // FP32/HF32 completion overhead beyond the throughput cost.
  int64_t cube_fp32_base_latency{67};
  // HiF8/FP8 throughput.
  int64_t cube_throughput_fp8{16384};
  // MX MAD completion overhead beyond the independent-issue cost.
  int64_t cube_mx_base_latency{25};
  // Minimum larger M/N extent in the MX compute cost after alignment.
  int64_t cube_mx_min_mn{48};
  // MX FP4 throughput.
  int64_t cube_throughput_mxfp4{32768};
  // INT8 MMAD is not supported on A5; zero selects the fallback.
  int64_t cube_throughput_int8{0};

  // VALU throughput by operation type.
  int64_t add_throughput{51};
  int64_t sub_throughput{51};
  int64_t mul_throughput{51};
  int64_t div_throughput{13};
  int64_t mod_throughput{13};
  int64_t min_max_throughput{51};
  int64_t cmp_throughput{51};
  int64_t logic_throughput{51};
  int64_t special_func_throughput{8};
  // Unknown operations use the OperationCounter one-cycle fallback.
  int64_t default_operation_throughput{0};
  // VRF <-> UB bandwidth per AIV.
  int64_t valu_bandwidth{256};

  // ND->NZ scatter cost, including conversion, interleaving and packing.
  struct Nd2NzLatencyParams {
    // Minimum completion latency.
    int64_t min_cycles;
    // Fixed overhead per invocation.
    int64_t setup_cycles;
    // Overhead per 256-byte source-vector pass across all rows.
    int64_t pass_cycles;
    // Cost per row per pass, scaled by four.
    int64_t row_cycles_x4;
  };
  Nd2NzLatencyParams nd2nz_same_dtype{62, 10, 10, 8};
  Nd2NzLatencyParams nd2nz_bf16_to_f32{98, 40, 4, 13};
  Nd2NzLatencyParams nd2nz_f32_to_bf16{73, 16, 4, 8};

  // MTE1 completion overhead.
  int64_t mte1_base_latency{5};
  // Issue overhead per L0 load descriptor.
  int64_t mte1_issue_overhead{3};
  // MX scale-factor load completion and issue overheads.
  int64_t mte1_sf_base_latency{28};
  int64_t mte1_sf_issue_overhead{1};
  // Default FixPipe completion overhead.
  int64_t fixpipe_base_latency{58};
  // Issue overhead per descriptor at the shared FixPipe endpoint.
  int64_t fixpipe_issue_overhead{2};
  // L0C->GM completion overhead.
  int64_t fixpipe_l0c_to_gm_base_latency{200};
  // Additional completion cycles per memory descriptor.
  int64_t mte_descriptor_cycles{0};

  // Physical path bandwidths; FixPipe uses a single destination here.
  int64_t l1_to_l0a_bandwidth{256};
  int64_t l1_to_l0b_bandwidth{256};
  int64_t l1_to_l0_sf_bandwidth{32};
  int64_t l1_to_bt_bandwidth{32};
  int64_t l1_to_fp_buf_bandwidth{32};
  int64_t aic_mte2_to_l1_bandwidth{100};
  int64_t fixpipe_bandwidth{128};
  // GM->L1 completion overhead and minimum issue interval per descriptor.
  int64_t mte2_gm_to_l1_base_latency{190};
  int64_t mte2_gm_to_l1_min_ii{64};

  // AIV MTE completion overheads and bandwidths for pure-Vector and Mixed
  // kernels. All AIV MTE bandwidths use the aggregate payload of two AIVs.
  int64_t mte2_gm_to_ub_base_latency{90};
  int64_t mte2_gm_to_ub_bandwidth{100};
  int64_t mte3_ub_to_gm_base_latency{180};
  int64_t mte3_ub_to_gm_bandwidth{115};
  int64_t mte3_ub_to_l1_base_latency{43};
  int64_t mte3_ub_to_l1_bandwidth{256};
  // Minimum issue intervals per GM->UB and UB->GM descriptor.
  int64_t mte2_gm_to_ub_min_ii{13};
  int64_t mte3_ub_to_gm_min_ii{10};

  // Dual-destination FixPipe bandwidth cap, per-AIV row-width multiplier,
  // and fallback bandwidth when the row width is unknown.
  int64_t fixpipe_dual_bandwidth{256};
  int64_t fixpipe_dual_width_scale{2};
  int64_t fixpipe_dual_unknown_width_bandwidth{128};
  // N-strided MTE transaction sizes are per-AIV contiguous row widths.
  // Bandwidths cover full transactions, sub-transaction plateaus, wide rows,
  // and unknown row widths, respectively.
  int64_t mte2_gm_to_ub_n_transaction_bytes{128};
  int64_t mte2_gm_to_ub_n_peak_bandwidth{102};
  int64_t mte2_gm_to_ub_n_plateau_bandwidth{50};
  int64_t mte2_gm_to_ub_n_wide_bandwidth{100};
  int64_t mte2_gm_to_ub_n_unknown_bandwidth{48};
  int64_t mte3_ub_to_gm_n_transaction_bytes{128};
  int64_t mte3_ub_to_gm_n_peak_bandwidth{86};
  int64_t mte3_ub_to_gm_n_plateau_bandwidth{40};
  int64_t mte3_ub_to_gm_n_wide_bandwidth{110};
  int64_t mte3_ub_to_gm_n_unknown_bandwidth{40};
  // Raw N-split UB->L1: below 160 B per-AIV rows the rate is 3/4 of the
  // width; from 160 B up the row streams at the full width rate, capped by
  // the 256 B/cycle endpoint.
  int64_t mte3_ub_to_l1_raw_n_full_width_bytes{160};
  int64_t mte3_ub_to_l1_raw_n_width_numerator{3};
  int64_t mte3_ub_to_l1_raw_n_width_denominator{4};
  int64_t mte3_ub_to_l1_raw_n_unknown_bandwidth{48};

  // Completion overheads for N-strided GM->UB, narrow N-strided UB->GM,
  // and dual-destination L0C->UB copies.
  int64_t mte2_gm_to_ub_n_base_latency{75};
  int64_t mte3_ub_to_gm_narrow_base_latency{440};
  int64_t fixpipe_dual_base_latency{55};
};

} // namespace tl
} // namespace tvm

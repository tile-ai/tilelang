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
  // throughput in operations/cycle (MACs/cycle for Cube).

  // Cube (MMAD). The physical FP16/BF16 granularity and throughput come from
  // the A5 architecture guide p.7: 16x16x16x2 = 8192 operations/cycle. The
  // dispatch overhead is measured; one MMAD can be issued each cycle.
  int64_t cube_base_latency{8};
  int64_t cube_fallback_throughput{8192};
  int64_t cube_min_ii{1};
  int64_t cube_pipeline_depth{1};
  int64_t cube_unit_m{16};
  int64_t cube_unit_k{16};
  int64_t cube_unit_n{16};
  int64_t cube_throughput_fp16{8192};
  int64_t cube_throughput_bf16{8192};
  // A5 has no native FP32 MMAD; zero selects the fallback throughput.
  int64_t cube_throughput_fp32{0};
  // HiF8/FP8 uses 16x32x16x2 operations/cycle.
  int64_t cube_throughput_fp8{16384};
  // INT8 MMAD is not supported on A5; zero selects the fallback.
  int64_t cube_throughput_int8{0};

  // VALU throughput measured with cannsim RVEC instruction timing. Common
  // fp32 operations sustain about 51 ops/cycle; division, modulo, and special
  // functions use iterative implementations and are slower.
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
  // A5 architecture guide p.8: VRF <-> UB is 256 bytes/cycle per AIV.
  int64_t valu_bandwidth{256};
  int64_t vector_queue_depth{16};

  // MTE1 retains its existing completion base pending a scheduler-stability
  // follow-up. The FixPipe bases below use Ascend950DT SYS_CNT P-sweeps.
  int64_t mte1_base_latency{5};
  int64_t fixpipe_base_latency{58};
  // Real-device L0C->GM latency fit.
  int64_t fixpipe_l0c_to_gm_base_latency{200};
  // Descriptor overhead is already absorbed into measured base latency.
  int64_t mte_descriptor_cycles{0};

  // Physical path bandwidths. L1->L0A/L0B and UB->L1 reach 256 B/cycle;
  // GM->L1 reaches 100 B/cycle; a single-destination FixPipe path sustains
  // 128 B/cycle. AIV MTE uses measured effective bandwidths below.
  int64_t l1_to_l0a_bandwidth{256};
  int64_t l1_to_l0b_bandwidth{256};
  int64_t l1_to_bt_bandwidth{32};
  int64_t l1_to_fp_buf_bandwidth{32};
  int64_t aic_mte2_to_l1_bandwidth{100};
  int64_t fixpipe_bandwidth{128};
  // GM->L1 completion and saturated issue rate use the same bytes/cycle fit;
  // small descriptors retain the measured II floor below.
  int64_t mte2_gm_to_l1_base_latency{190};
  int64_t mte2_gm_to_l1_min_ii{64};

  // Real-device AIV MTE model shared by pure-Vector and Mixed kernels. The
  // calibration unit is aggregate payload across the two normally active
  // AIVs. A rewritten copy contains one AIV's region, so the estimator
  // normalizes it to this common unit without inspecting cthread guards. The
  // bandwidths use Ascend950DT SYS_CNT P-sweeps (20 warmups, 2000 iterations,
  // an 8-slot UB ring through P=64): GM->UB reaches 100 B/cycle, while UB->GM
  // reaches 115 B/cycle for M-like and >=512B N-like rows, with a 110 B/cycle
  // dip at 256B N-like rows.
  int64_t mte2_gm_to_ub_base_latency{90};
  int64_t mte2_gm_to_ub_bandwidth{100};
  int64_t mte3_ub_to_gm_base_latency{180};
  int64_t mte3_ub_to_gm_bandwidth{115};
  int64_t mte3_ub_to_l1_base_latency{43};
  int64_t mte3_ub_to_l1_bandwidth{256};
  // Small packets remain descriptor-limited; above this payload II uses the
  // aggregate calibration bytes.
  int64_t mte2_gm_to_ub_packet_floor_bytes{512};
  int64_t mte3_ub_to_gm_packet_floor_bytes{512};

  // Strided MTE geometry parameters. Effective bandwidth is limited by the
  // shared endpoint and by the physically contiguous row width.
  //
  // Strided N-split rows travel in 128-byte MTE transactions: the row rate
  // peaks when the per-AIV width is a whole multiple of 128 B and dips
  // between multiples. A row narrower than one transaction pays a measured
  // half-rate plateau; from two transactions (256 B) up the row streams at
  // the wide steady rate. Values use the aggregate two-AIV calibration unit
  // described above.
  int64_t fixpipe_dual_bandwidth{256};
  int64_t fixpipe_dual_width_scale{2};
  int64_t fixpipe_dual_unknown_width_bandwidth{128};
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

  // Geometry-specific completion bases. UB->GM has a separate narrow-N value
  // because 64-byte strided rows incur a large first-descriptor penalty even
  // though steady-state II is represented by the bandwidth above.
  int64_t mte2_gm_to_ub_n_base_latency{75};
  int64_t mte3_ub_to_gm_narrow_base_latency{440};
  int64_t fixpipe_dual_base_latency{55};
};

} // namespace tl
} // namespace tvm

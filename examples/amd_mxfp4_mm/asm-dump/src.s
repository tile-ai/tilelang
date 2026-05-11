
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx950
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z9gemm_corePKhS0_PvS0_S0_iiiiiii ; -- Begin function _Z9gemm_corePKhS0_PvS0_S0_iiiiiii
	.globl	_Z9gemm_corePKhS0_PvS0_S0_iiiiiii
	.p2align	8
	.type	_Z9gemm_corePKhS0_PvS0_S0_iiiiiii,@function
_Z9gemm_corePKhS0_PvS0_S0_iiiiiii:      ; @_Z9gemm_corePKhS0_PvS0_S0_iiiiiii
; %bb.0:
	s_load_dwordx2 s[6:7], s[0:1], 0x10
	s_load_dwordx8 s[8:15], s[0:1], 0x28
	v_and_b32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s15, s2, 5
	s_lshl_b32 s23, s3, 5
	v_or_b32_e32 v2, s23, v1
	s_mul_i32 s22, s11, s4
	s_add_i32 s2, s22, s11
	s_min_i32 s5, s2, s10
	v_lshrrev_b32_e32 v14, 5, v0
	v_cmp_gt_i32_e64 s[2:3], s9, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_mov_b32_e32 v18, 0
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	s_cmp_ge_i32 s22, s5
	v_accvgpr_write_b32 a15, 0
	s_cbranch_scc0 .LBB0_4
; %bb.1:
	s_cmp_lg_u32 s12, 1
	s_mov_b64 s[0:1], -1
	s_cbranch_scc1 .LBB0_26
.LBB0_2:                                ; %Flow442
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccz .LBB0_60
.LBB0_3:                                ; %.loopexit
	s_endpgm
.LBB0_4:
	s_load_dwordx4 s[16:19], s[0:1], 0x0
	s_ashr_i32 s11, s10, 1
	v_or_b32_e32 v10, s15, v1
	v_mad_i64_i32 v[4:5], s[20:21], s11, v10, 0
	s_ashr_i32 s20, s22, 1
	v_lshlrev_b32_e32 v15, 4, v14
	v_add_u32_e32 v8, s20, v15
	v_cmp_gt_i32_e32 vcc, s8, v10
	v_mov_b32_e32 v19, v18
	v_mov_b32_e32 v20, v18
	v_mov_b32_e32 v21, v18
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[4:5], s[16:17], 0, v[4:5]
	v_ashrrev_i32_e32 v9, 31, v8
	s_and_saveexec_b64 s[16:17], vcc
	s_cbranch_execz .LBB0_6
; %bb.5:                                ; %.loopexit44.i
	v_lshl_add_u64 v[6:7], v[4:5], 0, v[8:9]
	global_load_dwordx4 v[18:21], v[6:7], off
.LBB0_6:
	s_or_b64 exec, exec, s[16:17]
	s_load_dwordx2 s[20:21], s[0:1], 0x18
	v_mad_i64_i32 v[6:7], s[16:17], s11, v2, 0
	v_mov_b32_e32 v26, 0
	v_mov_b32_e32 v27, v26
	v_mov_b32_e32 v28, v26
	v_mov_b32_e32 v29, v26
	v_lshl_add_u64 v[6:7], s[18:19], 0, v[6:7]
	s_and_saveexec_b64 s[16:17], s[2:3]
	s_cbranch_execz .LBB0_8
; %bb.7:                                ; %.loopexit.i
	v_lshl_add_u64 v[8:9], v[6:7], 0, v[8:9]
	global_load_dwordx4 v[26:29], v[8:9], off
.LBB0_8:
	s_or_b64 exec, exec, s[16:17]
	s_load_dwordx2 s[16:17], s[0:1], 0x20
	v_mad_i64_i32 v[8:9], s[0:1], s13, v10, 0
	s_ashr_i32 s0, s22, 5
	s_nop 0
	v_add_u32_e32 v10, s0, v14
	v_cmp_gt_i32_e64 s[0:1], s13, v10
	s_and_b64 s[18:19], vcc, s[0:1]
	v_mov_b32_e32 v17, 0x7f
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[8:9], s[20:21], 0, v[8:9]
	v_mov_b32_e32 v16, 0x7f
	s_and_saveexec_b64 s[0:1], s[18:19]
	s_cbranch_execz .LBB0_10
; %bb.9:
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[12:13], v[8:9], 0, v[10:11]
	global_load_ubyte v16, v[12:13], off
.LBB0_10:
	s_or_b64 exec, exec, s[0:1]
	s_ashr_i32 s0, s23, 5
	v_lshlrev_b32_e32 v0, 2, v0
	s_lshl_b32 s1, s14, 5
	v_and_b32_e32 v12, 60, v0
	v_mov_b32_e32 v13, 0
	v_mov_b32_e32 v0, s0
	s_ashr_i32 s10, s10, 5
	v_mad_i64_i32 v[34:35], s[0:1], s1, v0, v[12:13]
	v_lshrrev_b32_e32 v0, 4, v1
	v_or_b32_e32 v34, v34, v0
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a15, 0
	s_and_b64 s[18:19], s[2:3], s[0:1]
	v_lshl_add_u64 v[0:1], s[16:17], 0, v[34:35]
	s_and_saveexec_b64 s[0:1], s[18:19]
	s_cbranch_execz .LBB0_12
; %bb.11:
	v_lshlrev_b32_e32 v11, 5, v10
	v_and_b32_e32 v34, 0xffffff00, v11
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshlrev_b32_e32 v11, 6, v10
	v_and_b32_e32 v12, 0xc0, v11
	v_lshrrev_b32_e32 v10, 1, v10
	v_lshl_add_u64 v[34:35], v[0:1], 0, v[34:35]
	v_and_b32_e32 v10, 2, v10
	v_mov_b32_e32 v11, v13
	v_lshl_add_u64 v[12:13], v[34:35], 0, v[12:13]
	v_lshl_add_u64 v[10:11], v[12:13], 0, v[10:11]
	global_load_ubyte v17, v[10:11], off
.LBB0_12:                               ; %_Z9load_fragPKhS0_S0_S0_iiiillllibbRDv32_hS2_RhS3_.exit
	s_or_b64 exec, exec, s[0:1]
	s_add_i32 s11, s22, 64
	s_cmp_ge_i32 s11, s5
	s_cbranch_scc1 .LBB0_24
; %bb.13:                               ; %.lr.ph
	v_mov_b32_e32 v34, 0
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a14, 0
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a12, 0
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a10, 0
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a8, 0
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a6, 0
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a4, 0
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a2, 0
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a0, 0
.LBB0_14:                               ; =>This Inner Loop Header: Depth=1
	s_ashr_i32 s0, s11, 1
	v_add_u32_e32 v10, s0, v15
	v_mov_b32_e32 v35, v34
	v_mov_b32_e32 v36, v34
	v_mov_b32_e32 v37, v34
	v_mov_b64_e32 v[44:45], v[40:41]
	v_ashrrev_i32_e32 v11, 31, v10
	v_mov_b64_e32 v[42:43], v[38:39]
	v_mov_b64_e32 v[40:41], v[36:37]
	v_mov_b64_e32 v[38:39], v[34:35]
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_16
; %bb.15:                               ; %.loopexit44.i136
                                        ;   in Loop: Header=BB0_14 Depth=1
	v_lshl_add_u64 v[12:13], v[4:5], 0, v[10:11]
	global_load_dwordx4 v[38:41], v[12:13], off
.LBB0_16:                               ;   in Loop: Header=BB0_14 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[52:53], v[40:41]
	v_mov_b64_e32 v[50:51], v[38:39]
	v_mov_b64_e32 v[48:49], v[36:37]
	v_mov_b64_e32 v[46:47], v[34:35]
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_18
; %bb.17:                               ; %.loopexit.i134
                                        ;   in Loop: Header=BB0_14 Depth=1
	v_lshl_add_u64 v[10:11], v[6:7], 0, v[10:11]
	global_load_dwordx4 v[46:49], v[10:11], off
.LBB0_18:                               ;   in Loop: Header=BB0_14 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_ashr_i32 s0, s11, 5
	v_add_u32_e32 v10, s0, v14
	v_cmp_gt_i32_e64 s[0:1], s13, v10
	s_and_b64 s[16:17], vcc, s[0:1]
	v_mov_b32_e32 v12, 0x7f
	v_mov_b32_e32 v11, 0x7f
	s_and_saveexec_b64 s[0:1], s[16:17]
	s_cbranch_execz .LBB0_20
; %bb.19:                               ;   in Loop: Header=BB0_14 Depth=1
	v_ashrrev_i32_e32 v11, 31, v10
	v_lshl_add_u64 v[22:23], v[8:9], 0, v[10:11]
	global_load_ubyte v11, v[22:23], off
.LBB0_20:                               ;   in Loop: Header=BB0_14 Depth=1
	s_or_b64 exec, exec, s[0:1]
	v_cmp_gt_i32_e64 s[0:1], s10, v10
	s_and_b64 s[16:17], s[2:3], s[0:1]
	s_and_saveexec_b64 s[0:1], s[16:17]
	s_cbranch_execz .LBB0_22
; %bb.21:                               ;   in Loop: Header=BB0_14 Depth=1
	v_lshlrev_b32_e32 v12, 5, v10
	v_and_b32_e32 v12, 0xffffff00, v12
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshlrev_b32_e32 v22, 6, v10
	v_and_b32_e32 v22, 0xc0, v22
	v_mov_b32_e32 v23, v34
	v_lshrrev_b32_e32 v10, 1, v10
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[12:13]
	v_and_b32_e32 v24, 2, v10
	v_mov_b32_e32 v25, v34
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[22:23]
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[24:25]
	global_load_ubyte v12, v[12:13], off
.LBB0_22:                               ; %_Z9load_fragPKhS0_S0_S0_iiiillllibbRDv32_hS2_RhS3_.exit138
                                        ;   in Loop: Header=BB0_14 Depth=1
	s_or_b64 exec, exec, s[0:1]
	v_and_b32_e32 v10, 0xff, v16
	v_and_b32_e32 v13, 0xff, v17
	s_add_i32 s11, s11, 64
	s_cmp_ge_i32 s11, s5
	v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[18:21], v[26:29], a[0:15], v10, v13 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cbranch_scc1 .LBB0_25
; %bb.23:                               ;   in Loop: Header=BB0_14 Depth=1
	v_mov_b64_e32 v[18:19], v[38:39]
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[26:27], v[46:47]
	v_mov_b64_e32 v[20:21], v[40:41]
	v_mov_b64_e32 v[22:23], v[42:43]
	v_mov_b64_e32 v[24:25], v[44:45]
	v_mov_b64_e32 v[28:29], v[48:49]
	v_mov_b64_e32 v[30:31], v[50:51]
	v_mov_b64_e32 v[32:33], v[52:53]
	v_mov_b32_e32 v17, v12
	v_mov_b32_e32 v16, v11
	s_branch .LBB0_14
.LBB0_24:
	s_waitcnt vmcnt(0)
	v_mov_b64_e32 v[44:45], v[24:25]
	v_mov_b64_e32 v[52:53], v[32:33]
	v_mov_b64_e32 v[40:41], v[20:21]
	v_mov_b64_e32 v[38:39], v[18:19]
	v_mov_b64_e32 v[48:49], v[28:29]
	v_mov_b64_e32 v[46:47], v[26:27]
	v_mov_b32_e32 v12, v17
	v_mov_b32_e32 v11, v16
	v_mov_b64_e32 v[42:43], v[22:23]
	v_mov_b64_e32 v[50:51], v[30:31]
.LBB0_25:                               ; %Flow444
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v0, 0xff, v11
	v_and_b32_e32 v1, 0xff, v12
	s_nop 1
	v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[38:41], v[46:49], a[0:15], v0, v1 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cmp_lg_u32 s12, 1
	s_mov_b64 s[0:1], -1
	s_cbranch_scc0 .LBB0_2
.LBB0_26:
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_59
; %bb.27:                               ; %.preheader154
	v_lshl_add_u32 v4, v14, 2, s15
	s_mul_hi_i32 s5, s8, s4
	s_mul_i32 s4, s8, s4
	s_ashr_i32 s12, s9, 31
	v_lshl_add_u64 v[0:1], v[2:3], 2, s[6:7]
	v_cmp_gt_i32_e32 vcc, s8, v4
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_29
; %bb.28:
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[4:5]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a0, off
.LBB0_29:
	s_or_b64 exec, exec, s[10:11]
	v_or_b32_e32 v6, 1, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_31
; %bb.30:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a1, off
.LBB0_31:
	s_or_b64 exec, exec, s[10:11]
	v_or_b32_e32 v6, 2, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_33
; %bb.32:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a2, off
.LBB0_33:
	s_or_b64 exec, exec, s[10:11]
	v_or_b32_e32 v6, 3, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_35
; %bb.34:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a3, off
.LBB0_35:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 8, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_37
; %bb.36:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a4, off
.LBB0_37:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 9, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_39
; %bb.38:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a5, off
.LBB0_39:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 10, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_41
; %bb.40:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a6, off
.LBB0_41:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 11, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_43
; %bb.42:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a7, off
.LBB0_43:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 16, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_45
; %bb.44:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a8, off
.LBB0_45:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 17, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_47
; %bb.46:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a9, off
.LBB0_47:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 18, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_49
; %bb.48:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a10, off
.LBB0_49:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 19, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_51
; %bb.50:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a11, off
.LBB0_51:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 24, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_53
; %bb.52:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a12, off
.LBB0_53:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 25, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_55
; %bb.54:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a13, off
.LBB0_55:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 26, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_57
; %bb.56:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a14, off
.LBB0_57:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v4, 27, v4
	v_cmp_gt_i32_e32 vcc, s8, v4
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_59
; %bb.58:
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], s[4:5], 0, v[4:5]
	v_mul_lo_u32 v6, v5, s9
	v_mul_lo_u32 v7, v4, s12
	v_mad_u64_u32 v[4:5], s[4:5], v4, s9, 0
	v_add3_u32 v5, v5, v7, v6
	v_lshl_add_u64 v[0:1], v[4:5], 2, v[0:1]
	global_store_dword v[0:1], a15, off
.LBB0_59:                               ; %Flow439
	s_or_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_3
.LBB0_60:
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_3
; %bb.61:                               ; %.preheader
	v_lshl_add_u32 v4, v14, 2, s15
	v_lshl_add_u64 v[0:1], v[2:3], 1, s[6:7]
	v_cmp_gt_i32_e32 vcc, s8, v4
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_63
; %bb.62:
	v_mad_i64_i32 v[2:3], s[2:3], v4, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a0, off
.LBB0_63:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 1, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_65
; %bb.64:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a1, off
.LBB0_65:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 2, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_67
; %bb.66:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a2, off
.LBB0_67:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 3, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_69
; %bb.68:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a3, off
.LBB0_69:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 8, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_71
; %bb.70:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a4, off
.LBB0_71:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 9, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_73
; %bb.72:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a5, off
.LBB0_73:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 10, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_75
; %bb.74:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a6, off
.LBB0_75:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 11, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_77
; %bb.76:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a7, off
.LBB0_77:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 16, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_79
; %bb.78:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a8, off
.LBB0_79:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 17, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_81
; %bb.80:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a9, off
.LBB0_81:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 18, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_83
; %bb.82:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a10, off
.LBB0_83:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 19, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_85
; %bb.84:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a11, off
.LBB0_85:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 24, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_87
; %bb.86:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a12, off
.LBB0_87:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 25, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_89
; %bb.88:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a13, off
.LBB0_89:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 26, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_91
; %bb.90:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a14, off
.LBB0_91:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 27, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_3
; %bb.92:
	v_mad_i64_i32 v[2:3], s[0:1], v2, s9, 0
	v_lshl_add_u64 v[0:1], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[0:1], a15, off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z9gemm_corePKhS0_PvS0_S0_iiiiiii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 68
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 72
		.amdhsa_next_free_sgpr 24
		.amdhsa_accum_offset 56
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z9gemm_corePKhS0_PvS0_S0_iiiiiii, .Lfunc_end0-_Z9gemm_corePKhS0_PvS0_S0_iiiiiii
                                        ; -- End function
	.set _Z9gemm_corePKhS0_PvS0_S0_iiiiiii.num_vgpr, 54
	.set _Z9gemm_corePKhS0_PvS0_S0_iiiiiii.num_agpr, 16
	.set _Z9gemm_corePKhS0_PvS0_S0_iiiiiii.numbered_sgpr, 24
	.set _Z9gemm_corePKhS0_PvS0_S0_iiiiiii.num_named_barrier, 0
	.set _Z9gemm_corePKhS0_PvS0_S0_iiiiiii.private_seg_size, 0
	.set _Z9gemm_corePKhS0_PvS0_S0_iiiiiii.uses_vcc, 1
	.set _Z9gemm_corePKhS0_PvS0_S0_iiiiiii.uses_flat_scratch, 0
	.set _Z9gemm_corePKhS0_PvS0_S0_iiiiiii.has_dyn_sized_stack, 0
	.set _Z9gemm_corePKhS0_PvS0_S0_iiiiiii.has_recursion, 0
	.set _Z9gemm_corePKhS0_PvS0_S0_iiiiiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3424
; TotalNumSgprs: 30
; NumVgprs: 54
; NumAgprs: 16
; TotalNumVgprs: 72
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 3
; VGPRBlocks: 8
; NumSGPRsForWavesPerEU: 30
; NumVGPRsForWavesPerEU: 72
; AccumOffset: 56
; Occupancy: 7
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 13
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z9reduce_skPKfP12hip_bfloat16ii ; -- Begin function _Z9reduce_skPKfP12hip_bfloat16ii
	.globl	_Z9reduce_skPKfP12hip_bfloat16ii
	.p2align	8
	.type	_Z9reduce_skPKfP12hip_bfloat16ii,@function
_Z9reduce_skPKfP12hip_bfloat16ii:       ; @_Z9reduce_skPKfP12hip_bfloat16ii
; %bb.0:
	s_load_dwordx2 s[4:5], s[0:1], 0x10
	v_lshl_add_u32 v0, s2, 8, v0
	s_waitcnt lgkmcnt(0)
	v_cmp_gt_i32_e32 vcc, s4, v0
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB1_8
; %bb.1:                                ; %.preheader
	s_cmp_gt_i32 s5, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_cbranch_scc1 .LBB1_3
; %bb.2:                                ; %.preheader.._crit_edge_crit_edge
	s_load_dwordx2 s[2:3], s[0:1], 0x8
	v_mov_b32_e32 v2, 0
	s_cbranch_execz .LBB1_4
	s_branch .LBB1_7
.LBB1_3:
	s_load_dwordx2 s[2:3], s[0:1], 0x8
	v_mov_b32_e32 v2, 0
.LBB1_4:                                ; %.lr.ph
	s_load_dwordx2 s[6:7], s[0:1], 0x0
	s_ashr_i32 s1, s4, 31
	s_mov_b32 s0, s4
	s_lshl_b64 s[0:1], s[0:1], 2
	v_mov_b32_e32 v4, 0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[0:1], 2, s[6:7]
.LBB1_5:                                ; =>This Inner Loop Header: Depth=1
	global_load_dword v5, v[2:3], off
	s_add_i32 s5, s5, -1
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[0:1]
	s_cmp_eq_u32 s5, 0
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v4, v4, v5
	s_cbranch_scc0 .LBB1_5
; %bb.6:                                ; %._crit_edge.loopexit
	v_lshrrev_b32_e32 v2, 16, v4
.LBB1_7:                                ; %Flow26
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[2:3]
	global_store_short v[0:1], v2, off
.LBB1_8:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z9reduce_skPKfP12hip_bfloat16ii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 8
		.amdhsa_accum_offset 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end1:
	.size	_Z9reduce_skPKfP12hip_bfloat16ii, .Lfunc_end1-_Z9reduce_skPKfP12hip_bfloat16ii
                                        ; -- End function
	.set _Z9reduce_skPKfP12hip_bfloat16ii.num_vgpr, 6
	.set _Z9reduce_skPKfP12hip_bfloat16ii.num_agpr, 0
	.set _Z9reduce_skPKfP12hip_bfloat16ii.numbered_sgpr, 8
	.set _Z9reduce_skPKfP12hip_bfloat16ii.num_named_barrier, 0
	.set _Z9reduce_skPKfP12hip_bfloat16ii.private_seg_size, 0
	.set _Z9reduce_skPKfP12hip_bfloat16ii.uses_vcc, 1
	.set _Z9reduce_skPKfP12hip_bfloat16ii.uses_flat_scratch, 0
	.set _Z9reduce_skPKfP12hip_bfloat16ii.has_dyn_sized_stack, 0
	.set _Z9reduce_skPKfP12hip_bfloat16ii.has_recursion, 0
	.set _Z9reduce_skPKfP12hip_bfloat16ii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 176
; TotalNumSgprs: 14
; NumVgprs: 6
; NumAgprs: 0
; TotalNumVgprs: 6
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 14
; NumVGPRsForWavesPerEU: 6
; AccumOffset: 8
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 1
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_8b457700d3d88645,@object ; @__hip_cuid_8b457700d3d88645
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_8b457700d3d88645
__hip_cuid_8b457700d3d88645:
	.byte	0                               ; 0x0
	.size	__hip_cuid_8b457700d3d88645, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_8b457700d3d88645
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     16
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 68
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 64
    .name:           _Z9gemm_corePKhS0_PvS0_S0_iiiiiii
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .sgpr_spill_count: 0
    .symbol:         _Z9gemm_corePKhS0_PvS0_S0_iiiiiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     72
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         20
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z9reduce_skPKfP12hip_bfloat16ii
    .private_segment_fixed_size: 0
    .sgpr_count:     14
    .sgpr_spill_count: 0
    .symbol:         _Z9reduce_skPKfP12hip_bfloat16ii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata

# __CLANG_OFFLOAD_BUNDLE____END__ hip-amdgcn-amd-amdhsa--gfx950

# __CLANG_OFFLOAD_BUNDLE____START__ host-x86_64-unknown-linux-gnu-
	.file	"src.hip"
	.text
	.globl	_Z24__device_stub__gemm_corePKhS0_PvS0_S0_iiiiiii # -- Begin function _Z24__device_stub__gemm_corePKhS0_PvS0_S0_iiiiiii
	.p2align	4
	.type	_Z24__device_stub__gemm_corePKhS0_PvS0_S0_iiiiiii,@function
_Z24__device_stub__gemm_corePKhS0_PvS0_S0_iiiiiii: # @_Z24__device_stub__gemm_corePKhS0_PvS0_S0_iiiiiii
	.cfi_startproc
# %bb.0:
	subq	$200, %rsp
	.cfi_def_cfa_offset 208
	movq	%rdi, 88(%rsp)
	movq	%rsi, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	%r8, 56(%rsp)
	movl	%r9d, 4(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 112(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 120(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 128(%rsp)
	leaq	4(%rsp), %rax
	movq	%rax, 136(%rsp)
	leaq	208(%rsp), %rax
	movq	%rax, 144(%rsp)
	leaq	216(%rsp), %rax
	movq	%rax, 152(%rsp)
	leaq	224(%rsp), %rax
	movq	%rax, 160(%rsp)
	leaq	232(%rsp), %rax
	movq	%rax, 168(%rsp)
	leaq	240(%rsp), %rax
	movq	%rax, 176(%rsp)
	leaq	248(%rsp), %rax
	movq	%rax, 184(%rsp)
	leaq	40(%rsp), %rdi
	leaq	24(%rsp), %rsi
	leaq	16(%rsp), %rdx
	leaq	8(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	40(%rsp), %rsi
	movl	48(%rsp), %edx
	movq	24(%rsp), %rcx
	movl	32(%rsp), %r8d
	leaq	96(%rsp), %r9
	movl	$_Z9gemm_corePKhS0_PvS0_S0_iiiiiii, %edi
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$216, %rsp
	.cfi_adjust_cfa_offset -216
	retq
.Lfunc_end0:
	.size	_Z24__device_stub__gemm_corePKhS0_PvS0_S0_iiiiiii, .Lfunc_end0-_Z24__device_stub__gemm_corePKhS0_PvS0_S0_iiiiiii
	.cfi_endproc
                                        # -- End function
	.globl	_Z24__device_stub__reduce_skPKfP12hip_bfloat16ii # -- Begin function _Z24__device_stub__reduce_skPKfP12hip_bfloat16ii
	.p2align	4
	.type	_Z24__device_stub__reduce_skPKfP12hip_bfloat16ii,@function
_Z24__device_stub__reduce_skPKfP12hip_bfloat16ii: # @_Z24__device_stub__reduce_skPKfP12hip_bfloat16ii
	.cfi_startproc
# %bb.0:
	subq	$120, %rsp
	.cfi_def_cfa_offset 128
	movq	%rdi, 72(%rsp)
	movq	%rsi, 64(%rsp)
	movl	%edx, 12(%rsp)
	movl	%ecx, 8(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 88(%rsp)
	leaq	12(%rsp), %rax
	movq	%rax, 96(%rsp)
	leaq	8(%rsp), %rax
	movq	%rax, 104(%rsp)
	leaq	48(%rsp), %rdi
	leaq	32(%rsp), %rsi
	leaq	24(%rsp), %rdx
	leaq	16(%rsp), %rcx
	callq	__hipPopCallConfiguration
	movq	48(%rsp), %rsi
	movl	56(%rsp), %edx
	movq	32(%rsp), %rcx
	movl	40(%rsp), %r8d
	leaq	80(%rsp), %r9
	movl	$_Z9reduce_skPKfP12hip_bfloat16ii, %edi
	pushq	16(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	32(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$136, %rsp
	.cfi_adjust_cfa_offset -136
	retq
.Lfunc_end1:
	.size	_Z24__device_stub__reduce_skPKfP12hip_bfloat16ii, .Lfunc_end1-_Z24__device_stub__reduce_skPKfP12hip_bfloat16ii
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_ctor
	.type	__hip_module_ctor,@function
__hip_module_ctor:                      # @__hip_module_ctor
	.cfi_startproc
# %bb.0:
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	__hip_gpubin_handle_8b457700d3d88645(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB2_2
# %bb.1:
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle_8b457700d3d88645(%rip)
.LBB2_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z9gemm_corePKhS0_PvS0_S0_iiiiiii, %esi
	movl	$.L__unnamed_1, %edx
	movl	$.L__unnamed_1, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z9reduce_skPKfP12hip_bfloat16ii, %esi
	movl	$.L__unnamed_2, %edx
	movl	$.L__unnamed_2, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	movl	$__hip_module_dtor, %edi
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	jmp	atexit                          # TAILCALL
.Lfunc_end2:
	.size	__hip_module_ctor, .Lfunc_end2-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_8b457700d3d88645(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB3_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle_8b457700d3d88645(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB3_2:
	retq
.Lfunc_end3:
	.size	__hip_module_dtor, .Lfunc_end3-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_Z9gemm_corePKhS0_PvS0_S0_iiiiiii,@object # @_Z9gemm_corePKhS0_PvS0_S0_iiiiiii
	.section	.rodata,"a",@progbits
	.globl	_Z9gemm_corePKhS0_PvS0_S0_iiiiiii
	.p2align	3, 0x0
_Z9gemm_corePKhS0_PvS0_S0_iiiiiii:
	.quad	_Z24__device_stub__gemm_corePKhS0_PvS0_S0_iiiiiii
	.size	_Z9gemm_corePKhS0_PvS0_S0_iiiiiii, 8

	.type	_Z9reduce_skPKfP12hip_bfloat16ii,@object # @_Z9reduce_skPKfP12hip_bfloat16ii
	.globl	_Z9reduce_skPKfP12hip_bfloat16ii
	.p2align	3, 0x0
_Z9reduce_skPKfP12hip_bfloat16ii:
	.quad	_Z24__device_stub__reduce_skPKfP12hip_bfloat16ii
	.size	_Z9reduce_skPKfP12hip_bfloat16ii, 8

	.type	.L__unnamed_1,@object           # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_1:
	.asciz	"_Z9gemm_corePKhS0_PvS0_S0_iiiiiii"
	.size	.L__unnamed_1, 34

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"_Z9reduce_skPKfP12hip_bfloat16ii"
	.size	.L__unnamed_2, 33

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_8b457700d3d88645
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_8b457700d3d88645,@object # @__hip_gpubin_handle_8b457700d3d88645
	.local	__hip_gpubin_handle_8b457700d3d88645
	.comm	__hip_gpubin_handle_8b457700d3d88645,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_8b457700d3d88645,@object # @__hip_cuid_8b457700d3d88645
	.bss
	.globl	__hip_cuid_8b457700d3d88645
__hip_cuid_8b457700d3d88645:
	.byte	0                               # 0x0
	.size	__hip_cuid_8b457700d3d88645, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z24__device_stub__gemm_corePKhS0_PvS0_S0_iiiiiii
	.addrsig_sym _Z24__device_stub__reduce_skPKfP12hip_bfloat16ii
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Z9gemm_corePKhS0_PvS0_S0_iiiiiii
	.addrsig_sym _Z9reduce_skPKfP12hip_bfloat16ii
	.addrsig_sym __hip_fatbin_8b457700d3d88645
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_8b457700d3d88645

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-

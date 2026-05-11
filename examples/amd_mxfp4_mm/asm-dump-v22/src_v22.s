
# __CLANG_OFFLOAD_BUNDLE____START__ hip-amdgcn-amd-amdhsa--gfx950
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii ; -- Begin function _Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii
	.globl	_Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii
	.p2align	8
	.type	_Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii,@function
_Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii: ; @_Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii
; %bb.0:
	s_load_dwordx2 s[6:7], s[0:1], 0x10
	s_load_dwordx8 s[8:15], s[0:1], 0x28
	v_and_b32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s15, s2, 5
	s_lshl_b32 s17, s3, 5
	v_or_b32_e32 v2, s17, v1
	s_mul_i32 s5, s11, s4
	s_add_i32 s2, s5, s11
	s_min_i32 s11, s2, s10
	v_lshrrev_b32_e32 v14, 5, v0
	v_cmp_gt_i32_e64 s[2:3], s9, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_mov_b32_e32 v5, 0
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
	s_cmp_ge_i32 s5, s11
	v_accvgpr_write_b32 a15, 0
	s_cbranch_scc1 .LBB0_11
; %bb.1:                                ; %.preheader123
	s_load_dwordx4 s[20:23], s[0:1], 0x0
	s_load_dwordx4 s[24:27], s[0:1], 0x18
	v_lshlrev_b32_e32 v0, 2, v0
	s_ashr_i32 s16, s10, 5
	v_or_b32_e32 v12, s15, v1
	v_lshrrev_b32_e32 v4, 4, v1
	v_and_b32_e32 v10, 60, v0
	s_lshl_b32 s0, s14, 5
	s_ashr_i32 s1, s17, 5
	s_ashr_i32 s10, s10, 1
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[0:1], s[20:21]
	v_mov_b64_e32 v[6:7], s[22:23]
	v_mov_b64_e32 v[8:9], s[24:25]
	s_mul_hi_i32 s14, s0, s1
	s_mul_i32 s17, s0, s1
	v_mad_i64_i32 v[0:1], s[0:1], s10, v12, v[0:1]
	v_mad_i64_i32 v[6:7], s[0:1], s10, v2, v[6:7]
	v_mad_i64_i32 v[8:9], s[0:1], s13, v12, v[8:9]
	s_add_u32 s0, s26, s17
	v_mov_b32_e32 v11, v5
	s_addc_u32 s1, s27, s14
	v_lshl_add_u64 v[10:11], s[0:1], 0, v[10:11]
	v_cmp_gt_i32_e32 vcc, s8, v12
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
	v_lshlrev_b32_e32 v15, 4, v14
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[4:5]
	s_branch .LBB0_3
.LBB0_2:                                ; %_Z14load_frag_gmemPKhS0_S0_S0_iiiillllibbRDv32_hS2_RhS3_.exit
                                        ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[16:19], v[20:23], a[0:15], v13, v4 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_i32 s5, s5, 64
	s_cmp_lt_i32 s5, s11
	s_cbranch_scc0 .LBB0_11
.LBB0_3:                                ; =>This Inner Loop Header: Depth=1
	s_ashr_i32 s0, s5, 1
	v_add_u32_e32 v12, s0, v15
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v17, 0
	v_mov_b32_e32 v18, 0
	v_mov_b32_e32 v20, 0
	v_ashrrev_i32_e32 v13, 31, v12
	v_mov_b32_e32 v19, 0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_5
; %bb.4:                                ; %.loopexit45.i
                                        ;   in Loop: Header=BB0_3 Depth=1
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[12:13]
	global_load_dwordx4 v[16:19], v[16:17], off
.LBB0_5:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v21, 0
	v_mov_b32_e32 v22, 0
	v_mov_b32_e32 v23, 0
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_7
; %bb.6:                                ; %.loopexit.i
                                        ;   in Loop: Header=BB0_3 Depth=1
	v_lshl_add_u64 v[12:13], v[6:7], 0, v[12:13]
	global_load_dwordx4 v[20:23], v[12:13], off
.LBB0_7:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[0:1]
	s_ashr_i32 s0, s5, 5
	v_add_u32_e32 v12, s0, v14
	v_cmp_gt_i32_e64 s[0:1], s13, v12
	s_and_b64 s[18:19], vcc, s[0:1]
	v_mov_b32_e32 v4, 0x7f
	v_mov_b32_e32 v13, 0x7f
	s_and_saveexec_b64 s[0:1], s[18:19]
	s_cbranch_execz .LBB0_9
; %bb.8:                                ;   in Loop: Header=BB0_3 Depth=1
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshl_add_u64 v[24:25], v[8:9], 0, v[12:13]
	global_load_ubyte v13, v[24:25], off
.LBB0_9:                                ;   in Loop: Header=BB0_3 Depth=1
	s_or_b64 exec, exec, s[0:1]
	v_cmp_gt_i32_e64 s[0:1], s16, v12
	s_and_b64 s[18:19], s[2:3], s[0:1]
	s_and_saveexec_b64 s[0:1], s[18:19]
	s_cbranch_execz .LBB0_2
; %bb.10:                               ;   in Loop: Header=BB0_3 Depth=1
	v_lshlrev_b32_e32 v4, 5, v12
	v_and_b32_e32 v24, 0xffffff00, v4
	v_ashrrev_i32_e32 v25, 31, v24
	v_lshlrev_b32_e32 v4, 6, v12
	v_and_b32_e32 v4, 0xc0, v4
	v_lshrrev_b32_e32 v12, 1, v12
	v_lshl_add_u64 v[24:25], v[10:11], 0, v[24:25]
	v_and_b32_e32 v26, 2, v12
	v_mov_b32_e32 v27, v5
	v_lshl_add_u64 v[24:25], v[24:25], 0, v[4:5]
	v_lshl_add_u64 v[24:25], v[24:25], 0, v[26:27]
	global_load_ubyte v4, v[24:25], off
	s_branch .LBB0_2
.LBB0_11:                               ; %.loopexit124
	s_cmp_lg_u32 s12, 1
	s_mov_b64 s[0:1], -1
	s_cbranch_scc0 .LBB0_46
; %bb.12:
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_45
; %bb.13:                               ; %.preheader121
	v_lshl_add_u32 v4, v14, 2, s15
	s_mul_hi_i32 s5, s8, s4
	s_mul_i32 s4, s8, s4
	s_ashr_i32 s12, s9, 31
	v_lshl_add_u64 v[0:1], v[2:3], 2, s[6:7]
	v_cmp_gt_i32_e32 vcc, s8, v4
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_15
; %bb.14:
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[4:5]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a0, off
.LBB0_15:
	s_or_b64 exec, exec, s[10:11]
	v_or_b32_e32 v6, 1, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_17
; %bb.16:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a1, off
.LBB0_17:
	s_or_b64 exec, exec, s[10:11]
	v_or_b32_e32 v6, 2, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_19
; %bb.18:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a2, off
.LBB0_19:
	s_or_b64 exec, exec, s[10:11]
	v_or_b32_e32 v6, 3, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_21
; %bb.20:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a3, off
.LBB0_21:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 8, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_23
; %bb.22:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a4, off
.LBB0_23:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 9, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_25
; %bb.24:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a5, off
.LBB0_25:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 10, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_27
; %bb.26:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a6, off
.LBB0_27:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 11, v4
	v_cmp_gt_i32_e32 vcc, s8, v6
	s_and_saveexec_b64 s[10:11], vcc
	s_cbranch_execz .LBB0_29
; %bb.28:
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[6:7], s[4:5], 0, v[6:7]
	v_mul_lo_u32 v5, v7, s9
	v_mul_lo_u32 v8, v6, s12
	v_mad_u64_u32 v[6:7], s[16:17], v6, s9, 0
	v_add3_u32 v7, v7, v8, v5
	v_lshl_add_u64 v[6:7], v[6:7], 2, v[0:1]
	global_store_dword v[6:7], a7, off
.LBB0_29:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 16, v4
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
	global_store_dword v[6:7], a8, off
.LBB0_31:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 17, v4
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
	global_store_dword v[6:7], a9, off
.LBB0_33:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 18, v4
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
	global_store_dword v[6:7], a10, off
.LBB0_35:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 19, v4
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
	global_store_dword v[6:7], a11, off
.LBB0_37:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 24, v4
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
	global_store_dword v[6:7], a12, off
.LBB0_39:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 25, v4
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
	global_store_dword v[6:7], a13, off
.LBB0_41:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v6, 26, v4
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
	global_store_dword v[6:7], a14, off
.LBB0_43:
	s_or_b64 exec, exec, s[10:11]
	v_add_u32_e32 v4, 27, v4
	v_cmp_gt_i32_e32 vcc, s8, v4
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_45
; %bb.44:
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[4:5], s[4:5], 0, v[4:5]
	v_mul_lo_u32 v6, v5, s9
	v_mul_lo_u32 v7, v4, s12
	v_mad_u64_u32 v[4:5], s[4:5], v4, s9, 0
	v_add3_u32 v5, v5, v7, v6
	v_lshl_add_u64 v[0:1], v[4:5], 2, v[0:1]
	global_store_dword v[0:1], a15, off
.LBB0_45:                               ; %Flow403
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB0_46:                               ; %Flow406
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB0_80
; %bb.47:
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_80
; %bb.48:                               ; %.preheader
	v_lshl_add_u32 v4, v14, 2, s15
	v_lshl_add_u64 v[0:1], v[2:3], 1, s[6:7]
	v_cmp_gt_i32_e32 vcc, s8, v4
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_50
; %bb.49:
	v_mad_i64_i32 v[2:3], s[2:3], v4, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a0, off
.LBB0_50:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 1, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_52
; %bb.51:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a1, off
.LBB0_52:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 2, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_54
; %bb.53:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a2, off
.LBB0_54:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v2, 3, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_56
; %bb.55:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a3, off
.LBB0_56:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 8, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_58
; %bb.57:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a4, off
.LBB0_58:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 9, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_60
; %bb.59:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a5, off
.LBB0_60:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 10, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_62
; %bb.61:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a6, off
.LBB0_62:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 11, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_64
; %bb.63:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a7, off
.LBB0_64:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 16, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_66
; %bb.65:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a8, off
.LBB0_66:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 17, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_68
; %bb.67:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a9, off
.LBB0_68:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 18, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_70
; %bb.69:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a10, off
.LBB0_70:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 19, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_72
; %bb.71:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a11, off
.LBB0_72:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 24, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_74
; %bb.73:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a12, off
.LBB0_74:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 25, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_76
; %bb.75:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a13, off
.LBB0_76:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 26, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB0_78
; %bb.77:
	v_mad_i64_i32 v[2:3], s[2:3], v2, s9, 0
	v_lshl_add_u64 v[2:3], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[2:3], a14, off
.LBB0_78:
	s_or_b64 exec, exec, s[0:1]
	v_add_u32_e32 v2, 27, v4
	v_cmp_gt_i32_e32 vcc, s8, v2
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB0_80
; %bb.79:
	v_mad_i64_i32 v[2:3], s[0:1], v2, s9, 0
	v_lshl_add_u64 v[0:1], v[2:3], 1, v[0:1]
	global_store_short_d16_hi v[0:1], a15, off
.LBB0_80:                               ; %.loopexit
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii
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
		.amdhsa_next_free_vgpr 44
		.amdhsa_next_free_sgpr 28
		.amdhsa_accum_offset 28
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
	.size	_Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii, .Lfunc_end0-_Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii
                                        ; -- End function
	.set _Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii.num_vgpr, 28
	.set _Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii.num_agpr, 16
	.set _Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii.numbered_sgpr, 28
	.set _Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii.num_named_barrier, 0
	.set _Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii.private_seg_size, 0
	.set _Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii.uses_vcc, 1
	.set _Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii.uses_flat_scratch, 0
	.set _Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii.has_dyn_sized_stack, 0
	.set _Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii.has_recursion, 0
	.set _Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 2828
; TotalNumSgprs: 34
; NumVgprs: 28
; NumAgprs: 16
; TotalNumVgprs: 44
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 5
; NumSGPRsForWavesPerEU: 34
; NumVGPRsForWavesPerEU: 44
; AccumOffset: 28
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 6
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	_Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii ; -- Begin function _Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii
	.globl	_Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii
	.p2align	8
	.type	_Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii,@function
_Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii: ; @_Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii
; %bb.0:
	s_load_dwordx2 s[10:11], s[0:1], 0x10
	s_load_dwordx8 s[12:19], s[0:1], 0x28
	s_lshl_b32 s3, s3, 6
	v_lshrrev_b32_e32 v2, 1, v0
	v_and_b32_e32 v1, 31, v0
	v_and_or_b32 v5, v2, 32, s3
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s5, s15, s4
	s_add_i32 s6, s5, s15
	s_lshl_b32 s2, s2, 6
	v_lshrrev_b32_e32 v4, 2, v0
	v_or_b32_e32 v18, v5, v1
	s_min_i32 s20, s6, s14
	v_bfe_u32 v28, v0, 5, 1
	v_and_or_b32 v29, v4, 32, s2
	s_cmp_ge_i32 s5, s20
	v_cmp_gt_i32_e64 s[8:9], s13, v18
	s_cbranch_scc1 .LBB1_26
; %bb.1:                                ; %.lr.ph211
	s_load_dwordx4 s[24:27], s[0:1], 0x0
	s_load_dwordx4 s[28:31], s[0:1], 0x18
	s_ashr_i32 s21, s14, 5
	v_lshlrev_b32_e32 v11, 5, v0
	v_mov_b32_e32 v3, 0
	v_ashrrev_i32_e32 v12, 5, v5
	s_ashr_i32 s14, s14, 1
	v_add_u32_e32 v5, s2, v4
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[8:9], s[24:25]
	v_and_b32_e32 v32, 0x60, v11
	v_mad_i64_i32 v[8:9], s[6:7], s14, v5, v[8:9]
	v_mov_b32_e32 v33, v3
	v_add_u32_e32 v13, s3, v4
	v_lshl_add_u64 v[20:21], v[8:9], 0, v[32:33]
	v_mov_b64_e32 v[8:9], s[26:27]
	v_mad_i64_i32 v[8:9], s[6:7], s14, v13, v[8:9]
	v_or_b32_e32 v10, v29, v1
	v_cmp_gt_i32_e64 s[2:3], s12, v5
	v_lshl_add_u64 v[22:23], v[8:9], 0, v[32:33]
	v_lshlrev_b32_e32 v33, 7, v4
	v_mov_b64_e32 v[4:5], s[28:29]
	v_lshlrev_b32_e32 v6, 2, v0
	s_lshl_b32 s15, s18, 5
	v_mad_i64_i32 v[24:25], s[6:7], s17, v10, v[4:5]
	v_mov_b64_e32 v[4:5], s[30:31]
	v_and_b32_e32 v6, 60, v6
	v_mov_b32_e32 v7, v3
	v_mad_i64_i32 v[4:5], s[6:7], s15, v12, v[4:5]
	v_lshrrev_b32_e32 v2, 4, v1
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[6:7]
	v_lshlrev_b32_e32 v0, 6, v0
	v_lshlrev_b32_e32 v1, 7, v1
	s_movk_i32 s6, 0x1000
	s_movk_i32 s14, 0x2000
	v_lshl_add_u64 v[26:27], v[4:5], 0, v[2:3]
	v_lshlrev_b32_e32 v2, 4, v28
	v_and_or_b32 v0, v0, s6, v1
	v_or3_b32 v30, v0, v2, s14
	v_and_b32_e32 v0, 0x1000, v11
	v_or3_b32 v31, v0, v1, v2
	v_mov_b32_e32 v2, v3
	v_cmp_gt_i32_e32 vcc, s12, v10
	v_cmp_gt_i32_e64 s[0:1], s13, v13
	v_or_b32_e32 v34, 0x2000, v33
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v5, v3
	v_mov_b32_e32 v6, v3
	v_mov_b32_e32 v8, v3
	v_mov_b32_e32 v9, v3
	v_mov_b32_e32 v10, v3
	v_mov_b32_e32 v11, v3
	v_mov_b32_e32 v12, v3
	v_mov_b32_e32 v13, v3
	v_mov_b32_e32 v14, v3
	v_mov_b32_e32 v15, v3
	v_mov_b32_e32 v16, v3
	v_mov_b32_e32 v17, v3
	v_accvgpr_write_b32 a0, v2
	v_add_u32_e32 v19, 32, v32
	v_accvgpr_write_b32 a1, v3
	v_accvgpr_write_b32 a2, v4
	v_accvgpr_write_b32 a3, v5
	v_accvgpr_write_b32 a4, v6
	v_accvgpr_write_b32 a5, v7
	v_accvgpr_write_b32 a6, v8
	v_accvgpr_write_b32 a7, v9
	v_accvgpr_write_b32 a8, v10
	v_accvgpr_write_b32 a9, v11
	v_accvgpr_write_b32 a10, v12
	v_accvgpr_write_b32 a11, v13
	v_accvgpr_write_b32 a12, v14
	v_accvgpr_write_b32 a13, v15
	v_accvgpr_write_b32 a14, v16
	v_accvgpr_write_b32 a15, v17
	v_add_u32_e32 v4, v33, v32
	v_add_u32_e32 v5, v34, v32
	s_branch .LBB1_3
.LBB1_2:                                ; %._crit_edge
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_addk_i32 s5, 0x100
	s_cmp_ge_i32 s5, s20
	s_barrier
	s_cbranch_scc1 .LBB1_27
.LBB1_3:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_25 Depth 2
	s_sub_i32 s23, s20, s5
	s_min_i32 s22, s23, 0x100
	s_lshr_b32 s6, s22, 1
	v_cmp_ge_u32_e64 s[6:7], s6, v19
	s_ashr_i32 s14, s5, 1
	s_and_b64 s[24:25], s[2:3], s[6:7]
	s_ashr_i32 s15, s14, 31
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	s_and_saveexec_b64 s[18:19], s[24:25]
	s_cbranch_execz .LBB1_5
; %bb.4:                                ;   in Loop: Header=BB1_3 Depth=1
	v_lshl_add_u64 v[0:1], v[20:21], 0, s[14:15]
	global_load_dwordx4 v[6:9], v[0:1], off
	global_load_dwordx4 v[10:13], v[0:1], off offset:16
.LBB1_5:                                ; %_Z8copy_32bPhPKhb.exit
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[18:19]
	s_and_b64 s[18:19], s[0:1], s[6:7]
	s_waitcnt vmcnt(1)
	ds_write_b128 v4, v[6:9]
	s_waitcnt vmcnt(0)
	ds_write_b128 v4, v[10:13] offset:16
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v8, 0
	v_mov_b32_e32 v9, 0
	v_mov_b32_e32 v10, 0
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v12, 0
	v_mov_b32_e32 v13, 0
	s_and_saveexec_b64 s[6:7], s[18:19]
	s_cbranch_execz .LBB1_7
; %bb.6:                                ;   in Loop: Header=BB1_3 Depth=1
	v_lshl_add_u64 v[0:1], v[22:23], 0, s[14:15]
	global_load_dwordx4 v[6:9], v[0:1], off
	global_load_dwordx4 v[10:13], v[0:1], off offset:16
.LBB1_7:                                ; %_Z8copy_32bPhPKhb.exit197
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_ashr_i32 s6, s5, 5
	v_add_u32_e32 v0, s6, v28
	v_cmp_gt_i32_e64 s[6:7], s17, v0
	s_waitcnt vmcnt(1)
	ds_write_b128 v5, v[6:9]
	s_waitcnt vmcnt(0)
	ds_write_b128 v5, v[10:13] offset:16
	s_and_b64 s[14:15], vcc, s[6:7]
	v_mov_b32_e32 v7, 0x7f
	v_mov_b32_e32 v6, 0x7f
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_and_saveexec_b64 s[6:7], s[14:15]
	s_cbranch_execz .LBB1_9
; %bb.8:                                ;   in Loop: Header=BB1_3 Depth=1
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[8:9], v[24:25], 0, v[0:1]
	global_load_ubyte v6, v[8:9], off
.LBB1_9:                                ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_cmp_gt_i32_e64 s[6:7], s21, v0
	s_and_b64 s[14:15], s[8:9], s[6:7]
	s_and_saveexec_b64 s[6:7], s[14:15]
	s_cbranch_execz .LBB1_11
; %bb.10:                               ;   in Loop: Header=BB1_3 Depth=1
	v_lshlrev_b32_e32 v1, 5, v0
	v_and_b32_e32 v8, 0xffffff00, v1
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshlrev_b32_e32 v1, 6, v0
	v_and_b32_e32 v2, 0xc0, v1
	v_lshrrev_b32_e32 v1, 1, v0
	v_lshl_add_u64 v[8:9], v[26:27], 0, v[8:9]
	v_and_b32_e32 v10, 2, v1
	v_mov_b32_e32 v11, v3
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[10:11]
	global_load_ubyte v7, v[8:9], off
.LBB1_11:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_add_u32_e32 v10, 2, v0
	v_cmp_gt_i32_e64 s[6:7], s17, v10
	s_and_b64 s[14:15], vcc, s[6:7]
	v_mov_b32_e32 v9, 0x7f00
	v_mov_b32_e32 v8, 0x7f00
	s_and_saveexec_b64 s[6:7], s[14:15]
	s_cbranch_execz .LBB1_13
; %bb.12:                               ;   in Loop: Header=BB1_3 Depth=1
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[12:13], v[24:25], 0, v[0:1]
	global_load_ubyte v1, v[12:13], off offset:2
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v8, 8, v1
.LBB1_13:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_cmp_gt_i32_e64 s[6:7], s21, v10
	s_and_b64 s[14:15], s[8:9], s[6:7]
	s_and_saveexec_b64 s[6:7], s[14:15]
	s_cbranch_execz .LBB1_15
; %bb.14:                               ;   in Loop: Header=BB1_3 Depth=1
	v_lshlrev_b32_e32 v1, 5, v10
	v_and_b32_e32 v12, 0xffffff00, v1
	v_ashrrev_i32_e32 v13, 31, v12
	v_lshlrev_b32_e32 v1, 6, v10
	v_and_b32_e32 v2, 0xc0, v1
	v_lshrrev_b32_e32 v1, 1, v10
	v_lshl_add_u64 v[12:13], v[26:27], 0, v[12:13]
	v_and_b32_e32 v10, 2, v1
	v_mov_b32_e32 v11, v3
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[2:3]
	v_lshl_add_u64 v[10:11], v[12:13], 0, v[10:11]
	global_load_ubyte v1, v[10:11], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v9, 8, v1
.LBB1_15:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_add_u32_e32 v12, 4, v0
	v_cmp_gt_i32_e64 s[6:7], s17, v12
	s_and_b64 s[14:15], vcc, s[6:7]
	v_mov_b32_e32 v11, 0x7f0000
	v_mov_b32_e32 v10, 0x7f0000
	s_and_saveexec_b64 s[6:7], s[14:15]
	s_cbranch_execz .LBB1_17
; %bb.16:                               ;   in Loop: Header=BB1_3 Depth=1
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[14:15], v[24:25], 0, v[0:1]
	global_load_ubyte v1, v[14:15], off offset:4
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v10, 16, v1
.LBB1_17:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_cmp_gt_i32_e64 s[6:7], s21, v12
	s_and_b64 s[14:15], s[8:9], s[6:7]
	s_and_saveexec_b64 s[6:7], s[14:15]
	s_cbranch_execz .LBB1_19
; %bb.18:                               ;   in Loop: Header=BB1_3 Depth=1
	v_lshlrev_b32_e32 v1, 5, v12
	v_and_b32_e32 v14, 0xffffff00, v1
	v_ashrrev_i32_e32 v15, 31, v14
	v_lshlrev_b32_e32 v1, 6, v0
	v_and_b32_e32 v2, 0xc0, v1
	v_lshrrev_b32_e32 v1, 1, v12
	v_lshl_add_u64 v[14:15], v[26:27], 0, v[14:15]
	v_and_b32_e32 v12, 2, v1
	v_mov_b32_e32 v13, v3
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[2:3]
	v_lshl_add_u64 v[12:13], v[14:15], 0, v[12:13]
	global_load_ubyte v1, v[12:13], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v11, 16, v1
.LBB1_19:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_add_u32_e32 v12, 6, v0
	v_cmp_gt_i32_e64 s[6:7], s17, v12
	s_and_b64 s[14:15], vcc, s[6:7]
	v_mov_b32_e32 v2, 0x7f000000
	v_mov_b32_e32 v1, 0x7f000000
	s_and_saveexec_b64 s[6:7], s[14:15]
	s_cbranch_execz .LBB1_21
; %bb.20:                               ;   in Loop: Header=BB1_3 Depth=1
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[24:25], 0, v[0:1]
	global_load_ubyte v0, v[0:1], off offset:6
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v1, 24, v0
.LBB1_21:                               ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_cmp_gt_i32_e64 s[6:7], s21, v12
	s_and_b64 s[14:15], s[8:9], s[6:7]
	s_and_saveexec_b64 s[6:7], s[14:15]
	s_cbranch_execz .LBB1_23
; %bb.22:                               ;   in Loop: Header=BB1_3 Depth=1
	v_lshlrev_b32_e32 v0, 5, v12
	v_and_b32_e32 v14, 0xffffff00, v0
	v_ashrrev_i32_e32 v15, 31, v14
	v_lshlrev_b32_e32 v0, 6, v12
	v_and_b32_e32 v2, 0xc0, v0
	v_lshrrev_b32_e32 v0, 1, v12
	v_lshl_add_u64 v[14:15], v[26:27], 0, v[14:15]
	v_and_b32_e32 v12, 2, v0
	v_mov_b32_e32 v13, v3
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[2:3]
	v_lshl_add_u64 v[12:13], v[14:15], 0, v[12:13]
	global_load_ubyte v0, v[12:13], off
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 24, v0
.LBB1_23:                               ; %.preheader202
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_or_b64 exec, exec, s[6:7]
	s_cmp_lt_i32 s23, 1
	s_cbranch_scc1 .LBB1_2
; %bb.24:                               ; %.lr.ph.preheader
                                        ;   in Loop: Header=BB1_3 Depth=1
	s_waitcnt vmcnt(0)
	v_or_b32_e32 v0, v8, v6
	v_or_b32_e32 v6, v9, v7
	v_or3_b32 v0, v10, v0, v1
	v_or3_b32 v1, v11, v6, v2
	s_mov_b32 s6, 0
	v_mov_b32_e32 v2, v31
	v_mov_b32_e32 v6, v30
	s_mov_b32 s7, 0
.LBB1_25:                               ; %.lr.ph
                                        ;   Parent Loop BB1_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ds_read_b128 v[8:11], v2
	ds_read_b128 v[12:15], v6
	v_bfe_u32 v7, v0, s6, 8
	v_bfe_u32 v16, v1, s6, 8
	s_add_i32 s7, s7, 64
	s_add_i32 s6, s6, 8
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[8:11], v[12:15], a[0:15], v7, v16 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v6, 32, v6
	s_cmp_ge_i32 s7, s22
	v_add_u32_e32 v2, 32, v2
	s_cbranch_scc0 .LBB1_25
	s_branch .LBB1_2
.LBB1_26:
	v_accvgpr_write_b32 a0, 0
	v_accvgpr_mov_b32 a1, a0
	v_accvgpr_mov_b32 a2, a0
	v_accvgpr_mov_b32 a3, a0
	v_accvgpr_mov_b32 a4, a0
	v_accvgpr_mov_b32 a5, a0
	v_accvgpr_mov_b32 a6, a0
	v_accvgpr_mov_b32 a7, a0
	v_accvgpr_mov_b32 a8, a0
	v_accvgpr_mov_b32 a9, a0
	v_accvgpr_mov_b32 a10, a0
	v_accvgpr_mov_b32 a11, a0
	v_accvgpr_mov_b32 a12, a0
	v_accvgpr_mov_b32 a13, a0
	v_accvgpr_mov_b32 a14, a0
	v_accvgpr_mov_b32 a15, a0
.LBB1_27:                               ; %Flow413
	s_waitcnt vmcnt(0)
	s_nop 1
	v_accvgpr_read_b32 v0, a0
	v_accvgpr_read_b32 v1, a1
	v_accvgpr_read_b32 v2, a2
	v_accvgpr_read_b32 v3, a3
	v_accvgpr_read_b32 v4, a4
	v_accvgpr_read_b32 v5, a5
	v_accvgpr_read_b32 v6, a6
	v_accvgpr_read_b32 v7, a7
	v_accvgpr_read_b32 v8, a8
	v_accvgpr_read_b32 v9, a9
	v_accvgpr_read_b32 v10, a10
	v_accvgpr_read_b32 v11, a11
	v_accvgpr_read_b32 v12, a12
	v_accvgpr_read_b32 v13, a13
	v_accvgpr_read_b32 v14, a14
	v_accvgpr_read_b32 v15, a15
	s_cmp_lg_u32 s16, 1
	s_mov_b64 s[0:1], -1
	s_cbranch_scc0 .LBB1_62
; %bb.28:
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB1_61
; %bb.29:                               ; %.preheader200
	v_lshl_or_b32 v20, v28, 2, v29
	v_ashrrev_i32_e32 v19, 31, v18
	s_mul_hi_i32 s3, s12, s4
	s_mul_i32 s2, s12, s4
	s_ashr_i32 s6, s13, 31
	v_lshl_add_u64 v[16:17], v[18:19], 2, s[10:11]
	v_cmp_gt_i32_e32 vcc, s12, v20
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_31
; %bb.30:
	v_ashrrev_i32_e32 v21, 31, v20
	v_lshl_add_u64 v[22:23], s[2:3], 0, v[20:21]
	v_mul_lo_u32 v19, v23, s13
	v_mul_lo_u32 v21, v22, s6
	v_mad_u64_u32 v[22:23], s[14:15], v22, s13, 0
	v_add3_u32 v23, v23, v21, v19
	v_lshl_add_u64 v[22:23], v[22:23], 2, v[16:17]
	global_store_dword v[22:23], v0, off
.LBB1_31:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v22, 1, v20
	v_cmp_gt_i32_e32 vcc, s12, v22
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_33
; %bb.32:
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], s[2:3], 0, v[22:23]
	v_mul_lo_u32 v19, v23, s13
	v_mul_lo_u32 v21, v22, s6
	v_mad_u64_u32 v[22:23], s[14:15], v22, s13, 0
	v_add3_u32 v23, v23, v21, v19
	v_lshl_add_u64 v[22:23], v[22:23], 2, v[16:17]
	global_store_dword v[22:23], v1, off
.LBB1_33:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v22, 2, v20
	v_cmp_gt_i32_e32 vcc, s12, v22
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_35
; %bb.34:
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], s[2:3], 0, v[22:23]
	v_mul_lo_u32 v19, v23, s13
	v_mul_lo_u32 v21, v22, s6
	v_mad_u64_u32 v[22:23], s[14:15], v22, s13, 0
	v_add3_u32 v23, v23, v21, v19
	v_lshl_add_u64 v[22:23], v[22:23], 2, v[16:17]
	global_store_dword v[22:23], v2, off
.LBB1_35:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v22, 3, v20
	v_cmp_gt_i32_e32 vcc, s12, v22
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_37
; %bb.36:
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], s[2:3], 0, v[22:23]
	v_mul_lo_u32 v19, v23, s13
	v_mul_lo_u32 v21, v22, s6
	v_mad_u64_u32 v[22:23], s[14:15], v22, s13, 0
	v_add3_u32 v23, v23, v21, v19
	v_lshl_add_u64 v[22:23], v[22:23], 2, v[16:17]
	global_store_dword v[22:23], v3, off
.LBB1_37:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v22, 8, v20
	v_cmp_gt_i32_e32 vcc, s12, v22
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_39
; %bb.38:
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], s[2:3], 0, v[22:23]
	v_mul_lo_u32 v19, v23, s13
	v_mul_lo_u32 v21, v22, s6
	v_mad_u64_u32 v[22:23], s[14:15], v22, s13, 0
	v_add3_u32 v23, v23, v21, v19
	v_lshl_add_u64 v[22:23], v[22:23], 2, v[16:17]
	global_store_dword v[22:23], v4, off
.LBB1_39:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v22, 9, v20
	v_cmp_gt_i32_e32 vcc, s12, v22
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_41
; %bb.40:
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], s[2:3], 0, v[22:23]
	v_mul_lo_u32 v19, v23, s13
	v_mul_lo_u32 v21, v22, s6
	v_mad_u64_u32 v[22:23], s[14:15], v22, s13, 0
	v_add3_u32 v23, v23, v21, v19
	v_lshl_add_u64 v[22:23], v[22:23], 2, v[16:17]
	global_store_dword v[22:23], v5, off
.LBB1_41:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v22, 10, v20
	v_cmp_gt_i32_e32 vcc, s12, v22
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_43
; %bb.42:
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], s[2:3], 0, v[22:23]
	v_mul_lo_u32 v19, v23, s13
	v_mul_lo_u32 v21, v22, s6
	v_mad_u64_u32 v[22:23], s[14:15], v22, s13, 0
	v_add3_u32 v23, v23, v21, v19
	v_lshl_add_u64 v[22:23], v[22:23], 2, v[16:17]
	global_store_dword v[22:23], v6, off
.LBB1_43:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v22, 11, v20
	v_cmp_gt_i32_e32 vcc, s12, v22
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_45
; %bb.44:
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], s[2:3], 0, v[22:23]
	v_mul_lo_u32 v19, v23, s13
	v_mul_lo_u32 v21, v22, s6
	v_mad_u64_u32 v[22:23], s[14:15], v22, s13, 0
	v_add3_u32 v23, v23, v21, v19
	v_lshl_add_u64 v[22:23], v[22:23], 2, v[16:17]
	global_store_dword v[22:23], v7, off
.LBB1_45:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v22, 16, v20
	v_cmp_gt_i32_e32 vcc, s12, v22
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_47
; %bb.46:
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], s[2:3], 0, v[22:23]
	v_mul_lo_u32 v19, v23, s13
	v_mul_lo_u32 v21, v22, s6
	v_mad_u64_u32 v[22:23], s[14:15], v22, s13, 0
	v_add3_u32 v23, v23, v21, v19
	v_lshl_add_u64 v[22:23], v[22:23], 2, v[16:17]
	global_store_dword v[22:23], v8, off
.LBB1_47:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v22, 17, v20
	v_cmp_gt_i32_e32 vcc, s12, v22
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_49
; %bb.48:
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], s[2:3], 0, v[22:23]
	v_mul_lo_u32 v19, v23, s13
	v_mul_lo_u32 v21, v22, s6
	v_mad_u64_u32 v[22:23], s[14:15], v22, s13, 0
	v_add3_u32 v23, v23, v21, v19
	v_lshl_add_u64 v[22:23], v[22:23], 2, v[16:17]
	global_store_dword v[22:23], v9, off
.LBB1_49:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v22, 18, v20
	v_cmp_gt_i32_e32 vcc, s12, v22
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_51
; %bb.50:
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], s[2:3], 0, v[22:23]
	v_mul_lo_u32 v19, v23, s13
	v_mul_lo_u32 v21, v22, s6
	v_mad_u64_u32 v[22:23], s[14:15], v22, s13, 0
	v_add3_u32 v23, v23, v21, v19
	v_lshl_add_u64 v[22:23], v[22:23], 2, v[16:17]
	global_store_dword v[22:23], v10, off
.LBB1_51:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v22, 19, v20
	v_cmp_gt_i32_e32 vcc, s12, v22
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_53
; %bb.52:
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], s[2:3], 0, v[22:23]
	v_mul_lo_u32 v19, v23, s13
	v_mul_lo_u32 v21, v22, s6
	v_mad_u64_u32 v[22:23], s[14:15], v22, s13, 0
	v_add3_u32 v23, v23, v21, v19
	v_lshl_add_u64 v[22:23], v[22:23], 2, v[16:17]
	global_store_dword v[22:23], v11, off
.LBB1_53:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v22, 24, v20
	v_cmp_gt_i32_e32 vcc, s12, v22
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_55
; %bb.54:
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], s[2:3], 0, v[22:23]
	v_mul_lo_u32 v19, v23, s13
	v_mul_lo_u32 v21, v22, s6
	v_mad_u64_u32 v[22:23], s[14:15], v22, s13, 0
	v_add3_u32 v23, v23, v21, v19
	v_lshl_add_u64 v[22:23], v[22:23], 2, v[16:17]
	global_store_dword v[22:23], v12, off
.LBB1_55:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v22, 25, v20
	v_cmp_gt_i32_e32 vcc, s12, v22
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_57
; %bb.56:
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], s[2:3], 0, v[22:23]
	v_mul_lo_u32 v19, v23, s13
	v_mul_lo_u32 v21, v22, s6
	v_mad_u64_u32 v[22:23], s[14:15], v22, s13, 0
	v_add3_u32 v23, v23, v21, v19
	v_lshl_add_u64 v[22:23], v[22:23], 2, v[16:17]
	global_store_dword v[22:23], v13, off
.LBB1_57:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v22, 26, v20
	v_cmp_gt_i32_e32 vcc, s12, v22
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB1_59
; %bb.58:
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshl_add_u64 v[22:23], s[2:3], 0, v[22:23]
	v_mul_lo_u32 v19, v23, s13
	v_mul_lo_u32 v21, v22, s6
	v_mad_u64_u32 v[22:23], s[14:15], v22, s13, 0
	v_add3_u32 v23, v23, v21, v19
	v_lshl_add_u64 v[22:23], v[22:23], 2, v[16:17]
	global_store_dword v[22:23], v14, off
.LBB1_59:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v20, 27, v20
	v_cmp_gt_i32_e32 vcc, s12, v20
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_61
; %bb.60:
	v_ashrrev_i32_e32 v21, 31, v20
	v_lshl_add_u64 v[20:21], s[2:3], 0, v[20:21]
	v_mul_lo_u32 v19, v21, s13
	v_mul_lo_u32 v22, v20, s6
	v_mad_u64_u32 v[20:21], s[2:3], v20, s13, 0
	v_add3_u32 v21, v21, v22, v19
	v_lshl_add_u64 v[16:17], v[20:21], 2, v[16:17]
	global_store_dword v[16:17], v15, off
.LBB1_61:                               ; %Flow406
	s_or_b64 exec, exec, s[0:1]
	s_mov_b64 s[0:1], 0
.LBB1_62:                               ; %Flow409
	s_andn2_b64 vcc, exec, s[0:1]
	s_cbranch_vccnz .LBB1_96
; %bb.63:
	s_and_saveexec_b64 s[0:1], s[8:9]
	s_cbranch_execz .LBB1_96
; %bb.64:                               ; %.preheader
	v_lshl_or_b32 v20, v28, 2, v29
	v_ashrrev_i32_e32 v19, 31, v18
	v_lshl_add_u64 v[16:17], v[18:19], 1, s[10:11]
	v_cmp_gt_i32_e32 vcc, s12, v20
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_66
; %bb.65:
	v_mad_i64_i32 v[18:19], s[2:3], v20, s13, 0
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[16:17]
	global_store_short_d16_hi v[18:19], v0, off
.LBB1_66:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 1, v20
	v_cmp_gt_i32_e32 vcc, s12, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_68
; %bb.67:
	v_mad_i64_i32 v[18:19], s[2:3], v0, s13, 0
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[16:17]
	global_store_short_d16_hi v[18:19], v1, off
.LBB1_68:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 2, v20
	v_cmp_gt_i32_e32 vcc, s12, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_70
; %bb.69:
	v_mad_i64_i32 v[0:1], s[2:3], v0, s13, 0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[16:17]
	global_store_short_d16_hi v[0:1], v2, off
.LBB1_70:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 3, v20
	v_cmp_gt_i32_e32 vcc, s12, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_72
; %bb.71:
	v_mad_i64_i32 v[0:1], s[2:3], v0, s13, 0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[16:17]
	global_store_short_d16_hi v[0:1], v3, off
.LBB1_72:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 8, v20
	v_cmp_gt_i32_e32 vcc, s12, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_74
; %bb.73:
	v_mad_i64_i32 v[0:1], s[2:3], v0, s13, 0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[16:17]
	global_store_short_d16_hi v[0:1], v4, off
.LBB1_74:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 9, v20
	v_cmp_gt_i32_e32 vcc, s12, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_76
; %bb.75:
	v_mad_i64_i32 v[0:1], s[2:3], v0, s13, 0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[16:17]
	global_store_short_d16_hi v[0:1], v5, off
.LBB1_76:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 10, v20
	v_cmp_gt_i32_e32 vcc, s12, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_78
; %bb.77:
	v_mad_i64_i32 v[0:1], s[2:3], v0, s13, 0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[16:17]
	global_store_short_d16_hi v[0:1], v6, off
.LBB1_78:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 11, v20
	v_cmp_gt_i32_e32 vcc, s12, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_80
; %bb.79:
	v_mad_i64_i32 v[0:1], s[2:3], v0, s13, 0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[16:17]
	global_store_short_d16_hi v[0:1], v7, off
.LBB1_80:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 16, v20
	v_cmp_gt_i32_e32 vcc, s12, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_82
; %bb.81:
	v_mad_i64_i32 v[0:1], s[2:3], v0, s13, 0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[16:17]
	global_store_short_d16_hi v[0:1], v8, off
.LBB1_82:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 17, v20
	v_cmp_gt_i32_e32 vcc, s12, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_84
; %bb.83:
	v_mad_i64_i32 v[0:1], s[2:3], v0, s13, 0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[16:17]
	global_store_short_d16_hi v[0:1], v9, off
.LBB1_84:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 18, v20
	v_cmp_gt_i32_e32 vcc, s12, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_86
; %bb.85:
	v_mad_i64_i32 v[0:1], s[2:3], v0, s13, 0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[16:17]
	global_store_short_d16_hi v[0:1], v10, off
.LBB1_86:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 19, v20
	v_cmp_gt_i32_e32 vcc, s12, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_88
; %bb.87:
	v_mad_i64_i32 v[0:1], s[2:3], v0, s13, 0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[16:17]
	global_store_short_d16_hi v[0:1], v11, off
.LBB1_88:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 24, v20
	v_cmp_gt_i32_e32 vcc, s12, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_90
; %bb.89:
	v_mad_i64_i32 v[0:1], s[2:3], v0, s13, 0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[16:17]
	global_store_short_d16_hi v[0:1], v12, off
.LBB1_90:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 25, v20
	v_cmp_gt_i32_e32 vcc, s12, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_92
; %bb.91:
	v_mad_i64_i32 v[0:1], s[2:3], v0, s13, 0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[16:17]
	global_store_short_d16_hi v[0:1], v13, off
.LBB1_92:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 26, v20
	v_cmp_gt_i32_e32 vcc, s12, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_cbranch_execz .LBB1_94
; %bb.93:
	v_mad_i64_i32 v[0:1], s[2:3], v0, s13, 0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[16:17]
	global_store_short_d16_hi v[0:1], v14, off
.LBB1_94:
	s_or_b64 exec, exec, s[0:1]
	v_or_b32_e32 v0, 27, v20
	v_cmp_gt_i32_e32 vcc, s12, v0
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_96
; %bb.95:
	v_mad_i64_i32 v[0:1], s[0:1], v0, s13, 0
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[16:17]
	global_store_short_d16_hi v[0:1], v15, off
.LBB1_96:                               ; %.loopexit
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii
		.amdhsa_group_segment_fixed_size 16384
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
		.amdhsa_next_free_vgpr 52
		.amdhsa_next_free_sgpr 32
		.amdhsa_accum_offset 36
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
	.size	_Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii, .Lfunc_end1-_Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii
                                        ; -- End function
	.set _Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii.num_vgpr, 35
	.set _Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii.num_agpr, 16
	.set _Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii.numbered_sgpr, 32
	.set _Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii.num_named_barrier, 0
	.set _Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii.private_seg_size, 0
	.set _Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii.uses_vcc, 1
	.set _Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii.uses_flat_scratch, 0
	.set _Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii.has_dyn_sized_stack, 0
	.set _Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii.has_recursion, 0
	.set _Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3892
; TotalNumSgprs: 38
; NumVgprs: 35
; NumAgprs: 16
; TotalNumVgprs: 52
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 16384 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 6
; NumSGPRsForWavesPerEU: 38
; NumVGPRsForWavesPerEU: 52
; AccumOffset: 36
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 8
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
	s_cbranch_execz .LBB2_8
; %bb.1:                                ; %.preheader
	s_cmp_gt_i32 s5, 0
	v_ashrrev_i32_e32 v1, 31, v0
	s_cbranch_scc1 .LBB2_3
; %bb.2:                                ; %.preheader.._crit_edge_crit_edge
	s_load_dwordx2 s[2:3], s[0:1], 0x8
	v_mov_b32_e32 v2, 0
	s_cbranch_execz .LBB2_4
	s_branch .LBB2_7
.LBB2_3:
	s_load_dwordx2 s[2:3], s[0:1], 0x8
	v_mov_b32_e32 v2, 0
.LBB2_4:                                ; %.lr.ph
	s_load_dwordx2 s[6:7], s[0:1], 0x0
	s_ashr_i32 s1, s4, 31
	s_mov_b32 s0, s4
	s_lshl_b64 s[0:1], s[0:1], 2
	v_mov_b32_e32 v4, 0
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[0:1], 2, s[6:7]
.LBB2_5:                                ; =>This Inner Loop Header: Depth=1
	global_load_dword v5, v[2:3], off
	s_add_i32 s5, s5, -1
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[0:1]
	s_cmp_eq_u32 s5, 0
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v4, v4, v5
	s_cbranch_scc0 .LBB2_5
; %bb.6:                                ; %._crit_edge.loopexit
	v_lshrrev_b32_e32 v2, 16, v4
.LBB2_7:                                ; %Flow26
	s_waitcnt lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], 1, s[2:3]
	global_store_short v[0:1], v2, off
.LBB2_8:
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
.Lfunc_end2:
	.size	_Z9reduce_skPKfP12hip_bfloat16ii, .Lfunc_end2-_Z9reduce_skPKfP12hip_bfloat16ii
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
	.type	__hip_cuid_85127a709332f6f1,@object ; @__hip_cuid_85127a709332f6f1
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_85127a709332f6f1
__hip_cuid_85127a709332f6f1:
	.byte	0                               ; 0x0
	.size	__hip_cuid_85127a709332f6f1, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_85127a709332f6f1
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
    .name:           _Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii
    .private_segment_fixed_size: 0
    .sgpr_count:     34
    .sgpr_spill_count: 0
    .symbol:         _Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     44
    .vgpr_spill_count: 0
    .wavefront_size: 64
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
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 68
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii
    .private_segment_fixed_size: 0
    .sgpr_count:     38
    .sgpr_spill_count: 0
    .symbol:         _Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     52
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
	.file	"src_v22.hip"
	.text
	.globl	_Z29__device_stub__gemm_core_gmemPKhS0_PvS0_S0_iiiiiii # -- Begin function _Z29__device_stub__gemm_core_gmemPKhS0_PvS0_S0_iiiiiii
	.p2align	4
	.type	_Z29__device_stub__gemm_core_gmemPKhS0_PvS0_S0_iiiiiii,@function
_Z29__device_stub__gemm_core_gmemPKhS0_PvS0_S0_iiiiiii: # @_Z29__device_stub__gemm_core_gmemPKhS0_PvS0_S0_iiiiiii
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
	movl	$_Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii, %edi
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$216, %rsp
	.cfi_adjust_cfa_offset -216
	retq
.Lfunc_end0:
	.size	_Z29__device_stub__gemm_core_gmemPKhS0_PvS0_S0_iiiiiii, .Lfunc_end0-_Z29__device_stub__gemm_core_gmemPKhS0_PvS0_S0_iiiiiii
	.cfi_endproc
                                        # -- End function
	.globl	_Z28__device_stub__gemm_core_ldsPKhS0_PvS0_S0_iiiiiii # -- Begin function _Z28__device_stub__gemm_core_ldsPKhS0_PvS0_S0_iiiiiii
	.p2align	4
	.type	_Z28__device_stub__gemm_core_ldsPKhS0_PvS0_S0_iiiiiii,@function
_Z28__device_stub__gemm_core_ldsPKhS0_PvS0_S0_iiiiiii: # @_Z28__device_stub__gemm_core_ldsPKhS0_PvS0_S0_iiiiiii
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
	movl	$_Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii, %edi
	pushq	8(%rsp)
	.cfi_adjust_cfa_offset 8
	pushq	24(%rsp)
	.cfi_adjust_cfa_offset 8
	callq	hipLaunchKernel
	addq	$216, %rsp
	.cfi_adjust_cfa_offset -216
	retq
.Lfunc_end1:
	.size	_Z28__device_stub__gemm_core_ldsPKhS0_PvS0_S0_iiiiiii, .Lfunc_end1-_Z28__device_stub__gemm_core_ldsPKhS0_PvS0_S0_iiiiiii
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
.Lfunc_end2:
	.size	_Z24__device_stub__reduce_skPKfP12hip_bfloat16ii, .Lfunc_end2-_Z24__device_stub__reduce_skPKfP12hip_bfloat16ii
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
	movq	__hip_gpubin_handle_85127a709332f6f1(%rip), %rbx
	testq	%rbx, %rbx
	jne	.LBB3_2
# %bb.1:
	movl	$__hip_fatbin_wrapper, %edi
	callq	__hipRegisterFatBinary
	movq	%rax, %rbx
	movq	%rax, __hip_gpubin_handle_85127a709332f6f1(%rip)
.LBB3_2:
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii, %esi
	movl	$.L__unnamed_1, %edx
	movl	$.L__unnamed_1, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii, %esi
	movl	$.L__unnamed_2, %edx
	movl	$.L__unnamed_2, %ecx
	movq	%rbx, %rdi
	movl	$-1, %r8d
	xorl	%r9d, %r9d
	callq	__hipRegisterFunction
	xorps	%xmm0, %xmm0
	movups	%xmm0, 16(%rsp)
	movups	%xmm0, (%rsp)
	movl	$_Z9reduce_skPKfP12hip_bfloat16ii, %esi
	movl	$.L__unnamed_3, %edx
	movl	$.L__unnamed_3, %ecx
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
.Lfunc_end3:
	.size	__hip_module_ctor, .Lfunc_end3-__hip_module_ctor
	.cfi_endproc
                                        # -- End function
	.p2align	4                               # -- Begin function __hip_module_dtor
	.type	__hip_module_dtor,@function
__hip_module_dtor:                      # @__hip_module_dtor
	.cfi_startproc
# %bb.0:
	movq	__hip_gpubin_handle_85127a709332f6f1(%rip), %rdi
	testq	%rdi, %rdi
	je	.LBB4_2
# %bb.1:
	pushq	%rax
	.cfi_def_cfa_offset 16
	callq	__hipUnregisterFatBinary
	movq	$0, __hip_gpubin_handle_85127a709332f6f1(%rip)
	addq	$8, %rsp
	.cfi_def_cfa_offset 8
.LBB4_2:
	retq
.Lfunc_end4:
	.size	__hip_module_dtor, .Lfunc_end4-__hip_module_dtor
	.cfi_endproc
                                        # -- End function
	.type	_Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii,@object # @_Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii
	.section	.rodata,"a",@progbits
	.globl	_Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii
	.p2align	3, 0x0
_Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii:
	.quad	_Z29__device_stub__gemm_core_gmemPKhS0_PvS0_S0_iiiiiii
	.size	_Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii, 8

	.type	_Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii,@object # @_Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii
	.globl	_Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii
	.p2align	3, 0x0
_Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii:
	.quad	_Z28__device_stub__gemm_core_ldsPKhS0_PvS0_S0_iiiiiii
	.size	_Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii, 8

	.type	_Z9reduce_skPKfP12hip_bfloat16ii,@object # @_Z9reduce_skPKfP12hip_bfloat16ii
	.globl	_Z9reduce_skPKfP12hip_bfloat16ii
	.p2align	3, 0x0
_Z9reduce_skPKfP12hip_bfloat16ii:
	.quad	_Z24__device_stub__reduce_skPKfP12hip_bfloat16ii
	.size	_Z9reduce_skPKfP12hip_bfloat16ii, 8

	.type	.L__unnamed_1,@object           # @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_1:
	.asciz	"_Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii"
	.size	.L__unnamed_1, 40

	.type	.L__unnamed_2,@object           # @1
.L__unnamed_2:
	.asciz	"_Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii"
	.size	.L__unnamed_2, 39

	.type	.L__unnamed_3,@object           # @2
.L__unnamed_3:
	.asciz	"_Z9reduce_skPKfP12hip_bfloat16ii"
	.size	.L__unnamed_3, 33

	.type	__hip_fatbin_wrapper,@object    # @__hip_fatbin_wrapper
	.section	.hipFatBinSegment,"a",@progbits
	.p2align	3, 0x0
__hip_fatbin_wrapper:
	.long	1212764230                      # 0x48495046
	.long	1                               # 0x1
	.quad	__hip_fatbin_85127a709332f6f1
	.quad	0
	.size	__hip_fatbin_wrapper, 24

	.type	__hip_gpubin_handle_85127a709332f6f1,@object # @__hip_gpubin_handle_85127a709332f6f1
	.local	__hip_gpubin_handle_85127a709332f6f1
	.comm	__hip_gpubin_handle_85127a709332f6f1,8,8
	.section	.init_array,"aw",@init_array
	.p2align	3, 0x0
	.quad	__hip_module_ctor
	.type	__hip_cuid_85127a709332f6f1,@object # @__hip_cuid_85127a709332f6f1
	.bss
	.globl	__hip_cuid_85127a709332f6f1
__hip_cuid_85127a709332f6f1:
	.byte	0                               # 0x0
	.size	__hip_cuid_85127a709332f6f1, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym _Z29__device_stub__gemm_core_gmemPKhS0_PvS0_S0_iiiiiii
	.addrsig_sym _Z28__device_stub__gemm_core_ldsPKhS0_PvS0_S0_iiiiiii
	.addrsig_sym _Z24__device_stub__reduce_skPKfP12hip_bfloat16ii
	.addrsig_sym __hip_module_ctor
	.addrsig_sym __hip_module_dtor
	.addrsig_sym _Z14gemm_core_gmemPKhS0_PvS0_S0_iiiiiii
	.addrsig_sym _Z13gemm_core_ldsPKhS0_PvS0_S0_iiiiiii
	.addrsig_sym _Z9reduce_skPKfP12hip_bfloat16ii
	.addrsig_sym __hip_fatbin_85127a709332f6f1
	.addrsig_sym __hip_fatbin_wrapper
	.addrsig_sym __hip_cuid_85127a709332f6f1

# __CLANG_OFFLOAD_BUNDLE____END__ host-x86_64-unknown-linux-gnu-

from __future__ import annotations

from ptodsl import pto, scalar
from ptodsl._ops import _coerce_i64


class PTOGemmL1Template:
    """PTODSL helper for the dav-3510 Ascend L1 cube GEMM pipeline.

    The constructor arguments are the static template parameters. Calling
    ``run_l1_tile`` emits the L1->L0 sub-K pipeline for one already-loaded
    L1 tile.

    dav-3510 requires the TRANS_B=true path. The right operand must be
    staged in L1 as logical W[N, K]. If you write NN matmul with W[K, N],
    then during GM->L1 you must use DN2NZ to transpose B into this layout
    before calling the helper.
    """

    _instances: dict[tuple[int, ...], PTOGemmL1Template] = {}

    def __new__(
        cls,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        base_k: int,
        sub_k_tiles: int,
        input_c0: int,
        sub_k_c0_blocks: int,
        a_l0_stage_elems: int,
        b_l0_stage_elems: int,
        input_pack_factor: int = 1,
    ):
        key = (
            tile_m,
            tile_n,
            tile_k,
            base_k,
            sub_k_tiles,
            input_c0,
            sub_k_c0_blocks,
            a_l0_stage_elems,
            b_l0_stage_elems,
            input_pack_factor,
        )
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        base_k: int,
        sub_k_tiles: int,
        input_c0: int,
        sub_k_c0_blocks: int,
        a_l0_stage_elems: int,
        b_l0_stage_elems: int,
        input_pack_factor: int = 1,
    ):
        if getattr(self, "_initialized", False):
            return
        self._validate_static_params(
            tile_m,
            tile_n,
            tile_k,
            base_k,
            sub_k_tiles,
            input_c0,
            sub_k_c0_blocks,
            a_l0_stage_elems,
            b_l0_stage_elems,
            input_pack_factor,
        )
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.base_k = base_k
        self.sub_k_tiles = sub_k_tiles
        self.input_c0 = input_c0
        self.sub_k_c0_blocks = sub_k_c0_blocks
        self.a_l0_stage_elems = a_l0_stage_elems
        self.b_l0_stage_elems = b_l0_stage_elems
        self.input_pack_factor = input_pack_factor
        self.sub_k_storage_cols = sub_k_c0_blocks * input_pack_factor
        self._initialized = True

    @staticmethod
    def _require_positive_int(name: str, value: int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be a positive Python int, got {value!r}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    @classmethod
    def _validate_static_params(
        cls,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        base_k: int,
        sub_k_tiles: int,
        input_c0: int,
        sub_k_c0_blocks: int,
        a_l0_stage_elems: int,
        b_l0_stage_elems: int,
        input_pack_factor: int,
    ):
        for name, value in (
            ("tile_m", tile_m),
            ("tile_n", tile_n),
            ("tile_k", tile_k),
            ("base_k", base_k),
            ("sub_k_tiles", sub_k_tiles),
            ("input_c0", input_c0),
            ("sub_k_c0_blocks", sub_k_c0_blocks),
            ("a_l0_stage_elems", a_l0_stage_elems),
            ("b_l0_stage_elems", b_l0_stage_elems),
            ("input_pack_factor", input_pack_factor),
        ):
            cls._require_positive_int(name, value)

        if tile_m % 16 != 0:
            raise ValueError(f"tile_m must be a multiple of 16, got {tile_m}")
        if tile_n % 16 != 0:
            raise ValueError(f"tile_n must be a multiple of 16, got {tile_n}")
        if tile_k % base_k != 0:
            raise ValueError(f"tile_k must be divisible by base_k, got {tile_k} and {base_k}")
        if tile_k != base_k * sub_k_tiles:
            raise ValueError(f"tile_k must equal base_k * sub_k_tiles, got {tile_k} != {base_k} * {sub_k_tiles}")
        if base_k % input_c0 != 0:
            raise ValueError(f"base_k must be divisible by input_c0, got {base_k} and {input_c0}")
        if sub_k_c0_blocks != base_k // input_c0:
            raise ValueError(f"sub_k_c0_blocks must equal base_k // input_c0, got {sub_k_c0_blocks} != {base_k} // {input_c0}")
        if (tile_m * base_k) % input_pack_factor != 0:
            raise ValueError(f"tile_m * base_k must be divisible by input_pack_factor, got {tile_m} * {base_k} and {input_pack_factor}")
        if (tile_n * base_k) % input_pack_factor != 0:
            raise ValueError(f"tile_n * base_k must be divisible by input_pack_factor, got {tile_n} * {base_k} and {input_pack_factor}")
        expected_a_stage_elems = tile_m * base_k // input_pack_factor
        expected_b_stage_elems = tile_n * base_k // input_pack_factor
        if a_l0_stage_elems != expected_a_stage_elems:
            raise ValueError(
                f"a_l0_stage_elems must equal tile_m * base_k / input_pack_factor, got {a_l0_stage_elems} != {expected_a_stage_elems}"
            )
        if b_l0_stage_elems != expected_b_stage_elems:
            raise ValueError(
                f"b_l0_stage_elems must equal tile_n * base_k / input_pack_factor, got {b_l0_stage_elems} != {expected_b_stage_elems}"
            )

    def _emit_l1_to_l0b_static(self, b_mat, b_l0_0, stage: int, sk: int):
        sub_k_storage_col = sk * self.sub_k_storage_cols
        # dav-3510 TRANS_B=true path: GEMM interprets B as transposed, so L1
        # must hold W[N, K]. NN inputs W[K, N] are converted to this layout by
        # DN2NZ during GM->L1.
        if sk == 0:
            pto.mte_l1_l0b(
                b_mat,
                self._b_l0_stage(b_l0_0, stage),
                self.base_k,
                self.tile_n,
            )
        else:
            pto.mte_l1_l0b(
                b_mat,
                self._b_l0_stage(b_l0_0, stage),
                self.base_k,
                self.tile_n,
                start_col=sub_k_storage_col,
            )

    def _a_l0_stage(self, a_l0_0, stage: int):
        if stage == 0:
            return a_l0_0
        return pto.addptr(a_l0_0, self.a_l0_stage_elems)

    def _b_l0_stage(self, b_l0_0, stage: int):
        if stage == 0:
            return b_l0_0
        return pto.addptr(b_l0_0, self.b_l0_stage_elems)

    def _emit_l1_to_l0_static(self, a_mat, b_mat, a_l0_0, b_l0_0, sk: int):
        stage = sk & 1
        sub_k_storage_col = sk * self.sub_k_storage_cols
        pto.wait_flag("M", "MTE1", event_id=stage)
        if sk == 0:
            pto.mte_l1_l0a(a_mat, self._a_l0_stage(a_l0_0, stage), self.tile_m, self.base_k)
        else:
            pto.mte_l1_l0a(
                a_mat,
                self._a_l0_stage(a_l0_0, stage),
                self.tile_m,
                self.base_k,
                start_col=sub_k_storage_col,
            )
        self._emit_l1_to_l0b_static(b_mat, b_l0_0, stage, sk)
        pto.set_flag("MTE1", "M", event_id=stage)

    @staticmethod
    def _is_static_int(value):
        return type(value) is int

    @staticmethod
    def _validate_static_unit_flag_ctrl(unit_flag_ctrl: int):
        if isinstance(unit_flag_ctrl, bool):
            raise TypeError("unit_flag_ctrl must be 0, 2, or 3, not bool")
        if unit_flag_ctrl not in (0, 2, 3):
            raise ValueError(f"unit_flag_ctrl must be 0, 2, or 3, got {unit_flag_ctrl}")

    @staticmethod
    def _mad_unit_flag(unit_flag_ctrl: int, is_last_sub_k: bool):
        PTOGemmL1Template._validate_static_unit_flag_ctrl(unit_flag_ctrl)
        if unit_flag_ctrl == 0:
            return None
        if unit_flag_ctrl == 3 and not is_last_sub_k:
            unit_flag_ctrl = 2
        if unit_flag_ctrl == 2:
            return pto.MadUnitFlagMode.CHECK_ONLY
        if unit_flag_ctrl == 3:
            return pto.MadUnitFlagMode.CHECK_AND_SET
        raise ValueError(f"unsupported MAD unit_flag_ctrl={unit_flag_ctrl}")

    def _emit_mad_op(
        self,
        a_l0_0,
        b_l0_0,
        acc,
        sk: int,
        use_mad: bool,
        unit_flag,
        tf32_mode,
    ):
        stage = sk & 1
        if use_mad:
            pto.mad(
                self._a_l0_stage(a_l0_0, stage),
                self._b_l0_stage(b_l0_0, stage),
                acc,
                self.tile_m,
                self.tile_n,
                self.base_k,
                unit_flag=unit_flag,
                tf32_mode=tf32_mode,
            )
            return
        pto.mad_acc(
            self._a_l0_stage(a_l0_0, stage),
            self._b_l0_stage(b_l0_0, stage),
            acc,
            self.tile_m,
            self.tile_n,
            self.base_k,
            unit_flag=unit_flag,
            tf32_mode=tf32_mode,
        )

    def _emit_mad_with_unit_flag(
        self,
        a_l0_0,
        b_l0_0,
        acc,
        sk: int,
        use_mad: bool,
        unit_flag_ctrl,
        tf32_mode,
    ):
        is_last_sub_k = sk == self.sub_k_tiles - 1
        if self._is_static_int(unit_flag_ctrl):
            self._emit_mad_op(
                a_l0_0,
                b_l0_0,
                acc,
                sk,
                use_mad,
                self._mad_unit_flag(unit_flag_ctrl, is_last_sub_k),
                tf32_mode,
            )
            return
        if isinstance(unit_flag_ctrl, bool):
            raise TypeError("unit_flag_ctrl must be 0, 2, or 3, not bool")

        with pto.if_(unit_flag_ctrl == 0) as uf_zero:
            with uf_zero.then_:
                self._emit_mad_op(a_l0_0, b_l0_0, acc, sk, use_mad, None, tf32_mode)
            with uf_zero.else_, pto.if_(unit_flag_ctrl == 3) as uf_set:
                with uf_set.then_:
                    self._emit_mad_op(
                        a_l0_0,
                        b_l0_0,
                        acc,
                        sk,
                        use_mad,
                        pto.MadUnitFlagMode.CHECK_AND_SET if is_last_sub_k else pto.MadUnitFlagMode.CHECK_ONLY,
                        tf32_mode,
                    )
                with uf_set.else_:
                    self._emit_mad_op(
                        a_l0_0,
                        b_l0_0,
                        acc,
                        sk,
                        use_mad,
                        pto.MadUnitFlagMode.CHECK_ONLY,
                        tf32_mode,
                    )

    def _emit_mad_static(
        self,
        a_l0_0,
        b_l0_0,
        acc,
        sk: int,
        clear_accum,
        unit_flag_ctrl,
        tf32_mode,
    ):
        stage = sk & 1
        pto.wait_flag("MTE1", "M", event_id=stage)
        if sk == 0 and not isinstance(clear_accum, bool):
            with pto.if_(clear_accum) as clear_br:
                with clear_br.then_:
                    self._emit_mad_with_unit_flag(a_l0_0, b_l0_0, acc, sk, True, unit_flag_ctrl, tf32_mode)
                with clear_br.else_:
                    self._emit_mad_with_unit_flag(a_l0_0, b_l0_0, acc, sk, False, unit_flag_ctrl, tf32_mode)
        else:
            use_mad = sk == 0 and bool(clear_accum)
            self._emit_mad_with_unit_flag(a_l0_0, b_l0_0, acc, sk, use_mad, unit_flag_ctrl, tf32_mode)
        pto.set_flag("M", "MTE1", event_id=stage)

    def _emit_pipeline_init(self):
        pto.set_flag("M", "MTE1", event_id=0)
        pto.set_flag("M", "MTE1", event_id=1)

    def _emit_pipeline_drain(self):
        pto.wait_flag("M", "MTE1", event_id=0)
        pto.wait_flag("M", "MTE1", event_id=1)

    def _emit_pipeline_init_if_enabled_first_k(self, clear_accum):
        if isinstance(clear_accum, (bool, int)):
            if bool(clear_accum):
                self._emit_pipeline_init()
            return

        with pto.if_(clear_accum) as first_k, first_k.then_:
            self._emit_pipeline_init()

    def _emit_pipeline_init_for_tile(self, clear_accum, unit_flag_ctrl):
        if self._is_static_int(unit_flag_ctrl):
            self._validate_static_unit_flag_ctrl(unit_flag_ctrl)
            if unit_flag_ctrl == 0:
                self._emit_pipeline_init()
            else:
                self._emit_pipeline_init_if_enabled_first_k(clear_accum)
            return
        if isinstance(unit_flag_ctrl, bool):
            raise TypeError("unit_flag_ctrl must be 0, 2, or 3, not bool")

        with pto.if_(unit_flag_ctrl == 0) as uf_disabled:
            with uf_disabled.then_:
                self._emit_pipeline_init()
            with uf_disabled.else_:
                self._emit_pipeline_init_if_enabled_first_k(clear_accum)

    def _emit_pipeline_drain_for_tile(self, unit_flag_ctrl):
        if self._is_static_int(unit_flag_ctrl):
            self._validate_static_unit_flag_ctrl(unit_flag_ctrl)
            if unit_flag_ctrl == 0 or unit_flag_ctrl == 3:
                self._emit_pipeline_drain()
            return
        if isinstance(unit_flag_ctrl, bool):
            raise TypeError("unit_flag_ctrl must be 0, 2, or 3, not bool")

        with pto.if_(unit_flag_ctrl == 0) as uf_disabled:
            with uf_disabled.then_:
                self._emit_pipeline_drain()
            with uf_disabled.else_, pto.if_(unit_flag_ctrl == 3) as last_k, last_k.then_:
                self._emit_pipeline_drain()

    def run_l1_tile(
        self,
        a_mat,
        b_mat,
        a_l0_0,
        b_l0_0,
        acc,
        *,
        clear_accum,
        unit_flag_ctrl=0,
        tf32_mode=None,
    ):
        """Emit GEMM for one L1 A/B tile into ``acc``."""

        self._emit_pipeline_init_for_tile(clear_accum, unit_flag_ctrl)

        self._emit_l1_to_l0_static(a_mat, b_mat, a_l0_0, b_l0_0, 0)
        if self.sub_k_tiles == 1:
            self._emit_mad_static(a_l0_0, b_l0_0, acc, 0, clear_accum, unit_flag_ctrl, tf32_mode)
            self._emit_pipeline_drain_for_tile(unit_flag_ctrl)
            return

        self._emit_l1_to_l0_static(a_mat, b_mat, a_l0_0, b_l0_0, 1)
        self._emit_mad_static(a_l0_0, b_l0_0, acc, 0, clear_accum, unit_flag_ctrl, tf32_mode)
        with pto.for_(2, self.sub_k_tiles, step=1) as sk:
            l0_stage = sk % 2
            l0_stage_i64 = _coerce_i64(l0_stage, context="L0 stage index")
            sub_k_storage_col = sk * self.sub_k_storage_cols
            a_l0 = pto.addptr(
                a_l0_0,
                scalar.muli(l0_stage_i64, pto.const(self.a_l0_stage_elems, dtype=pto.int64)),
            )
            b_l0 = pto.addptr(
                b_l0_0,
                scalar.muli(l0_stage_i64, pto.const(self.b_l0_stage_elems, dtype=pto.int64)),
            )
            pto.wait_flag("M", "MTE1", event_id=l0_stage)
            pto.mte_l1_l0a(
                a_mat,
                a_l0,
                self.tile_m,
                self.base_k,
                start_col=sub_k_storage_col,
            )
            pto.mte_l1_l0b(
                b_mat,
                b_l0,
                self.base_k,
                self.tile_n,
                start_col=sub_k_storage_col,
            )
            pto.set_flag("MTE1", "M", event_id=l0_stage)

            prev_stage = (sk - 1) % 2
            prev_stage_i64 = _coerce_i64(prev_stage, context="previous L0 stage index")
            a_l0_prev = pto.addptr(
                a_l0_0,
                scalar.muli(prev_stage_i64, pto.const(self.a_l0_stage_elems, dtype=pto.int64)),
            )
            b_l0_prev = pto.addptr(
                b_l0_0,
                scalar.muli(prev_stage_i64, pto.const(self.b_l0_stage_elems, dtype=pto.int64)),
            )
            pto.wait_flag("MTE1", "M", event_id=prev_stage)
            if self._is_static_int(unit_flag_ctrl):
                pto.mad_acc(
                    a_l0_prev,
                    b_l0_prev,
                    acc,
                    self.tile_m,
                    self.tile_n,
                    self.base_k,
                    unit_flag=self._mad_unit_flag(unit_flag_ctrl, False),
                    tf32_mode=tf32_mode,
                )
            elif isinstance(unit_flag_ctrl, bool):
                raise TypeError("unit_flag_ctrl must be 0, 2, or 3, not bool")
            else:
                with pto.if_(unit_flag_ctrl == 0) as loop_uf_zero:
                    with loop_uf_zero.then_:
                        pto.mad_acc(
                            a_l0_prev,
                            b_l0_prev,
                            acc,
                            self.tile_m,
                            self.tile_n,
                            self.base_k,
                            tf32_mode=tf32_mode,
                        )
                    with loop_uf_zero.else_:
                        pto.mad_acc(
                            a_l0_prev,
                            b_l0_prev,
                            acc,
                            self.tile_m,
                            self.tile_n,
                            self.base_k,
                            unit_flag=pto.MadUnitFlagMode.CHECK_ONLY,
                            tf32_mode=tf32_mode,
                        )
            pto.set_flag("M", "MTE1", event_id=prev_stage)

        self._emit_mad_static(
            a_l0_0,
            b_l0_0,
            acc,
            self.sub_k_tiles - 1,
            False,
            unit_flag_ctrl,
            tf32_mode,
        )
        self._emit_pipeline_drain_for_tile(unit_flag_ctrl)


class PTOBlockscaledGemmL1Template(PTOGemmL1Template):
    """PTODSL helper for an E4M3 or E2M1 blockscaled L1 cube GEMM tile.

    Matrix data uses the regular L1-to-L0 loads inherited from
    :class:`PTOGemmL1Template`. Pair-packed ``uint16`` scale storage is viewed
    as E8M0 only at the MX load boundary, where the caller supplies an
    ``pto.ptr(pto.f8e8m0, "mat")`` pointer to the same L1 allocation. FP4
    uses ``input_pack_factor=2`` so data-load start columns and L0 stage
    offsets remain in packed-storage units.
    """

    _instances: dict[tuple[int, ...], PTOBlockscaledGemmL1Template] = {}

    def __new__(
        cls,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        base_k: int,
        sub_k_tiles: int,
        input_c0: int,
        sub_k_c0_blocks: int,
        a_l0_stage_elems: int,
        b_l0_stage_elems: int,
        sf_nz_stride: int,
        input_pack_factor: int = 1,
    ):
        key = (
            tile_m,
            tile_n,
            tile_k,
            base_k,
            sub_k_tiles,
            input_c0,
            sub_k_c0_blocks,
            a_l0_stage_elems,
            b_l0_stage_elems,
            sf_nz_stride,
            input_pack_factor,
        )
        if key not in cls._instances:
            cls._instances[key] = object.__new__(cls)
        return cls._instances[key]

    def __init__(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        base_k: int,
        sub_k_tiles: int,
        input_c0: int,
        sub_k_c0_blocks: int,
        a_l0_stage_elems: int,
        b_l0_stage_elems: int,
        sf_nz_stride: int,
        input_pack_factor: int = 1,
    ):
        if getattr(self, "_blockscaled_initialized", False):
            return
        super().__init__(
            tile_m,
            tile_n,
            tile_k,
            base_k,
            sub_k_tiles,
            input_c0,
            sub_k_c0_blocks,
            a_l0_stage_elems,
            b_l0_stage_elems,
            input_pack_factor,
        )
        self._require_positive_int("sf_nz_stride", sf_nz_stride)
        if base_k % 64 != 0:
            raise ValueError(f"base_k must be divisible by 64, got {base_k}")
        self.sf_nz_stride = sf_nz_stride
        self.sf_pairs_per_inner = base_k // 64
        self._blockscaled_initialized = True

    def _emit_l1_to_l0_static(
        self,
        a_mat,
        b_mat,
        sfa_e8m0_mat,
        sfb_e8m0_mat,
        a_l0_0,
        b_l0_0,
        sf_k_offset,
        sk: int,
    ):
        stage = sk & 1
        sub_k_storage_col = sk * self.sub_k_storage_cols
        sf_y = sf_k_offset + sk * self.sf_pairs_per_inner
        a_l0 = self._a_l0_stage(a_l0_0, stage)
        b_l0 = self._b_l0_stage(b_l0_0, stage)

        pto.wait_flag("M", "MTE1", event_id=stage)
        if sk == 0:
            pto.mte_l1_l0a(a_mat, a_l0, self.tile_m, self.base_k)
            pto.mte_l1_l0b(b_mat, b_l0, self.base_k, self.tile_n)
        else:
            pto.mte_l1_l0a(
                a_mat,
                a_l0,
                self.tile_m,
                self.base_k,
                start_col=sub_k_storage_col,
            )
            pto.mte_l1_l0b(
                b_mat,
                b_l0,
                self.base_k,
                self.tile_n,
                start_col=sub_k_storage_col,
            )
        pto.mte_l1_l0a_mx(
            sfa_e8m0_mat,
            a_l0,
            x_start=0,
            y_start=sf_y,
            x_step=self.tile_m // 16,
            y_step=self.sf_pairs_per_inner,
            src_stride=self.sf_nz_stride,
            dst_stride=self.sf_pairs_per_inner,
        )
        pto.mte_l1_l0b_mx(
            sfb_e8m0_mat,
            b_l0,
            x_start=0,
            y_start=sf_y,
            x_step=self.tile_n // 16,
            y_step=self.sf_pairs_per_inner,
            src_stride=self.sf_nz_stride,
            dst_stride=self.sf_pairs_per_inner,
        )
        pto.set_flag("MTE1", "M", event_id=stage)

    def _emit_mad_op(
        self,
        a_l0_0,
        b_l0_0,
        acc,
        sk: int,
        use_mad: bool,
        unit_flag,
        *_unused_hf32_mode,
    ):
        stage = sk & 1
        a_l0 = self._a_l0_stage(a_l0_0, stage)
        b_l0 = self._b_l0_stage(b_l0_0, stage)
        if use_mad:
            pto.mad_mx(
                a_l0,
                b_l0,
                acc,
                self.tile_m,
                self.tile_n,
                self.base_k,
                unit_flag=unit_flag,
                disable_gemv=True,
                sat="sat",
            )
            return
        pto.mad_mx_acc(
            a_l0,
            b_l0,
            acc,
            self.tile_m,
            self.tile_n,
            self.base_k,
            unit_flag=unit_flag,
            disable_gemv=True,
            sat="sat",
        )

    def run_l1_tile(
        self,
        a_mat,
        b_mat,
        sfa_e8m0_mat,
        sfb_e8m0_mat,
        a_l0_0,
        b_l0_0,
        acc,
        *,
        sf_k_offset,
        clear_accum,
        unit_flag_ctrl=0,
    ):
        """Emit one blockscaled L1 tile with E8M0 scale staging."""

        self._emit_pipeline_init_for_tile(clear_accum, unit_flag_ctrl)

        # Keep each MX stage ordered until a prefetching schedule is validated.
        for sk in range(self.sub_k_tiles):
            self._emit_l1_to_l0_static(
                a_mat,
                b_mat,
                sfa_e8m0_mat,
                sfb_e8m0_mat,
                a_l0_0,
                b_l0_0,
                sf_k_offset,
                sk,
            )
            self._emit_mad_static(
                a_l0_0,
                b_l0_0,
                acc,
                sk,
                clear_accum if sk == 0 else False,
                unit_flag_ctrl,
                None,
            )
        self._emit_pipeline_drain_for_tile(unit_flag_ctrl)

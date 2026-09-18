"""TileIR optimization-hint validation and numerical tests."""

import pytest
import tilelang
from tilelang.tileir import language as T

from tileir_test_utils import _setup_gpu, _skip_if_tileir_toolchain_unavailable, skip_no_cuda_tile


def _vec_add_hints_prim_func(n=256, tileir_hints=None, num_ctas=None):
    """1-D vector add whose T.Kernel carries a per-arch `tileir_hints` dict."""

    @T.prim_func
    def kern(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 128), threads=128, tileir_hints=tileir_hints, num_ctas=num_ctas) as bx:
            for i in T.Parallel(128):
                idx = bx * 128 + i
                if idx < n:
                    C[idx] = A[idx] + B[idx]

    return kern


def _hints_lowering_options(pf):
    """Materialize the launch nest and run `_lowering_options` on *pf*."""
    from tilelang.tileir.lowering import _lowering_options
    from tilelang.tileir.semantic import materialize_launch_nest
    from tilelang.tileir.launch import _split_grid_sync_primfunc

    pf = materialize_launch_nest(pf)
    pf = _split_grid_sync_primfunc(pf)
    return pf, _lowering_options(pf, {})


@skip_no_cuda_tile
def test_tileir_hints_two_arch_mlir():
    """A two-arch `tileir_hints` dict must appear verbatim in the entry's
    optimization_hints: BOTH sm_100 and sm_120 sub-dicts, each with its own
    num_cta_in_cga value."""
    import re
    from tilelang.tileir.pipeline import build_tileir_module

    pf = _vec_add_hints_prim_func(
        tileir_hints={
            "sm_100": {"num_cta_in_cga": 2},
            "sm_120": {"num_cta_in_cga": 4},
        }
    )
    pf, options = _hints_lowering_options(pf)
    assert options.hints is not None

    mlir = str(build_tileir_module(pf, arch="sm_100", hints=options.hints))
    assert "optimization_hints" in mlir
    # Extract the optimization_hints=<...> substring to verify arch keys are inside it.
    hints_match = re.search(r"optimization_hints=<[^>]+>", mlir)
    assert hints_match is not None, f"optimization_hints attribute not found in MLIR.\nMLIR:\n{mlir[:2000]}"
    hints_attr = hints_match.group(0)
    assert "sm_100" in hints_attr, f"sm_100 arch not found inside optimization_hints: {hints_attr}"
    assert "sm_120" in hints_attr, f"sm_120 arch not found inside optimization_hints: {hints_attr}"
    assert "num_cta_in_cga = 2" in hints_attr, f"sm_100 hint not found inside optimization_hints: {hints_attr}"
    assert "num_cta_in_cga = 4" in hints_attr, f"sm_120 hint not found inside optimization_hints: {hints_attr}"


@skip_no_cuda_tile
def test_tileir_hints_canonicalize_ordering():
    """Two hint dicts with different insertion orders must produce identical
    TileIRLoweringOptions.hints and MLIR text (canonical sorted ordering)."""
    from tilelang.tileir.pipeline import build_tileir_module

    # First dict: z-last insertion order.
    pf1 = _vec_add_hints_prim_func(
        tileir_hints={
            "sm_120": {"occupancy": 2, "num_cta_in_cga": 4},
            "sm_100": {"num_cta_in_cga": 2, "occupancy": 1},
        }
    )
    pf1, options1 = _hints_lowering_options(pf1)

    # Second dict: reverse insertion order, different per-hint ordering.
    pf2 = _vec_add_hints_prim_func(
        tileir_hints={
            "sm_100": {"occupancy": 1, "num_cta_in_cga": 2},
            "sm_120": {"num_cta_in_cga": 4, "occupancy": 2},
        }
    )
    pf2, options2 = _hints_lowering_options(pf2)

    # Both should produce the same canonical frozen hints tuple.
    assert options1.hints == options2.hints, "Different insertion orders should yield identical hints"

    # Both should produce the same MLIR text.
    mlir1 = str(build_tileir_module(pf1, arch="sm_100", hints=options1.hints))
    mlir2 = str(build_tileir_module(pf2, arch="sm_100", hints=options2.hints))
    # Extract optimization_hints substrings and compare.
    import re

    match1 = re.search(r"optimization_hints=<[^>]+>", mlir1)
    match2 = re.search(r"optimization_hints=<[^>]+>", mlir2)
    assert match1 and match2, "Both should have optimization_hints"
    assert match1.group(0) == match2.group(0), "Different insertion orders should produce identical MLIR text"


@skip_no_cuda_tile
def test_tileir_hints_entry_scoped_keys_mlir():
    """Entry-scoped hint keys (num_cta_in_cga, num_worker_warps_per_cta,
    occupancy) plus the `default` arch key must parse and survive into MLIR."""
    from tilelang.tileir.pipeline import build_tileir_module

    pf = _vec_add_hints_prim_func(
        tileir_hints={
            "sm_100": {
                "num_cta_in_cga": 2,
                "num_worker_warps_per_cta": 8,
                "occupancy": 4,
            },
            "default": {"occupancy": 2},
        }
    )
    pf, options = _hints_lowering_options(pf)

    mlir = str(build_tileir_module(pf, arch="sm_100", hints=options.hints))
    assert "sm_100" in mlir
    assert "default" in mlir
    assert "num_cta_in_cga = 2" in mlir
    assert "num_worker_warps_per_cta = 8" in mlir
    assert "occupancy = 4" in mlir
    assert "occupancy = 2" in mlir


@skip_no_cuda_tile
def test_tileir_hints_load_store_scoped_keys_rejected():
    """Load/store-scoped hint keys (allow_tma, latency) must be rejected
    on T.Kernel entry with a clear error message directing to per-copy hints."""
    from tilelang.tileir.errors import TileIRLoweringError

    for key, value in (("allow_tma", True), ("latency", 3)):
        pf = _vec_add_hints_prim_func(tileir_hints={"sm_100": {key: value}})
        with pytest.raises(TileIRLoweringError, match=rf"{key}.*load/store-scoped.*T.copy"):
            _hints_lowering_options(pf)


@skip_no_cuda_tile
def test_tileir_hints_invalid_arch_key_raises():
    from tilelang.tileir.errors import TileIRLoweringError

    pf = _vec_add_hints_prim_func(tileir_hints={"sm_999": {"num_cta_in_cga": 2}})
    with pytest.raises(TileIRLoweringError, match="sm_90, sm_100, sm_103, sm_110, sm_120, sm_121, default"):
        _hints_lowering_options(pf)


@skip_no_cuda_tile
def test_tileir_hints_arch_mismatch_raises():
    """An sm_100 compile whose `tileir.hints` covers ONLY sm_120 (no
    "default" fallback) must raise -- the hints would silently never apply
    to this compile. MLIR-text recipe: call `build_tileir_module` directly
    with `arch="sm_100"` (mirrors `test_tileir_hints_two_arch_mlir` above),
    no real GPU compile needed."""
    from tilelang.tileir.errors import TileIRLoweringError
    from tilelang.tileir.pipeline import build_tileir_module

    pf = _vec_add_hints_prim_func(tileir_hints={"sm_120": {"num_cta_in_cga": 2}})
    pf, options = _hints_lowering_options(pf)
    assert options.hints is not None

    with pytest.raises(TileIRLoweringError, match="sm_100"):
        build_tileir_module(pf, arch="sm_100", hints=options.hints)

    # Sanity: a "default" fallback entry rescues the same sm_100 compile.
    pf2 = _vec_add_hints_prim_func(tileir_hints={"sm_120": {"num_cta_in_cga": 2}, "default": {"occupancy": 2}})
    pf2, options2 = _hints_lowering_options(pf2)
    mlir = str(build_tileir_module(pf2, arch="sm_100", hints=options2.hints))
    assert "optimization_hints" in mlir

    # Sanity: an "sm_100a" (variant-suffixed) compile arch is normalized to
    # "sm_100" before matching against the hints dict's plain "sm_100" key.
    pf3 = _vec_add_hints_prim_func(tileir_hints={"sm_100": {"num_cta_in_cga": 2}})
    pf3, options3 = _hints_lowering_options(pf3)
    mlir3 = str(build_tileir_module(pf3, arch="sm_100a", hints=options3.hints))
    assert "optimization_hints" in mlir3


@skip_no_cuda_tile
def test_tileir_hints_invalid_hint_key_raises():
    from tilelang.tileir.errors import TileIRLoweringError

    pf = _vec_add_hints_prim_func(tileir_hints={"sm_100": {"bogus_knob": 1}})
    with pytest.raises(TileIRLoweringError, match="bogus_knob"):
        _hints_lowering_options(pf)


@skip_no_cuda_tile
def test_tileir_hints_out_of_range_value_raises():
    from tilelang.tileir.errors import TileIRLoweringError

    invalid_values = (
        ("occupancy", 33),
        ("num_cta_in_cga", 3),
        ("num_worker_warps_per_cta", 6),
        ("occupancy", 0),
    )
    for key, value in invalid_values:
        pf = _vec_add_hints_prim_func(tileir_hints={"sm_100": {key: value}})
        with pytest.raises(TileIRLoweringError, match=key):
            _hints_lowering_options(pf)


@skip_no_cuda_tile
def test_tileir_hints_conflict_with_single_knob_raises():
    from tilelang.tileir.errors import TileIRLoweringError

    pf = _vec_add_hints_prim_func(tileir_hints={"sm_100": {"num_cta_in_cga": 2}}, num_ctas=4)
    with pytest.raises(TileIRLoweringError, match="tileir.hints"):
        _hints_lowering_options(pf)


@skip_no_cuda_tile
def test_tileir_hints_numeric_smoke():
    """Compile AND run a per-arch-hinted kernel on the local GPU: proves
    tileiras accepts the parsed per-arch optimization_hints attribute."""
    _skip_if_tileir_toolchain_unavailable()
    torch, major, minor, _target_str = _setup_gpu()

    n = 256
    arch_key = f"sm_{major}{minor}"
    hints = {arch_key: {"num_cta_in_cga": 2, "occupancy": 4}, "default": {"occupancy": 2}}

    pf = _vec_add_hints_prim_func(n=n, tileir_hints=hints)
    kernel = tilelang.compile(pf, execution_backend="tileir")

    a = torch.randn(n, dtype=torch.float32, device="cuda")
    b = torch.randn(n, dtype=torch.float32, device="cuda")
    c = torch.empty(n, dtype=torch.float32, device="cuda")
    kernel(a, b, c)
    torch.testing.assert_close(c, a + b, rtol=1e-6, atol=1e-6)

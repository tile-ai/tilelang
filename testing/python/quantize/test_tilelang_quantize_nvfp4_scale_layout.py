from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
tilelang_testing = pytest.importorskip("tilelang.testing")

from tilelang.quantize import nvfp4 as nvfp4_utils

_BLOCKSCALED_CHUNK_WORDS = nvfp4_utils._BLOCKSCALED_CHUNK_WORDS
from tilelang.quantize import (
    pack_blockscaled_chunk_kmajor_scale_bytes,
    quantize_bf16_to_nvfp4_blockscaled,
    swizzle_blockscaled_chunk_kmajor_scale_words,
    unswizzle_blockscaled_chunk_kmajor_scale_words,
)


def _load_maint_quantizer():
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "maint/gemm/gemm_sm120/tilelang_nvfp4_quantizer.py"
    spec = importlib.util.spec_from_file_location("tilelang_nvfp4_quantizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.tilelang_quantize_bf16_to_nvfp4_blockscaled


from tilelang.quantize.nvfp4 import (
    blockscaled_chunk_kmajor_word_offset,
    decode_packed_fp4_e2m1,
    decode_ue4m3_scale_bytes,
    encode_fp4_e2m1_values,
    encode_ue4m3_scale_bytes,
    pack_nvfp4_scale_bytes,
)


# ---------------------------------------------------------------------------
# SM120 OMMA scale-selector contract oracles.
# These document the compact-selector lane semantics the packed scale layout
# was derived from; they are exercised only by the contract tests below and
# live here to keep tilelang.quantize.nvfp4 focused on quantization.
# ---------------------------------------------------------------------------


def _sm120_sfa_row_in_lane(lane: int) -> int:
    return 8 * (lane & 1) + (lane >> 2)


def _sm120_sfb_col_in_lane(lane: int) -> int:
    return lane >> 2


def _sm120_sfa_selector_source_lane(lane: int, scale_a_thread_id: int) -> int:
    """Return the source lane selected by OMMA.SF's A-side thread selector."""

    return (lane & ~3) | ((scale_a_thread_id & 1) << 1) | (lane & 1)


def _sm120_sfb_selector_source_lane(lane: int, scale_b_thread_id: int) -> int:
    """Return the source lane selected by OMMA.SF's B-side thread selector."""

    return (lane & ~3) | (scale_b_thread_id & 3)


def _sm120_compact_sfa_owner_row(lane: int, warp_m: int, reg_group: int) -> int:
    qlane = lane & 3
    return warp_m * 64 + reg_group * 32 + (qlane >> 1) * 16 + _sm120_sfa_row_in_lane(lane)


def _sm120_compact_sfb_owner_row(lane: int, warp_n: int, reg_group: int) -> int:
    qlane = lane & 3
    return warp_n * 64 + reg_group * 32 + qlane * 8 + _sm120_sfb_col_in_lane(lane)


def _sm120_compact_sfa_effective_row(lane: int, warp_m: int, mma_i: int) -> int:
    source_lane = _sm120_sfa_selector_source_lane(lane, mma_i & 1)
    return _sm120_compact_sfa_owner_row(source_lane, warp_m, mma_i >> 1)


def _sm120_compact_sfb_effective_row(lane: int, warp_n: int, mma_j: int, half: int) -> int:
    source_lane = _sm120_sfb_selector_source_lane(lane, (mma_j & 1) * 2 + half)
    return _sm120_compact_sfb_owner_row(source_lane, warp_n, mma_j >> 1)


def _sm120_compact_scale_issue_contract(lane: int, warp_m: int, warp_n: int) -> list[dict[str, int | str]]:
    """Return the compact-selector OMMA.SF scale contract for one lane.

    This is a pure specification helper for tests and future lowering work. It
    models which lane supplies the SFA/SFB scale register selected by each
    ``mma.sync.m16n8k64.kind::mxf4nvf4.block_scale`` issue.
    """

    issues: list[dict[str, int | str]] = []
    for mma_i in range(4):
        sa_reg_group = mma_i >> 1
        sa_tid = mma_i & 1
        sa_source_lane = _sm120_sfa_selector_source_lane(lane, sa_tid)
        for mma_j in range(4):
            sb_reg_group = mma_j >> 1
            for half in range(2):
                sb_tid = (mma_j & 1) * 2 + half
                sb_source_lane = _sm120_sfb_selector_source_lane(lane, sb_tid)
                issues.append(
                    {
                        "mma_i": mma_i,
                        "mma_j": mma_j,
                        "half": half,
                        "sa_reg": f"sa{sa_reg_group}",
                        "sa_tid": sa_tid,
                        "sa_source_lane": sa_source_lane,
                        "sfa_row": _sm120_compact_sfa_owner_row(sa_source_lane, warp_m, sa_reg_group),
                        "sb_reg": f"sb{sb_reg_group}",
                        "sb_tid": sb_tid,
                        "sb_source_lane": sb_source_lane,
                        "sfb_row": _sm120_compact_sfb_owner_row(sb_source_lane, warp_n, sb_reg_group),
                    }
                )
    return issues


def _sm120_compact_scale_copy_view_contract(lane: int, warp_m: int, warp_n: int, k64_word: int) -> list[dict[str, object]]:
    """Return per-issue scale rows plus BlockScaledBasicChunk word addresses."""

    issues: list[dict[str, object]] = []
    for issue in _sm120_compact_scale_issue_contract(lane, warp_m, warp_n):
        sfa_word_coord = blockscaled_chunk_kmajor_word_offset(int(issue["sfa_row"]), k64_word)
        sfb_word_coord = blockscaled_chunk_kmajor_word_offset(int(issue["sfb_row"]), k64_word)
        issues.append(
            {
                **issue,
                "k64_word": k64_word,
                "sfa_word_coord": sfa_word_coord,
                "sfa_word_offset": sfa_word_coord[0] * _BLOCKSCALED_CHUNK_WORDS + sfa_word_coord[1],
                "sfb_word_coord": sfb_word_coord,
                "sfb_word_offset": sfb_word_coord[0] * _BLOCKSCALED_CHUNK_WORDS + sfb_word_coord[1],
            }
        )
    return issues


def _sm120_compact_scale_producer_contract(warp_m: int, warp_n: int) -> list[dict[str, object]]:
    """Return the source-lane register assignment required by current OMMA.SF.

    This is the inverse view of ``_sm120_compact_scale_issue_contract``. It says
    which producer lane supplies the scale register consumed by each issue. A
    future compact scale-copy lowering must preserve these semantic rows while
    changing how the producer lanes load/package the registers.
    """

    rows: list[dict[str, object]] = []
    for mma_i in range(4):
        sa_tid = mma_i & 1
        sa_reg_group = mma_i >> 1
        for producer_lane in range(32):
            consumers = [lane for lane in range(32) if _sm120_sfa_selector_source_lane(lane, sa_tid) == producer_lane]
            if not consumers:
                continue
            rows.append(
                {
                    "kind": "SFA",
                    "mma_i": mma_i,
                    "sa_tid": sa_tid,
                    "sa_reg": f"sa{sa_reg_group}",
                    "producer_lane": producer_lane,
                    "consumers": tuple(consumers),
                    "row": _sm120_compact_sfa_owner_row(producer_lane, warp_m, sa_reg_group),
                }
            )

    for mma_j in range(4):
        sb_reg_group = mma_j >> 1
        for half in range(2):
            sb_tid = (mma_j & 1) * 2 + half
            for producer_lane in range(32):
                consumers = [lane for lane in range(32) if _sm120_sfb_selector_source_lane(lane, sb_tid) == producer_lane]
                if not consumers:
                    continue
                rows.append(
                    {
                        "kind": "SFB",
                        "mma_j": mma_j,
                        "half": half,
                        "sb_tid": sb_tid,
                        "sb_reg": f"sb{sb_reg_group}",
                        "producer_lane": producer_lane,
                        "consumers": tuple(consumers),
                        "row": _sm120_compact_sfb_owner_row(producer_lane, warp_n, sb_reg_group),
                    }
                )
    return rows


def _sm120_compact_package_consumer_byte_offsets(lane: int, warp_m: int, warp_n: int, k64_word: int) -> dict[str, tuple[int, ...]]:
    """Return the current package's per-consumer byte offsets for one K64 word."""

    issues = _sm120_compact_scale_copy_view_contract(lane, warp_m, warp_n, k64_word)
    sfa_offsets = tuple(int(issue["sfa_word_offset"]) * 4 for issue in issues[0::8])
    sfb_offsets = tuple(int(issue["sfb_word_offset"]) * 4 for issue in issues[:8])
    return {"SFA": sfa_offsets, "SFB": sfb_offsets}


def _sm120_cutlass_issue_site_contract(site: int) -> dict[str, int]:
    """Return CUTLASS/CuTe SM120 fulltile issue-site coordinates.

    The traversal covers four K64 atoms. Each K64 atom issues eight N8 atoms,
    and each N8 atom visits four M atoms. Odd N8 atoms reverse the M order.
    """

    if site < 0 or site >= 128:
        raise ValueError(f"site must be in [0, 128), got {site}")
    k_block = site // 32
    site_in_k = site % 32
    n8_atom = site_in_k // 4
    m_serp = site_in_k % 4
    m_atom = m_serp if n8_atom % 2 == 0 else 3 - m_serp
    n16_col = n8_atom // 2
    n8_half = n8_atom % 2
    return {
        "site": site,
        "k_block": k_block,
        "m_atom": m_atom,
        "n8_atom": n8_atom,
        "n16_col": n16_col,
        "n8_half": n8_half,
        "A_slot0": 128 * m_atom + 32 * k_block,
        "B_slot0": 64 * n8_half + 128 * n16_col + 16 * k_block,
        "SFA_slot0": 4 * m_atom + 16 * k_block,
        "SFB_slot0": 4 * n16_col + 16 * n8_half + 32 * k_block,
        "C_slot0": 4 * m_atom + 32 * n16_col + 16 * n8_half,
    }


def _sm120_tilelang_current_issue_site_contract(site: int) -> dict[str, int]:
    """Return the current package helper's issue coordinates for comparison."""

    if site < 0 or site >= 128:
        raise ValueError(f"site must be in [0, 128), got {site}")
    k_block = site // 32
    site_in_k = site % 32
    mma_i = site_in_k // 8
    n_site = site_in_k % 8
    mma_j = n_site // 2
    half = n_site & 1
    return {
        "site": site,
        "k_block": k_block,
        "mma_i": mma_i,
        "mma_j": mma_j,
        "half": half,
        "sa_reg_group": mma_i >> 1,
        "sa_tid": mma_i & 1,
        "sb_reg_group": mma_j >> 1,
        "sb_tid": (mma_j & 1) * 2 + half,
    }


def _pack_semantic_scale_words(scale_bytes):
    scale_i64 = scale_bytes.to(torch.int64).reshape(scale_bytes.shape[0], scale_bytes.shape[1] // 4, 4)
    words = scale_i64[:, :, 0]
    words = words | (scale_i64[:, :, 1] << 8)
    words = words | (scale_i64[:, :, 2] << 16)
    words = words | (scale_i64[:, :, 3] << 24)
    return words.to(torch.uint32).contiguous()


def _expected_blockscaled_chunk_kmajor_byte_location(row: int, k16_idx: int, k64_cols: int = 4) -> tuple[int, int, int]:
    k64_word = k16_idx // 4
    byte_lane = k16_idx % 4
    row_block = row // 128
    row_in_block = row % 128
    flat_word = row_block * 128 * k64_cols + k64_word * 128 + (row_in_block % 32) * 4 + (row_in_block // 32)
    physical_row = flat_word // k64_cols
    physical_word = flat_word % k64_cols
    return physical_row, physical_word, byte_lane


def _flat_word_offset(row: int, k64_word: int) -> int:
    physical_row, physical_word = blockscaled_chunk_kmajor_word_offset(row, k64_word)
    return physical_row * 4 + physical_word


def _packed_byte(packed, row: int, word: int, byte_lane: int) -> int:
    return (int(packed[row, word].item()) >> (8 * byte_lane)) & 0xFF


def _packed_word(packed, logical_row: int, k64_word: int) -> int:
    k64_cols = packed.shape[1]
    row_in_block = logical_row % 128
    flat_word = (logical_row // 128) * 128 * k64_cols + k64_word * 128 + (row_in_block % 32) * 4 + (row_in_block // 32)
    return int(packed.reshape(-1)[flat_word].item())


def _sfa_row_in_lane(lane: int) -> int:
    return _sm120_sfa_row_in_lane(lane)


def _sfb_col_in_lane(lane: int) -> int:
    return _sm120_sfb_col_in_lane(lane)


def _sfa_selector_source_lane(lane: int, scale_a_thread_id: int) -> int:
    return _sm120_sfa_selector_source_lane(lane, scale_a_thread_id)


def _sfb_selector_source_lane(lane: int, scale_b_thread_id: int) -> int:
    return _sm120_sfb_selector_source_lane(lane, scale_b_thread_id)


def _current_compact_sfa_row(lane: int, warp_m: int, mma_i: int) -> int:
    return _sm120_compact_sfa_effective_row(lane, warp_m, mma_i)


def _current_compact_sfb_row(lane: int, warp_n: int, mma_j: int, half: int) -> int:
    return _sm120_compact_sfb_effective_row(lane, warp_n, mma_j, half)


def _compact_issue_contract(lane: int, warp_m: int, warp_n: int):
    return _sm120_compact_scale_issue_contract(lane, warp_m, warp_n)


def _compact_copy_view_contract(lane: int, warp_m: int, warp_n: int, k64_word: int):
    return _sm120_compact_scale_copy_view_contract(lane, warp_m, warp_n, k64_word)


def _compact_producer_contract(warp_m: int, warp_n: int):
    return _sm120_compact_scale_producer_contract(warp_m, warp_n)


def _compact_package_consumer_byte_offsets(lane: int, warp_m: int, warp_n: int, k64_word: int):
    return _sm120_compact_package_consumer_byte_offsets(lane, warp_m, warp_n, k64_word)


def _cutlass_issue_site_contract(site: int):
    return _sm120_cutlass_issue_site_contract(site)


def _tilelang_current_issue_site_contract(site: int):
    return _sm120_tilelang_current_issue_site_contract(site)


def _unpack_with_oracle(packed, rows: int, k16_cols: int):
    out = torch.empty((rows, k16_cols), dtype=torch.uint8)
    k64_cols = k16_cols // 4
    for row in range(rows):
        for k16_idx in range(k16_cols):
            physical_row, physical_word, byte_lane = _expected_blockscaled_chunk_kmajor_byte_location(row, k16_idx, k64_cols)
            out[row, k16_idx] = _packed_byte(packed, physical_row, physical_word, byte_lane)
    return out


def test_blockscaled_chunk_kmajor_word_offset_fixed_cases():
    expected = {
        (0, 0): (0, 0),
        (0, 1): (32, 0),
        (0, 2): (64, 0),
        (0, 3): (96, 0),
        (31, 0): (31, 0),
        (32, 0): (0, 1),
        (63, 1): (63, 1),
        (64, 2): (64, 2),
        (96, 3): (96, 3),
        (127, 3): (127, 3),
    }
    for (row, k64_word), physical in expected.items():
        assert blockscaled_chunk_kmajor_word_offset(row, k64_word) == physical


def test_sm120_contract_helpers_stay_out_of_the_quantize_module():
    # The OMMA scale-selector contract oracles live in this test file; the
    # quantize module stays focused on quantization and layout packing.
    import tilelang.quantize as quantize

    assert not hasattr(nvfp4_utils, "_sm120_compact_scale_issue_contract")
    assert not hasattr(nvfp4_utils, "_sm120_compact_scale_copy_view_contract")
    assert not hasattr(quantize, "_sm120_compact_scale_issue_contract")
    assert not hasattr(quantize, "_sm120_compact_scale_copy_view_contract")


def test_blockscaled_chunk_kmajor_byte_word_and_flat_offsets():
    # CUTLASS documents BlockScaledBasicChunk at scale-byte granularity. TileLang
    # stores four K/16 scale bytes in one uint32, so the CUDA helper consumes the
    # corresponding flat word offset.
    byte_offsets_for_k16_zero = {
        0: 0,
        32: 4,
        64: 8,
        96: 12,
        127: 508,
    }
    for row, byte_offset in byte_offsets_for_k16_zero.items():
        physical_row, physical_word, byte_lane = _expected_blockscaled_chunk_kmajor_byte_location(row, 0)
        assert byte_lane == 0
        assert physical_row * 16 + physical_word * 4 + byte_lane == byte_offset
        assert _flat_word_offset(row, 0) == byte_offset // 4

    assert [_flat_word_offset(row, 0) for row in (0, 32, 64, 96)] == [0, 1, 2, 3]
    assert [_flat_word_offset(row, 0) * 4 for row in (0, 32, 64, 96)] == [0, 4, 8, 12]


def test_blockscaled_chunk_kmajor_explains_compact_source_address_examples():
    # These are source-layout examples only: they show why BlockScaledBasicChunk
    # K-major can produce compact scale-word addresses such as the CuTe-style
    # examples A=[0,4,8,12] and B=[0,4,32,36]. A real OMMA.SF scale package must
    # additionally prove that the selected source lanes name the right semantic
    # SFA/SFB rows for every issue.
    sfa_source_rows = [0, 32, 64, 96]
    sfb_source_rows = [0, 32, 2, 34]

    assert [_flat_word_offset(row, 0) * 4 for row in sfa_source_rows] == [0, 4, 8, 12]
    assert [_flat_word_offset(row, 0) * 4 for row in sfb_source_rows] == [0, 4, 32, 36]

    # The production compact-selector package still names the rows selected by
    # OMMA.SF's thread-id network. For lane 0 this is a different access pattern;
    # the source layout is already swizzled, but a CuTe-style scale slot package is
    # still a separate lowering contract.
    current_lane0_sfa_rows = [_current_compact_sfa_row(0, 0, mma_i) for mma_i in range(4)]
    current_lane0_sfb_rows = [_current_compact_sfb_row(0, 0, mma_j, 0) for mma_j in range(4)]

    assert current_lane0_sfa_rows == [0, 16, 32, 48]
    assert [_flat_word_offset(row, 0) * 4 for row in current_lane0_sfa_rows] == [0, 256, 4, 260]
    assert current_lane0_sfb_rows == [0, 16, 32, 48]
    assert [_flat_word_offset(row, 0) * 4 for row in current_lane0_sfb_rows] == [0, 256, 4, 260]


def test_compact_package_producer_contract_inverts_omma_sf_source_lane_selectors():
    producer_rows = _compact_producer_contract(warp_m=0, warp_n=0)

    sfa_issue0 = [row for row in producer_rows if row["kind"] == "SFA" and row["mma_i"] == 0 and row["producer_lane"] < 4]
    sfa_issue1 = [row for row in producer_rows if row["kind"] == "SFA" and row["mma_i"] == 1 and row["producer_lane"] < 4]
    assert [(row["producer_lane"], row["consumers"], row["row"]) for row in sfa_issue0] == [
        (0, (0, 2), 0),
        (1, (1, 3), 8),
    ]
    assert [(row["producer_lane"], row["consumers"], row["row"]) for row in sfa_issue1] == [
        (2, (0, 2), 16),
        (3, (1, 3), 24),
    ]

    sfb_issue0 = [row for row in producer_rows if row["kind"] == "SFB" and row["mma_j"] == 0 and row["producer_lane"] < 4]
    assert [(row["producer_lane"], row["consumers"], row["row"]) for row in sfb_issue0] == [
        (0, (0, 1, 2, 3), 0),
        (1, (0, 1, 2, 3), 8),
    ]

    # The inverse contract is the semantic constraint a future scale package must
    # keep. Compact source addresses alone are insufficient unless they are tied
    # back to these producer-lane rows.
    for row in producer_rows:
        if row["kind"] == "SFA":
            for consumer_lane in row["consumers"]:
                assert _current_compact_sfa_row(consumer_lane, 0, row["mma_i"]) == row["row"]
        else:
            for consumer_lane in row["consumers"]:
                assert _current_compact_sfb_row(consumer_lane, 0, row["mma_j"], row["half"]) == row["row"]


def test_current_register_package_shape_cannot_be_relabelled_as_cute_compact_slots():
    # With the current issue order and selector ids, lane 0 consumes scale rows
    # through the existing sa0/sa1 and sb0/sb1 slots. These are the true offsets
    # seen by the current OMMA.SF package after applying the source-lane selector.
    current_offsets = _compact_package_consumer_byte_offsets(lane=0, warp_m=0, warp_n=0, k64_word=0)

    assert current_offsets["SFA"] == (0, 256, 4, 260)
    assert current_offsets["SFB"] == (0, 128, 256, 384, 4, 132, 260, 388)

    # The compact source-address examples are real properties of the swizzled
    # source layout, but they are not reachable by merely changing the current
    # smem load offsets while preserving this package's semantic rows.
    assert current_offsets["SFA"] != (0, 4, 8, 12)
    assert current_offsets["SFB"][:4] != (0, 4, 32, 36)


def test_cutlass_fulltile_issue_site_contract_matches_known_static_dump_points():
    expected = {
        0: (0, 0, 0, 0, 0, 0, 0, 0),
        1: (0, 1, 0, 0, 4, 0, 0, 4),
        3: (0, 3, 0, 0, 12, 0, 0, 12),
        4: (0, 3, 0, 1, 12, 16, 64, 28),
        7: (0, 0, 0, 1, 0, 16, 64, 16),
        8: (0, 0, 1, 0, 0, 4, 128, 32),
        31: (0, 0, 3, 1, 0, 28, 448, 112),
        32: (1, 0, 0, 0, 16, 32, 16, 0),
        63: (1, 0, 3, 1, 16, 60, 464, 112),
        96: (3, 0, 0, 0, 48, 96, 48, 0),
        127: (3, 0, 3, 1, 48, 124, 496, 112),
    }
    for site, values in expected.items():
        issue = _cutlass_issue_site_contract(site)
        assert (
            issue["k_block"],
            issue["m_atom"],
            issue["n16_col"],
            issue["n8_half"],
            issue["SFA_slot0"],
            issue["SFB_slot0"],
            issue["B_slot0"],
            issue["C_slot0"],
        ) == values


def test_current_tilelang_issue_order_differs_from_cutlass_serpentine_order():
    # Sites 0..3 agree on m order. Starting at site 4 CUTLASS reverses m_atom for
    # the odd N8 half, while the current helper continues row-major over I.
    assert [_cutlass_issue_site_contract(site)["m_atom"] for site in range(8)] == [0, 1, 2, 3, 3, 2, 1, 0]
    assert [_tilelang_current_issue_site_contract(site)["mma_i"] for site in range(8)] == [0] * 8

    assert [_cutlass_issue_site_contract(site)["SFA_slot0"] for site in range(8)] == [0, 4, 8, 12, 12, 8, 4, 0]
    assert [_tilelang_current_issue_site_contract(site)["sa_tid"] for site in range(8)] == [0] * 8


def test_sfa_compact_source_example_spans_warp_m_groups_not_one_consumer_package():
    # The SFA [0,4,8,12] byte-address example corresponds to rows
    # [0,32,64,96]. For the current TileLang warp decomposition, lane 0 reaches
    # those rows only by grouping the same producer-lane/reg-group slots across
    # warp_m=0 and warp_m=1. A single warp_m consumer issue sequence remains
    # [0,16,32,48] and therefore has the non-compact offsets checked above.
    compact_rows = [
        _current_compact_sfa_row(lane=0, warp_m=0, mma_i=0),
        _current_compact_sfa_row(lane=0, warp_m=0, mma_i=2),
        _current_compact_sfa_row(lane=0, warp_m=1, mma_i=0),
        _current_compact_sfa_row(lane=0, warp_m=1, mma_i=2),
    ]
    assert compact_rows == [0, 32, 64, 96]
    assert [_flat_word_offset(row, 0) * 4 for row in compact_rows] == [0, 4, 8, 12]

    one_warp_rows = [_current_compact_sfa_row(lane=0, warp_m=0, mma_i=mma_i) for mma_i in range(4)]
    assert one_warp_rows == [0, 16, 32, 48]
    assert [_flat_word_offset(row, 0) * 4 for row in one_warp_rows] == [0, 256, 4, 260]


def test_sfb_compact_source_example_spans_b_operand_lane_groups():
    # The SFB [0,4,32,36] byte-address example corresponds to rows
    # [0,32,2,34]. In the current semantic model those are the two SFB reg groups
    # owned by producer lanes 0 and 8. This matches the B operand's lane-grouped
    # view: lanes separated by 8 advance the B column-in-lane coordinate while
    # keeping qlane fixed.
    compact_rows = [
        _current_compact_sfb_row(lane=0, warp_n=0, mma_j=0, half=0),
        _current_compact_sfb_row(lane=0, warp_n=0, mma_j=2, half=0),
        _current_compact_sfb_row(lane=8, warp_n=0, mma_j=0, half=0),
        _current_compact_sfb_row(lane=8, warp_n=0, mma_j=2, half=0),
    ]
    assert compact_rows == [0, 32, 2, 34]
    assert [_flat_word_offset(row, 0) * 4 for row in compact_rows] == [0, 4, 32, 36]

    one_consumer_rows = [_current_compact_sfb_row(lane=0, warp_n=0, mma_j=mma_j, half=0) for mma_j in range(4)]
    assert one_consumer_rows == [0, 16, 32, 48]
    assert [_flat_word_offset(row, 0) * 4 for row in one_consumer_rows] == [0, 256, 4, 260]


def test_pack_blockscaled_chunk_kmajor_scale_bytes_fixed_byte_offsets():
    rows = 256
    k16_cols = 32
    scale_bytes = torch.zeros((rows, k16_cols), dtype=torch.uint8)
    cases = [
        (0, 0, 0x11),
        (32, 0, 0x22),
        (64, 8, 0x33),
        (96, 12, 0x44),
        (127, 15, 0x55),
        (128, 16, 0x66),
        (159, 31, 0x77),
        (255, 27, 0x88),
    ]
    for row, k16_idx, value in cases:
        scale_bytes[row, k16_idx] = value

    packed = pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes)

    assert packed.shape == (rows, k16_cols // 4)
    assert packed.dtype == torch.uint32
    for row, k16_idx, value in cases:
        physical_row, physical_word, byte_lane = _expected_blockscaled_chunk_kmajor_byte_location(row, k16_idx, packed.shape[1])
        assert _packed_byte(packed, physical_row, physical_word, byte_lane) == value


def test_pack_blockscaled_chunk_kmajor_scale_bytes_random_binary_512x32_matches_oracle():
    rows = 512
    k16_cols = 512 // 16
    generator = torch.Generator(device="cpu").manual_seed(17)
    scale_bytes = torch.randint(0, 2, (rows, k16_cols), generator=generator, dtype=torch.uint8) * 0x38

    packed = pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes)

    assert packed.shape == (rows, k16_cols // 4)
    assert packed.dtype == torch.uint32
    assert torch.equal(_unpack_with_oracle(packed, rows, k16_cols), scale_bytes)


def test_pack_blockscaled_chunk_kmajor_scale_bytes_matches_cutedsl_blocked_sf_layout():
    """Byte-level cross-compatibility with the CuTeDSL/CUTLASS canonical SF layout.

    CuTeDSL builds SFA/SFB with ``blockscaled_utils.tile_atom_to_shape_SF``:
    atom ``((32,4),(16,4)):((16,4),(0,1))`` tiled with order ``(2,1,3)``, e.g.
    for ``(MN=256, K=512, L=1)`` the layout prints as
    ``(((32,4),2),((16,4),8),(1,1)):(((16,4),4096),((0,1),512),(0,0))``.
    The packed uint32 tensor must carry exactly those bytes so one buffer can
    feed both the TileLang SM120 path and a CuTeDSL NVFP4 blockscaled GEMM
    (``tl_words.view(torch.uint8)`` / ``sf_u8.view(torch.uint32)`` are
    zero-copy bridges between the two views).
    """
    for rows, k in ((128, 256), (256, 512), (384, 1024)):
        k16_cols = k // 16
        rest_k = k16_cols // 4
        generator = torch.Generator(device="cpu").manual_seed(rows + k)
        scale_bytes = torch.randint(0, 256, (rows, k16_cols), generator=generator, dtype=torch.uint8)

        packed_bytes = pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes).view(torch.uint8).reshape(-1)

        m = torch.arange(rows).unsqueeze(1)
        k16 = torch.arange(k16_cols).unsqueeze(0)
        cutedsl_offset = (m % 32) * 16 + ((m // 32) % 4) * 4 + (m // 128) * (512 * rest_k) + (k16 % 4) + (k16 // 4) * 512
        assert torch.equal(packed_bytes[cutedsl_offset.reshape(-1)].reshape(rows, k16_cols), scale_bytes)


def test_pack_blockscaled_chunk_kmajor_scale_bytes_matches_word_swizzle():
    rows = 512
    k16_cols = 512 // 16
    scale_bytes = torch.arange(rows * k16_cols, dtype=torch.uint8).reshape(rows, k16_cols)

    semantic_words = _pack_semantic_scale_words(scale_bytes)
    assert torch.equal(pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes), swizzle_blockscaled_chunk_kmajor_scale_words(semantic_words))


def test_pack_nvfp4_scale_bytes_default_matches_blockscaled_chunk_kmajor_layout():
    rows = 512
    k16_cols = 512 // 16
    scale_bytes = torch.arange(rows * k16_cols, dtype=torch.uint8).reshape(rows, k16_cols)

    expected = pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes)
    actual = pack_nvfp4_scale_bytes(scale_bytes)

    assert torch.equal(actual, expected)


def test_blockscaled_chunk_kmajor_matches_current_omma_sf_selector_contract():
    rows = 128
    k = 256
    k64_words = k // 64
    semantic_words = torch.empty((rows, k64_words), dtype=torch.uint32)
    for row in range(rows):
        for k64_word in range(k64_words):
            semantic_words[row, k64_word] = row * 1000 + k64_word

    packed = swizzle_blockscaled_chunk_kmajor_scale_words(semantic_words)

    # Current TileLang package keeps two SFA words and two SFB words per lane.
    # OMMA.SF's *_thread_id selects another lane in the same quad, so the
    # effective rows are derived from the selector id, not from the local lane's
    # qlane alone. This is the compact-selector contract used by the production
    # package path in gemm_sm120.h.
    for lane in range(32):
        qlane = lane & 3
        sfa_base_row = _sfa_row_in_lane(lane)
        sfb_base_col = _sfb_col_in_lane(lane)
        for warp_m in range(2):
            for warp_n in range(2):
                for k64_word in range(k64_words):
                    local_sfa_owner = qlane >> 1
                    local_sfb_owner = qlane
                    local_sa0_row = warp_m * 64 + local_sfa_owner * 16 + sfa_base_row
                    local_sa1_row = local_sa0_row + 32
                    local_sb0_row = warp_n * 64 + local_sfb_owner * 8 + sfb_base_col
                    local_sb1_row = local_sb0_row + 32

                    assert _packed_word(packed, local_sa0_row, k64_word) == int(semantic_words[local_sa0_row, k64_word])
                    assert _packed_word(packed, local_sa1_row, k64_word) == int(semantic_words[local_sa1_row, k64_word])
                    assert _packed_word(packed, local_sb0_row, k64_word) == int(semantic_words[local_sb0_row, k64_word])
                    assert _packed_word(packed, local_sb1_row, k64_word) == int(semantic_words[local_sb1_row, k64_word])

                    for mma_i in range(4):
                        effective_sfa_row = _current_compact_sfa_row(lane, warp_m, mma_i)
                        assert _packed_word(packed, effective_sfa_row, k64_word) == int(semantic_words[effective_sfa_row, k64_word])

                    for mma_j in range(4):
                        for half in range(2):
                            effective_sfb_row = _current_compact_sfb_row(lane, warp_n, mma_j, half)
                            assert _packed_word(packed, effective_sfb_row, k64_word) == int(semantic_words[effective_sfb_row, k64_word])


def test_compact_package_issue_table_models_omma_sf_source_lane_selectors():
    issues = _compact_issue_contract(lane=0, warp_m=0, warp_n=0)

    assert len(issues) == 4 * 4 * 2
    assert issues[:4] == [
        {
            "mma_i": 0,
            "mma_j": 0,
            "half": 0,
            "sa_reg": "sa0",
            "sa_tid": 0,
            "sa_source_lane": 0,
            "sfa_row": 0,
            "sb_reg": "sb0",
            "sb_tid": 0,
            "sb_source_lane": 0,
            "sfb_row": 0,
        },
        {
            "mma_i": 0,
            "mma_j": 0,
            "half": 1,
            "sa_reg": "sa0",
            "sa_tid": 0,
            "sa_source_lane": 0,
            "sfa_row": 0,
            "sb_reg": "sb0",
            "sb_tid": 1,
            "sb_source_lane": 1,
            "sfb_row": 8,
        },
        {
            "mma_i": 0,
            "mma_j": 1,
            "half": 0,
            "sa_reg": "sa0",
            "sa_tid": 0,
            "sa_source_lane": 0,
            "sfa_row": 0,
            "sb_reg": "sb0",
            "sb_tid": 2,
            "sb_source_lane": 2,
            "sfb_row": 16,
        },
        {
            "mma_i": 0,
            "mma_j": 1,
            "half": 1,
            "sa_reg": "sa0",
            "sa_tid": 0,
            "sa_source_lane": 0,
            "sfa_row": 0,
            "sb_reg": "sb0",
            "sb_tid": 3,
            "sb_source_lane": 3,
            "sfb_row": 24,
        },
    ]

    assert [issue["sfa_row"] for issue in issues[0::8]] == [0, 16, 32, 48]
    assert [issue["sfb_row"] for issue in issues[:8]] == [0, 8, 16, 24, 32, 40, 48, 56]


def test_compact_package_issue_table_matches_cpp_macro_issue_order():
    issues = _compact_issue_contract(lane=0, warp_m=0, warp_n=0)
    assert [
        (
            issue["mma_i"],
            issue["mma_j"],
            issue["half"],
            issue["sa_reg"],
            issue["sa_tid"],
            issue["sb_reg"],
            issue["sb_tid"],
        )
        for issue in issues
    ] == [
        (0, 0, 0, "sa0", 0, "sb0", 0),
        (0, 0, 1, "sa0", 0, "sb0", 1),
        (0, 1, 0, "sa0", 0, "sb0", 2),
        (0, 1, 1, "sa0", 0, "sb0", 3),
        (0, 2, 0, "sa0", 0, "sb1", 0),
        (0, 2, 1, "sa0", 0, "sb1", 1),
        (0, 3, 0, "sa0", 0, "sb1", 2),
        (0, 3, 1, "sa0", 0, "sb1", 3),
        (1, 0, 0, "sa0", 1, "sb0", 0),
        (1, 0, 1, "sa0", 1, "sb0", 1),
        (1, 1, 0, "sa0", 1, "sb0", 2),
        (1, 1, 1, "sa0", 1, "sb0", 3),
        (1, 2, 0, "sa0", 1, "sb1", 0),
        (1, 2, 1, "sa0", 1, "sb1", 1),
        (1, 3, 0, "sa0", 1, "sb1", 2),
        (1, 3, 1, "sa0", 1, "sb1", 3),
        (2, 0, 0, "sa1", 0, "sb0", 0),
        (2, 0, 1, "sa1", 0, "sb0", 1),
        (2, 1, 0, "sa1", 0, "sb0", 2),
        (2, 1, 1, "sa1", 0, "sb0", 3),
        (2, 2, 0, "sa1", 0, "sb1", 0),
        (2, 2, 1, "sa1", 0, "sb1", 1),
        (2, 3, 0, "sa1", 0, "sb1", 2),
        (2, 3, 1, "sa1", 0, "sb1", 3),
        (3, 0, 0, "sa1", 1, "sb0", 0),
        (3, 0, 1, "sa1", 1, "sb0", 1),
        (3, 1, 0, "sa1", 1, "sb0", 2),
        (3, 1, 1, "sa1", 1, "sb0", 3),
        (3, 2, 0, "sa1", 1, "sb1", 0),
        (3, 2, 1, "sa1", 1, "sb1", 1),
        (3, 3, 0, "sa1", 1, "sb1", 2),
        (3, 3, 1, "sa1", 1, "sb1", 3),
    ]


def test_cpp_package_macro_keeps_compact_selector_issue_order():
    header = Path(__file__).resolve().parents[3] / "src/tl_templates/cuda/gemm_sm120.h"
    source = header.read_text()
    row_macro = source[source.index("#define TL_SM120_PKG_MMA_ROW") : source.index("  TL_SM120_PKG_MMA_ROW(0")]

    position = 0
    for mma_j, half, sb_reg, sb_tid in [
        (0, 0, "scale_pkg.sb0", 0),
        (0, 1, "scale_pkg.sb0", 1),
        (1, 0, "scale_pkg.sb0", 2),
        (1, 1, "scale_pkg.sb0", 3),
        (2, 0, "scale_pkg.sb1", 0),
        (2, 1, "scale_pkg.sb1", 1),
        (3, 0, "scale_pkg.sb1", 2),
        (3, 1, "scale_pkg.sb1", 3),
    ]:
        call = f"TL_SM120_PKG_MMA_N8(I, {mma_j}, {half},"
        position = row_macro.index(call, position)
        selector = f"{sb_reg}, SA_TID, {sb_tid});"
        assert selector in row_macro[position:]

    row_calls = source[source.index("  TL_SM120_PKG_MMA_ROW(0") : source.index("#undef TL_SM120_PKG_MMA_ROW")]
    position = 0
    for mma_i, sa_reg, sa_tid in [
        (0, "scale_pkg.sa0", 0),
        (1, "scale_pkg.sa0", 1),
        (2, "scale_pkg.sa1", 0),
        (3, "scale_pkg.sa1", 1),
    ]:
        call = f"TL_SM120_PKG_MMA_ROW({mma_i},"
        position = row_calls.index(call, position)
        assert sa_reg in row_calls[position:]
        assert f"{sa_tid});" in row_calls[position:]


def test_compact_package_issue_table_matches_effective_row_helpers():
    for lane in range(32):
        for warp_m in range(2):
            for warp_n in range(2):
                for issue in _compact_issue_contract(lane, warp_m, warp_n):
                    assert issue["sfa_row"] == _current_compact_sfa_row(lane, warp_m, issue["mma_i"])
                    assert issue["sfb_row"] == _current_compact_sfb_row(
                        lane,
                        warp_n,
                        issue["mma_j"],
                        issue["half"],
                    )


def test_compact_package_copy_view_contract_maps_issue_rows_to_kmajor_words():
    issues = _compact_copy_view_contract(lane=0, warp_m=0, warp_n=0, k64_word=0)

    assert [
        (
            issue["mma_i"],
            issue["mma_j"],
            issue["half"],
            issue["sfa_row"],
            issue["sfa_word_coord"],
            issue["sfa_word_offset"],
            issue["sfb_row"],
            issue["sfb_word_coord"],
            issue["sfb_word_offset"],
        )
        for issue in issues[:8]
    ] == [
        (0, 0, 0, 0, (0, 0), 0, 0, (0, 0), 0),
        (0, 0, 1, 0, (0, 0), 0, 8, (8, 0), 32),
        (0, 1, 0, 0, (0, 0), 0, 16, (16, 0), 64),
        (0, 1, 1, 0, (0, 0), 0, 24, (24, 0), 96),
        (0, 2, 0, 0, (0, 0), 0, 32, (0, 1), 1),
        (0, 2, 1, 0, (0, 0), 0, 40, (8, 1), 33),
        (0, 3, 0, 0, (0, 0), 0, 48, (16, 1), 65),
        (0, 3, 1, 0, (0, 0), 0, 56, (24, 1), 97),
    ]

    assert [issue["sfa_word_offset"] for issue in issues[0::8]] == [0, 64, 1, 65]


def test_sm120_cuda_helper_keeps_scale_tv_package_boundary():
    header = Path(__file__).resolve().parents[3] / "src/tl_templates/cuda/gemm_sm120.h"
    source = header.read_text()

    assert "struct SM120ScaleTVPackage" in source
    assert "template <class ScalePkg>" in source
    assert "sm120_load_scale_tv_package" in source
    assert "sm120_copy_scale_tv_package(SM120ScaleTVPackage &pkg," in source
    assert "sm120_gemm_fulltile_ab_owner_wide_package(" in source
    assert "const detail::SM120ScaleTVPackage &scale_pkg" in source
    assert (
        "uint32_t sa0, sa1;"
        not in source[source.index("struct SM120FulltileABOwnerWidePackage") : source.index("sm120_copy_fulltile_ab_owner_wide_package")]
    )
    assert "detail::SM120ScaleTVPackage scale_pkg0;" in source
    assert "detail::SM120ScaleTVPackage scale_pkg1;" in source
    assert source.index("detail::sm120_copy_scale_tv_package(scale_pkg0") < source.index(
        "sm120_gemm_fulltile_ab_owner_wide_package(c, pkg0, scale_pkg0)"
    )
    assert "pkg.sa0 =" in source
    assert "pkg.sb1 =" in source


def test_blockscaled_chunk_kmajor_scale_packer_rejects_invalid_shapes():
    # rows that are not a multiple of 128 are zero-padded, not rejected
    padded = pack_blockscaled_chunk_kmajor_scale_bytes(torch.zeros((127, 16), dtype=torch.uint8))
    assert padded.shape == (128, 4)

    with pytest.raises(ValueError, match="K/16 columns multiple of 16"):
        pack_blockscaled_chunk_kmajor_scale_bytes(torch.zeros((128, 12), dtype=torch.uint8))

    with pytest.raises(TypeError, match="torch.uint8"):
        pack_blockscaled_chunk_kmajor_scale_bytes(torch.zeros((128, 16), dtype=torch.int32))


def test_encode_fp4_e2m1_values_and_pack_order():
    values = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0]], dtype=torch.float32)
    codes = encode_fp4_e2m1_values(values)
    assert torch.equal(codes, torch.tensor([[0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x9, 0xA]], dtype=torch.uint8))

    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous().view(torch.int8)
    assert torch.equal(decode_packed_fp4_e2m1(packed), values)


def test_encode_ue4m3_scale_bytes_known_values():
    values = torch.tensor([0.0, 2.0**-9, 2.0**-6, 1.0, 2.0, 448.0], dtype=torch.float32)
    encoded = encode_ue4m3_scale_bytes(values, rounding="nearest")
    assert torch.equal(encoded, torch.tensor([0x00, 0x01, 0x08, 0x38, 0x40, 0x7E], dtype=torch.uint8))
    torch.testing.assert_close(decode_ue4m3_scale_bytes(encoded), values)
    assert torch.isnan(decode_ue4m3_scale_bytes(torch.tensor([0x7F], dtype=torch.uint8))).all()


def test_quantize_nvfp4_blockscaled_bf16_activation_contract():
    rows = 128
    cols = 256
    x = torch.zeros((rows, cols), dtype=torch.bfloat16)
    pattern = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0])
    x[0, :16] = pattern.to(torch.bfloat16)

    packed_fp4, packed_scales, scale_bytes = quantize_bf16_to_nvfp4_blockscaled(x, return_scale_bytes=True)

    assert packed_fp4.shape == (rows, cols // 2)
    assert packed_fp4.dtype == torch.int8
    assert packed_scales.shape == (rows, cols // 64)
    assert packed_scales.dtype == torch.uint32
    assert scale_bytes.shape == (rows, cols // 16)
    assert scale_bytes.dtype == torch.uint8
    assert scale_bytes[0, 0].item() == 0x38
    assert torch.equal(packed_scales, pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes))

    decoded = decode_packed_fp4_e2m1(packed_fp4) * decode_ue4m3_scale_bytes(scale_bytes).repeat_interleave(16, dim=1)
    torch.testing.assert_close(decoded[0, :16], pattern, rtol=0.0, atol=0.0)


def test_quantize_nvfp4_blockscaled_random_bf16_has_bounded_error():
    rows = 128
    cols = 256
    generator = torch.Generator(device="cpu").manual_seed(19)
    x = (torch.randn((rows, cols), generator=generator, dtype=torch.float32) * 2.0).to(torch.bfloat16)

    packed_fp4, scale_source, scale_bytes = quantize_bf16_to_nvfp4_blockscaled(x, return_scale_bytes=True)
    decoded = decode_packed_fp4_e2m1(packed_fp4) * decode_ue4m3_scale_bytes(scale_bytes).repeat_interleave(16, dim=1)

    scale = decode_ue4m3_scale_bytes(scale_bytes).repeat_interleave(16, dim=1)
    error = (decoded - x.to(torch.float32)).abs()
    assert torch.isfinite(decoded).all()
    assert torch.all(error <= scale + 1e-6)
    assert torch.equal(scale_source, pack_blockscaled_chunk_kmajor_scale_bytes(scale_bytes))


def test_quantize_nvfp4_blockscaled_explicit_layout_matches_default():
    rows = 128
    cols = 256
    generator = torch.Generator(device="cpu").manual_seed(23)
    x = (torch.randn((rows, cols), generator=generator, dtype=torch.float32) * 2.0).to(torch.bfloat16)

    default = quantize_bf16_to_nvfp4_blockscaled(x)
    explicit = quantize_bf16_to_nvfp4_blockscaled(x, scale_layout="blockscaled_chunk_kmajor")

    assert torch.equal(explicit[0], default[0])
    assert torch.equal(explicit[1], default[1])


@tilelang_testing.requires_cuda
@tilelang_testing.requires_cuda_compute_version_ge(10, 0)
@pytest.mark.parametrize("rows, cols", [(128, 256), (256, 512)])
def test_tilelang_quantize_nvfp4_blockscaled_matches_reference_layout_and_error_bound(rows, cols):
    generator = torch.Generator(device="cuda").manual_seed(rows + cols)
    x = (torch.randn((rows, cols), generator=generator, device="cuda", dtype=torch.float32) * 2.0).to(torch.bfloat16)

    tilelang_quantize = _load_maint_quantizer()
    packed_tl, scale_source_tl = tilelang_quantize(x)
    _, scale_source_ref, scale_bytes_ref = quantize_bf16_to_nvfp4_blockscaled(x, return_scale_bytes=True)

    assert packed_tl.shape == (rows, cols // 2)
    assert packed_tl.dtype == torch.int8
    assert scale_source_tl.shape == (rows, cols // 64)
    assert scale_source_tl.dtype == torch.uint32
    assert torch.equal(scale_source_tl.cpu(), scale_source_ref.cpu())

    semantic_words = unswizzle_blockscaled_chunk_kmajor_scale_words(scale_source_tl)
    assert torch.equal(swizzle_blockscaled_chunk_kmajor_scale_words(semantic_words).cpu(), scale_source_tl.cpu())

    scale = decode_ue4m3_scale_bytes(scale_bytes_ref).repeat_interleave(16, dim=1)
    decoded = decode_packed_fp4_e2m1(packed_tl) * scale
    error = (decoded - x.to(torch.float32)).abs()
    assert torch.isfinite(decoded).all()
    assert torch.all(error <= scale + 1e-6)


def test_swizzle_blockscaled_chunk_kmajor_pads_rows_to_full_tiles():
    rows, cols = 130, 8
    generator = torch.Generator(device="cpu").manual_seed(23)
    words = torch.randint(0, 2**31, (rows, cols), generator=generator, dtype=torch.int64).to(torch.uint32)

    swizzled = swizzle_blockscaled_chunk_kmajor_scale_words(words)
    assert swizzled.shape == (256, cols)

    back = unswizzle_blockscaled_chunk_kmajor_scale_words(swizzled)
    assert torch.equal(back[:rows], words)
    assert bool((back[rows:] == 0).all())

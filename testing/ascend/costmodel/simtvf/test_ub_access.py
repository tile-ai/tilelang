"""DAV-3510 normal SIMT UB geometry tests; no device or CANN required."""

import pytest
from tilelang import tvm


def analyze(addresses, width=4, store=False):
    function = tvm.get_global_func("tl.ascend.AnalyzeSimtUbInstruction")
    result = function([-1 if x is None else x for x in addresses], width, store)
    return {str(k): int(v) for k, v in result.items()}


@pytest.mark.parametrize(
    "width,stride,entries,partial,requests",
    [
        (4, 4, 1, 0, 1),
        (8, 8, 4, 0, 2),
        (2, 2, 1, 0, 1),
        (1, 1, 1, 0, 1),
        (2, 4, 1, 1, 1),
        (4, 128, 32, 0, 1),
        (4, 512, 32, 0, 1),
        (4, 0, 1, 0, 1),
        (8, 0, 2, 0, 2),
    ],
)
def test_geometry(width, stride, entries, partial, requests):
    addresses = [lane * stride for lane in range(32)]
    assert analyze(addresses, width, True) == dict(
        instructions=1,
        requests=requests,
        entries=entries,
        partial=partial,
    )
    load = analyze(addresses, width, False)
    assert load["partial"] == 0
    assert load["entries"] == entries


def test_masked_halfword_store():
    addresses = [2 * lane if lane % 2 == 0 else None for lane in range(32)]
    assert analyze(addresses, 2, True) == dict(
        instructions=1,
        requests=1,
        entries=1,
        partial=1,
    )


def test_empty_warp_has_no_requests():
    assert analyze([None] * 32, 8, True) == dict(
        instructions=1,
        requests=0,
        entries=0,
        partial=0,
    )


def test_full_line_identity_is_not_just_key():
    # 0 and 512 have the same key/group but must use different entries.
    addresses = [None] * 30 + [0, 512]
    result = analyze(addresses)
    assert result["entries"] == 2
    addresses[-1] = 4  # Same 8B line can coalesce.
    result = analyze(addresses)
    assert result["entries"] == 1


def test_repeated_elements_still_consume_capacity():
    # An odd 4B span occupies three 2B-aligned elements; 32 repeated spans
    # exceed 64 element slots even though they touch only one full line.
    result = analyze([1] * 32, 4, True)
    assert result["entries"] == 2
    assert result["partial"] == 2


@pytest.mark.parametrize("shift", [8, 128, 512, 4096])
def test_common_aligned_translation_preserves_geometry(shift):
    addresses = [None if i % 3 == 0 else i * 130 + i % 2 for i in range(32)]
    translated = [None if x is None else x + shift for x in addresses]
    assert analyze(addresses, 8, True) == analyze(translated, 8, True)


def test_b64_slices_are_independent_but_one_instruction():
    addresses = [None if i % 5 == 0 else i * 520 for i in range(32)]
    wide = analyze(addresses, 8, True)
    low = analyze(addresses, 4, True)
    high = analyze([None if x is None else x + 4 for x in addresses], 4, True)
    assert wide["instructions"] == 1
    for key in ("requests", "entries", "partial"):
        assert wide[key] == low[key] + high[key]


@pytest.mark.parametrize("width", [0, 3, 16, -1])
def test_unsupported_machine_width_is_not_free(width):
    with pytest.raises(Exception, match="Unsupported SIMT UB"):
        analyze([0] * 32, width)


def test_address_overflow_is_rejected():
    with pytest.raises(Exception, match="Unsupported SIMT UB"):
        analyze([(1 << 63) - 1] * 32, 8)


def latency(addresses, width=4, store=False):
    function = tvm.get_global_func("tl.ascend.EstimateSimtUbInstructionLatency")
    return float(function([-1 if x is None else x for x in addresses], width, store))


@pytest.mark.parametrize("store", [False, True])
@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_supported_memory_instructions_have_finite_latency(store, width):
    import math

    value = latency([width * i for i in range(32)], width, store)
    assert math.isfinite(value) and value > 0


def test_inactive_memory_access_has_zero_latency():
    assert latency([None] * 32) == 0


@pytest.mark.parametrize("store", [False, True])
def test_entries_pipeline_their_service_instead_of_repeating_base(store):
    contiguous = [4 * i for i in range(32)]
    strided = [8 * i for i in range(32)]
    # Two entries need one more publishing cycle, not another instruction's
    # whole service path. This asserts the ideal one-entry-per-cycle mechanism.
    assert analyze(strided)["entries"] == 2
    assert latency(strided, store=store) == latency(contiguous, store=store) + 1


def test_partial_store_latency_is_unknown_but_geometry_is_available():
    addresses = [4 * i for i in range(32)]
    assert analyze(addresses, 2, True)["partial"] == 1
    assert latency(addresses, 2, True) == -1
    assert latency(addresses, 4, True) > 0


def test_b64_pays_instruction_base_once():
    addresses = [8 * i for i in range(32)]
    assert latency(addresses, 8) < latency(addresses, 4) + latency([x + 4 for x in addresses], 4)


@pytest.mark.parametrize("store", [False, True])
def test_b64_subrequests_publish_without_an_extra_pipeline_restart(store):
    # Broadcast b64 has one entry in each of its two independent requests.
    assert analyze([0] * 32, 8, store)["entries"] == 2
    assert latency([0] * 32, 8, store) == latency([0] * 32, 4, store) + 1


def batch(instructions, store=False):
    """Each tuple is (lane addresses, machine width, independent instances)."""
    function = tvm.get_global_func("tl.ascend.EstimateSimtUbAccessLatency")
    result = function(
        [[-1 if x is None else x for x in addresses] for addresses, _, _ in instructions],
        [width for _, width, _ in instructions],
        [count for _, _, count in instructions],
        store,
    )
    return {str(k): v.value for k, v in result.items()}


@pytest.mark.parametrize("store", [False, True])
def test_independent_instructions_overlap_completion_paths(store):
    addresses = [4 * lane for lane in range(32)]
    single = batch([(addresses, 4, 1)], store)
    many = batch([(addresses, 4, 16)], store)
    assert single["estimated_latency"] == latency(addresses, store=store)
    assert many["service_cycles"] == 16 * single["service_cycles"]
    assert many["estimated_latency"] == single["estimated_latency"] + 15 * single["service_cycles"]
    assert many["estimated_latency"] < 16 * single["estimated_latency"]


@pytest.mark.parametrize("store", [False, True])
def test_equal_work_narrow_access_can_beat_b64(store):
    # Same 256 bytes: one b64 warp versus two independent b32 warps.
    wide = batch([([8 * lane for lane in range(32)], 8, 1)], store)
    narrow = batch([([4 * lane for lane in range(32)], 4, 2)], store)
    assert narrow["instructions"] == 2 * wide["instructions"]
    assert narrow["service_cycles"] < wide["service_cycles"]
    assert narrow["estimated_latency"] < wide["estimated_latency"]


@pytest.mark.parametrize("store", [False, True])
def test_independent_period_compression_preserves_pipeline_cost(store):
    a = ([8 * lane for lane in range(32)], 8, 1)
    b = ([4 * lane for lane in range(32)], 4, 1)
    expanded = batch([a, b] * 8, store)
    compressed = batch([(a[0], a[1], 8), (b[0], b[1], 8)], store)
    assert compressed == expanded


def test_empty_and_inactive_batches_do_not_add_a_pipeline_tail():
    assert batch([])["estimated_latency"] == 0
    assert batch([([None] * 32, 4, 16)])["service_cycles"] == 0
    active = ([4 * lane for lane in range(32)], 4, 1)
    assert batch([active, ([None] * 32, 4, 16)])["estimated_latency"] == batch([active])["estimated_latency"]


@pytest.mark.parametrize("count", [0, -1, (1 << 63) - 1])
def test_invalid_or_overflowing_batch_count_is_rejected(count):
    with pytest.raises(Exception, match="Unsupported SIMT UB batch"):
        batch([([8 * lane for lane in range(32)], 8, count)])


def test_partial_store_cannot_be_priced_as_a_full_store_batch():
    with pytest.raises(Exception, match="Unsupported SIMT UB batch"):
        batch([([4 * lane for lane in range(32)], 2, 8)], store=True)


@pytest.mark.parametrize("width", [0, 3, 16, (1 << 32) + 4])
def test_invalid_batch_width_is_not_truncated(width):
    with pytest.raises(Exception, match="Unsupported SIMT UB batch"):
        batch([([4 * lane for lane in range(32)], width, 1)])

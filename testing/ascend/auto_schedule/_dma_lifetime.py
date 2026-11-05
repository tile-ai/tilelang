"""Check finite scalar/MTE3 reuse against the actual generated protocol."""

import tilelang
from tilelang.ascend import transform
from testing.ascend._ir import annotations, schedule
from testing.ascend.auto_schedule._dma_lifetime_events import extract
from testing.ascend.auto_schedule._dma_lifetime_order import prove


def certificate(before, after):
    """Require unchanged accesses, complete readers and a valid flag protocol.

    This checker covers only fixed, small, zero-stage loops on the common
    AIV core mask, scalar accesses and UB-to-GM MTE3 copies. The extractor
    rejects unknown domains, runtime guards, stages, cores or operations.
    """
    before_events, after_events = extract(before), extract(after)
    before_memory = [event for event in before_events if event["kind"] in ("scalar", "copy")]
    after_memory = [event for event in after_events if event["kind"] in ("scalar", "copy")]
    assert before_memory == after_memory, "InsertSync must preserve all accesses and loop coordinates"
    report = prove(after_events)
    assert report["safe"], {key: report[key] for key in ("cycle", "token_errors", "read_errors", "unordered_pairs")}
    return report


def assert_reuse_preserves_dma_readers(program, reader_counts):
    """Allow a/b allocation aliasing only with a proof from this exact pass.

    Rejecting reuse remains valid. Distinct global parameters can alias,
    however, and their dependency can force an earlier MTE3-to-S acquire
    that finishes every reader before the next allocation is overwritten.
    """
    with tilelang.transform.PassContext(config={"tl.disable_shared_memory_reuse": False}):
        before = schedule(program, through="ResolveCore", auto_schedule=False)
        after = transform.InsertSync()(before)
    pairs = {
        tuple(sorted((storage.name, other.name)))
        for fields in annotations(after)
        for storage, compatible in fields.get("tl.buffer_alias_map", {}).items()
        for other in compatible
    }
    if ("a", "b") in pairs:
        report = certificate(before, after)
        assert report["reader_counts"] == reader_counts

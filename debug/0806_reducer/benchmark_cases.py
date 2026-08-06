"""Benchmark matrix for the legacy reducer and reducer v2 comparison.

This module intentionally has no TileLang imports.  The benchmark driver and
the two isolated worker processes can therefore share exactly the same case
definitions without loading either TileLang checkout into the wrong process.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BenchmarkCase:
    """One reducer benchmark configuration."""

    name: str
    family: str
    blocks: int
    m: int
    k: int
    threads: int
    legacy_replication: str
    description: str
    suites: tuple[str, ...] = ("full",)
    tile_k: int = 0
    num_stages: int = 0
    batch: int = 1
    expected_legacy_correct: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        result = asdict(self)
        result["suites"] = list(self.suites)
        return result


CASES: tuple[BenchmarkCase, ...] = (
    # Unique-owner cases expose the most obvious optimization missing from the
    # v2 full-array baseline.  The legacy `replication="none"` strategy can
    # distribute outputs and finalize without a collective.
    BenchmarkCase(
        name="unique_owner_m8_b512_t128",
        family="unique_owner",
        blocks=512,
        m=8,
        k=1,
        threads=128,
        legacy_replication="none",
        description="Eight independent outputs; no cross-thread reduction is required.",
        suites=("quick", "full"),
    ),
    BenchmarkCase(
        name="unique_owner_m32_b512_t128",
        family="unique_owner",
        blocks=512,
        m=32,
        k=1,
        threads=128,
        legacy_replication="none",
        description="Thirty-two independent outputs with a unique logical owner.",
        suites=("quick", "full"),
    ),
    BenchmarkCase(
        name="unique_owner_m128_b256_t128",
        family="unique_owner",
        blocks=256,
        m=128,
        k=1,
        threads=128,
        legacy_replication="none",
        description="One output per thread; legacy finalization should require no collective.",
        suites=("full",),
    ),
    # This is the exact contribution-multiplicity stress pattern from #2408.
    # The legacy result is expected to be wrong, so its latency is not used for
    # a speedup unless --benchmark-incorrect is explicitly requested.
    BenchmarkCase(
        name="replica_stress_m1_k8_b512_t128",
        family="row_reduce",
        blocks=512,
        m=1,
        k=8,
        threads=128,
        legacy_replication="all",
        description="BD=8/threads=128 replicated-contribution wrong-code stress case.",
        suites=("quick", "full", "diagnostic"),
        expected_legacy_correct=False,
    ),
    # Cross-thread reductions compare the old fully replicated reducer with
    # the v2 canonical full-participant fallback on cases where the old result
    # is expected to be numerically valid.
    BenchmarkCase(
        name="row_reduce_m1_k128_b512_t128",
        family="row_reduce",
        blocks=512,
        m=1,
        k=128,
        threads=128,
        legacy_replication="all",
        description="One scalar output with one logical reduction value per thread.",
        suites=("quick", "full"),
    ),
    BenchmarkCase(
        name="row_reduce_m8_k128_b256_t128",
        family="row_reduce",
        blocks=256,
        m=8,
        k=128,
        threads=128,
        legacy_replication="all",
        description="Eight output rows reduced across a 128-element axis.",
        suites=("full",),
    ),
    BenchmarkCase(
        name="row_reduce_m32_k128_b128_t128",
        family="row_reduce",
        blocks=128,
        m=32,
        k=128,
        threads=128,
        legacy_replication="all",
        description="Moderate output array; stresses v2 full-array initialization/finalization.",
        suites=("quick", "full"),
    ),
    BenchmarkCase(
        name="row_reduce_m128_k64_b64_t256",
        family="row_reduce",
        blocks=64,
        m=128,
        k=64,
        threads=256,
        legacy_replication="all",
        description="Large logical output array and a 256-thread participant range.",
        suites=("full",),
    ),
    # Streaming GEMV keeps the reducer epoch alive across several K tiles.  It
    # measures the intended deferred-reduction use case rather than only the
    # final collective in isolation.
    BenchmarkCase(
        name="streaming_gemv_m32_k512_tk64_b128_t128",
        family="streaming_gemv",
        blocks=128,
        m=32,
        k=512,
        tile_k=64,
        num_stages=2,
        threads=128,
        legacy_replication="all",
        description="Eight tiled updates per epoch with a 32-element output vector.",
        suites=("quick", "full"),
    ),
    BenchmarkCase(
        name="streaming_gemv_m128_k1024_tk64_b32_t256",
        family="streaming_gemv",
        blocks=32,
        m=128,
        k=1024,
        tile_k=64,
        num_stages=2,
        threads=256,
        legacy_replication="all",
        description="Large deferred GEMV epoch; exposes register pressure and scalar-finalize cost.",
        suites=("full",),
    ),
    # The legacy batched path is retained as a diagnostic case because it is
    # known to have correctness problems.  It is deliberately outside the
    # quick/full performance suites.
    BenchmarkCase(
        name="legacy_batch4_m128_k64_b64_t256",
        family="row_reduce",
        blocks=64,
        m=128,
        k=64,
        threads=256,
        legacy_replication="all",
        batch=4,
        description="Diagnostic for legacy run_batch versus the v2 scalar batch-hint fallback.",
        suites=("diagnostic",),
        expected_legacy_correct=False,
    ),
)


CASE_BY_NAME = {case.name: case for case in CASES}


def select_cases(suite: str) -> list[BenchmarkCase]:
    """Select cases belonging to a named suite."""

    if suite == "all":
        return list(CASES)
    return [case for case in CASES if suite in case.suites]

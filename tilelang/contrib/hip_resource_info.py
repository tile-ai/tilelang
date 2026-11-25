"""Parse AMD GPU per-kernel resource usage out of clang's
``-Rpass-analysis=kernel-resource-usage`` remarks and expose them on
JITKernel.

clang emits a block like::
    remark: src.cc:9:0: Function Name: main_kernel [-Rpass-analysis=kernel-resource-usage]
    remark: src.cc:9:0:     TotalSGPRs: 16 [-Rpass-analysis=kernel-resource-usage]
    remark: src.cc:9:0:     VGPRs: 5 [-Rpass-analysis=kernel-resource-usage]
    remark: src.cc:9:0:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
    remark: src.cc:9:0:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
    remark: src.cc:9:0:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
    ...

right alongside any real warnings/errors. We *parse and strip* those
lines before printing or raising, so autotune logs don't drown in
hundreds of remark blocks while real diagnostics still surface.
"""

from __future__ import annotations

import contextlib
import json
import re
import threading
from dataclasses import asdict, dataclass, field

# A line is a kernel-resource-usage remark iff it carries this exact tag.
# clang appends the option name as ``[-Rpass-analysis=kernel-resource-usage]``.
_REMARK_TAG = "[-Rpass-analysis=kernel-resource-usage]"
# clang prints `<path>:<line>:<col>: remark:     <key>: <value> [-Rpass-...]`
_REMARK_LINE_RE = re.compile(r"\bremark:\s*(?P<key>[^:]+?):\s*(?P<value>.*?)\s*" + re.escape(_REMARK_TAG) + r"\s*$")


@dataclass
class KernelResourceUsage:
    """Resource counts as reported by clang's kernel-resource-usage pass.

    Field names mirror the remark labels (lower-cased, normalized) so we
    can extend without breaking callers.
    """

    n_regs: int = 0  # VGPRs
    # Total VGPR-equivalent spill pressure: the explicit `VGPRs Spill` count
    # plus scratch memory in dwords (`ScratchSize [bytes/lane]` / 4). Matches
    # how triton accounts for spills (its n_spills is scratch_bytes / 4) but
    # also folds in clang's explicit spill count when present.
    n_spills: int = 0
    scratch_bytes: int = 0  # raw `ScratchSize [bytes/lane]`
    n_max_threads: int | None = None  # not in remarks; kept for API symmetry
    extra: dict[str, str] = field(default_factory=dict)  # raw remark key→value


_FLAG = "-Rpass-analysis=kernel-resource-usage"

_RECORDER = threading.local()


def hipcc_remark_flag() -> str:
    """The clang flag callers should pass to hipcc to enable the remark
    output we parse here."""
    return _FLAG


def reset_recorder() -> None:
    """Begin a fresh recording window on this thread."""
    _RECORDER.items = {}


def pop_recorded() -> dict[str, KernelResourceUsage]:
    """Return everything recorded since the last ``reset_recorder`` and
    clear the buffer."""
    items = getattr(_RECORDER, "items", {})
    _RECORDER.items = {}
    return dict(items)


def filter_and_record(output: str) -> str:
    """Strip kernel-resource-usage remarks from ``output``, parse them,
    and append the parsed entries to the active recorder (if any).
    Returns the filtered output with the remark lines removed."""
    if _REMARK_TAG not in output:
        return output

    kept_lines: list[str] = []
    current_name: str | None = None
    current: KernelResourceUsage | None = None
    items = getattr(_RECORDER, "items", None)

    for line in output.splitlines(keepends=True):
        m = _REMARK_LINE_RE.search(line.rstrip("\n").rstrip("\r"))
        if m is None:
            kept_lines.append(line)
            continue
        key = m.group("key").strip()
        value = m.group("value").strip()
        if key == "Function Name":
            # finalize previous block
            if items is not None and current_name and current is not None:
                items[current_name] = current
            current_name = value
            current = KernelResourceUsage()
        elif current is not None:
            current.extra[key] = value
            if key == "VGPRs":
                with contextlib.suppress(ValueError):
                    current.n_regs = int(value)
            elif key == "VGPRs Spill":
                with contextlib.suppress(ValueError):
                    current.n_spills += int(value)
            elif key.startswith("ScratchSize"):
                # ScratchSize [bytes/lane] — fold scratch dwords into n_spills.
                with contextlib.suppress(ValueError):
                    current.scratch_bytes = int(value)
                    current.n_spills += int(value) // 4
        # remark line is dropped (not added to kept_lines)

    if items is not None and current_name and current is not None:
        items[current_name] = current

    return "".join(kept_lines)


@dataclass
class OccupancyReport:
    """How many blocks of a kernel fit on one compute unit, and what limits it."""

    # Waves per SIMD as the compiler reports it: the least of what the registers,
    # the work-group size and the LDS declared statically in the kernel allow.
    occupancy: int
    waves_per_block: int
    blocks_by_occupancy: int
    blocks_by_lds: int | None  # None when the LDS size per block is not given
    blocks_per_cu: int
    warnings: list[str] = field(default_factory=list)


def occupancy_report(
    usage: KernelResourceUsage,
    threads: int,
    lds_bytes: int = 0,
    wave_size: int = 64,
    simds_per_cu: int = 4,
    lds_bytes_per_cu: int = 65536,
    lds_granularity: int = 512,
) -> OccupancyReport:
    """Blocks per compute unit from the compiler's occupancy and the LDS size.

    A block needs ``ceil(threads / wave_size)`` wavefronts, a compute unit holds
    ``occupancy * simds_per_cu`` of them, and the blocks on it share the LDS,
    which is allocated in units of ``lds_granularity`` bytes. The compiler's
    occupancy already accounts for LDS it can see, but not for dynamic shared
    memory, so pass the full size per block as ``lds_bytes``.
    The defaults describe CDNA, including MI200's 64 KB of LDS per compute unit
    in units of 128 dwords.
    """
    occupancy = int(usage.extra["Occupancy [waves/SIMD]"])
    waves_per_block = (threads + wave_size - 1) // wave_size
    by_occupancy = occupancy * simds_per_cu // waves_per_block
    by_lds = None
    if lds_bytes > 0:
        lds_allocated = (lds_bytes + lds_granularity - 1) // lds_granularity * lds_granularity
        by_lds = lds_bytes_per_cu // lds_allocated
    blocks = by_occupancy if by_lds is None else min(by_occupancy, by_lds)
    warnings = []
    if usage.scratch_bytes > 0:
        warnings.append(f"kernel spills to scratch ({usage.scratch_bytes} bytes/lane), which is usually slow")
    if by_occupancy == 0:
        warnings.append(f"occupancy {occupancy} does not fit a block of {waves_per_block} wavefronts on one compute unit")
    elif by_lds is not None and by_occupancy < by_lds:
        warnings.append(
            f"the compiler's occupancy limits this kernel to {by_occupancy} blocks per compute unit while LDS allows {by_lds}; "
            "if registers set that occupancy, see T.annotate_min_blocks_per_sm"
        )
    return OccupancyReport(occupancy, waves_per_block, by_occupancy, by_lds, blocks, warnings)


def dump_to_file(usage: dict[str, KernelResourceUsage], path: str) -> None:
    """Persist parsed resource usage so it survives kernel-cache hits."""
    data = {name: asdict(u) for name, u in usage.items()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_from_file(path: str) -> dict[str, KernelResourceUsage]:
    """Inverse of ``dump_to_file``. Tolerant of missing / unknown fields
    so older cache entries keep working when the dataclass evolves."""
    with open(path) as f:
        data = json.load(f)
    out: dict[str, KernelResourceUsage] = {}
    for name, entry in data.items():
        out[name] = KernelResourceUsage(
            n_regs=int(entry.get("n_regs", 0)),
            n_spills=int(entry.get("n_spills", 0)),
            scratch_bytes=int(entry.get("scratch_bytes", 0)),
            n_max_threads=entry.get("n_max_threads"),
            extra=dict(entry.get("extra", {})),
        )
    return out

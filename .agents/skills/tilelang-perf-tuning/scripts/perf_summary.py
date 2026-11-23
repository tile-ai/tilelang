# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Archive msprof op CSV output and summarize one TileLang kernel.

Usage:
    python3 perf_summary.py <OPPROF_dir> <variant_output_dir> \
        --kernel-name <expected_kernel> [--round-name <name>]

The script copies the unmodified CSV files to
<variant_output_dir>/docs/perf/<round>/, then generates a summary from only
the requested kernel. AIC and AIV data are summarized independently when a
TileLang kernel uses both core types.
"""

import argparse
import csv
import glob
import math
import os
import re
import shutil
import statistics
import tempfile
from typing import Any
from collections.abc import Iterable, Sequence


KERNEL_NAME_FIELDS = (
    "Op Name",
    "OpName",
    "Kernel Name",
    "KernelName",
    "op_name",
    "kernel_name",
    "Task Name",
    "task_name",
)

CSV_NAMES = (
    "OpBasicInfo.csv",
    "PipeUtilization.csv",
    "ArithmeticUtilization.csv",
    "Memory.csv",
    "MemoryL0.csv",
    "MemoryUB.csv",
    "L2Cache.csv",
    "ResourceConflictRatio.csv",
)


def optional_float(value: Any) -> float | None:
    """Return a finite float, or None when the value is missing or invalid."""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "N/A", "NA", "-"):
        return None
    try:
        parsed = float(text)
    except (ValueError, TypeError):
        return None
    return parsed if math.isfinite(parsed) else None


def required_positive_float(value: Any, field: str, source: str) -> float:
    """Parse a required timing field instead of defaulting it to zero."""
    parsed = optional_float(value)
    if parsed is None or parsed <= 0:
        raise ValueError(f"{source}: required field '{field}' must be a positive finite number, got {value!r}")
    return parsed


def required_positive_int(value: Any, field: str, source: str) -> int:
    parsed = required_positive_float(value, field, source)
    if not parsed.is_integer():
        raise ValueError(f"{source}: required field '{field}' must be an integer, got {value!r}")
    return int(parsed)


def metric_values(rows: Sequence[dict[str, str]], field: str, multiply: float = 1.0) -> list[float]:
    """Return the finite values actually present in a column; missing entries are skipped."""
    values = []
    for row in rows:
        parsed = optional_float(row.get(field))
        if parsed is not None:
            values.append(parsed * multiply)
    return values


def present_field(rows: Sequence[dict[str, str]], candidates: Sequence[str]) -> str | None:
    """Return the first candidate column that this export actually provides."""
    for field in candidates:
        if any(field in row for row in rows):
            return field
    return None


def metric_series(
    rows: Sequence[dict[str, str]],
    candidates: Sequence[str],
    multiply: float = 1.0,
) -> tuple[list[float], str | None]:
    """Read the first supported column and report which field was used."""
    field = present_field(rows, candidates)
    if field is None:
        return [], None
    return metric_values(rows, field, multiply), field


def paired_ratio(
    rows: Sequence[dict[str, str]],
    numerator_field: str,
    denominator_field: str,
) -> tuple[float, int] | None:
    """Average a ratio over the rows that carry both measurements."""
    numerator = 0.0
    denominator = 0.0
    pairs = 0
    for row in rows:
        top = optional_float(row.get(numerator_field))
        bottom = optional_float(row.get(denominator_field))
        if top is None or bottom is None or bottom <= 0:
            continue
        numerator += top
        denominator += bottom
        pairs += 1
    if not pairs:
        return None
    return numerator / denominator, pairs


def read_csv(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as file:
        rows = []
        for row in csv.DictReader(file):
            rows.append(
                {str(key).strip(): value.strip() if isinstance(value, str) else value for key, value in row.items() if key is not None}
            )
        return rows


def metric_csv_path(opprof_dir: str, csv_name: str) -> str:
    """Resolve a fixed CSV name or one timestamp-suffixed CSV."""
    exact = os.path.join(opprof_dir, csv_name)
    if os.path.isfile(exact):
        return exact
    stem, extension = os.path.splitext(csv_name)
    matches = glob.glob(os.path.join(glob.escape(opprof_dir), f"{stem}_*{extension}"))
    if len(matches) != 1:
        raise ValueError(f"expected one {csv_name} in {opprof_dir}, got {len(matches)}")
    return matches[0]


def stat_line(
    name: str,
    values: Sequence[float],
    fmt: str = ".1f",
    unit: str = "",
    multiply: float = 1.0,
) -> str | None:
    """Generate a min/avg/max line, omitting an all-zero metric."""
    scaled = [value * multiply for value in values]
    if not scaled or all(abs(value) < 0.001 for value in scaled):
        return None
    suffix = f"    ({unit})" if unit else ""
    return f"{'  ' + name:<24s} {min(scaled):>10{fmt}} {statistics.mean(scaled):>10{fmt}} {max(scaled):>10{fmt}}{suffix}"


def row_kernel_name(row: dict[str, str]) -> str | None:
    for field in KERNEL_NAME_FIELDS:
        raw_value = row.get(field)
        value = raw_value.strip() if isinstance(raw_value, str) else ""
        if value:
            return value
    return None


def unique_kernel_names(rows: Iterable[dict[str, str]]) -> list[str]:
    return sorted({name for row in rows if (name := row_kernel_name(row))})


def unique_match(expected: str, names: Sequence[str], source: str) -> str:
    """Resolve expected to exactly one runtime kernel name."""
    exact = [name for name in names if name == expected]
    if exact:
        return exact[0]

    folded_expected = expected.casefold()
    casefold_exact = [name for name in names if name.casefold() == folded_expected]
    if len(casefold_exact) == 1:
        return casefold_exact[0]

    contains = [name for name in names if folded_expected in name.casefold()]
    if len(contains) == 1:
        return contains[0]
    if not contains:
        raise ValueError(f"{source}: kernel '{expected}' not found; available kernels: {', '.join(names) if names else '(none)'}")
    raise ValueError(f"{source}: kernel '{expected}' is ambiguous; matches: {', '.join(contains)}")


def match_resolved_kernel(
    resolved: str,
    names: Sequence[str],
    source: str,
    basic_names: Sequence[str] | None = None,
) -> str:
    """Require the resolved kernel identity in each named table."""
    if resolved in names:
        return resolved
    casefold = [name for name in names if name.casefold() == resolved.casefold()]
    if len(casefold) == 1:
        candidate = casefold[0]
        if basic_names is not None and candidate in basic_names and candidate != resolved:
            raise ValueError(
                f"{source}: resolved kernel '{resolved}' is a distinct kernel from '{candidate}' in OpBasicInfo.csv; "
                "refusing to accept a case-normalized name that maps to a different identity"
            )
        return candidate
    raise ValueError(
        f"{source}: resolved kernel '{resolved}' missing; available kernels: {', '.join(names)}. "
        "Refusing to attribute a different kernel's metrics"
    )


def filter_rows_for_kernel(
    rows: list[dict[str, str]],
    resolved: str,
    source: str,
    raw_kernel_count: int,
    basic_names: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Filter a CSV by kernel name when it exposes a supported name column."""
    if not rows:
        return []
    names = unique_kernel_names(rows)
    if not names:
        if raw_kernel_count > 1:
            raise ValueError(
                f"{source}: no kernel-name column, but OpBasicInfo.csv contains {raw_kernel_count} kernels; refusing to mix their metrics"
            )
        if basic_names is not None and any(name.casefold() != resolved.casefold() for name in basic_names):
            raise ValueError(
                f"{source}: no kernel-name column and OpBasicInfo.csv cannot establish a single kernel identity; "
                "refusing the unnamed-table fallback"
            )
        return rows

    selected_name = match_resolved_kernel(resolved, names, source, basic_names)
    filtered = [row for row in rows if row_kernel_name(row) == selected_name]
    if not filtered:
        raise ValueError(f"{source}: no rows remain after selecting kernel '{selected_name}'")
    return filtered


def load_kernel_tables(opprof_dir: str, expected: str) -> tuple[dict[str, list[dict[str, str]]], str]:
    basic_name = "OpBasicInfo.csv"
    basic_rows = read_csv(metric_csv_path(opprof_dir, basic_name))
    if not basic_rows:
        raise ValueError(f"{basic_name} is missing or empty")
    unnamed = [index for index, row in enumerate(basic_rows) if not row_kernel_name(row)]
    if unnamed:
        raise ValueError(
            f"{basic_name}: rows {unnamed[:5]} have no kernel name; refusing to infer kernel identity from incomplete basic rows"
        )
    basic_kernels = unique_kernel_names(basic_rows)
    if not basic_kernels:
        raise ValueError(f"{basic_name} has no supported kernel-name column; expected one of: {', '.join(KERNEL_NAME_FIELDS)}")

    resolved = unique_match(expected, basic_kernels, basic_name)
    raw_kernel_count = len(basic_kernels)
    tables: dict[str, list[dict[str, str]]] = {}
    for csv_name in CSV_NAMES:
        rows = read_csv(metric_csv_path(opprof_dir, csv_name))
        tables[csv_name] = filter_rows_for_kernel(rows, resolved, csv_name, raw_kernel_count, basic_kernels)
    return tables, resolved


def active_core_prefixes(rows: Sequence[dict[str, str]]) -> list[str]:
    """Return every active core type instead of collapsing mixed AIC/AIV data."""
    active = []
    for prefix in ("aiv", "aic"):
        values = [
            parsed
            for row in rows
            for field, value in row.items()
            if field.startswith(f"{prefix}_") and (parsed := optional_float(value)) is not None
        ]
        if any(value > 0 for value in values):
            active.append(prefix)
    return active


def core_rows(rows: Sequence[dict[str, str]], prefix: str) -> list[dict[str, str]]:
    time_field = f"{prefix}_time(us)"
    return [row for row in rows if (value := optional_float(row.get(time_field))) is not None and value > 0]


def find_next_round(perf_dir: str) -> str:
    if not os.path.exists(perf_dir):
        return os.path.join(perf_dir, "round_001")
    existing = [name for name in os.listdir(perf_dir) if re.fullmatch(r"round_\d+", name)]
    if not existing:
        return os.path.join(perf_dir, "round_001")
    numbers = [int(name.split("_")[1]) for name in existing]
    return os.path.join(perf_dir, f"round_{max(numbers) + 1:03d}")


def publish_round(opprof_dir: str, round_dir: str, summary: str) -> list[str]:
    """Stage the CSVs and summary, then publish the round atomically."""
    parent = os.path.dirname(round_dir)
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(round_dir):
        raise ValueError(f"archive directory already exists: {round_dir}")
    staging = tempfile.mkdtemp(prefix=f".{os.path.basename(round_dir)}.staging-", dir=parent)
    try:
        copied = []
        for csv_name in CSV_NAMES:
            shutil.copy2(metric_csv_path(opprof_dir, csv_name), os.path.join(staging, csv_name))
            copied.append(csv_name)
        with open(os.path.join(staging, "summary.txt"), "w", encoding="utf-8") as file:
            file.write(summary)
        if os.path.exists(round_dir):
            raise ValueError(f"archive directory already exists: {round_dir}")
        os.rename(staging, round_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return copied


def append_ratio_and_bandwidth(lines: list[str], rows: Sequence[dict[str, str]], prefix: str) -> None:
    if prefix == "aiv":
        ratio_fields = [
            ("vec_ratio%", "aiv_vec_ratio"),
            ("scalar_ratio%", "aiv_scalar_ratio"),
            ("mte2_ratio%", "aiv_mte2_ratio"),
            ("mte3_ratio%", "aiv_mte3_ratio"),
            ("icache_miss%", "aiv_icache_miss_rate"),
        ]
        bandwidth_fields = [
            ("mte2_active_bw", "aiv_mte2_active_bw(GB/s)"),
            ("mte3_active_bw", "aiv_mte3_active_bw(GB/s)"),
        ]
    else:
        ratio_fields = [
            ("cube_ratio%", "aic_cube_ratio"),
            ("scalar_ratio%", "aic_scalar_ratio"),
            ("mte1_ratio%", "aic_mte1_ratio"),
            ("mte2_ratio%", "aic_mte2_ratio"),
            ("mte3_ratio%", "aic_mte3_ratio"),
            ("fixpipe_ratio%", "aic_fixpipe_ratio"),
            ("icache_miss%", "aic_icache_miss_rate"),
        ]
        bandwidth_fields = [
            ("mte1_active_bw", "aic_mte1_active_bw(GB/s)"),
            ("mte2_active_bw", "aic_mte2_active_bw(GB/s)"),
            ("mte3_active_bw", "aic_mte3_active_bw(GB/s)"),
            ("fixpipe_active_bw", "aic_fixpipe_active_bw(GB/s)"),
        ]

    for display_name, field in ratio_fields:
        line = stat_line(
            display_name,
            metric_values(rows, field),
            ".2f",
            multiply=100,
        )
        if line:
            lines.append(line)
    for display_name, field in bandwidth_fields:
        line = stat_line(
            display_name,
            metric_values(rows, field),
            ".1f",
            unit="GB/s",
        )
        if line:
            lines.append(line)


def append_scalar_breakdown(lines: list[str], rows: Sequence[dict[str, str]], prefix: str) -> None:
    fields = [
        ("single", f"{prefix}_scalar_single_time(us)"),
        ("dual", f"{prefix}_scalar_dual_time(us)"),
        ("wait", f"{prefix}_scalar_wait_time(us)"),
        ("mte2_stall", f"{prefix}_scalar_mte2_stall_time(us)"),
        ("mte3_stall", f"{prefix}_scalar_mte3_stall_time(us)"),
    ]
    if prefix == "aiv":
        fields.extend(
            [
                ("vec_stall", "aiv_scalar_vector_stall_time(us)"),
                ("ub_stall", "aiv_scalar_stall_by_ub_time(us)"),
            ]
        )
    else:
        fields.extend(
            [
                ("cube_stall", "aic_scalar_cube_stall_time(us)"),
                ("mte1_stall", "aic_scalar_mte1_stall_time(us)"),
            ]
        )
    fields.append(("wait_ib", f"{prefix}_scalar_wait_ib_time(us)"))

    parts = []
    for display_name, field in fields:
        values = metric_values(rows, field)
        if not values:
            continue
        average = statistics.mean(values)
        if average > 0.001:
            parts.append(f"{display_name}: {average:.2f}us")
    if parts:
        lines.append(f"  SCALAR breakdown ({prefix}, avg): " + " | ".join(parts))
    else:
        lines.append(f"  SCALAR breakdown ({prefix}): unavailable (no supported stall column in PipeUtilization.csv)")


def generate_summary(
    tables: dict[str, list[dict[str, str]]],
    round_dir: str,
    expected_kernel: str,
    resolved_kernel: str,
) -> str:
    lines = ["=== On-Device Performance Summary ==="]

    basic_rows = tables["OpBasicInfo.csv"]
    first = basic_rows[0]
    op_type = first.get("Op Type", "unknown")
    durations = [required_positive_float(row.get("Task Duration(us)"), "Task Duration(us)", "OpBasicInfo.csv") for row in basic_rows]
    block_dims = [required_positive_int(row.get("Block Dim"), "Block Dim", "OpBasicInfo.csv") for row in basic_rows]
    current_freqs = metric_values(basic_rows, "Current Freq")
    rated_freqs = metric_values(basic_rows, "Rated Freq")
    lines.append(f"Requested Kernel: {expected_kernel}")
    lines.append(f"Resolved Kernel:  {resolved_kernel} | Type: {op_type}")
    lines.append(
        f"Task Duration(us): min={min(durations):.2f} | avg={statistics.mean(durations):.2f} | "
        f"max={max(durations):.2f} | samples={len(durations)}"
    )
    lines.append(f"BlockDim: {min(block_dims)}..{max(block_dims)}")
    if current_freqs and rated_freqs:
        lines.append(f"Freq(avg): {statistics.mean(current_freqs):.0f}/{statistics.mean(rated_freqs):.0f}")
    elif current_freqs:
        lines.append(f"Freq(avg): {statistics.mean(current_freqs):.0f}/unknown")

    pipe_rows = tables["PipeUtilization.csv"]
    prefixes = active_core_prefixes(pipe_rows)
    for prefix in prefixes:
        selected_rows = core_rows(pipe_rows, prefix)
        if not selected_rows:
            raise ValueError(f"PipeUtilization.csv: no positive '{prefix}_time(us)' samples for the resolved kernel")
        core_times = [
            required_positive_float(row.get(f"{prefix}_time(us)"), f"{prefix}_time(us)", "PipeUtilization.csv") for row in selected_rows
        ]
        lines.append("")
        lines.append(f"--- PipeUtilization ({prefix.upper()}, {len(selected_rows)} cores) ---")
        lines.append(f"  {'':24s} {'min':>10s} {'avg':>10s} {'max':>10s}")
        line = stat_line(f"{prefix}_time(us)", core_times, ".2f")
        if line:
            lines.append(line)
        append_ratio_and_bandwidth(lines, selected_rows, prefix)
        append_scalar_breakdown(lines, selected_rows, prefix)

    if pipe_rows and not prefixes:
        lines.extend(["", "--- PipeUtilization ---", "  No nonzero AIV/AIC metrics detected"])
    if prefixes:
        lines.extend(
            [
                "",
                "Task duration and per-core time are reported separately; no launch-level critical path or",
                "launch overhead is inferred because these CSV samples are not paired by launch ID.",
            ]
        )

    mem_rows = tables["Memory.csv"]
    if mem_rows:
        lines.extend(["", "--- Memory ---"])
        transfers = [
            ("GM→UB", "GM_to_UB_datas(KB)", "GM_to_UB_bw_usage_rate(%)"),
            ("UB→GM", "UB_to_GM_datas(KB)", "UB_to_GM_bw_usage_rate(%)"),
            ("GM→L1", "GM_to_L1_datas(KB)", "GM_to_L1_bw_usage_rate(%)"),
        ]
        for display, data_field, usage_field in transfers:
            data = metric_values(mem_rows, data_field)
            if data and sum(data) > 0:
                usage = metric_values(mem_rows, usage_field)
                usage_text = f"{statistics.mean(usage):.2f}% (samples={len(usage)})" if usage else "unavailable"
                lines.append(
                    f"  {display}: {sum(data):.1f}KB total ({statistics.mean(data):.1f}KB/core, samples={len(data)}), "
                    f"BW usage: {usage_text}"
                )
        for display, field in [
            ("Main-memory read", "read_main_memory_datas(KB)"),
            ("Main-memory write", "write_main_memory_datas(KB)"),
        ]:
            data = metric_values(mem_rows, field)
            if data and sum(data) > 0:
                lines.append(f"  {display}: {sum(data):.1f}KB total")

        mte2_lines = []
        for label, byte_field, instruction_field in (
            ("AIV GM→UB", "GM_to_UB_datas(KB)", "aiv_mte2_instructions"),
            ("AIC GM→L1", "GM_to_L1_datas(KB)", "aic_mte2_instructions"),
        ):
            paired = paired_ratio(mem_rows, byte_field, instruction_field)
            if paired:
                average, samples = paired
                mte2_lines.append(f"  Avg MTE2 transfer ({label}): {average:.2f}KB/instr (paired rows={samples})")
        if mte2_lines:
            lines.extend(mte2_lines)
        elif present_field(mem_rows, ("GM_to_UB_datas(KB)", "GM_to_L1_datas(KB)")) is not None:
            lines.append("  Avg MTE2 transfer: unavailable (no row carries both transfer bytes and MTE2 instruction counts)")

        for display, candidates in [
            ("GM→UB avg BW", ("aiv_gm_to_ub_bw(GB/s)",)),
            ("UB→GM avg BW", ("aiv_ub_to_gm_bw(GB/s)",)),
            ("GM→L1 avg BW", ("aic_gm_to_l1_bw(GB/s)", "aic_l1_read_bw(GB/s)")),
        ]:
            values, _ = metric_series(mem_rows, candidates)
            if values and sum(values) > 0:
                lines.append(f"  {display}: {statistics.mean(values):.2f} GB/s (samples={len(values)})")

    ub_rows = tables["MemoryUB.csv"]
    if ub_rows:
        parts = []
        for display, candidates in [
            ("UB read BW (vector)", ("aiv_ub_read_bw_vector(GB/s)",)),
            ("UB write BW (vector)", ("aiv_ub_write_bw_vector(GB/s)",)),
            ("UB read BW (GM)", ("aiv_ub_read_bw_gm(GB/s)", "aiv_ub_read_bw_scalar(GB/s)")),
            ("UB write BW (GM)", ("aiv_ub_write_bw_gm(GB/s)", "aiv_ub_write_bw_scalar(GB/s)")),
        ]:
            values, _ = metric_series(ub_rows, candidates)
            if values and statistics.mean(values) > 0.001:
                parts.append(f"  {display}: avg={statistics.mean(values):.1f} GB/s (samples={len(values)})")
        lines.extend(["", "--- MemoryUB ---"])
        lines.extend(parts or ["  unavailable: no supported UB bandwidth column in MemoryUB.csv"])

    l0_rows = tables["MemoryL0.csv"]
    if l0_rows:
        parts = []
        for display, candidates in [
            ("L0A read BW", ("aic_l0a_read_bw(GB/s)",)),
            ("L0A write BW", ("aic_l0a_write_bw(GB/s)",)),
            ("L0B read BW", ("aic_l0b_read_bw(GB/s)",)),
            ("L0B write BW", ("aic_l0b_write_bw(GB/s)",)),
            ("L0C read BW (cube)", ("aic_l0c_read_bw_cube(GB/s)",)),
            ("L0C write BW (cube)", ("aic_l0c_write_bw_cube(GB/s)",)),
        ]:
            values, _ = metric_series(l0_rows, candidates)
            if values and statistics.mean(values) > 0.001:
                parts.append(f"  {display}: avg={statistics.mean(values):.1f} GB/s (samples={len(values)})")
        lines.extend(["", "--- MemoryL0 ---"])
        lines.extend(parts or ["  unavailable: no supported L0 bandwidth column in MemoryL0.csv"])

    l2_rows = tables["L2Cache.csv"]
    if l2_rows:
        l2_prefixes = active_core_prefixes(l2_rows) or prefixes
        lines.extend(["", "--- L2Cache ---"])
        l2_reported = False
        for prefix in l2_prefixes:
            parts = []
            for display, candidates in [
                ("read_hit", (f"{prefix}_read_hit_rate(%)",)),
                ("write_hit", (f"{prefix}_write_hit_rate(%)",)),
                ("total_hit", (f"{prefix}_total_hit_rate(%)",)),
            ]:
                values, _ = metric_series(l2_rows, candidates)
                if any(value > 0 for value in values):
                    parts.append(
                        f"{display}: avg={statistics.mean(values):.1f}% "
                        f"(min={min(values):.1f}%, max={max(values):.1f}%, samples={len(values)})"
                    )
            if parts:
                l2_reported = True
                lines.append(f"  {prefix.upper()}: " + " | ".join(parts))
            hits = sum(
                sum(metric_values(l2_rows, field))
                for field in (
                    f"{prefix}_write_close_hit",
                    f"{prefix}_write_far_hit",
                    f"{prefix}_write_cache_hit",
                )
            )
            misses = sum(
                sum(metric_values(l2_rows, field))
                for field in (
                    f"{prefix}_write_close_miss",
                    f"{prefix}_write_far_miss",
                    f"{prefix}_write_cache_miss_allocate",
                )
            )
            if hits + misses > 0:
                l2_reported = True
                lines.append(f"  {prefix.upper()} write cache: hit={int(hits)} miss={int(misses)}")
        if not l2_reported:
            lines.append("  unavailable: no supported L2 hit-rate column in L2Cache.csv")

    conflict_rows = tables["ResourceConflictRatio.csv"]
    if conflict_rows:
        lines.extend(["", "--- ResourceConflict ---"])
        conflict_reported = False
        if "aiv" in prefixes or active_core_prefixes(conflict_rows) == ["aiv"]:
            parts = []
            for display, candidates in [
                ("vec_stu_cflt", ("aiv_vec_stu_cflt_ratio",)),
                ("vec_ldu_cflt", ("aiv_vec_ldu_cflt_ratio",)),
                ("vec_sfu_cflt", ("aiv_vec_sfu_cflt_ratio",)),
                ("vec_total_cflt", ("aiv_vec_total_cflt_ratio",)),
                ("bankgroup_cflt", ("aiv_vec_bankgroup_cflt_ratio",)),
                ("bank_cflt", ("aiv_vec_bank_cflt_ratio",)),
                ("resc_cflt", ("aiv_vec_resc_cflt_ratio",)),
                ("mte_cflt", ("aiv_vec_mte_cflt_ratio",)),
            ]:
                values, _ = metric_series(conflict_rows, candidates, 100)
                if any(value > 0 for value in values):
                    parts.append(f"{display}: {statistics.mean(values):.2f}% (samples={len(values)})")
            if parts:
                conflict_reported = True
                lines.append("  AIV conflicts: " + " | ".join(parts))

        for prefix in active_core_prefixes(conflict_rows) or prefixes:
            wait_fields = (
                [
                    ("vec_wait", "aiv_vec_wait_ratio"),
                    ("mte2_wait", "aiv_mte2_wait_ratio"),
                    ("mte3_wait", "aiv_mte3_wait_ratio"),
                ]
                if prefix == "aiv"
                else [
                    ("cube_wait", "aic_cube_wait_ratio"),
                    ("mte1_wait", "aic_mte1_wait_ratio"),
                    ("mte2_wait", "aic_mte2_wait_ratio"),
                    ("mte3_wait", "aic_mte3_wait_ratio"),
                ]
            )
            parts = []
            for display, field in wait_fields:
                values = metric_values(conflict_rows, field, 100)
                if any(value > 0 for value in values):
                    parts.append(f"{display}: {statistics.mean(values):.2f}% (samples={len(values)})")
            if parts:
                conflict_reported = True
                lines.append(f"  {prefix.upper()} waits: " + " | ".join(parts))
        if not conflict_reported:
            lines.append("  unavailable: no supported conflict or wait column in ResourceConflictRatio.csv")

    arithmetic_rows = tables["ArithmeticUtilization.csv"]
    if arithmetic_rows:
        lines.extend(["", "--- ArithmeticUtilization ---"])
        groups = (
            (
                "AIV",
                (
                    ("vec_ratio", ("aiv_vec_ratio",)),
                    ("vec_vf", ("aiv_vec_vf_ratio",)),
                    ("vec_sfu", ("aiv_vec_sfu_ratio",)),
                    ("vec_simt_vf", ("aiv_vec_simt_vf_ratio",)),
                    ("vec_fp32", ("aiv_vec_fp32_ratio",)),
                    ("vec_fp16", ("aiv_vec_fp16_ratio",)),
                    ("vec_int32", ("aiv_vec_int32_ratio",)),
                    ("vec_int16", ("aiv_vec_int16_ratio",)),
                    ("vec_misc", ("aiv_vec_misc_ratio",)),
                ),
                (("vec_fops", ("aiv_vec_fops",)),),
            ),
            (
                "AIC",
                (
                    ("cube_ratio", ("aic_cube_ratio",)),
                    ("cube_fp", ("aic_cube_fp_ratio", "aic_cube_fp16_ratio")),
                    ("cube_int", ("aic_cube_int_ratio", "aic_cube_int8_ratio")),
                ),
                (
                    ("cube_total_instr", ("aic_cube_total_instr_number",)),
                    ("cube_fp_instr", ("aic_cube_fp_instr_number",)),
                    ("cube_int_instr", ("aic_cube_int_instr_number",)),
                ),
            ),
        )
        arithmetic_reported = False
        for label, ratio_specs, counter_specs in groups:
            parts = []
            for display, candidates in ratio_specs:
                values, _ = metric_series(arithmetic_rows, candidates, 100)
                if values and statistics.mean(values) > 0.01:
                    parts.append(f"{display}: {statistics.mean(values):.1f}% (samples={len(values)})")
            for display, candidates in counter_specs:
                values, _ = metric_series(arithmetic_rows, candidates)
                if values and statistics.mean(values) > 0:
                    parts.append(f"{display}: {statistics.mean(values):.0f}/core (samples={len(values)})")
            if parts:
                arithmetic_reported = True
                lines.append(f"  {label}: " + " | ".join(parts))
        if not arithmetic_reported:
            lines.append("  unavailable: no supported ratio column in ArithmeticUtilization.csv")

    lines.extend(
        [
            "",
            "--- Raw Data Location ---",
            f"  CSV files: {round_dir}/",
            "  Archived files preserve the original msprof content; summary.txt reports only the Resolved Kernel.",
            "  For per-core details, read the corresponding CSV files and continue using the same kernel filter.",
        ]
    )
    return "\n".join(lines)


def validate_round_name(round_name: str) -> None:
    if round_name in ("", ".", "..") or os.path.basename(round_name) != round_name:
        raise ValueError("--round-name must be one directory name, not a path")


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive msprof CSVs and summarize exactly one TileLang kernel.")
    parser.add_argument("opprof_dir", help="Path to a flat OPPROF or launch directory")
    parser.add_argument(
        "variant_output_dir",
        help="Current baseline/optimization variant output directory",
    )
    parser.add_argument(
        "--kernel-name",
        required=True,
        help="Expected runtime kernel name; zero or ambiguous matches are rejected",
    )
    parser.add_argument("--round-name", help="Override round directory name (default: auto-increment)")
    args = parser.parse_args()

    opprof_dir = os.path.abspath(args.opprof_dir)
    variant_output_dir = os.path.abspath(args.variant_output_dir)
    if not os.path.isdir(opprof_dir):
        parser.error(f"'{opprof_dir}' is not an OPPROF directory")
    if not os.path.isdir(variant_output_dir):
        parser.error(f"'{variant_output_dir}' is not a variant output directory")
    try:
        metric_csv_path(opprof_dir, "PipeUtilization.csv")
        tables, resolved_kernel = load_kernel_tables(opprof_dir, args.kernel_name)
        perf_dir = os.path.join(variant_output_dir, "docs", "perf")
        if args.round_name:
            validate_round_name(args.round_name)
            round_dir = os.path.join(perf_dir, args.round_name)
        else:
            round_dir = find_next_round(perf_dir)
        if os.path.exists(round_dir):
            raise ValueError(f"archive directory already exists: {round_dir}")

        summary = generate_summary(tables, round_dir, args.kernel_name, resolved_kernel)
        copied = publish_round(opprof_dir, round_dir, summary)
        summary_path = os.path.join(round_dir, "summary.txt")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Kernel verified: requested='{args.kernel_name}', resolved='{resolved_kernel}'")
    print(f"Archived {len(copied)} CSV files to: {round_dir}")
    print(f"Summary written to: {summary_path}")
    print()
    print(summary)


if __name__ == "__main__":
    main()

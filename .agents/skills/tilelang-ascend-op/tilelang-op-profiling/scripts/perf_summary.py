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
import os
import re
import shutil
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or str(val).strip() in ("", "N/A", "NA"):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as file:
        rows = []
        for row in csv.DictReader(file):
            rows.append({
                str(key).strip(): value.strip() if isinstance(value, str) else value
                for key, value in row.items()
                if key is not None
            })
        return rows


def metric_csv_path(opprof_dir: str, csv_name: str) -> str:
    """Resolve a fixed CSV name or one timestamp-suffixed CSV."""
    exact = os.path.join(opprof_dir, csv_name)
    if os.path.isfile(exact):
        return exact
    stem, extension = os.path.splitext(csv_name)
    matches = glob.glob(os.path.join(opprof_dir, f"{stem}_*{extension}"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one {csv_name} in {opprof_dir}, got {len(matches)}"
        )
    return matches[0]


def stat_line(
    name: str,
    values: Sequence[float],
    fmt: str = ".1f",
    unit: str = "",
    multiply: float = 1.0,
) -> Optional[str]:
    """Generate a min/avg/max line, omitting an all-zero metric."""
    scaled = [value * multiply for value in values]
    if not scaled or all(abs(value) < 0.001 for value in scaled):
        return None
    suffix = f"    ({unit})" if unit else ""
    return (
        f"{'  ' + name:<24s} {min(scaled):>10{fmt}} "
        f"{statistics.mean(scaled):>10{fmt}} {max(scaled):>10{fmt}}{suffix}"
    )


def row_kernel_name(row: Dict[str, str]) -> Optional[str]:
    for field in KERNEL_NAME_FIELDS:
        raw_value = row.get(field)
        value = raw_value.strip() if isinstance(raw_value, str) else ""
        if value:
            return value
    return None


def unique_kernel_names(rows: Iterable[Dict[str, str]]) -> List[str]:
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
        raise ValueError(
            f"{source}: kernel '{expected}' not found; available kernels: "
            f"{', '.join(names) if names else '(none)'}"
        )
    raise ValueError(
        f"{source}: kernel '{expected}' is ambiguous; matches: {', '.join(contains)}"
    )


def filter_rows_for_kernel(
    rows: List[Dict[str, str]],
    expected: str,
    resolved: str,
    source: str,
    raw_kernel_count: int,
) -> List[Dict[str, str]]:
    """Filter a CSV by kernel name when it exposes a supported name column."""
    if not rows:
        return []
    names = unique_kernel_names(rows)
    if not names:
        if raw_kernel_count > 1:
            raise ValueError(
                f"{source}: no kernel-name column, but OpBasicInfo.csv contains "
                f"{raw_kernel_count} kernels; refusing to mix their metrics"
            )
        return rows

    selected_name = resolved if resolved in names else unique_match(expected, names, source)
    filtered = [row for row in rows if row_kernel_name(row) == selected_name]
    if not filtered:
        raise ValueError(f"{source}: no rows remain after selecting kernel '{selected_name}'")
    return filtered


def load_kernel_tables(
    opprof_dir: str, expected: str
) -> Tuple[Dict[str, List[Dict[str, str]]], str]:
    basic_name = "OpBasicInfo.csv"
    basic_rows = read_csv(metric_csv_path(opprof_dir, basic_name))
    if not basic_rows:
        raise ValueError(f"{basic_name} is missing or empty")
    basic_kernels = unique_kernel_names(basic_rows)
    if not basic_kernels:
        raise ValueError(
            f"{basic_name} has no supported kernel-name column; expected one of: "
            f"{', '.join(KERNEL_NAME_FIELDS)}"
        )

    resolved = unique_match(expected, basic_kernels, basic_name)
    raw_kernel_count = len(basic_kernels)
    tables: Dict[str, List[Dict[str, str]]] = {}
    for csv_name in CSV_NAMES:
        rows = read_csv(metric_csv_path(opprof_dir, csv_name))
        tables[csv_name] = filter_rows_for_kernel(
            rows, expected, resolved, csv_name, raw_kernel_count
        )
    return tables, resolved


def active_core_prefixes(rows: Sequence[Dict[str, str]]) -> List[str]:
    """Return every active core type instead of collapsing mixed AIC/AIV data."""
    active = []
    for prefix in ("aiv", "aic"):
        if any(
            safe_float(value) > 0
            for row in rows
            for field, value in row.items()
            if field.startswith(f"{prefix}_")
        ):
            active.append(prefix)
    return active


def core_rows(rows: Sequence[Dict[str, str]], prefix: str) -> List[Dict[str, str]]:
    time_field = f"{prefix}_time(us)"
    selected = [row for row in rows if safe_float(row.get(time_field)) > 0]
    return selected or list(rows)


def find_next_round(perf_dir: str) -> str:
    if not os.path.exists(perf_dir):
        return os.path.join(perf_dir, "round_001")
    existing = [name for name in os.listdir(perf_dir) if re.fullmatch(r"round_\d+", name)]
    if not existing:
        return os.path.join(perf_dir, "round_001")
    numbers = [int(name.split("_")[1]) for name in existing]
    return os.path.join(perf_dir, f"round_{max(numbers) + 1:03d}")


def archive_csvs(opprof_dir: str, round_dir: str) -> List[str]:
    os.makedirs(round_dir, exist_ok=False)
    copied = []
    for csv_name in CSV_NAMES:
        shutil.copy2(
            metric_csv_path(opprof_dir, csv_name), os.path.join(round_dir, csv_name)
        )
        copied.append(csv_name)
    return copied


def append_ratio_and_bandwidth(
    lines: List[str], rows: Sequence[Dict[str, str]], prefix: str
) -> None:
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
        line = stat_line(display_name, [safe_float(row.get(field)) for row in rows], ".2f", multiply=100)
        if line:
            lines.append(line)
    for display_name, field in bandwidth_fields:
        line = stat_line(
            display_name,
            [safe_float(row.get(field)) for row in rows],
            ".1f",
            unit="GB/s",
        )
        if line:
            lines.append(line)


def append_scalar_breakdown(
    lines: List[str], rows: Sequence[Dict[str, str]], prefix: str
) -> None:
    fields = [
        ("single", f"{prefix}_scalar_single_time(us)"),
        ("dual", f"{prefix}_scalar_dual_time(us)"),
        ("wait", f"{prefix}_scalar_wait_time(us)"),
        ("mte2_stall", f"{prefix}_scalar_mte2_stall_time(us)"),
        ("mte3_stall", f"{prefix}_scalar_mte3_stall_time(us)"),
    ]
    if prefix == "aiv":
        fields.extend([
            ("vec_stall", "aiv_scalar_vector_stall_time(us)"),
            ("ub_stall", "aiv_scalar_stall_by_ub_time(us)"),
        ])
    else:
        fields.extend([
            ("cube_stall", "aic_scalar_cube_stall_time(us)"),
            ("mte1_stall", "aic_scalar_mte1_stall_time(us)"),
        ])
    fields.append(("wait_ib", f"{prefix}_scalar_wait_ib_time(us)"))

    parts = []
    for display_name, field in fields:
        average = statistics.mean(safe_float(row.get(field)) for row in rows)
        if average > 0.001:
            parts.append(f"{display_name}: {average:.2f}us")
    if parts:
        lines.append(f"  SCALAR breakdown ({prefix}, avg): " + " | ".join(parts))


def generate_summary(
    tables: Dict[str, List[Dict[str, str]]],
    round_dir: str,
    expected_kernel: str,
    resolved_kernel: str,
) -> str:
    lines = ["=== On-Device Performance Summary ==="]

    basic_rows = tables["OpBasicInfo.csv"]
    first = basic_rows[0]
    op_type = first.get("Op Type", "unknown")
    durations = [safe_float(row.get("Task Duration(us)")) for row in basic_rows]
    duration = statistics.mean(durations) if durations else 0.0
    block_dims = [int(safe_float(row.get("Block Dim", "1"))) for row in basic_rows]
    current_freqs = [safe_float(row.get("Current Freq")) for row in basic_rows]
    rated_freqs = [safe_float(row.get("Rated Freq")) for row in basic_rows]
    lines.append(f"Requested Kernel: {expected_kernel}")
    lines.append(f"Resolved Kernel:  {resolved_kernel} | Type: {op_type}")
    if durations:
        lines.append(
            f"Task Duration(us): min={min(durations):.2f} | avg={duration:.2f} | "
            f"max={max(durations):.2f} | samples={len(durations)}"
        )
    if block_dims:
        lines.append(f"BlockDim: {min(block_dims)}..{max(block_dims)}")
    if current_freqs or rated_freqs:
        lines.append(
            f"Freq(avg): {statistics.mean(current_freqs):.0f}/"
            f"{statistics.mean(rated_freqs):.0f}"
        )

    pipe_rows = tables["PipeUtilization.csv"]
    prefixes = active_core_prefixes(pipe_rows)
    critical_times: List[float] = []
    for prefix in prefixes:
        selected_rows = core_rows(pipe_rows, prefix)
        core_times = [safe_float(row.get(f"{prefix}_time(us)")) for row in selected_rows]
        critical_times.extend(core_times)
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
    if critical_times and duration > 0:
        critical_path = max(critical_times)
        overhead = max(0.0, duration - critical_path)
        mode = "max(AIV, AIC)" if len(prefixes) == 2 else prefixes[0].upper()
        lines.extend([
            "",
            "--- Critical Path and Launch Overhead ---",
            f"  Core-time methodology: {mode}",
            f"  Task Duration(avg): {duration:.2f}us | Critical path: {critical_path:.2f}us | "
            f"Launch overhead: {overhead:.2f}us ({overhead / duration * 100:.1f}%)",
        ])

    mem_rows = tables["Memory.csv"]
    if mem_rows:
        lines.extend(["", "--- Memory ---"])
        transfers = [
            ("GM→UB", "GM_to_UB_datas(KB)", "GM_to_UB_bw_usage_rate(%)"),
            ("UB→GM", "UB_to_GM_datas(KB)", "UB_to_GM_bw_usage_rate(%)"),
            ("GM→L1", "GM_to_L1_datas(KB)", "GM_to_L1_bw_usage_rate(%)"),
        ]
        for display, data_field, usage_field in transfers:
            data = [safe_float(row.get(data_field)) for row in mem_rows]
            if sum(data) > 0:
                usage = [safe_float(row.get(usage_field)) for row in mem_rows]
                lines.append(
                    f"  {display}: {sum(data):.1f}KB total "
                    f"({statistics.mean(data):.1f}KB/core), BW usage: {statistics.mean(usage):.2f}%"
                )
        for display, field in [
            ("Main-memory read", "read_main_memory_datas(KB)"),
            ("Main-memory write", "write_main_memory_datas(KB)"),
        ]:
            data = [safe_float(row.get(field)) for row in mem_rows]
            if sum(data) > 0:
                lines.append(f"  {display}: {sum(data):.1f}KB total")

        gm_ub_total = sum(safe_float(row.get("GM_to_UB_datas(KB)")) for row in mem_rows)
        mte2_instructions = sum(
            safe_float(row.get("aiv_mte2_instructions"))
            + safe_float(row.get("aic_mte2_instructions"))
            for row in mem_rows
        )
        if gm_ub_total > 0 and mte2_instructions > 0:
            average_transfer = gm_ub_total / mte2_instructions
            lines.append(
                f"  Avg MTE2 transfer: {average_transfer:.2f}KB "
                f"({int(mte2_instructions)} instructions total)"
            )

        for display, field in [
            ("GM→UB avg BW", "aiv_gm_to_ub_bw(GB/s)"),
            ("UB→GM avg BW", "aiv_ub_to_gm_bw(GB/s)"),
            ("GM→L1 avg BW", "aic_gm_to_l1_bw(GB/s)"),
        ]:
            values = [safe_float(row.get(field)) for row in mem_rows]
            if sum(values) > 0:
                lines.append(f"  {display}: {statistics.mean(values):.2f} GB/s")

    ub_rows = tables["MemoryUB.csv"]
    if ub_rows:
        parts = []
        for display, field in [
            ("UB read BW (vector)", "aiv_ub_read_bw_vector(GB/s)"),
            ("UB write BW (vector)", "aiv_ub_write_bw_vector(GB/s)"),
            ("UB read BW (scalar)", "aiv_ub_read_bw_scalar(GB/s)"),
            ("UB write BW (scalar)", "aiv_ub_write_bw_scalar(GB/s)"),
        ]:
            average = statistics.mean(safe_float(row.get(field)) for row in ub_rows)
            if average > 0.001:
                parts.append(f"  {display}: avg={average:.1f} GB/s")
        if parts:
            lines.extend(["", "--- MemoryUB ---", *parts])

    l0_rows = tables["MemoryL0.csv"]
    if l0_rows:
        parts = []
        for display, field in [
            ("L0A read BW", "aic_l0a_read_bw(GB/s)"),
            ("L0A write BW", "aic_l0a_write_bw(GB/s)"),
            ("L0B read BW", "aic_l0b_read_bw(GB/s)"),
            ("L0B write BW", "aic_l0b_write_bw(GB/s)"),
            ("L0C read BW (cube)", "aic_l0c_read_bw_cube(GB/s)"),
            ("L0C write BW (cube)", "aic_l0c_write_bw_cube(GB/s)"),
        ]:
            average = statistics.mean(safe_float(row.get(field)) for row in l0_rows)
            if average > 0.001:
                parts.append(f"  {display}: {average:.1f} GB/s")
        if parts:
            lines.extend(["", "--- MemoryL0 ---", *parts])

    l2_rows = tables["L2Cache.csv"]
    if l2_rows:
        l2_prefixes = active_core_prefixes(l2_rows) or prefixes
        lines.extend(["", "--- L2Cache ---"])
        for prefix in l2_prefixes:
            parts = []
            for display, field in [
                ("total_hit", f"{prefix}_total_hit_rate(%)"),
                ("read_hit", f"{prefix}_read_hit_rate(%)"),
                ("write_hit", f"{prefix}_write_hit_rate(%)"),
            ]:
                values = [safe_float(row.get(field)) for row in l2_rows]
                if any(value > 0 for value in values):
                    parts.append(
                        f"{display}: avg={statistics.mean(values):.1f}% "
                        f"(min={min(values):.1f}%, max={max(values):.1f}%)"
                    )
            if parts:
                lines.append(f"  {prefix.upper()}: " + " | ".join(parts))
            hits = sum(safe_float(row.get(f"{prefix}_write_cache_hit")) for row in l2_rows)
            misses = sum(
                safe_float(row.get(f"{prefix}_write_cache_miss_allocate")) for row in l2_rows
            )
            if hits + misses > 0:
                lines.append(f"  {prefix.upper()} write cache: hit={int(hits)} miss={int(misses)}")

    conflict_rows = tables["ResourceConflictRatio.csv"]
    if conflict_rows:
        lines.extend(["", "--- ResourceConflict ---"])
        if "aiv" in prefixes or active_core_prefixes(conflict_rows) == ["aiv"]:
            parts = []
            for display, field in [
                ("vec_total_cflt", "aiv_vec_total_cflt_ratio"),
                ("bankgroup_cflt", "aiv_vec_bankgroup_cflt_ratio"),
                ("bank_cflt", "aiv_vec_bank_cflt_ratio"),
                ("resc_cflt", "aiv_vec_resc_cflt_ratio"),
                ("mte_cflt", "aiv_vec_mte_cflt_ratio"),
            ]:
                values = [safe_float(row.get(field)) * 100 for row in conflict_rows]
                if any(value > 0 for value in values):
                    parts.append(f"{display}: {statistics.mean(values):.2f}%")
            if parts:
                lines.append("  AIV conflicts: " + " | ".join(parts))

        for prefix in active_core_prefixes(conflict_rows) or prefixes:
            wait_fields = (
                [("vec_wait", "aiv_vec_wait_ratio"), ("mte2_wait", "aiv_mte2_wait_ratio"), ("mte3_wait", "aiv_mte3_wait_ratio")]
                if prefix == "aiv"
                else [("cube_wait", "aic_cube_wait_ratio"), ("mte1_wait", "aic_mte1_wait_ratio"), ("mte2_wait", "aic_mte2_wait_ratio"), ("mte3_wait", "aic_mte3_wait_ratio")]
            )
            parts = []
            for display, field in wait_fields:
                values = [safe_float(row.get(field)) * 100 for row in conflict_rows]
                if any(value > 0 for value in values):
                    parts.append(f"{display}: {statistics.mean(values):.2f}%")
            if parts:
                lines.append(f"  {prefix.upper()} waits: " + " | ".join(parts))

    arithmetic_rows = tables["ArithmeticUtilization.csv"]
    if arithmetic_rows:
        lines.extend(["", "--- ArithmeticUtilization ---"])
        vector_parts = []
        for display, field in [
            ("vec_fp32", "aiv_vec_fp32_ratio"),
            ("vec_fp16", "aiv_vec_fp16_ratio"),
            ("vec_int32", "aiv_vec_int32_ratio"),
            ("vec_int16", "aiv_vec_int16_ratio"),
            ("vec_misc", "aiv_vec_misc_ratio"),
        ]:
            average = statistics.mean(safe_float(row.get(field)) * 100 for row in arithmetic_rows)
            if average > 0.01:
                vector_parts.append(f"{display}: {average:.1f}%")
        vector_fops = statistics.mean(safe_float(row.get("aiv_vec_fops")) for row in arithmetic_rows)
        if vector_fops > 0:
            vector_parts.append(f"vec_fops: {vector_fops:.0f}/core")
        if vector_parts:
            lines.append("  AIV: " + " | ".join(vector_parts))

        cube_parts = []
        for display, field in [
            ("cube_fp16", "aic_cube_fp16_ratio"),
            ("cube_int8", "aic_cube_int8_ratio"),
        ]:
            average = statistics.mean(safe_float(row.get(field)) * 100 for row in arithmetic_rows)
            if average > 0.01:
                cube_parts.append(f"{display}: {average:.1f}%")
        cube_fops = statistics.mean(safe_float(row.get("aic_cube_fops")) for row in arithmetic_rows)
        if cube_fops > 0:
            cube_parts.append(f"cube_fops: {cube_fops:.0f}/core")
        if cube_parts:
            lines.append("  AIC: " + " | ".join(cube_parts))

    lines.extend([
        "",
        "--- Raw Data Location ---",
        f"  CSV files: {round_dir}/",
        "  Archived files preserve the original msprof content; summary.txt reports only the Resolved Kernel.",
        "  For per-core details, read the corresponding CSV files and continue using the same kernel filter.",
    ])
    return "\n".join(lines)


def validate_round_name(round_name: str) -> None:
    if round_name in ("", ".", "..") or os.path.basename(round_name) != round_name:
        raise ValueError("--round-name must be one directory name, not a path")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive msprof CSVs and summarize exactly one TileLang kernel."
    )
    parser.add_argument("opprof_dir", help="Path to a flat OPPROF or launch directory")
    parser.add_argument("variant_output_dir", help="Current baseline/optimization variant output directory")
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

        summary = generate_summary(
            tables, round_dir, args.kernel_name, resolved_kernel
        )
        copied = archive_csvs(opprof_dir, round_dir)
        summary_path = os.path.join(round_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as file:
            file.write(summary)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Kernel verified: requested='{args.kernel_name}', resolved='{resolved_kernel}'")
    print(f"Archived {len(copied)} CSV files to: {round_dir}")
    print(f"Summary written to: {summary_path}")
    print()
    print(summary)


if __name__ == "__main__":
    main()

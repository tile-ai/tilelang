#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Validate the structural search record and its append-only candidate event log."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


AXES = (
    "work_granularity",
    "task_mapping",
    "physical_dataflow",
    "precision",
    "storage_pipeline",
    "tail_strategy",
)
CASE_FIELDS = (
    "case_id",
    "outer_items",
    "inner_independent_chunks_per_item",
    "inner_chunks_independent",
    "available_cores",
    "outer_task_waves",
    "flattened_task_waves",
    "per_task_payload_bytes",
    "kernel_time_us",
    "lower_bounds_us",
    "lower_bound_evidence",
    "main_pipes",
    "performance_target",
    "best_metrics",
    "best_metric_evidence",
    "source_window",
    "obligation_ids",
)
OBLIGATION_STATUSES = {
    "PENDING",
    "ACTIVE",
    "BLOCKED",
    "SATISFIED",
    "CLOSED_WITH_EVIDENCE",
}
CANDIDATE_RESULTS = {
    "PROMOTED",
    "REJECTED_WITH_EVIDENCE",
    "INAPPLICABLE_WITH_EVIDENCE",
    "BLOCKED_BY_ENVIRONMENT",
}
CASE_DISPOSITIONS = {
    "PENDING",
    "BLOCKED",
    "BASELINE_MEASURED",
    "TESTED_PATH",
    "CLOSED_WITH_EVIDENCE",
}
CLOSED_OBLIGATION_STATUSES = {"SATISFIED", "CLOSED_WITH_EVIDENCE"}
AUTO_STATUSES = {"NOT_TRIED", "COMPLETE", "FAILED", "PARTIAL", "INAPPLICABLE"}
MANUAL_STATUSES = {
    "NOT_TRIED",
    "BLOCKED",
    "NOT_NEEDED_AUTO_COMPLETE",
    "TESTED",
    "CLOSED_WITH_EVIDENCE",
}
ROUTE_KINDS = {
    "SCALAR_OR_SIMT",
    "MEMORY_INDEXED_VECTOR",
    "CONTIGUOUS_LOAD_REGISTER_REORDER",
    "MATERIALIZED_TRANSFORM",
}
ALL_ROUTE_KINDS = ROUTE_KINDS | {"OTHER"}
SOURCE_ACCESS_KINDS = {"SCALAR", "INDEXED", "CONTIGUOUS", "BULK_COPY", "MIXED"}
WRITEBACK_KINDS = {
    "SCALAR",
    "CONTIGUOUS",
    "STRIDED",
    "INDEXED_SCATTER",
    "MATERIALIZED_COPY",
    "MIXED",
}
BOUNDARY_KINDS = {
    "FULL_CONTIGUOUS_UNIT",
    "SIMD",
    "DMA",
    "CAPACITY",
    "PARALLEL",
}
RESULT_CASE_DISPOSITIONS = {
    "TESTED_PATH",
    "NOT_RUN_WITH_EVIDENCE",
    "NOT_RUN_BLOCKED",
    "INAPPLICABLE_WITH_EVIDENCE",
}
BLOCKER_KINDS = {
    "DEVICE_UNAVAILABLE",
    "PROFILER_UNAVAILABLE",
    "TOOLCHAIN_ENVIRONMENT",
    "EXTERNAL_RESOURCE",
    "OTHER_EXTERNAL",
}
REJECTION_STAGES = {
    "COMPILATION",
    "KERNEL_EXECUTION",
    "PRECISION",
    "PROFILING_VALIDITY",
}
PIPELINE_REALIZATIONS = {"SINGLE_STAGE", "AUTOMATIC", "MANUAL"}
PIPELINE_PAIRING_AXES = ("work_granularity", "precision", "task_mapping")
OBLIGATION_KINDS = {
    "inner_parallel_flatten",
    "work_granularity_search",
    "physical_dataflow_routes",
    "granularity_dataflow_interaction",
    "pipeline_realization",
}
ACCESS_PATTERNS = {
    "NOT_APPLICABLE",
    "SCALAR_ONLY",
    "DIRECT_CONTIGUOUS",
    "INDEXED_OR_REORDERED",
}
TARGET_OPERATORS = {"LE", "GE"}
TARGET_LOGICS = {"ANY", "ALL"}
HARD_EVIDENCE_KINDS = {"CAPACITY", "LOWERING", "SEMANTIC", "STRICT_COST_DOMINANCE"}
BASELINE_ID = "B0"


class ValidationError(Exception):
    pass


class SearchIncomplete(ValidationError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def require_choice(value: object, allowed: set[str] | frozenset[str], message: str) -> str:
    """Require a string enum member so malformed JSON fails as INVALID, not TypeError."""
    require(isinstance(value, str) and value in allowed, message)
    return value


def require_string_list(value: object, message: str) -> list[str]:
    """Require a non-empty list of distinct non-empty strings."""
    require(isinstance(value, list) and value, message)
    require(all(nonempty_text(item) for item in value), message)
    require(len(set(value)) == len(value), message)
    return value


def require_strict_int(value: object, minimum: int, message: str) -> int:
    """Require an integer counter; bool is an int subclass but never a counter."""
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum, message)
    return value


def nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def nonnegative_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def evidence_list(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(nonempty_text(item) for item in value)


def valid_boundary_value(value: object) -> bool:
    return positive_number(value) or nonempty_text(value)


def validate_blocking_evidence(owner: str, value: object) -> None:
    require(isinstance(value, dict), f"{owner}: blocking_evidence must be an object")
    require_choice(value.get("kind"), BLOCKER_KINDS, f"{owner}: invalid blocker kind")
    for field in (
        "failure_stage",
        "observed_error",
        "control_check",
        "resume_condition",
    ):
        require(
            nonempty_text(value.get(field)),
            f"{owner}: blocking_evidence missing {field}",
        )


def validate_rejection_evidence(owner: str, value: object) -> None:
    require(isinstance(value, dict), f"{owner}: rejection_evidence must be an object")
    require_choice(
        value.get("failure_stage"),
        REJECTION_STAGES,
        f"{owner}: invalid rejection failure_stage",
    )
    for field in ("observed_error", "control_check", "candidate_finding"):
        require(
            nonempty_text(value.get(field)),
            f"{owner}: rejection_evidence missing {field}",
        )


def validate_basis_binding(
    owner: str,
    value: dict,
    basis_by_case: dict[str, str],
) -> None:
    """Require an infeasibility proof to describe the basis and cases it covers."""
    case_ids = value.get("case_ids")
    require_string_list(
        case_ids,
        f"{owner}: hard infeasibility must list the case_ids it covers",
    )
    require(
        set(case_ids) == set(basis_by_case),
        f"{owner}: proof covers {sorted(set(case_ids))}, but the obligation covers {sorted(basis_by_case)}",
    )
    binding = value.get("basis_candidate_id")
    if isinstance(binding, str):
        require(nonempty_text(binding), f"{owner}: invalid basis_candidate_id binding")
        resolved = {case_id: binding for case_id in basis_by_case}
    elif isinstance(binding, dict):
        require(
            set(binding) == set(basis_by_case) and all(nonempty_text(item) for item in binding.values()),
            f"{owner}: basis_candidate_id binding must map exactly {sorted(basis_by_case)}",
        )
        resolved = dict(binding)
    else:
        raise ValidationError(f"{owner}: hard infeasibility must bind basis_candidate_id to the basis it describes")
    stale = sorted(case_id for case_id, basis in basis_by_case.items() if resolved[case_id] != basis)
    require(
        not stale,
        f"{owner}: proof is bound to {resolved}, but the obligation now uses "
        f"{ {case_id: basis_by_case[case_id] for case_id in stale} }; "
        "re-derive the proof for the current basis instead of reusing it",
    )


def validate_hard_infeasibility(
    owner: str,
    value: object,
    basis_by_case: dict[str, str] | None = None,
) -> None:
    require(isinstance(value, dict), f"{owner}: hard_infeasibility must be an object")
    kind = require_choice(value.get("kind"), HARD_EVIDENCE_KINDS, f"{owner}: invalid hard-infeasibility kind")
    require(
        evidence_list(value.get("artifacts")),
        f"{owner}: hard infeasibility needs evidence artifacts",
    )
    require(
        nonempty_text(value.get("finding")),
        f"{owner}: hard infeasibility needs a finding",
    )
    require(
        value.get("exact_scope_only") is True,
        f"{owner}: hard infeasibility must state exact_scope_only=true",
    )
    if kind == "CAPACITY":
        require(
            positive_number(value.get("capacity_bytes")),
            f"{owner}: capacity_bytes must be positive",
        )
        require(
            positive_number(value.get("required_bytes")),
            f"{owner}: required_bytes must be positive",
        )
        require(
            value["required_bytes"] > value["capacity_bytes"],
            f"{owner}: capacity proof must exceed available capacity",
        )
        require(
            value.get("streaming_or_decomposition_checked") is True,
            f"{owner}: capacity proof must check streaming/decomposition",
        )
    elif kind == "LOWERING":
        require(
            nonempty_text(value.get("minimal_reproducer")),
            f"{owner}: lowering proof needs minimal_reproducer",
        )
        require(
            evidence_list(value.get("checked_symbols")),
            f"{owner}: lowering proof needs checked_symbols",
        )
    elif kind == "SEMANTIC":
        require(
            nonempty_text(value.get("constraint")),
            f"{owner}: semantic proof needs constraint",
        )
        require(nonempty_text(value.get("proof")), f"{owner}: semantic proof needs proof")
    elif kind == "STRICT_COST_DOMINANCE":
        require(
            value.get("same_payload") is True,
            f"{owner}: cost dominance requires same_payload=true",
        )
        require(
            nonnegative_number(value.get("reference_cost")),
            f"{owner}: invalid reference_cost",
        )
        require(
            nonnegative_number(value.get("candidate_cost")),
            f"{owner}: invalid candidate_cost",
        )
        require(
            value["candidate_cost"] > value["reference_cost"],
            f"{owner}: candidate cost must be strictly greater than reference cost",
        )
        require(
            nonempty_text(value.get("cost_unit")),
            f"{owner}: cost dominance needs cost_unit",
        )
    if basis_by_case is not None:
        validate_basis_binding(owner, value, basis_by_case)


def validate_performance_target(case: dict) -> bool:
    case_id = case.get("case_id", "<missing>")
    target = case["performance_target"]
    metrics = case["best_metrics"]
    require(isinstance(metrics, dict), f"case {case_id}: best_metrics must be an object")
    require(
        positive_number(metrics.get("kernel_time_us")),
        f"case {case_id}: best kernel_time_us must be positive",
    )
    require(
        evidence_list(case["best_metric_evidence"]),
        f"case {case_id}: best metric evidence required",
    )
    # An absent user target must not be replaced with an invented threshold.
    if target is None:
        return False
    require(
        isinstance(target, dict),
        f"case {case_id}: performance_target must be an object or null",
    )
    require_choice(target.get("logic"), TARGET_LOGICS, f"case {case_id}: target logic must be ANY or ALL")
    conditions = target.get("conditions")
    require(
        isinstance(conditions, list) and conditions,
        f"case {case_id}: target conditions required",
    )
    outcomes: list[bool] = []
    seen_metrics: set[str] = set()
    for index, condition in enumerate(conditions):
        require(
            isinstance(condition, dict),
            f"case {case_id}: target condition {index} must be an object",
        )
        metric = condition.get("metric")
        operator = condition.get("operator")
        threshold = condition.get("threshold")
        require(
            nonempty_text(metric),
            f"case {case_id}: target condition {index} needs metric",
        )
        require(
            metric not in seen_metrics,
            f"case {case_id}: duplicate target metric {metric}",
        )
        seen_metrics.add(metric)
        require_choice(
            operator,
            TARGET_OPERATORS,
            f"case {case_id}: invalid target operator {operator}",
        )
        require(
            nonnegative_number(threshold),
            f"case {case_id}: target threshold must be non-negative",
        )
        require(
            nonnegative_number(metrics.get(metric)),
            f"case {case_id}: best metric {metric} missing or invalid",
        )
        outcomes.append(metrics[metric] <= threshold if operator == "LE" else metrics[metric] >= threshold)
    return any(outcomes) if target["logic"] == "ANY" else all(outcomes)


def validate_case(case: dict, obligation_ids: set[str]) -> None:
    case_id = case.get("case_id", "<missing>")
    for field in CASE_FIELDS:
        require(field in case, f"case {case_id}: missing {field}")

    for field in ("outer_items", "inner_independent_chunks_per_item"):
        require_strict_int(
            case[field],
            0,
            f"case {case_id}: {field} must be a non-negative integer",
        )
    require_strict_int(
        case["available_cores"],
        1,
        f"case {case_id}: available_cores must be positive",
    )
    for field in ("outer_task_waves", "flattened_task_waves", "per_task_payload_bytes"):
        require(
            nonnegative_number(case[field]),
            f"case {case_id}: {field} must be non-negative",
        )
    require(
        positive_number(case["kernel_time_us"]),
        f"case {case_id}: kernel_time_us must be positive",
    )
    require(
        isinstance(case["inner_chunks_independent"], bool),
        f"case {case_id}: inner_chunks_independent must be boolean",
    )

    outer_waves = case["outer_items"] / case["available_cores"]
    flat_waves = case["outer_items"] * case["inner_independent_chunks_per_item"] / case["available_cores"]
    require(
        math.isclose(case["outer_task_waves"], outer_waves, rel_tol=1e-3, abs_tol=1e-6),
        f"case {case_id}: outer_task_waves is inconsistent with items/cores",
    )
    require(
        math.isclose(case["flattened_task_waves"], flat_waves, rel_tol=1e-3, abs_tol=1e-6),
        f"case {case_id}: flattened_task_waves is inconsistent with items*chunks/cores",
    )

    lower = case["lower_bounds_us"]
    require(isinstance(lower, dict), f"case {case_id}: lower_bounds_us must be an object")
    for key in ("movement", "compute"):
        require(key in lower, f"case {case_id}: lower_bounds_us missing {key}")
        require(
            lower[key] is None or nonnegative_number(lower[key]),
            f"case {case_id}: lower bound {key} must be non-negative or null",
        )
    require(
        nonempty_text(case["lower_bound_evidence"]),
        f"case {case_id}: lower_bound_evidence must explain values or nulls",
    )
    require(
        isinstance(case["main_pipes"], list) and case["main_pipes"] and all(nonempty_text(item) for item in case["main_pipes"]),
        f"case {case_id}: main_pipes must be a non-empty string list",
    )
    validate_performance_target(case)
    source = case["source_window"]
    require(isinstance(source, dict), f"case {case_id}: source_window must be an object")
    access_pattern = require_choice(
        source.get("access_pattern"),
        ACCESS_PATTERNS,
        f"case {case_id}: invalid source-window access_pattern",
    )
    require(
        isinstance(source.get("bounded"), bool),
        f"case {case_id}: source_window.bounded must be boolean",
    )
    require(
        nonempty_text(source.get("index_regularity")),
        f"case {case_id}: source_window needs index_regularity",
    )
    require(
        nonempty_text(source.get("evidence")),
        f"case {case_id}: source_window needs evidence",
    )
    if access_pattern == "INDEXED_OR_REORDERED" and source["bounded"]:
        span = source.get("span_bytes")
        consumed = source.get("consumed_bytes")
        density = source.get("density")
        window_vregs = source.get("window_vregs")
        require(
            positive_number(span),
            f"case {case_id}: bounded source span must be positive",
        )
        require(
            positive_number(consumed) and consumed <= span,
            f"case {case_id}: invalid consumed source bytes",
        )
        require(
            isinstance(density, (int, float)) and not isinstance(density, bool) and 0 < density <= 1,
            f"case {case_id}: invalid source density",
        )
        require(
            math.isclose(density, consumed / span, rel_tol=1e-3, abs_tol=1e-6),
            f"case {case_id}: source density is inconsistent with consumed/span",
        )
        require_strict_int(
            window_vregs,
            1,
            f"case {case_id}: bounded source window needs positive window_vregs",
        )
        require(
            isinstance(source.get("register_window_feasible"), bool),
            f"case {case_id}: bounded source window needs register_window_feasible",
        )
        require(
            nonempty_text(source.get("window_scope")),
            f"case {case_id}: source window needs window_scope",
        )
        require(
            source.get("decomposition_checked") is True,
            f"case {case_id}: source window must check row/chunk decomposition",
        )
        if source["register_window_feasible"] is False:
            validate_hard_infeasibility(
                f"case {case_id}/register_window",
                source.get("register_infeasibility"),
            )
    require_choice(
        source.get("baseline_route_kind"),
        ROUTE_KINDS,
        f"case {case_id}: source_window needs baseline_route_kind",
    )
    require_choice(
        source.get("baseline_writeback_kind"),
        WRITEBACK_KINDS,
        f"case {case_id}: source_window needs baseline_writeback_kind",
    )
    require_choice(
        source.get("baseline_boundary_kind"),
        BOUNDARY_KINDS,
        f"case {case_id}: source_window needs baseline_boundary_kind",
    )
    baseline_value = source.get("baseline_boundary_value")
    require(
        valid_boundary_value(baseline_value),
        f"case {case_id}: source_window needs a positive or textual baseline_boundary_value",
    )
    require_string_list(
        case["obligation_ids"],
        f"case {case_id}: at least one distinct coverage obligation is required",
    )
    unknown = set(case["obligation_ids"]) - obligation_ids
    require(not unknown, f"case {case_id}: unknown obligations {sorted(unknown)}")


def validate_physical_identity(candidate_id: str, identity: dict) -> None:
    physical = identity.get("physical_dataflow")
    require(
        isinstance(physical, dict),
        f"candidate {candidate_id}: physical_dataflow must be structured",
    )
    require(
        nonempty_text(physical.get("description")),
        f"candidate {candidate_id}: physical_dataflow missing description",
    )
    route = require_choice(
        physical.get("route_kind"),
        ALL_ROUTE_KINDS,
        f"candidate {candidate_id}: invalid physical route {physical.get('route_kind')}",
    )
    source = require_choice(
        physical.get("source_access"),
        SOURCE_ACCESS_KINDS,
        f"candidate {candidate_id}: invalid source_access {physical.get('source_access')}",
    )
    require_choice(
        physical.get("writeback_kind"),
        WRITEBACK_KINDS,
        f"candidate {candidate_id}: invalid writeback_kind {physical.get('writeback_kind')}",
    )
    for field in (
        "full_payload_materialization",
        "register_lane_reorder",
        "indexed_lane_access",
        "route_experiment",
    ):
        require(
            isinstance(physical.get(field), bool),
            f"candidate {candidate_id}: physical_dataflow.{field} must be boolean",
        )

    if route == "MEMORY_INDEXED_VECTOR":
        require(
            physical["indexed_lane_access"],
            f"candidate {candidate_id}: memory-indexed route needs indexed lane access",
        )
    if route == "CONTIGUOUS_LOAD_REGISTER_REORDER":
        require(
            source == "CONTIGUOUS",
            f"candidate {candidate_id}: direct register route must load a contiguous source window",
        )
        require(
            physical["register_lane_reorder"],
            f"candidate {candidate_id}: direct register route needs register lane reorder",
        )
        require(
            not physical["full_payload_materialization"],
            f"candidate {candidate_id}: materializing a full intermediate payload is not the direct register route",
        )
        require(
            not physical["indexed_lane_access"],
            f"candidate {candidate_id}: indexed lane reads are memory-indexed, not direct register reorder",
        )
    if route == "MATERIALIZED_TRANSFORM":
        require(
            physical["full_payload_materialization"],
            f"candidate {candidate_id}: materialized transform must declare the intermediate payload pass",
        )


def validate_granularity_identity(candidate_id: str, identity: dict, applicable_cases: set[str]) -> None:
    facts = identity.get("granularity_facts")
    require(
        isinstance(facts, dict),
        f"candidate {candidate_id}: identity missing granularity_facts",
    )
    require(
        nonempty_text(facts.get("unit")),
        f"candidate {candidate_id}: granularity_facts missing unit",
    )
    kinds = facts.get("boundary_kinds")
    values = facts.get("values_by_case")
    require(
        isinstance(kinds, list) and kinds and all(isinstance(item, str) for item in kinds) and set(kinds) <= BOUNDARY_KINDS,
        f"candidate {candidate_id}: invalid granularity boundary_kinds",
    )
    require(
        isinstance(values, dict),
        f"candidate {candidate_id}: values_by_case must be an object",
    )
    require(
        set(values) == applicable_cases,
        f"candidate {candidate_id}: granularity values must cover applicable_cases exactly",
    )
    require(
        all(valid_boundary_value(value) for value in values.values()),
        f"candidate {candidate_id}: invalid granularity value",
    )


def validate_pipeline_identity(candidate_id: str, identity: dict) -> None:
    facts = identity.get("pipeline_facts")
    require(
        isinstance(facts, dict),
        f"candidate {candidate_id}: identity missing pipeline_facts",
    )
    stages = facts.get("stages")
    realization = require_choice(
        facts.get("realization"),
        PIPELINE_REALIZATIONS,
        f"candidate {candidate_id}: invalid pipeline realization",
    )
    require(
        isinstance(stages, int) and not isinstance(stages, bool) and stages > 0,
        f"candidate {candidate_id}: pipeline stages must be positive",
    )
    if stages == 1:
        require(
            realization == "SINGLE_STAGE",
            f"candidate {candidate_id}: stage 1 must use SINGLE_STAGE",
        )
    if stages > 1:
        require(
            realization != "SINGLE_STAGE",
            f"candidate {candidate_id}: multistage needs automatic/manual realization",
        )


def validate_events(events: list, case_ids: set[str], final: bool) -> tuple[dict, dict, dict, dict]:
    created_by_candidate: dict[str, dict] = {}
    result_by_candidate: dict[str, dict] = {}
    created_index: dict[str, int] = {}
    promoted_index: dict[str, int] = {}
    for index, event in enumerate(events):
        candidate_id = event.get("candidate_id")
        event_type = event.get("event")
        identity = event.get("identity")
        applicable = event.get("applicable_cases")
        require(
            nonempty_text(candidate_id),
            f"candidate event {index}: missing candidate_id",
        )
        require(
            candidate_id != BASELINE_ID,
            f"candidate event {index}: '{BASELINE_ID}' is reserved for the baseline reference; allocate a real candidate ID",
        )
        require_choice(
            event_type,
            {"CREATED", "RESULT"},
            f"candidate {candidate_id}: event must be CREATED or RESULT",
        )
        require(
            isinstance(identity, dict),
            f"candidate {candidate_id}: identity must be an object",
        )
        for axis in AXES:
            if axis != "physical_dataflow":
                require(
                    nonempty_text(identity.get(axis)),
                    f"candidate {candidate_id}: identity missing {axis}",
                )
        require(
            isinstance(applicable, list) and applicable and all(nonempty_text(item) for item in applicable),
            f"candidate {candidate_id}: applicable_cases must reference known cases",
        )
        require(
            not (set(applicable) - case_ids),
            f"candidate {candidate_id}: applicable_cases must reference known cases",
        )
        require(
            len(applicable) == len(set(applicable)),
            f"candidate {candidate_id}: duplicate applicable case",
        )
        validate_physical_identity(candidate_id, identity)
        validate_granularity_identity(candidate_id, identity, set(applicable))
        validate_pipeline_identity(candidate_id, identity)

        if candidate_id in created_by_candidate:
            created = created_by_candidate[candidate_id]
            require(
                event_type == "RESULT",
                f"candidate {candidate_id}: duplicate CREATED event",
            )
            require(
                identity == created["identity"],
                f"candidate {candidate_id}: identity changed; allocate a new ID",
            )
            require(
                applicable == created["applicable_cases"],
                f"candidate {candidate_id}: applicable_cases changed",
            )
            require(
                candidate_id not in result_by_candidate,
                f"candidate {candidate_id}: duplicate RESULT event",
            )
        else:
            require(
                event_type == "CREATED",
                f"candidate {candidate_id}: first event must be CREATED",
            )
            require(
                nonempty_text(event.get("hypothesis")),
                f"candidate {candidate_id}: CREATED needs hypothesis",
            )
            require(
                nonempty_text(event.get("gain_upper_bound")),
                f"candidate {candidate_id}: CREATED needs gain_upper_bound",
            )
            unresolved = sorted(cid for cid in created_by_candidate if cid not in result_by_candidate)
            require(
                not unresolved,
                f"candidate {candidate_id}: created before previous candidates were resolved: {unresolved}",
            )
            created_by_candidate[candidate_id] = event
            created_index[candidate_id] = index

        if event_type == "RESULT":
            result_status = require_choice(
                event.get("result_status"),
                CANDIDATE_RESULTS,
                f"candidate {candidate_id}: invalid result_status",
            )
            require(
                evidence_list(event.get("evidence")),
                f"candidate {candidate_id}: result needs evidence",
            )
            case_results = event.get("case_results")
            require(
                isinstance(case_results, list),
                f"candidate {candidate_id}: RESULT needs case_results",
            )
            by_case = {item.get("case_id"): item for item in case_results if isinstance(item, dict)}
            require(
                len(by_case) == len(case_results),
                f"candidate {candidate_id}: invalid or duplicate case_results",
            )
            require(
                set(by_case) == set(applicable),
                f"candidate {candidate_id}: case_results must cover applicable_cases exactly",
            )
            for case_id, item in by_case.items():
                disposition = require_choice(
                    item.get("disposition"),
                    RESULT_CASE_DISPOSITIONS,
                    f"candidate {candidate_id}/{case_id}: invalid case disposition",
                )
                require(
                    evidence_list(item.get("evidence")),
                    f"candidate {candidate_id}/{case_id}: case result needs evidence",
                )
                if disposition == "TESTED_PATH":
                    measurement = item.get("performance_measurement")
                    require(
                        isinstance(measurement, dict),
                        f"candidate {candidate_id}/{case_id}: TESTED_PATH needs performance_measurement",
                    )
                    require(
                        positive_number(measurement.get("kernel_time_us")),
                        f"candidate {candidate_id}/{case_id}: invalid measured kernel_time_us",
                    )
                    require(
                        nonempty_text(measurement.get("artifact")),
                        f"candidate {candidate_id}/{case_id}: performance artifact required",
                    )
            blocked_cases = [case_id for case_id, item in by_case.items() if item.get("disposition") == "NOT_RUN_BLOCKED"]
            if result_status == "BLOCKED_BY_ENVIRONMENT":
                require(
                    blocked_cases,
                    f"candidate {candidate_id}: blocked result needs NOT_RUN_BLOCKED cases",
                )
                validate_blocking_evidence(f"candidate {candidate_id}", event.get("blocking_evidence"))
            else:
                require(
                    not blocked_cases,
                    f"candidate {candidate_id}: NOT_RUN_BLOCKED requires BLOCKED_BY_ENVIRONMENT",
                )
            tested_cases = [case_id for case_id, item in by_case.items() if item.get("disposition") == "TESTED_PATH"]
            rejection = event.get("rejection_evidence")
            if rejection is not None:
                validate_rejection_evidence(f"candidate {candidate_id}", rejection)
            if result_status == "REJECTED_WITH_EVIDENCE" and not tested_cases:
                require(
                    rejection is not None,
                    f"candidate {candidate_id}: unexecuted rejection needs structured rejection_evidence",
                )
            if result_status == "PROMOTED":
                require(
                    tested_cases,
                    f"candidate {candidate_id}: PROMOTED needs at least one TESTED_PATH case",
                )
                promoted_index[candidate_id] = index
            result_by_candidate[candidate_id] = event

    if final:
        require(
            not (set(created_by_candidate) - set(result_by_candidate)),
            "a CREATED candidate has no RESULT event",
        )
    return created_by_candidate, result_by_candidate, created_index, promoted_index


def candidate_tested_case(candidate_id: str, case_id: str, results: dict) -> bool:
    result = results.get(candidate_id)
    if result is None:
        return False
    return any(item["case_id"] == case_id and item["disposition"] == "TESTED_PATH" for item in result["case_results"])


def candidate_blocked_case(candidate_id: str, case_id: str, results: dict) -> bool:
    """True only when this candidate's own environment block left the case unmeasured."""
    result = results.get(candidate_id)
    if result is None or result.get("result_status") != "BLOCKED_BY_ENVIRONMENT":
        return False
    return any(
        isinstance(item, dict) and item.get("case_id") == case_id and item.get("disposition") == "NOT_RUN_BLOCKED"
        for item in result.get("case_results", [])
    )


def measured_kernel_time(candidate_id: str, case_id: str, results: dict) -> float | None:
    """Return the candidate's recorded TESTED_PATH measurement for one case."""
    result = results.get(candidate_id)
    if result is None:
        return None
    for item in result.get("case_results", []):
        if isinstance(item, dict) and item.get("case_id") == case_id and item.get("disposition") == "TESTED_PATH":
            return item["performance_measurement"]["kernel_time_us"]
    return None


def selected_best_implementation(case: dict, promoted_cases: dict[str, set[str]], results: dict) -> tuple[str, float]:
    """Return the fastest tested implementation: B0 or the best promoted candidate."""
    case_id = case["case_id"]
    selected_id = BASELINE_ID
    selected_time = case["kernel_time_us"]
    for candidate_id in promoted_cases:
        if case_id not in promoted_cases[candidate_id]:
            continue
        measured = measured_kernel_time(candidate_id, case_id, results)
        if measured is not None and measured < selected_time:
            selected_id = candidate_id
            selected_time = measured
    return selected_id, selected_time


def validate_best_measurement_binding(cases: list[dict], promoted_cases: dict[str, set[str]], results: dict) -> None:
    """Bind the claimed best time to the selected implementation's measurement.

    A fresh final measurement may supersede the original promotion value, but it must be
    recorded on the mutable case entry and must not rewrite the immutable RESULT event.
    """
    for case in cases:
        case_id = case["case_id"]
        selected_id, measured = selected_best_implementation(case, promoted_cases, results)
        claim = case["best_metrics"]["kernel_time_us"]
        fresh = case.get("final_measurement")
        if fresh is None:
            require(
                math.isclose(claim, measured, rel_tol=1e-3, abs_tol=1e-6),
                f"case {case_id}: best kernel_time_us {claim} is not the measured {measured} of the selected "
                f"best implementation {selected_id}; record a final_measurement for a fresh measurement",
            )
            continue
        require(isinstance(fresh, dict), f"case {case_id}: final_measurement must be an object")
        require(
            fresh.get("implementation_id") == selected_id,
            f"case {case_id}: final_measurement names {fresh.get('implementation_id')!r}, "
            f"but the selected best implementation is {selected_id}",
        )
        require(
            positive_number(fresh.get("kernel_time_us")),
            f"case {case_id}: final_measurement needs a positive kernel_time_us",
        )
        require(
            nonempty_text(fresh.get("artifact")),
            f"case {case_id}: final_measurement needs an artifact",
        )
        require(
            math.isclose(fresh["kernel_time_us"], claim, rel_tol=1e-3, abs_tol=1e-6),
            f"case {case_id}: best kernel_time_us {claim} disagrees with the final_measurement {fresh['kernel_time_us']}",
        )


def validate_disposition(
    owner: str,
    disposition: dict,
    valid_cases: set[str],
    obligation_candidates: set[str],
    created: dict,
    results: dict,
    final: bool,
) -> None:
    status = require_choice(disposition.get("status"), CASE_DISPOSITIONS, f"{owner}: invalid status {disposition.get('status')}")
    candidate_ids = disposition.get("candidate_ids", [])
    require(
        isinstance(candidate_ids, list) and all(nonempty_text(item) for item in candidate_ids),
        f"{owner}: candidate_ids must be a string list",
    )
    require(
        len(set(candidate_ids)) == len(candidate_ids),
        f"{owner}: duplicate candidate_ids",
    )
    require(
        set(candidate_ids) <= obligation_candidates,
        f"{owner}: candidate not linked by obligation",
    )
    require(set(candidate_ids) <= set(created), f"{owner}: unknown candidate")
    if status == "TESTED_PATH":
        require(candidate_ids, f"{owner}: TESTED_PATH needs a candidate")
        for case_id in valid_cases:
            require(
                any(candidate_tested_case(cid, case_id, results) for cid in candidate_ids),
                f"{owner}: no referenced candidate tested path for {case_id}",
            )
    if status in {"BASELINE_MEASURED", "CLOSED_WITH_EVIDENCE"}:
        require(
            evidence_list(disposition.get("evidence")),
            f"{owner}: closure/baseline needs evidence",
        )
    if status == "CLOSED_WITH_EVIDENCE":
        validate_hard_infeasibility(owner, disposition.get("hard_infeasibility"))
    if status == "BLOCKED":
        require(candidate_ids, f"{owner}: BLOCKED needs a blocked candidate")
        require(
            evidence_list(disposition.get("evidence")),
            f"{owner}: BLOCKED needs evidence",
        )
        require(
            all(any(candidate_blocked_case(cid, case_id, results) for cid in candidate_ids) for case_id in valid_cases),
            f"{owner}: BLOCKED must reference a BLOCKED_BY_ENVIRONMENT result for {sorted(valid_cases)}",
        )
    if final:
        require(
            status not in {"PENDING", "BLOCKED"},
            f"{owner}: still {status} at final gate",
        )


def validate_case_dispositions(
    obligation: dict,
    linked_cases: set[str],
    obligation_candidates: set[str],
    created: dict,
    results: dict,
    final: bool,
) -> dict[str, dict]:
    oid = obligation["obligation_id"]
    dispositions = obligation.get("case_dispositions")
    require(
        isinstance(dispositions, list),
        f"obligation {oid}: case_dispositions must be a list",
    )
    by_case = {item.get("case_id"): item for item in dispositions if isinstance(item, dict)}
    require(
        len(by_case) == len(dispositions),
        f"obligation {oid}: invalid or duplicate case_dispositions",
    )
    require(
        set(by_case) == linked_cases,
        f"obligation {oid}: case_dispositions must cover case_ids exactly",
    )
    for case_id, disposition in by_case.items():
        validate_disposition(
            f"obligation {oid}/{case_id}",
            disposition,
            {case_id},
            obligation_candidates,
            created,
            results,
            final,
        )
    return by_case


def validate_route_plan(
    obligation: dict,
    linked_cases: set[str],
    obligation_candidates: set[str],
    created: dict,
    results: dict,
    case_by_id: dict[str, dict],
    final: bool,
) -> None:
    oid = obligation["obligation_id"]
    plan = obligation.get("route_plan")
    require(isinstance(plan, list) and plan, f"obligation {oid}: route_plan is required")
    seen_ids: set[str] = set()
    covered_by_route = {route: set() for route in ROUTE_KINDS}
    for item in plan:
        plan_id = item.get("plan_id")
        route = item.get("route_kind")
        writeback = item.get("writeback_kind")
        item_cases = item.get("case_ids")
        require(
            nonempty_text(plan_id) and plan_id not in seen_ids,
            f"obligation {oid}: invalid route plan_id",
        )
        seen_ids.add(plan_id)
        require_choice(route, ROUTE_KINDS, f"obligation {oid}/{plan_id}: invalid route_kind")
        require_choice(
            writeback,
            WRITEBACK_KINDS,
            f"obligation {oid}/{plan_id}: invalid writeback_kind",
        )
        require(
            isinstance(item_cases, list) and item_cases,
            f"obligation {oid}/{plan_id}: case_ids required",
        )
        require(set(item_cases) <= linked_cases, f"obligation {oid}/{plan_id}: unknown case")
        covered_by_route[route].update(item_cases)
        validate_disposition(
            f"obligation {oid}/{plan_id}",
            item,
            set(item_cases),
            obligation_candidates,
            created,
            results,
            final,
        )
        if item.get("status") == "BASELINE_MEASURED":
            for case_id in item_cases:
                source = case_by_id[case_id]["source_window"]
                require(
                    route == source["baseline_route_kind"] and writeback == source["baseline_writeback_kind"],
                    f"obligation {oid}/{plan_id}: BASELINE_MEASURED route/writeback must match the recorded baseline for {case_id}",
                )
        if route == "CONTIGUOUS_LOAD_REGISTER_REORDER" and item.get("status") == "CLOSED_WITH_EVIDENCE":
            feasible_cases = [
                case_id for case_id in item_cases if case_by_id[case_id]["source_window"].get("register_window_feasible") is True
            ]
            if feasible_cases:
                api_search = item.get("api_search_evidence", {})
                require(
                    isinstance(api_search.get("paths"), list)
                    and api_search["paths"]
                    and all(nonempty_text(value) for value in api_search["paths"])
                    and isinstance(api_search.get("symbols"), list)
                    and api_search["symbols"]
                    and all(nonempty_text(value) for value in api_search["symbols"])
                    and nonempty_text(api_search.get("lowering_finding")),
                    f"obligation {oid}/{plan_id}: feasible direct route closure needs exact API/lowering search",
                )
        if item.get("status") == "TESTED_PATH":
            for cid in item.get("candidate_ids", []):
                require(
                    created[cid]["identity"]["physical_dataflow"]["route_kind"] == route,
                    f"obligation {oid}/{plan_id}: candidate {cid} has a different physical route",
                )
                require(
                    created[cid]["identity"]["physical_dataflow"]["writeback_kind"] == writeback,
                    f"obligation {oid}/{plan_id}: candidate {cid} has a different output writeback",
                )
    for route, covered_cases in covered_by_route.items():
        require(
            covered_cases == linked_cases,
            f"obligation {oid}: route {route} must disposition every linked case",
        )


def validate_boundary_plan(
    obligation: dict,
    linked_cases: set[str],
    obligation_candidates: set[str],
    created: dict,
    results: dict,
    case_by_id: dict[str, dict],
    final: bool,
) -> None:
    oid = obligation["obligation_id"]
    plan = obligation.get("boundary_plan")
    require(isinstance(plan, list) and plan, f"obligation {oid}: boundary_plan is required")
    seen_ids: set[str] = set()
    covered_by_kind = {kind: set() for kind in BOUNDARY_KINDS}
    for item in plan:
        plan_id = item.get("plan_id")
        kind = item.get("boundary_kind")
        case_values = item.get("case_values")
        require(
            nonempty_text(plan_id) and plan_id not in seen_ids,
            f"obligation {oid}: invalid boundary plan_id",
        )
        seen_ids.add(plan_id)
        require_choice(kind, BOUNDARY_KINDS, f"obligation {oid}/{plan_id}: invalid boundary_kind")
        require(
            isinstance(case_values, dict) and case_values,
            f"obligation {oid}/{plan_id}: case_values required",
        )
        item_cases = set(case_values)
        require(item_cases <= linked_cases, f"obligation {oid}/{plan_id}: unknown case")
        require(
            all(valid_boundary_value(value) for value in case_values.values()),
            f"obligation {oid}/{plan_id}: invalid value",
        )
        covered_by_kind[kind].update(item_cases)
        validate_disposition(
            f"obligation {oid}/{plan_id}",
            item,
            item_cases,
            obligation_candidates,
            created,
            results,
            final,
        )
        if item.get("status") == "BASELINE_MEASURED":
            for case_id, value in case_values.items():
                baseline = case_by_id[case_id]["source_window"]
                require(
                    kind == baseline["baseline_boundary_kind"] and value == baseline["baseline_boundary_value"],
                    f"obligation {oid}/{plan_id}: BASELINE_MEASURED boundary must match the recorded "
                    f"baseline {baseline['baseline_boundary_kind']}={baseline['baseline_boundary_value']!r} "
                    f"for {case_id}",
                )
        if item.get("status") == "TESTED_PATH":
            for case_id, value in case_values.items():
                matching = [
                    cid
                    for cid in item.get("candidate_ids", [])
                    if kind in created[cid]["identity"]["granularity_facts"]["boundary_kinds"]
                    and created[cid]["identity"]["granularity_facts"]["values_by_case"].get(case_id) == value
                    and candidate_tested_case(cid, case_id, results)
                ]
                require(
                    matching,
                    f"obligation {oid}/{plan_id}: no candidate tested {kind}={value} for {case_id}",
                )
    for kind, covered_cases in covered_by_kind.items():
        require(
            covered_cases == linked_cases,
            f"obligation {oid}: boundary {kind} must disposition every linked case",
        )


def validate_interaction_plan(
    obligation: dict,
    linked_cases: set[str],
    obligation_candidates: set[str],
    created: dict,
    results: dict,
    case_by_id: dict[str, dict],
    final: bool,
) -> None:
    oid = obligation["obligation_id"]
    plan = obligation.get("interaction_plan")
    require(
        isinstance(plan, list) and plan,
        f"obligation {oid}: interaction_plan is required",
    )
    covered_cases: set[str] = set()
    seen_ids: set[str] = set()
    for item in plan:
        plan_id = item.get("plan_id")
        route = item.get("route_kind")
        writeback = item.get("writeback_kind")
        kind = item.get("boundary_kind")
        case_values = item.get("case_values")
        require(
            nonempty_text(plan_id) and plan_id not in seen_ids,
            f"obligation {oid}: invalid interaction plan_id",
        )
        seen_ids.add(plan_id)
        require_choice(route, ROUTE_KINDS, f"obligation {oid}/{plan_id}: invalid route_kind")
        require_choice(
            writeback,
            WRITEBACK_KINDS,
            f"obligation {oid}/{plan_id}: invalid writeback_kind",
        )
        require_choice(kind, BOUNDARY_KINDS, f"obligation {oid}/{plan_id}: invalid boundary_kind")
        require(
            isinstance(case_values, dict) and case_values,
            f"obligation {oid}/{plan_id}: case_values required",
        )
        item_cases = set(case_values)
        require(item_cases <= linked_cases, f"obligation {oid}/{plan_id}: unknown case")
        require(
            all(valid_boundary_value(value) for value in case_values.values()),
            f"obligation {oid}/{plan_id}: invalid value",
        )
        covered_cases.update(item_cases)
        validate_disposition(
            f"obligation {oid}/{plan_id}",
            item,
            item_cases,
            obligation_candidates,
            created,
            results,
            final,
        )
        if item.get("status") == "BASELINE_MEASURED":
            for case_id, value in case_values.items():
                baseline = case_by_id[case_id]["source_window"]
                require(
                    route == baseline["baseline_route_kind"] and writeback == baseline["baseline_writeback_kind"],
                    f"obligation {oid}/{plan_id}: BASELINE_MEASURED route/writeback must match the recorded baseline for {case_id}",
                )
                require(
                    kind == baseline["baseline_boundary_kind"] and value == baseline["baseline_boundary_value"],
                    f"obligation {oid}/{plan_id}: BASELINE_MEASURED boundary must match the recorded "
                    f"baseline {baseline['baseline_boundary_kind']}={baseline['baseline_boundary_value']!r} "
                    f"for {case_id}",
                )
        if item.get("status") == "TESTED_PATH":
            for case_id, value in case_values.items():
                matching = []
                for cid in item.get("candidate_ids", []):
                    identity = created[cid]["identity"]
                    physical = identity["physical_dataflow"]
                    granularity = identity["granularity_facts"]
                    if (
                        physical["route_kind"] == route
                        and physical["writeback_kind"] == writeback
                        and kind in granularity["boundary_kinds"]
                        and granularity["values_by_case"].get(case_id) == value
                        and candidate_tested_case(cid, case_id, results)
                    ):
                        matching.append(cid)
                require(
                    matching,
                    f"obligation {oid}/{plan_id}: no candidate tested required combination for {case_id}",
                )
    require(
        covered_cases == linked_cases,
        f"obligation {oid}: interaction_plan must disposition every linked case",
    )


def validate_direct_route_priority(
    cases: list[dict],
    obligations: list[dict],
    created: dict[str, dict],
) -> None:
    """A feasible register route must be the first non-baseline physical route."""
    case_by_id = {case["case_id"]: case for case in cases}
    for candidate_id, event in created.items():
        physical = event["identity"]["physical_dataflow"]
        changed_from_baseline = any(
            physical["route_kind"] != case_by_id[case_id]["source_window"]["baseline_route_kind"]
            or physical["writeback_kind"] != case_by_id[case_id]["source_window"]["baseline_writeback_kind"]
            for case_id in event["applicable_cases"]
        )
        require(
            physical["route_experiment"] is changed_from_baseline,
            f"candidate {candidate_id}: route_experiment disagrees with baseline route/writeback facts",
        )

    for case in cases:
        case_id = case["case_id"]
        source = case["source_window"]
        if source.get("register_window_feasible") is not True:
            continue

        direct_dispositions = [
            item
            for obligation in obligations
            if obligation.get("kind") == "physical_dataflow_routes" and case_id in obligation.get("case_ids", [])
            for item in obligation.get("route_plan", [])
            if item.get("route_kind") == "CONTIGUOUS_LOAD_REGISTER_REORDER" and case_id in item.get("case_ids", [])
        ]
        require(
            direct_dispositions,
            f"case {case_id}: direct-register route disposition required",
        )
        closed_statuses = {"BASELINE_MEASURED", "CLOSED_WITH_EVIDENCE"}
        if all(item.get("status") in closed_statuses for item in direct_dispositions):
            continue

        route_experiments = [
            (candidate_id, event)
            for candidate_id, event in created.items()
            if case_id in event["applicable_cases"]
            and (
                event["identity"]["physical_dataflow"]["route_kind"] != source["baseline_route_kind"]
                or event["identity"]["physical_dataflow"]["writeback_kind"] != source["baseline_writeback_kind"]
            )
        ]
        if not route_experiments:
            continue
        first_id, first_event = route_experiments[0]
        require(
            first_event["identity"]["physical_dataflow"]["route_kind"] == "CONTIGUOUS_LOAD_REGISTER_REORDER",
            f"case {case_id}: first physical-route experiment {first_id} must test the feasible direct-register route",
        )


def blocker_map(events: list, cases: list[dict]) -> dict[str, list[str]]:
    """Map each case to the candidates whose environment block left it unmeasured."""
    blocked = {case["case_id"]: blocked_candidates_for_case(events, case["case_id"]) for case in cases}
    return {case_id: candidates for case_id, candidates in blocked.items() if candidates}


def validate_blocked_resolutions(entries: object, blocked: dict[str, list[str]]) -> None:
    """Require a recorded recovery for every blocked case, met target or not."""
    if not blocked:
        require(
            entries is None or entries == [],
            "blocked_resolutions is only allowed for an environment-blocked case",
        )
        return
    require(
        isinstance(entries, list) and entries,
        f"environment-blocked case(s) {sorted(blocked)} need a recorded resolution before convergence",
    )
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        require(
            isinstance(entry, dict),
            "every blocked_resolutions entry must be an object",
        )
        case_id = entry.get("case_id")
        require(
            case_id in blocked,
            f"blocked resolution names unblocked or unknown case {case_id!r}",
        )
        candidate_id = entry.get("candidate_id")
        require(
            candidate_id in blocked[case_id],
            f"case {case_id}: blocked resolution must name one of {blocked[case_id]}",
        )
        require(
            (case_id, candidate_id) not in seen,
            f"case {case_id}: duplicate blocked resolution for {candidate_id}",
        )
        seen.add((case_id, candidate_id))
        require(
            nonempty_text(entry.get("recovery")),
            f"case {case_id}: blocked resolution for {candidate_id} needs the recovery description",
        )
        require(
            evidence_list(entry.get("evidence")),
            f"case {case_id}: blocked resolution for {candidate_id} needs evidence",
        )
    missing = sorted({(case_id, cid) for case_id, cids in blocked.items() for cid in cids} - seen)
    require(
        not missing,
        f"blocked resolutions missing for {missing}; keep the original BLOCKED_BY_ENVIRONMENT RESULT in the event log",
    )


def validate_convergence_evidence(data: dict, cases: list[dict]) -> None:
    evidence_key = "convergence_evidence" if all(case["performance_target"] is None for case in cases) else "unmet_convergence_evidence"
    convergence = data.get(evidence_key)
    require(isinstance(convergence, dict), f"convergence needs {evidence_key}")
    entries = convergence.get("case_bounds")
    require(isinstance(entries, list), "unmet convergence needs case_bounds")
    by_case = {item.get("case_id"): item for item in entries if isinstance(item, dict)}
    require(
        len(by_case) == len(entries),
        "every case_bounds entry must be an object with a case_id",
    )
    unmet_cases = {case["case_id"] for case in cases if not validate_performance_target(case)}
    require(set(by_case) == unmet_cases, "case_bounds must cover every and only unmet case")
    for case_id, entry in by_case.items():
        require(
            positive_number(entry.get("verified_lower_bound_us")),
            f"case {case_id}: verified lower bound required",
        )
        require(
            positive_number(entry.get("best_kernel_time_us")),
            f"case {case_id}: best kernel time required",
        )
        case = next(item for item in cases if item["case_id"] == case_id)
        require(
            math.isclose(
                entry["best_kernel_time_us"],
                case["best_metrics"]["kernel_time_us"],
                rel_tol=1e-3,
                abs_tol=1e-6,
            ),
            f"case {case_id}: convergence best time disagrees with best_metrics",
        )
        require(
            entry["best_kernel_time_us"] >= entry["verified_lower_bound_us"],
            f"case {case_id}: lower bound exceeds measured best time",
        )
        require(
            evidence_list(entry.get("evidence")),
            f"case {case_id}: lower-bound evidence required",
        )
        require(
            nonempty_text(entry.get("remaining_gap_explanation")),
            f"case {case_id}: remaining gap explanation required",
        )


def resolve_pipeline_basis(
    oid: str,
    obligation: dict,
    linked_cases: set[str],
    created: dict[str, dict],
) -> dict[str, str]:
    """Resolve one pipeline basis for every linked case."""
    basis = obligation.get("basis_candidate_id")
    if isinstance(basis, str):
        require(
            nonempty_text(basis),
            f"obligation {oid}: basis_candidate_id must not be empty",
        )
        resolved = {case_id: basis for case_id in linked_cases}
    elif isinstance(basis, dict):
        require(
            set(basis) == linked_cases,
            f"obligation {oid}: per-case basis_candidate_id must map every linked case exactly",
        )
        require(
            all(nonempty_text(value) for value in basis.values()),
            f"obligation {oid}: every per-case basis must name B0 or a candidate",
        )
        resolved = dict(basis)
    else:
        raise ValidationError(f"obligation {oid}: basis_candidate_id must be B0, a candidate ID, or a per-case map")
    for case_id, value in resolved.items():
        require(
            value == BASELINE_ID or value in created,
            f"obligation {oid}/{case_id}: basis {value} must be B0 or a known candidate",
        )
        require(
            value == BASELINE_ID or case_id in created[value]["applicable_cases"],
            f"obligation {oid}/{case_id}: basis candidate {value} does not activate this case",
        )
    return resolved


def pipeline_b0_identities(
    oid: str,
    obligation: dict,
    b0_cases: set[str],
) -> dict[str, dict]:
    """Read the recorded baseline identity for every case measured on B0."""
    raw = obligation.get("basis_identity")
    require(
        isinstance(raw, dict),
        f"obligation {oid}: the B0 basis needs basis_identity",
    )
    shared = "granularity_facts" in raw or any(axis in raw for axis in PIPELINE_PAIRING_AXES)
    if set(raw) == b0_cases:
        entries = {case_id: raw[case_id] for case_id in b0_cases}
        per_case = True
    elif shared:
        entries = {case_id: raw for case_id in b0_cases}
        per_case = False
    else:
        raise ValidationError(
            f"obligation {oid}: basis_identity must either describe one shared B0 identity or map exactly the B0 cases {sorted(b0_cases)}"
        )
    identities = {}
    for case_id, entry in entries.items():
        require(
            isinstance(entry, dict),
            f"obligation {oid}/{case_id}: basis_identity entry must be an object",
        )
        identity = {}
        for axis in PIPELINE_PAIRING_AXES:
            require(
                nonempty_text(entry.get(axis)),
                f"obligation {oid}/{case_id}: basis_identity missing {axis}",
            )
            identity[axis] = entry[axis]
        facts = entry.get("granularity_facts")
        require(
            isinstance(facts, dict),
            f"obligation {oid}/{case_id}: B0 basis_identity needs granularity_facts",
        )
        require(
            nonempty_text(facts.get("unit")),
            f"obligation {oid}/{case_id}: B0 granularity_facts missing unit",
        )
        values = facts.get("values_by_case")
        expected_cases = {case_id} if per_case else b0_cases
        require(
            isinstance(values, dict) and set(values) == expected_cases,
            f"obligation {oid}/{case_id}: B0 granularity_facts.values_by_case must cover exactly {sorted(expected_cases)}",
        )
        require(
            all(valid_boundary_value(values[item]) for item in expected_cases),
            f"obligation {oid}/{case_id}: invalid B0 granularity value",
        )
        identity["granularity_unit"] = facts["unit"]
        identity["granularity_value"] = values[case_id]
        identities[case_id] = identity
    return identities


def candidate_pipeline_identity(candidate_id: str, case_id: str, created: dict[str, dict]) -> dict:
    """Return the evidence identity that must match a candidate basis."""
    identity = created[candidate_id]["identity"]
    facts = identity["granularity_facts"]
    return {
        **{axis: identity[axis] for axis in PIPELINE_PAIRING_AXES},
        "granularity_unit": facts["unit"],
        "granularity_value": facts["values_by_case"][case_id],
    }


def case_promotion_indices(
    case_id: str,
    promoted_cases: dict[str, set[str]],
    promoted_index: dict[str, int],
) -> list[int]:
    """Return the promotions that actually measured this case."""
    return sorted(index for candidate_id, index in promoted_index.items() if case_id in promoted_cases[candidate_id])


def blocked_candidates_for_case(events: list, case_id: str) -> list[str]:
    """Return candidates whose environment block left the case unmeasured."""
    blocked = set()
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("event") != "RESULT" or event.get("result_status") != "BLOCKED_BY_ENVIRONMENT":
            continue
        for item in event.get("case_results", []):
            if isinstance(item, dict) and item.get("case_id") == case_id and item.get("disposition") == "NOT_RUN_BLOCKED":
                blocked.add(event.get("candidate_id"))
    return sorted(blocked)


def validate(path: Path, final: bool, allow_unmet_convergence: bool) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), "record must be a JSON object")
    require(data.get("schema_version") == 3, "schema_version must be 3")
    cases = data.get("cases")
    obligations = data.get("obligations")
    events = data.get("candidate_events")
    require(isinstance(cases, list) and cases, "cases must be a non-empty list")
    require(
        isinstance(obligations, list) and obligations,
        "obligations must be a non-empty list",
    )
    require(isinstance(events, list), "candidate_events must be a list")
    for label, items in (
        ("cases", cases),
        ("obligations", obligations),
        ("candidate_events", events),
    ):
        require(
            all(isinstance(item, dict) for item in items),
            f"every entry in {label} must be an object",
        )

    obligation_by_id: dict[str, dict] = {}
    for obligation in obligations:
        oid = obligation.get("obligation_id")
        require(nonempty_text(oid), "every obligation needs obligation_id")
        require(oid not in obligation_by_id, f"duplicate obligation_id {oid}")
        obligation_by_id[oid] = obligation

    case_ids: set[str] = set()
    case_by_id: dict[str, dict] = {}
    for case in cases:
        case_id = case.get("case_id")
        require(nonempty_text(case_id), "every case needs case_id")
        require(case_id not in case_ids, f"duplicate case_id {case_id}")
        case_ids.add(case_id)
        case_by_id[case_id] = case
        validate_case(case, set(obligation_by_id))

    targets_met = all(validate_performance_target(case) for case in cases)
    no_numeric_target = all(case["performance_target"] is None for case in cases)
    strict_closure = final and not targets_met and (allow_unmet_convergence or no_numeric_target)

    created, results, created_index, promoted_index = validate_events(events, case_ids, final)
    promoted_cases = {
        candidate_id: {item["case_id"] for item in results[candidate_id]["case_results"] if item["disposition"] == "TESTED_PATH"}
        for candidate_id in promoted_index
    }
    obligation_kinds: dict[str, list[dict]] = {}
    pipeline_bases: dict[tuple[str, str], str] = {}
    for obligation in obligations:
        oid = obligation["obligation_id"]
        kind = obligation.get("kind")
        status = obligation.get("status")
        linked_cases_list = obligation.get("case_ids")
        candidate_ids_list = obligation.get("candidate_ids")
        kind = require_choice(
            obligation.get("kind"),
            OBLIGATION_KINDS,
            f"obligation {oid}: unsupported kind {obligation.get('kind')}",
        )
        status = require_choice(
            obligation.get("status"),
            OBLIGATION_STATUSES,
            f"obligation {oid}: invalid status {obligation.get('status')}",
        )
        require_string_list(
            linked_cases_list,
            f"obligation {oid}: distinct case_ids required",
        )
        linked_cases = set(linked_cases_list)
        require(not (linked_cases - case_ids), f"obligation {oid}: references unknown cases")
        for linked_case in linked_cases:
            case = next(item for item in cases if item["case_id"] == linked_case)
            require(
                oid in case["obligation_ids"],
                f"obligation {oid}: missing reverse link from {linked_case}",
            )
        require(
            isinstance(candidate_ids_list, list)
            and all(nonempty_text(item) for item in candidate_ids_list)
            and len(set(candidate_ids_list)) == len(candidate_ids_list),
            f"obligation {oid}: candidate_ids must be a distinct string list",
        )
        obligation_candidates = set(candidate_ids_list)
        require(
            obligation_candidates <= set(created),
            f"obligation {oid}: references unknown candidate",
        )
        case_dispositions = validate_case_dispositions(
            obligation,
            linked_cases,
            obligation_candidates,
            created,
            results,
            strict_closure,
        )

        if status == "SATISFIED":
            for case_id in linked_cases:
                disposition = case_dispositions[case_id]
                case_status = disposition.get("status")
                if case_status == "CLOSED_WITH_EVIDENCE":
                    continue
                require(
                    case_status == "TESTED_PATH",
                    f"obligation {oid}: SATISFIED cannot rest on {case_status} for {case_id}; "
                    "record a linked candidate that tested the case or a validated closure proof",
                )
                require(
                    any(candidate_tested_case(cid, case_id, results) for cid in disposition.get("candidate_ids", [])),
                    f"obligation {oid}: SATISFIED needs a linked candidate that tested {case_id}",
                )
        if status == "CLOSED_WITH_EVIDENCE":
            require(
                evidence_list(obligation.get("closure_evidence")),
                f"obligation {oid}: closure evidence required",
            )
            validate_hard_infeasibility(f"obligation {oid}", obligation.get("hard_infeasibility"))
        if status == "BLOCKED":
            require(
                candidate_ids_list,
                f"obligation {oid}: BLOCKED needs a blocked candidate",
            )
            require(
                evidence_list(obligation.get("blocking_reasons")),
                f"obligation {oid}: blocking reasons required",
            )
            require(
                all(any(candidate_blocked_case(cid, case_id, results) for cid in candidate_ids_list) for case_id in linked_cases),
                f"obligation {oid}: BLOCKED must reference a BLOCKED_BY_ENVIRONMENT result for {sorted(linked_cases)}",
            )
        closure_basis = obligation.get("closure_basis")
        if closure_basis is not None:
            require_choice(
                closure_basis,
                {"STRICT_COST_DOMINANCE"},
                f"obligation {oid}: invalid closure_basis {closure_basis}",
            )
        if closure_basis == "STRICT_COST_DOMINANCE":
            costs = obligation.get("cost_evidence", {})
            require(
                costs.get("same_payload") is True
                and nonempty_text(costs.get("reference_path"))
                and nonempty_text(costs.get("candidate_path")),
                f"obligation {oid}: strict dominance needs same-payload path costs",
            )

        if kind == "physical_dataflow_routes":
            validate_route_plan(
                obligation,
                linked_cases,
                obligation_candidates,
                created,
                results,
                case_by_id,
                strict_closure,
            )
        if kind == "work_granularity_search":
            validate_boundary_plan(
                obligation,
                linked_cases,
                obligation_candidates,
                created,
                results,
                case_by_id,
                strict_closure,
            )
        if kind == "granularity_dataflow_interaction":
            validate_interaction_plan(
                obligation,
                linked_cases,
                obligation_candidates,
                created,
                results,
                case_by_id,
                strict_closure,
            )
        if kind == "pipeline_realization":
            plan = obligation.get("pipeline_plan", {})
            require(isinstance(plan, dict), f"obligation {oid}: pipeline_plan must be an object")
            automatic = plan.get("automatic", {})
            manual_plan = plan.get("manual", {})
            require(isinstance(automatic, dict), f"obligation {oid}: pipeline_plan.automatic must be an object")
            require(isinstance(manual_plan, dict), f"obligation {oid}: pipeline_plan.manual must be an object")
            auto = automatic.get("status")
            manual = manual_plan.get("status")
            basis_by_case = resolve_pipeline_basis(oid, obligation, linked_cases, created)
            b0_cases = {case_id for case_id, basis in basis_by_case.items() if basis == BASELINE_ID}
            if b0_cases:
                b0_identities = pipeline_b0_identities(oid, obligation, b0_cases)
            else:
                require(
                    obligation.get("basis_identity") is None,
                    f"obligation {oid}: basis_identity is only allowed for cases measured on B0",
                )
                b0_identities = {}
            basis_identities = {}
            for case_id, basis in basis_by_case.items():
                basis_identities[case_id] = (
                    b0_identities[case_id] if basis == BASELINE_ID else candidate_pipeline_identity(basis, case_id, created)
                )
                pipeline_bases[(oid, case_id)] = basis
                if basis != BASELINE_ID and basis in promoted_index:
                    require(
                        candidate_tested_case(basis, case_id, results),
                        f"obligation {oid}: basis candidate {basis} was promoted without a TESTED_PATH for "
                        f"{case_id}; a promotion advances only the cases it measured, so it cannot carry that "
                        "case's evidence",
                    )
                if final and status in CLOSED_OBLIGATION_STATUSES and basis != BASELINE_ID:
                    require(
                        basis in promoted_index,
                        f"obligation {oid}: basis candidate {basis} was never promoted for {case_id}; "
                        "a closed obligation cannot rest on a rejected candidate at the final gate",
                    )
            require(auto in AUTO_STATUSES, f"obligation {oid}: invalid automatic status")
            require(manual in MANUAL_STATUSES, f"obligation {oid}: invalid manual status")
            for label, item, realization in (
                ("automatic", automatic, "AUTOMATIC"),
                ("manual", manual_plan, "MANUAL"),
            ):
                candidate_ids = item.get("candidate_ids", [])
                require(
                    isinstance(candidate_ids, list),
                    f"obligation {oid}: {label}.candidate_ids must be a list",
                )
                require(
                    set(candidate_ids) <= obligation_candidates,
                    f"obligation {oid}: {label} candidate not linked",
                )
                if item.get("status") in {"FAILED", "PARTIAL"} and label == "automatic":
                    require(
                        candidate_ids,
                        f"obligation {oid}: {label} {item.get('status')} needs at least one linked attempt candidate",
                    )
                    for candidate_id in candidate_ids:
                        require(
                            candidate_id in results,
                            f"obligation {oid}: {label} {item.get('status')} candidate {candidate_id} has no recorded RESULT",
                        )
                if item.get("status") in {"COMPLETE", "TESTED"}:
                    require(
                        candidate_ids,
                        f"obligation {oid}: {label} status needs a candidate",
                    )
                    for case_id in linked_cases:
                        basis = basis_by_case[case_id]
                        if basis == BASELINE_ID:
                            start = None
                        else:
                            require(
                                basis in promoted_index,
                                f"obligation {oid}: basis candidate {basis} has no PROMOTED result; "
                                "keep the obligation pending until the basis is promoted",
                            )
                            start = promoted_index[basis]
                        timeline = case_promotion_indices(case_id, promoted_cases, promoted_index)
                        end = next((index for index in timeline if start is None or index > start), None)
                        eligible_ids = [
                            cid
                            for cid in candidate_ids
                            if cid == basis or ((start is None or created_index[cid] > start) and (end is None or created_index[cid] < end))
                        ]
                        expected_identity = basis_identities[case_id]
                        require(
                            any(
                                created[cid]["identity"]["pipeline_facts"]["realization"] == realization
                                and candidate_tested_case(cid, case_id, results)
                                and candidate_pipeline_identity(cid, case_id, created) == expected_identity
                                for cid in eligible_ids
                            ),
                            f"obligation {oid}: {label} has no tested {realization} candidate for {case_id} "
                            f"measured on basis {basis}; evidence must be created inside that basis's promotion "
                            "interval for the case and match the basis on work_granularity, precision, "
                            "task_mapping, granularity unit, and granularity value",
                        )
                if item.get("status") in {"CLOSED_WITH_EVIDENCE", "INAPPLICABLE"}:
                    validate_hard_infeasibility(
                        f"obligation {oid}/{label}",
                        item.get("hard_infeasibility"),
                        {case_id: basis_by_case[case_id] for case_id in linked_cases},
                    )
            if strict_closure:
                require(
                    auto != "NOT_TRIED",
                    f"obligation {oid}: automatic pipeline realization is still NOT_TRIED at the final gate; "
                    "test it or record INAPPLICABLE with a proof for the current basis",
                )
                if auto != "COMPLETE":
                    require(
                        manual in {"TESTED", "CLOSED_WITH_EVIDENCE"},
                        f"obligation {oid}: incomplete automatic pipeline requires manual trial/evidence",
                    )
                else:
                    require(
                        manual in {"NOT_NEEDED_AUTO_COMPLETE", "TESTED"},
                        f"obligation {oid}: complete automatic pipeline needs explicit manual disposition",
                    )
        obligation_kinds.setdefault(kind, []).append(obligation)

    linked_by_case: dict[str, set[str]] = {case_id: set() for case_id in case_ids}
    for obligation in obligations:
        for case_id in obligation["case_ids"]:
            linked_by_case[case_id].add(obligation["obligation_id"])
    for case in cases:
        declared = set(case["obligation_ids"])
        expected = linked_by_case[case["case_id"]]
        require(
            declared == expected,
            f"case {case['case_id']}: obligation_ids must match the obligations that include it; "
            f"missing {sorted(expected - declared)}, extra {sorted(declared - expected)}",
        )

    for case in cases:
        if (
            case["outer_items"] < case["available_cores"]
            and case["inner_independent_chunks_per_item"] > 1
            and case["inner_chunks_independent"]
        ):
            covered = any(
                obligation["kind"] == "inner_parallel_flatten" and case["case_id"] in obligation["case_ids"] for obligation in obligations
            )
            require(
                covered,
                f"case {case['case_id']}: underfilled work requires inner_parallel_flatten",
            )
        if case["source_window"]["access_pattern"] == "INDEXED_OR_REORDERED":
            covered = any(
                obligation["kind"] == "physical_dataflow_routes" and case["case_id"] in obligation["case_ids"] for obligation in obligations
            )
            require(
                covered,
                f"case {case['case_id']}: indexed/reordered access requires physical routes",
            )

    physical_cases = {case_id for obligation in obligation_kinds.get("physical_dataflow_routes", []) for case_id in obligation["case_ids"]}
    granularity_cases = {
        case_id for obligation in obligation_kinds.get("work_granularity_search", []) for case_id in obligation["case_ids"]
    }
    interacting_cases = physical_cases & granularity_cases
    if interacting_cases:
        interaction_cases = {
            case_id for obligation in obligation_kinds.get("granularity_dataflow_interaction", []) for case_id in obligation["case_ids"]
        }
        require(
            interacting_cases <= interaction_cases,
            "cases with both granularity and physical-route obligations need granularity_dataflow_interaction",
        )

    validate_direct_route_priority(cases, obligations, created)

    if final:
        validate_best_measurement_binding(cases, promoted_cases, results)
        validate_blocked_resolutions(data.get("blocked_resolutions"), blocker_map(events, cases))

        pipeline_obligations = obligation_kinds.get("pipeline_realization", [])
        for candidate_id, result in results.items():
            if result["result_status"] != "PROMOTED":
                continue
            facts = created[candidate_id]["identity"]["pipeline_facts"]
            if facts["stages"] != 1:
                continue
            for obligation in pipeline_obligations:
                oid = obligation["obligation_id"]
                affected = set(obligation["case_ids"]) & promoted_cases[candidate_id]
                for case_id in sorted(affected):
                    basis = pipeline_bases[(oid, case_id)]
                    promoted_at = None if basis == BASELINE_ID else promoted_index.get(basis)
                    require(
                        promoted_at is not None and promoted_at >= promoted_index[candidate_id],
                        f"obligation {oid}: basis {basis} is superseded on {case_id} by promoted single-stage "
                        f"candidate {candidate_id}; rebase the obligation to a basis promoted at or after that "
                        "promotion and re-measure instead of reusing older evidence",
                    )

    if strict_closure:
        open_obligations = [item["obligation_id"] for item in obligations if item["status"] in {"PENDING", "ACTIVE", "BLOCKED"}]
        require(not open_obligations, f"open obligations at final gate: {open_obligations}")
        validate_convergence_evidence(data, cases)

    if final and targets_met:
        return "TARGET_MET"
    if strict_closure and no_numeric_target:
        return "CONVERGED_WITHOUT_NUMERIC_TARGET"
    if final and not allow_unmet_convergence:
        unmet = [case["case_id"] for case in cases if not validate_performance_target(case)]
        raise SearchIncomplete(f"performance targets unmet for {unmet}; keep search open")
    if strict_closure:
        return "UNMET_BUT_CONVERGED"
    return "VALID_RECORD"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path, help="optimization-search-coverage.json")
    parser.add_argument("--final", action="store_true", help="enforce stopping gate")
    parser.add_argument(
        "--allow-unmet-convergence",
        action="store_true",
        help="allow exceptional convergence below target when every route has hard evidence",
    )
    args = parser.parse_args()
    if args.allow_unmet_convergence and not args.final:
        parser.error("--allow-unmet-convergence requires --final")
    try:
        outcome = validate(args.record, args.final, args.allow_unmet_convergence)
    except SearchIncomplete as error:
        print(f"SEARCH_INCOMPLETE: {error}", file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 1
    except (TypeError, KeyError, AttributeError, IndexError, OverflowError) as error:
        print(f"INVALID: malformed record: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

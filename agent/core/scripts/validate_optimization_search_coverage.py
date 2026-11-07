#!/usr/bin/env python3
"""Validate the append-only structural search record used by Flash tuning."""

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
OBLIGATION_STATUSES = {"PENDING", "ACTIVE", "BLOCKED", "SATISFIED", "CLOSED_WITH_EVIDENCE"}
CANDIDATE_RESULTS = {
    "PROMOTED",
    "REJECTED_WITH_EVIDENCE",
    "INAPPLICABLE_WITH_EVIDENCE",
    "BLOCKED_BY_ENVIRONMENT",
}
CASE_DISPOSITIONS = {"PENDING", "BLOCKED", "BASELINE_MEASURED", "TESTED_PATH", "CLOSED_WITH_EVIDENCE"}
AUTO_STATUSES = {"NOT_TRIED", "COMPLETE", "FAILED", "PARTIAL", "INAPPLICABLE"}
MANUAL_STATUSES = {"NOT_TRIED", "BLOCKED", "NOT_NEEDED_AUTO_COMPLETE", "TESTED", "CLOSED_WITH_EVIDENCE"}
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
REJECTION_STAGES = {"COMPILATION", "KERNEL_EXECUTION", "PRECISION", "PROFILING_VALIDITY"}
PIPELINE_REALIZATIONS = {"SINGLE_STAGE", "AUTOMATIC", "MANUAL"}
ACCESS_PATTERNS = {"NOT_APPLICABLE", "SCALAR_ONLY", "DIRECT_CONTIGUOUS", "INDEXED_OR_REORDERED"}
TARGET_OPERATORS = {"LE", "GE"}
HARD_EVIDENCE_KINDS = {"CAPACITY", "LOWERING", "SEMANTIC", "STRICT_COST_DOMINANCE"}


class ValidationError(Exception):
    pass


class SearchIncomplete(ValidationError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def positive_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def nonnegative_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0


def evidence_list(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(nonempty_text(item) for item in value)


def valid_boundary_value(value: object) -> bool:
    return positive_number(value) or nonempty_text(value)


def validate_blocking_evidence(owner: str, value: object) -> None:
    require(isinstance(value, dict), f"{owner}: blocking_evidence must be an object")
    require(value.get("kind") in BLOCKER_KINDS, f"{owner}: invalid blocker kind")
    for field in ("failure_stage", "observed_error", "control_check", "resume_condition"):
        require(nonempty_text(value.get(field)), f"{owner}: blocking_evidence missing {field}")


def validate_rejection_evidence(owner: str, value: object) -> None:
    require(isinstance(value, dict), f"{owner}: rejection_evidence must be an object")
    require(value.get("failure_stage") in REJECTION_STAGES, f"{owner}: invalid rejection failure_stage")
    for field in ("observed_error", "control_check", "candidate_finding"):
        require(nonempty_text(value.get(field)), f"{owner}: rejection_evidence missing {field}")


def validate_hard_infeasibility(owner: str, value: object) -> None:
    require(isinstance(value, dict), f"{owner}: hard_infeasibility must be an object")
    kind = value.get("kind")
    require(kind in HARD_EVIDENCE_KINDS, f"{owner}: invalid hard-infeasibility kind")
    require(evidence_list(value.get("artifacts")), f"{owner}: hard infeasibility needs evidence artifacts")
    require(nonempty_text(value.get("finding")), f"{owner}: hard infeasibility needs a finding")
    require(
        value.get("exact_scope_only") is True,
        f"{owner}: hard infeasibility must state exact_scope_only=true",
    )
    if kind == "CAPACITY":
        require(positive_number(value.get("capacity_bytes")), f"{owner}: capacity_bytes must be positive")
        require(positive_number(value.get("required_bytes")), f"{owner}: required_bytes must be positive")
        require(
            value["required_bytes"] > value["capacity_bytes"],
            f"{owner}: capacity proof must exceed available capacity",
        )
        require(
            value.get("streaming_or_decomposition_checked") is True,
            f"{owner}: capacity proof must check streaming/decomposition",
        )
    elif kind == "LOWERING":
        require(nonempty_text(value.get("minimal_reproducer")), f"{owner}: lowering proof needs minimal_reproducer")
        require(evidence_list(value.get("checked_symbols")), f"{owner}: lowering proof needs checked_symbols")
    elif kind == "SEMANTIC":
        require(nonempty_text(value.get("constraint")), f"{owner}: semantic proof needs constraint")
        require(nonempty_text(value.get("proof")), f"{owner}: semantic proof needs proof")
    elif kind == "STRICT_COST_DOMINANCE":
        require(value.get("same_payload") is True, f"{owner}: cost dominance requires same_payload=true")
        require(nonnegative_number(value.get("reference_cost")), f"{owner}: invalid reference_cost")
        require(nonnegative_number(value.get("candidate_cost")), f"{owner}: invalid candidate_cost")
        require(
            value["candidate_cost"] > value["reference_cost"],
            f"{owner}: candidate cost must be strictly greater than reference cost",
        )
        require(nonempty_text(value.get("cost_unit")), f"{owner}: cost dominance needs cost_unit")


def validate_performance_target(case: dict) -> bool:
    case_id = case.get("case_id", "<missing>")
    target = case["performance_target"]
    metrics = case["best_metrics"]
    require(isinstance(target, dict), f"case {case_id}: performance_target must be an object")
    require(target.get("logic") in {"ANY", "ALL"}, f"case {case_id}: target logic must be ANY or ALL")
    conditions = target.get("conditions")
    require(isinstance(conditions, list) and conditions, f"case {case_id}: target conditions required")
    require(isinstance(metrics, dict), f"case {case_id}: best_metrics must be an object")
    require(positive_number(metrics.get("kernel_time_us")), f"case {case_id}: best kernel_time_us must be positive")
    require(evidence_list(case["best_metric_evidence"]), f"case {case_id}: best metric evidence required")
    outcomes: list[bool] = []
    seen_metrics: set[str] = set()
    for index, condition in enumerate(conditions):
        require(isinstance(condition, dict), f"case {case_id}: target condition {index} must be an object")
        metric = condition.get("metric")
        operator = condition.get("operator")
        threshold = condition.get("threshold")
        require(nonempty_text(metric), f"case {case_id}: target condition {index} needs metric")
        require(metric not in seen_metrics, f"case {case_id}: duplicate target metric {metric}")
        seen_metrics.add(metric)
        require(operator in TARGET_OPERATORS, f"case {case_id}: invalid target operator {operator}")
        require(nonnegative_number(threshold), f"case {case_id}: target threshold must be non-negative")
        require(nonnegative_number(metrics.get(metric)), f"case {case_id}: best metric {metric} missing or invalid")
        outcomes.append(metrics[metric] <= threshold if operator == "LE" else metrics[metric] >= threshold)
    return any(outcomes) if target["logic"] == "ANY" else all(outcomes)


def validate_case(case: dict, obligation_ids: set[str]) -> None:
    case_id = case.get("case_id", "<missing>")
    for field in CASE_FIELDS:
        require(field in case, f"case {case_id}: missing {field}")

    for field in ("outer_items", "inner_independent_chunks_per_item"):
        require(
            isinstance(case[field], int) and case[field] >= 0,
            f"case {case_id}: {field} must be a non-negative integer",
        )
    require(
        isinstance(case["available_cores"], int) and case["available_cores"] > 0,
        f"case {case_id}: available_cores must be positive",
    )
    for field in ("outer_task_waves", "flattened_task_waves", "per_task_payload_bytes"):
        require(nonnegative_number(case[field]), f"case {case_id}: {field} must be non-negative")
    require(positive_number(case["kernel_time_us"]), f"case {case_id}: kernel_time_us must be positive")
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
    access_pattern = source.get("access_pattern")
    require(access_pattern in ACCESS_PATTERNS, f"case {case_id}: invalid source-window access_pattern")
    require(isinstance(source.get("bounded"), bool), f"case {case_id}: source_window.bounded must be boolean")
    require(nonempty_text(source.get("index_regularity")), f"case {case_id}: source_window needs index_regularity")
    require(nonempty_text(source.get("evidence")), f"case {case_id}: source_window needs evidence")
    if access_pattern == "INDEXED_OR_REORDERED" and source["bounded"]:
        span = source.get("span_bytes")
        consumed = source.get("consumed_bytes")
        density = source.get("density")
        window_vregs = source.get("window_vregs")
        require(positive_number(span), f"case {case_id}: bounded source span must be positive")
        require(positive_number(consumed) and consumed <= span, f"case {case_id}: invalid consumed source bytes")
        require(
            isinstance(density, (int, float)) and not isinstance(density, bool) and 0 < density <= 1,
            f"case {case_id}: invalid source density",
        )
        require(
            math.isclose(density, consumed / span, rel_tol=1e-3, abs_tol=1e-6),
            f"case {case_id}: source density is inconsistent with consumed/span",
        )
        require(
            isinstance(window_vregs, int) and window_vregs > 0,
            f"case {case_id}: bounded source window needs positive window_vregs",
        )
        require(
            isinstance(source.get("register_window_feasible"), bool),
            f"case {case_id}: bounded source window needs register_window_feasible",
        )
        require(nonempty_text(source.get("window_scope")), f"case {case_id}: source window needs window_scope")
        require(
            source.get("decomposition_checked") is True,
            f"case {case_id}: source window must check row/chunk decomposition",
        )
        if source["register_window_feasible"] is False:
            validate_hard_infeasibility(
                f"case {case_id}/register_window",
                source.get("register_infeasibility"),
            )
    require(
        source.get("baseline_route_kind") in ROUTE_KINDS,
        f"case {case_id}: source_window needs baseline_route_kind",
    )
    require(
        source.get("baseline_writeback_kind") in WRITEBACK_KINDS,
        f"case {case_id}: source_window needs baseline_writeback_kind",
    )
    require(
        isinstance(case["obligation_ids"], list) and case["obligation_ids"],
        f"case {case_id}: at least one coverage obligation is required",
    )
    unknown = set(case["obligation_ids"]) - obligation_ids
    require(not unknown, f"case {case_id}: unknown obligations {sorted(unknown)}")


def validate_physical_identity(candidate_id: str, identity: dict) -> None:
    physical = identity.get("physical_dataflow")
    require(isinstance(physical, dict), f"candidate {candidate_id}: physical_dataflow must be structured")
    require(nonempty_text(physical.get("description")), f"candidate {candidate_id}: physical_dataflow missing description")
    route = physical.get("route_kind")
    source = physical.get("source_access")
    writeback = physical.get("writeback_kind")
    require(route in ALL_ROUTE_KINDS, f"candidate {candidate_id}: invalid physical route {route}")
    require(source in SOURCE_ACCESS_KINDS, f"candidate {candidate_id}: invalid source_access {source}")
    require(writeback in WRITEBACK_KINDS, f"candidate {candidate_id}: invalid writeback_kind {writeback}")
    for field in (
        "full_payload_materialization",
        "register_lane_reorder",
        "indexed_lane_access",
        "route_experiment",
    ):
        require(isinstance(physical.get(field), bool), f"candidate {candidate_id}: physical_dataflow.{field} must be boolean")

    if route == "MEMORY_INDEXED_VECTOR":
        require(physical["indexed_lane_access"], f"candidate {candidate_id}: memory-indexed route needs indexed lane access")
    if route == "CONTIGUOUS_LOAD_REGISTER_REORDER":
        require(source == "CONTIGUOUS", f"candidate {candidate_id}: direct register route must load a contiguous source window")
        require(physical["register_lane_reorder"], f"candidate {candidate_id}: direct register route needs register lane reorder")
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
    require(isinstance(facts, dict), f"candidate {candidate_id}: identity missing granularity_facts")
    require(nonempty_text(facts.get("unit")), f"candidate {candidate_id}: granularity_facts missing unit")
    kinds = facts.get("boundary_kinds")
    values = facts.get("values_by_case")
    require(
        isinstance(kinds, list) and kinds and set(kinds) <= BOUNDARY_KINDS,
        f"candidate {candidate_id}: invalid granularity boundary_kinds",
    )
    require(isinstance(values, dict), f"candidate {candidate_id}: values_by_case must be an object")
    require(set(values) == applicable_cases, f"candidate {candidate_id}: granularity values must cover applicable_cases exactly")
    require(all(valid_boundary_value(value) for value in values.values()), f"candidate {candidate_id}: invalid granularity value")


def validate_pipeline_identity(candidate_id: str, identity: dict) -> None:
    facts = identity.get("pipeline_facts")
    require(isinstance(facts, dict), f"candidate {candidate_id}: identity missing pipeline_facts")
    stages = facts.get("stages")
    realization = facts.get("realization")
    require(isinstance(stages, int) and stages > 0, f"candidate {candidate_id}: pipeline stages must be positive")
    require(realization in PIPELINE_REALIZATIONS, f"candidate {candidate_id}: invalid pipeline realization")
    if stages == 1:
        require(realization == "SINGLE_STAGE", f"candidate {candidate_id}: stage 1 must use SINGLE_STAGE")
    if stages > 1:
        require(realization != "SINGLE_STAGE", f"candidate {candidate_id}: multistage needs automatic/manual realization")


def validate_events(events: list, case_ids: set[str], final: bool) -> tuple[dict, dict]:
    created_by_candidate: dict[str, dict] = {}
    result_by_candidate: dict[str, dict] = {}
    for index, event in enumerate(events):
        candidate_id = event.get("candidate_id")
        event_type = event.get("event")
        identity = event.get("identity")
        applicable = event.get("applicable_cases")
        require(nonempty_text(candidate_id), f"candidate event {index}: missing candidate_id")
        require(event_type in {"CREATED", "RESULT"}, f"candidate {candidate_id}: event must be CREATED or RESULT")
        require(isinstance(identity, dict), f"candidate {candidate_id}: identity must be an object")
        for axis in AXES:
            if axis != "physical_dataflow":
                require(nonempty_text(identity.get(axis)), f"candidate {candidate_id}: identity missing {axis}")
        require(
            isinstance(applicable, list) and applicable and not (set(applicable) - case_ids),
            f"candidate {candidate_id}: applicable_cases must reference known cases",
        )
        require(len(applicable) == len(set(applicable)), f"candidate {candidate_id}: duplicate applicable case")
        validate_physical_identity(candidate_id, identity)
        validate_granularity_identity(candidate_id, identity, set(applicable))
        validate_pipeline_identity(candidate_id, identity)

        if candidate_id in created_by_candidate:
            created = created_by_candidate[candidate_id]
            require(event_type == "RESULT", f"candidate {candidate_id}: duplicate CREATED event")
            require(identity == created["identity"], f"candidate {candidate_id}: identity changed; allocate a new ID")
            require(applicable == created["applicable_cases"], f"candidate {candidate_id}: applicable_cases changed")
            require(candidate_id not in result_by_candidate, f"candidate {candidate_id}: duplicate RESULT event")
        else:
            require(event_type == "CREATED", f"candidate {candidate_id}: first event must be CREATED")
            created_by_candidate[candidate_id] = event

        if event_type == "RESULT":
            result_status = event.get("result_status")
            require(result_status in CANDIDATE_RESULTS, f"candidate {candidate_id}: invalid result_status")
            require(evidence_list(event.get("evidence")), f"candidate {candidate_id}: result needs evidence")
            case_results = event.get("case_results")
            require(isinstance(case_results, list), f"candidate {candidate_id}: RESULT needs case_results")
            by_case = {item.get("case_id"): item for item in case_results if isinstance(item, dict)}
            require(len(by_case) == len(case_results), f"candidate {candidate_id}: invalid or duplicate case_results")
            require(set(by_case) == set(applicable), f"candidate {candidate_id}: case_results must cover applicable_cases exactly")
            for case_id, item in by_case.items():
                disposition = item.get("disposition")
                require(
                    disposition in RESULT_CASE_DISPOSITIONS,
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
                require(blocked_cases, f"candidate {candidate_id}: blocked result needs NOT_RUN_BLOCKED cases")
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
            result_by_candidate[candidate_id] = event

    if final:
        require(
            not (set(created_by_candidate) - set(result_by_candidate)),
            "a CREATED candidate has no RESULT event",
        )
    return created_by_candidate, result_by_candidate


def candidate_tested_case(candidate_id: str, case_id: str, results: dict) -> bool:
    result = results.get(candidate_id)
    if result is None:
        return False
    return any(item["case_id"] == case_id and item["disposition"] == "TESTED_PATH" for item in result["case_results"])


def validate_disposition(
    owner: str,
    disposition: dict,
    valid_cases: set[str],
    obligation_candidates: set[str],
    created: dict,
    results: dict,
    final: bool,
) -> None:
    status = disposition.get("status")
    candidate_ids = disposition.get("candidate_ids", [])
    require(status in CASE_DISPOSITIONS, f"{owner}: invalid status {status}")
    require(isinstance(candidate_ids, list), f"{owner}: candidate_ids must be a list")
    require(set(candidate_ids) <= obligation_candidates, f"{owner}: candidate not linked by obligation")
    require(set(candidate_ids) <= set(created), f"{owner}: unknown candidate")
    if status == "TESTED_PATH":
        require(candidate_ids, f"{owner}: TESTED_PATH needs a candidate")
        for case_id in valid_cases:
            require(
                any(candidate_tested_case(cid, case_id, results) for cid in candidate_ids),
                f"{owner}: no referenced candidate tested path for {case_id}",
            )
    if status in {"BASELINE_MEASURED", "CLOSED_WITH_EVIDENCE"}:
        require(evidence_list(disposition.get("evidence")), f"{owner}: closure/baseline needs evidence")
    if status == "CLOSED_WITH_EVIDENCE":
        validate_hard_infeasibility(owner, disposition.get("hard_infeasibility"))
    if status == "BLOCKED":
        require(candidate_ids, f"{owner}: BLOCKED needs a blocked candidate")
        require(evidence_list(disposition.get("evidence")), f"{owner}: BLOCKED needs evidence")
        require(
            any(cid in results and results[cid].get("result_status") == "BLOCKED_BY_ENVIRONMENT" for cid in candidate_ids),
            f"{owner}: BLOCKED must reference a BLOCKED_BY_ENVIRONMENT result",
        )
    if final:
        require(status not in {"PENDING", "BLOCKED"}, f"{owner}: still {status} at final gate")


def validate_case_dispositions(
    obligation: dict,
    linked_cases: set[str],
    obligation_candidates: set[str],
    created: dict,
    results: dict,
    final: bool,
) -> None:
    oid = obligation["obligation_id"]
    dispositions = obligation.get("case_dispositions")
    require(isinstance(dispositions, list), f"obligation {oid}: case_dispositions must be a list")
    by_case = {item.get("case_id"): item for item in dispositions if isinstance(item, dict)}
    require(len(by_case) == len(dispositions), f"obligation {oid}: invalid or duplicate case_dispositions")
    require(set(by_case) == linked_cases, f"obligation {oid}: case_dispositions must cover case_ids exactly")
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
        require(nonempty_text(plan_id) and plan_id not in seen_ids, f"obligation {oid}: invalid route plan_id")
        seen_ids.add(plan_id)
        require(route in ROUTE_KINDS, f"obligation {oid}/{plan_id}: invalid route_kind")
        require(writeback in WRITEBACK_KINDS, f"obligation {oid}/{plan_id}: invalid writeback_kind")
        require(isinstance(item_cases, list) and item_cases, f"obligation {oid}/{plan_id}: case_ids required")
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
        require(nonempty_text(plan_id) and plan_id not in seen_ids, f"obligation {oid}: invalid boundary plan_id")
        seen_ids.add(plan_id)
        require(kind in BOUNDARY_KINDS, f"obligation {oid}/{plan_id}: invalid boundary_kind")
        require(isinstance(case_values, dict) and case_values, f"obligation {oid}/{plan_id}: case_values required")
        item_cases = set(case_values)
        require(item_cases <= linked_cases, f"obligation {oid}/{plan_id}: unknown case")
        require(all(valid_boundary_value(value) for value in case_values.values()), f"obligation {oid}/{plan_id}: invalid value")
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
        if item.get("status") == "TESTED_PATH":
            for case_id, value in case_values.items():
                matching = [
                    cid
                    for cid in item.get("candidate_ids", [])
                    if kind in created[cid]["identity"]["granularity_facts"]["boundary_kinds"]
                    and created[cid]["identity"]["granularity_facts"]["values_by_case"].get(case_id) == value
                    and candidate_tested_case(cid, case_id, results)
                ]
                require(matching, f"obligation {oid}/{plan_id}: no candidate tested {kind}={value} for {case_id}")
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
    final: bool,
) -> None:
    oid = obligation["obligation_id"]
    plan = obligation.get("interaction_plan")
    require(isinstance(plan, list) and plan, f"obligation {oid}: interaction_plan is required")
    covered_cases: set[str] = set()
    seen_ids: set[str] = set()
    for item in plan:
        plan_id = item.get("plan_id")
        route = item.get("route_kind")
        writeback = item.get("writeback_kind")
        kind = item.get("boundary_kind")
        case_values = item.get("case_values")
        require(nonempty_text(plan_id) and plan_id not in seen_ids, f"obligation {oid}: invalid interaction plan_id")
        seen_ids.add(plan_id)
        require(route in ROUTE_KINDS, f"obligation {oid}/{plan_id}: invalid route_kind")
        require(writeback in WRITEBACK_KINDS, f"obligation {oid}/{plan_id}: invalid writeback_kind")
        require(kind in BOUNDARY_KINDS, f"obligation {oid}/{plan_id}: invalid boundary_kind")
        require(isinstance(case_values, dict) and case_values, f"obligation {oid}/{plan_id}: case_values required")
        item_cases = set(case_values)
        require(item_cases <= linked_cases, f"obligation {oid}/{plan_id}: unknown case")
        require(all(valid_boundary_value(value) for value in case_values.values()), f"obligation {oid}/{plan_id}: invalid value")
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
                require(matching, f"obligation {oid}/{plan_id}: no candidate tested required combination for {case_id}")
    require(covered_cases == linked_cases, f"obligation {oid}: interaction_plan must disposition every linked case")


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
        require(direct_dispositions, f"case {case_id}: direct-register route disposition required")
        if any(item.get("status") in {"BASELINE_MEASURED", "CLOSED_WITH_EVIDENCE"} for item in direct_dispositions):
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


def validate_convergence_evidence(data: dict, cases: list[dict]) -> None:
    convergence = data.get("unmet_convergence_evidence")
    require(isinstance(convergence, dict), "unmet convergence needs unmet_convergence_evidence")
    entries = convergence.get("case_bounds")
    require(isinstance(entries, list), "unmet convergence needs case_bounds")
    by_case = {item.get("case_id"): item for item in entries if isinstance(item, dict)}
    unmet_cases = {case["case_id"] for case in cases if not validate_performance_target(case)}
    require(set(by_case) == unmet_cases, "case_bounds must cover every and only unmet case")
    for case_id, entry in by_case.items():
        require(positive_number(entry.get("verified_lower_bound_us")), f"case {case_id}: verified lower bound required")
        require(positive_number(entry.get("best_kernel_time_us")), f"case {case_id}: best kernel time required")
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
        require(evidence_list(entry.get("evidence")), f"case {case_id}: lower-bound evidence required")
        require(nonempty_text(entry.get("remaining_gap_explanation")), f"case {case_id}: remaining gap explanation required")


def validate(path: Path, final: bool, allow_unmet_convergence: bool) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    require(data.get("schema_version") == 3, "schema_version must be 3")
    cases = data.get("cases")
    obligations = data.get("obligations")
    events = data.get("candidate_events")
    require(isinstance(cases, list) and cases, "cases must be a non-empty list")
    require(isinstance(obligations, list) and obligations, "obligations must be a non-empty list")
    require(isinstance(events, list), "candidate_events must be a list")

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
    strict_closure = final and not targets_met and allow_unmet_convergence

    created, results = validate_events(events, case_ids, final)
    obligation_kinds: dict[str, list[dict]] = {}
    for obligation in obligations:
        oid = obligation["obligation_id"]
        kind = obligation.get("kind")
        status = obligation.get("status")
        linked_cases_list = obligation.get("case_ids")
        candidate_ids_list = obligation.get("candidate_ids")
        require(nonempty_text(kind), f"obligation {oid}: missing kind")
        require(status in OBLIGATION_STATUSES, f"obligation {oid}: invalid status {status}")
        require(isinstance(linked_cases_list, list) and linked_cases_list, f"obligation {oid}: case_ids required")
        linked_cases = set(linked_cases_list)
        require(not (linked_cases - case_ids), f"obligation {oid}: references unknown cases")
        for linked_case in linked_cases:
            case = next(item for item in cases if item["case_id"] == linked_case)
            require(oid in case["obligation_ids"], f"obligation {oid}: missing reverse link from {linked_case}")
        require(isinstance(candidate_ids_list, list), f"obligation {oid}: candidate_ids must be a list")
        obligation_candidates = set(candidate_ids_list)
        require(obligation_candidates <= set(created), f"obligation {oid}: references unknown candidate")
        validate_case_dispositions(
            obligation,
            linked_cases,
            obligation_candidates,
            created,
            results,
            strict_closure,
        )

        if status == "SATISFIED":
            require(candidate_ids_list, f"obligation {oid}: SATISFIED needs a tested candidate")
        if status == "CLOSED_WITH_EVIDENCE":
            require(evidence_list(obligation.get("closure_evidence")), f"obligation {oid}: closure evidence required")
            validate_hard_infeasibility(f"obligation {oid}", obligation.get("hard_infeasibility"))
        if status == "BLOCKED":
            require(candidate_ids_list, f"obligation {oid}: BLOCKED needs a blocked candidate")
            require(evidence_list(obligation.get("blocking_reasons")), f"obligation {oid}: blocking reasons required")
            require(
                any(cid in results and results[cid].get("result_status") == "BLOCKED_BY_ENVIRONMENT" for cid in candidate_ids_list),
                f"obligation {oid}: BLOCKED must reference a BLOCKED_BY_ENVIRONMENT result",
            )
        if obligation.get("closure_basis") == "STRICT_COST_DOMINANCE":
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
            validate_boundary_plan(obligation, linked_cases, obligation_candidates, created, results, strict_closure)
        if kind == "granularity_dataflow_interaction":
            validate_interaction_plan(obligation, linked_cases, obligation_candidates, created, results, strict_closure)
        if kind == "pipeline_realization":
            plan = obligation.get("pipeline_plan", {})
            basis_candidate_id = obligation.get("basis_candidate_id")
            automatic = plan.get("automatic", {})
            manual_plan = plan.get("manual", {})
            auto = automatic.get("status")
            manual = manual_plan.get("status")
            require(
                basis_candidate_id == "B0" or basis_candidate_id in created,
                f"obligation {oid}: basis_candidate_id must be B0 or a candidate",
            )
            if basis_candidate_id != "B0":
                require(
                    linked_cases <= set(created[basis_candidate_id]["applicable_cases"]),
                    f"obligation {oid}: basis candidate does not activate every linked case",
                )
                if strict_closure:
                    for case_id in linked_cases:
                        require(
                            candidate_tested_case(basis_candidate_id, case_id, results),
                            f"obligation {oid}: basis candidate path was not tested for {case_id}",
                        )
            require(auto in AUTO_STATUSES, f"obligation {oid}: invalid automatic status")
            require(manual in MANUAL_STATUSES, f"obligation {oid}: invalid manual status")
            for label, item, realization in (
                ("automatic", automatic, "AUTOMATIC"),
                ("manual", manual_plan, "MANUAL"),
            ):
                candidate_ids = item.get("candidate_ids", [])
                require(isinstance(candidate_ids, list), f"obligation {oid}: {label}.candidate_ids must be a list")
                require(set(candidate_ids) <= obligation_candidates, f"obligation {oid}: {label} candidate not linked")
                if item.get("status") in {"COMPLETE", "TESTED"}:
                    require(candidate_ids, f"obligation {oid}: {label} status needs a candidate")
                    for case_id in linked_cases:
                        require(
                            any(
                                created[cid]["identity"]["pipeline_facts"]["realization"] == realization
                                and candidate_tested_case(cid, case_id, results)
                                for cid in candidate_ids
                            ),
                            f"obligation {oid}: {label} has no actually tested {realization} candidate for {case_id}",
                        )
                if item.get("status") in {"CLOSED_WITH_EVIDENCE", "INAPPLICABLE"}:
                    validate_hard_infeasibility(
                        f"obligation {oid}/{label}",
                        item.get("hard_infeasibility"),
                    )
            if strict_closure and auto != "COMPLETE":
                require(
                    manual in {"TESTED", "CLOSED_WITH_EVIDENCE"},
                    f"obligation {oid}: incomplete automatic pipeline requires manual trial/evidence",
                )
            if strict_closure and auto == "COMPLETE":
                require(
                    manual in {"NOT_NEEDED_AUTO_COMPLETE", "TESTED"},
                    f"obligation {oid}: complete automatic pipeline needs explicit manual disposition",
                )
        obligation_kinds.setdefault(kind, []).append(obligation)

    for case in cases:
        if (
            case["outer_items"] < case["available_cores"]
            and case["inner_independent_chunks_per_item"] > 1
            and case["inner_chunks_independent"]
        ):
            covered = any(
                obligation["kind"] == "inner_parallel_flatten" and case["case_id"] in obligation["case_ids"] for obligation in obligations
            )
            require(covered, f"case {case['case_id']}: underfilled work requires inner_parallel_flatten")
        if case["source_window"]["access_pattern"] == "INDEXED_OR_REORDERED":
            covered = any(
                obligation["kind"] == "physical_dataflow_routes" and case["case_id"] in obligation["case_ids"] for obligation in obligations
            )
            require(covered, f"case {case['case_id']}: indexed/reordered access requires physical routes")

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

    if not targets_met:
        pipeline_obligations = obligation_kinds.get("pipeline_realization", [])
        pipeline_cases = {case_id for obligation in pipeline_obligations for case_id in obligation["case_ids"]}
        for candidate_id, result in results.items():
            if result["result_status"] != "PROMOTED":
                continue
            facts = created[candidate_id]["identity"]["pipeline_facts"]
            if facts["stages"] != 1:
                continue
            affected = set(created[candidate_id]["applicable_cases"]) & pipeline_cases
            if not affected:
                continue
            reevaluated = {
                case_id
                for obligation in pipeline_obligations
                if obligation.get("basis_candidate_id") == candidate_id
                for case_id in obligation["case_ids"]
            }
            require(
                affected <= reevaluated,
                f"promoted single-stage candidate {candidate_id} changes admitted pipeline cases; re-evaluate pipeline",
            )

    if strict_closure:
        open_obligations = [item["obligation_id"] for item in obligations if item["status"] in {"PENDING", "ACTIVE", "BLOCKED"}]
        require(not open_obligations, f"open obligations at final gate: {open_obligations}")
        validate_convergence_evidence(data, cases)

    if final and targets_met:
        return "TARGET_MET"
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
    print(outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

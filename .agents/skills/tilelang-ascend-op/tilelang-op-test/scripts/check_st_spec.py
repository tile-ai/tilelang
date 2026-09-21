#!/usr/bin/env python3
"""Validate an ST contract and case manifest for the current repository.

This checker validates reviewable evidence and identifies common unsupported
claims. It does not run pytest and cannot prove that a reference or operator
implementation is correct.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SOURCE_KINDS = {
    'user_requirement',
    'interface_doc',
    'design_doc',
    'public_docstring',
    'api_validation',
    'mathematical_definition',
    'reference',
    'cuda_implementation',
    'ascend_implementation',
    'existing_test',
    'implementation',
}
PRIMARY_ORACLES = {
    'pytorch_reference',
    'cpu_reference',
    'exact_expected',
    'mathematical',
    'metamorphic',
    'exception_contract',
}
SUPPORTING_ORACLES = {'cuda_differential', 'legacy_differential'}
CASE_KINDS = {'normal', 'boundary', 'negative', 'gradient', 'stateful'}
ASSERTIONS = {
    'shape',
    'dtype',
    'device',
    'values',
    'gradients',
    'aux_outputs',
    'state_changed',
    'state_unchanged',
    'protected_storage',
    'exception_type',
    'exception_message',
}
LIFECYCLE = {'designed': 0, 'implemented': 1, 'collected': 2, 'executed': 3}
RESULTS = {'not_run', 'passed', 'failed', 'skipped', 'xfailed', 'error'}
COVERAGE_DIMENSIONS = (
    'functionality',
    'precision',
    'boundary',
    'gradient',
    'state_mutation',
    'invalid_rejection',
    'layout_interface',
    'backend_branch',
    'randomness',
    'execution_evidence',
)
COVERAGE_STATUSES = {'covered', 'partial', 'missing', 'unknown', 'not_applicable'}
SEMANTIC_SOURCE_KINDS = {
    'user_requirement',
    'interface_doc',
    'design_doc',
    'public_docstring',
    'mathematical_definition',
    'reference',
}


class Findings:
    def __init__(self) -> None:
        self.errors: list[dict[str, str]] = []
        self.warnings: list[dict[str, str]] = []
        self.case_errors: dict[str, int] = {}

    def add(self, severity: str, code: str, message: str, context: str = '') -> None:
        record = {'code': code, 'message': message}
        if context:
            record['context'] = context
        if severity == 'error':
            self.errors.append(record)
            if context.startswith('case:'):
                case_id = context.split(':', 1)[1]
                self.case_errors[case_id] = self.case_errors.get(case_id, 0) + 1
        else:
            self.warnings.append(record)

    def error(self, code: str, message: str, context: str = '') -> None:
        self.add('error', code, message, context)

    def warn(self, code: str, message: str, context: str = '') -> None:
        self.add('warning', code, message, context)


def require_nonempty_string(value: Any, field: str, findings: Findings, context: str = '') -> bool:
    if isinstance(value, str) and value.strip():
        return True
    findings.error('invalid_field', f'{field} must be a nonempty string', context)
    return False


def unique_records(records: Any, field: str, findings: Findings, context_prefix: str) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list):
        findings.error('invalid_field', f'{field} must be a list')
        return {}
    result: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        context = f'{context_prefix}:{index}'
        if not isinstance(record, dict):
            findings.error('invalid_record', f'{field}[{index}] must be an object', context)
            continue
        record_id = record.get('id')
        if not require_nonempty_string(record_id, f'{field}[{index}].id', findings, context):
            continue
        if record_id in result:
            findings.error('duplicate_id', f'{field} contains duplicate ID {record_id!r}', context)
            continue
        result[record_id] = record
    return result


def validate_sources(data: dict[str, Any], repo_root: Path, findings: Findings) -> dict[str, dict[str, Any]]:
    sources = unique_records(data.get('contract_sources'), 'contract_sources', findings, 'source')
    for source_id, source in sources.items():
        context = f'source:{source_id}'
        kind = source.get('kind')
        if kind not in SOURCE_KINDS:
            findings.error('invalid_source_kind', f'unsupported source kind {kind!r}', context)
        require_nonempty_string(source.get('claim'), 'claim', findings, context)
        path_value = source.get('path')
        if path_value is None:
            if kind != 'user_requirement':
                findings.warn('source_without_path', 'local source does not provide a path', context)
        elif not isinstance(path_value, str) or not path_value.strip():
            findings.error('invalid_source_path', 'path must be a nonempty string', context)
        elif not (repo_root / path_value).exists():
            findings.error('missing_source_path', f'source path does not exist: {path_value}', context)
    return sources


def validate_requirements(
    data: dict[str, Any], sources: dict[str, dict[str, Any]], findings: Findings
) -> dict[str, dict[str, Any]]:
    requirements = unique_records(data.get('requirements'), 'requirements', findings, 'requirement')
    for requirement_id, requirement in requirements.items():
        context = f'requirement:{requirement_id}'
        require_nonempty_string(requirement.get('statement'), 'statement', findings, context)
        applicability = requirement.get('applicability')
        if applicability not in {'confirmed', 'assumed', 'unknown', 'not_applicable'}:
            findings.error('invalid_applicability', f'unsupported applicability value {applicability!r}', context)
        source_ids = requirement.get('source_ids')
        if not isinstance(source_ids, list) or not all(isinstance(item, str) for item in source_ids):
            findings.error('invalid_source_ids', 'source_ids must be a list of strings', context)
            continue
        missing = [source_id for source_id in source_ids if source_id not in sources]
        if missing:
            findings.error('unknown_source', f'unknown source IDs: {missing}', context)
        if applicability == 'confirmed' and not source_ids:
            findings.error('confirmed_without_source', 'a requirement with status confirmed must reference at least one source', context)
        source_kinds = {sources[item]['kind'] for item in source_ids if item in sources}
        if applicability == 'confirmed' and source_kinds and not (source_kinds & SEMANTIC_SOURCE_KINDS):
            findings.warn(
                'implementation_only_contract',
                'a requirement with status confirmed relies only on implementation or current-test evidence and should be treated as a provisional contract',
                context,
            )
    return requirements


def validate_oracles(
    case: dict[str, Any], sources: dict[str, dict[str, Any]], findings: Findings, context: str
) -> set[str]:
    oracles = case.get('oracles')
    if not isinstance(oracles, list) or not oracles:
        findings.error('missing_oracle', 'a case must declare at least one oracle', context)
        return set()
    kinds: set[str] = set()
    for index, oracle in enumerate(oracles):
        if not isinstance(oracle, dict):
            findings.error('invalid_oracle', f'oracles[{index}] must be an object', context)
            continue
        kind = oracle.get('kind')
        if kind not in PRIMARY_ORACLES | SUPPORTING_ORACLES:
            findings.error('invalid_oracle_kind', f'unsupported oracle kind {kind!r}', context)
        else:
            kinds.add(kind)
        source_id = oracle.get('source_id')
        if not isinstance(source_id, str) or source_id not in sources:
            findings.error('unknown_oracle_source', f'oracle references unknown source {source_id!r}', context)
    if kinds and not (kinds & PRIMARY_ORACLES):
        findings.error('differential_only_oracle', 'differential evidence from CUDA or a legacy implementation cannot be the only primary oracle', context)
    return kinds


def validate_assertions(case: dict[str, Any], findings: Findings, context: str) -> set[str]:
    assertions = case.get('assertions')
    if not isinstance(assertions, list) or not assertions or not all(isinstance(item, str) for item in assertions):
        findings.error('invalid_assertions', 'assertions must be a nonempty list of strings', context)
        return set()
    names = set(assertions)
    unknown = sorted(names - ASSERTIONS)
    if unknown:
        findings.error('unknown_assertion', f'unknown assertion names: {unknown}', context)
    kind = case.get('kind')
    if kind in {'normal', 'boundary'} and not ({'values', 'state_changed'} & names):
        findings.error('shape_only_correctness', f'a {kind} case must check values or state changes, not only metadata', context)
    if kind == 'gradient' and 'gradients' not in names:
        findings.error('missing_gradient_assertion', 'a gradient case must check gradients', context)
    if kind == 'stateful' and 'state_changed' not in names:
        findings.error('missing_state_assertion', 'a stateful case must check the expected state change', context)
    if kind == 'negative' and 'exception_type' not in names:
        findings.error('weak_negative_assertion', 'a negative case must check a specific exception type', context)
    return names


def validate_path_evidence(
    case: dict[str, Any], sources: dict[str, dict[str, Any]], findings: Findings, context: str
) -> None:
    evidence = case.get('path_evidence')
    if evidence is None:
        return
    if not isinstance(evidence, dict):
        findings.error('invalid_path_evidence', 'path_evidence must be an object', context)
        return
    source_id = evidence.get('source_id')
    if not isinstance(source_id, str) or source_id not in sources:
        findings.error('unknown_path_source', f'path_evidence references unknown source {source_id!r}', context)
    if evidence.get('kind') != 'tail':
        return
    logical_size = evidence.get('logical_size')
    block_size = evidence.get('block_size')
    if not isinstance(logical_size, int) or logical_size < 0:
        findings.error('invalid_tail_size', 'tail logical_size must be a nonnegative integer', context)
    if not isinstance(block_size, int) or block_size <= 0:
        findings.error('invalid_block_size', 'tail block_size must be a positive integer', context)
    if isinstance(logical_size, int) and isinstance(block_size, int) and block_size > 0 and logical_size % block_size == 0:
        findings.error('false_tail_claim', f'{logical_size} is divisible by block size {block_size} and does not form a tail', context)
    require_nonempty_string(evidence.get('axis'), 'path_evidence.axis', findings, context)


def validate_lifecycle(case: dict[str, Any], findings: Findings, context: str) -> None:
    status = case.get('status')
    result = case.get('result')
    if status not in LIFECYCLE:
        findings.error('invalid_status', f'unsupported lifecycle status {status!r}', context)
        return
    if result not in RESULTS:
        findings.error('invalid_result', f'unsupported run result {result!r}', context)
        return
    if LIFECYCLE[status] >= LIFECYCLE['collected'] and not isinstance(case.get('test_nodeid'), str):
        findings.error('missing_nodeid', 'a case with status collected or executed must include test_nodeid', context)
    if status == 'executed' and result == 'not_run':
        findings.error('executed_without_result', 'a case with status executed must include an execution result', context)
    if status != 'executed' and result != 'not_run':
        findings.error('result_without_execution', 'only a case with status executed may report passed, failed, skipped, or error', context)


def validate_cases(
    data: dict[str, Any], sources: dict[str, dict[str, Any]], requirements: dict[str, dict[str, Any]], findings: Findings
) -> list[dict[str, Any]]:
    cases_by_id = unique_records(data.get('cases'), 'cases', findings, 'case-index')
    cases = list(cases_by_id.values())
    for case_id, case in cases_by_id.items():
        context = f'case:{case_id}'
        kind = case.get('kind')
        if kind not in CASE_KINDS:
            findings.error('invalid_case_kind', f'unsupported case kind {kind!r}', context)
        level = case.get('level')
        if level not in {0, 1, 2, None}:
            findings.error('invalid_level', 'level must be 0, 1, 2, or null', context)
        requirement_ids = case.get('requirement_ids')
        if not isinstance(requirement_ids, list) or not requirement_ids:
            findings.error('missing_requirements', 'a case must map to at least one requirement', context)
            requirement_ids = []
        unknown_requirements = [item for item in requirement_ids if item not in requirements]
        if unknown_requirements:
            findings.error('unknown_requirement', f'unknown requirement IDs: {unknown_requirements}', context)
        for requirement_id in requirement_ids:
            requirement = requirements.get(requirement_id)
            if requirement and requirement.get('applicability') == 'unknown':
                findings.error('unknown_contract', f'case asserts requirement {requirement_id}, whose contract is still unknown', context)
            if requirement and requirement.get('applicability') == 'not_applicable':
                findings.error('not_applicable_requirement', f'case asserts requirement {requirement_id}, which is not applicable', context)
        oracle_kinds = validate_oracles(case, sources, findings, context)
        validate_assertions(case, findings, context)
        if kind == 'negative' and 'exception_contract' not in oracle_kinds:
            findings.error('missing_exception_contract', 'a negative case must use an exception_contract oracle', context)
        validate_path_evidence(case, sources, findings, context)
        validate_lifecycle(case, findings, context)
    return cases


def validate_coverage(
    data: dict[str, Any], cases: list[dict[str, Any]], findings: Findings, require_complete: bool
) -> dict[str, dict[str, Any]]:
    coverage = data.get('coverage')
    if coverage is None:
        if require_complete:
            findings.error('missing_coverage', 'a complete delivery must include a coverage assessment for all ten dimensions')
        return {}
    if not isinstance(coverage, dict):
        findings.error('invalid_coverage', 'coverage must be an object')
        return {}

    unknown_dimensions = sorted(set(coverage) - set(COVERAGE_DIMENSIONS))
    if unknown_dimensions:
        findings.error('unknown_coverage_dimension', f'unknown coverage dimensions: {unknown_dimensions}')

    cases_by_id = {str(case.get('id')): case for case in cases}
    validated: dict[str, dict[str, Any]] = {}
    for dimension in COVERAGE_DIMENSIONS:
        context = f'coverage:{dimension}'
        entry = coverage.get(dimension)
        if entry is None:
            if require_complete:
                findings.error('missing_coverage_dimension', f'missing coverage dimension {dimension!r}', context)
            else:
                findings.warn('missing_coverage_dimension', f'missing coverage dimension {dimension!r}', context)
            continue
        if not isinstance(entry, dict):
            findings.error('invalid_coverage_entry', 'coverage entry must be an object', context)
            continue
        validated[dimension] = entry
        status = entry.get('status')
        if status not in COVERAGE_STATUSES:
            findings.error('invalid_coverage_status', f'unsupported coverage status {status!r}', context)
        case_ids = entry.get('case_ids')
        if not isinstance(case_ids, list) or not all(isinstance(item, str) for item in case_ids):
            findings.error('invalid_coverage_cases', 'case_ids must be a list of strings', context)
            case_ids = []
        unknown_cases = [case_id for case_id in case_ids if case_id not in cases_by_id]
        if unknown_cases:
            findings.error('unknown_coverage_case', f'unknown case IDs: {unknown_cases}', context)
        if status == 'covered' and not case_ids:
            findings.error('covered_without_case', 'a dimension with status covered must reference at least one case', context)
        if status in {'missing', 'unknown', 'not_applicable'}:
            require_nonempty_string(entry.get('rationale'), 'rationale', findings, context)
        if status == 'not_applicable' and case_ids:
            findings.error('not_applicable_with_cases', 'a dimension with status not_applicable must not reference cases', context)
        untrusted_cases = [case_id for case_id in case_ids if findings.case_errors.get(case_id, 0)]
        if untrusted_cases:
            findings.error('coverage_uses_untrusted_case', f'coverage entry references invalid cases: {untrusted_cases}', context)
        if dimension == 'execution_evidence' and status == 'covered':
            not_passed = [
                case_id
                for case_id in case_ids
                if case_id in cases_by_id
                and (
                    cases_by_id[case_id].get('status') != 'executed'
                    or cases_by_id[case_id].get('result') != 'passed'
                )
            ]
            if not_passed:
                findings.error(
                    'execution_coverage_not_passed',
                    f'execution evidence references cases that have not reached executed/passed status: {not_passed}',
                    context,
                )
        if require_complete and status not in {'covered', 'not_applicable'}:
            findings.error(
                'incomplete_coverage',
                f'delivery coverage status must be covered or not_applicable; got {status!r}',
                context,
            )
    return validated


def validate(data: Any, repo_root: Path, require_complete_coverage: bool = False) -> dict[str, Any]:
    findings = Findings()
    if not isinstance(data, dict):
        findings.error('invalid_document', 'top-level JSON value must be an object')
        data = {}
    if data.get('schema_version') != 1:
        findings.error('schema_version', 'schema_version must be 1')
    require_nonempty_string(data.get('operator'), 'operator', findings)
    sources = validate_sources(data, repo_root, findings)
    requirements = validate_requirements(data, sources, findings)
    cases = validate_cases(data, sources, requirements, findings)
    coverage = validate_coverage(data, cases, findings, require_complete_coverage)
    credible_cases = [case.get('id') for case in cases if findings.case_errors.get(str(case.get('id')), 0) == 0]
    return {
        'operator': data.get('operator'),
        'valid': not findings.errors,
        'summary': {
            'sources': len(sources),
            'requirements': len(requirements),
            'cases': len(cases),
            'credible_cases': len(credible_cases),
            'coverage_dimensions': len(coverage),
            'errors': len(findings.errors),
            'warnings': len(findings.warnings),
        },
        'credible_case_ids': credible_cases,
        'errors': findings.errors,
        'warnings': findings.warnings,
        'limitations': [
            'A valid manifest does not prove that pytest was collected or executed, that the target path is reachable, or that the operator is correct.',
            'Code review is still required to confirm the independence and semantic accuracy of the reference.',
        ],
    }


def render_text(report: dict[str, Any]) -> str:
    summary = report['summary']
    verdict = 'PASS' if report['valid'] else 'FAIL'
    lines = [
        f"{verdict}: {report.get('operator') or '<unknown operator>'}",
        (
            f"sources={summary['sources']} requirements={summary['requirements']} cases={summary['cases']} "
            f"credible_cases={summary['credible_cases']} coverage_dimensions={summary['coverage_dimensions']} "
            f"errors={summary['errors']} warnings={summary['warnings']}"
        ),
    ]
    severity_labels = {'errors': 'ERROR', 'warnings': 'WARNING'}
    for severity in ('errors', 'warnings'):
        for finding in report[severity]:
            context = f" [{finding['context']}]" if finding.get('context') else ''
            lines.append(f"{severity_labels[severity]} {finding['code']}{context}: {finding['message']}")
    lines.append('Passing the checker means only that the manifest is valid; it does not prove that pytest ran or that the operator is correct.')
    return '\n'.join(lines) + '\n'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    parser.add_argument('--repo-root', type=Path, default=Path('.'))
    parser.add_argument('--format', choices=('text', 'json'), default='text')
    parser.add_argument('--strict-warnings', action='store_true')
    parser.add_argument(
        '--require-complete-coverage',
        action='store_true',
        help='Require all ten dimensions to be covered or explicitly marked not_applicable',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data = json.loads(args.spec.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f'FAIL: unable to load {args.spec}: {exc}')
        return 2
    report = validate(data, args.repo_root.resolve(), args.require_complete_coverage)
    if args.format == 'json':
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(render_text(report), end='')
    if not report['valid'] or (args.strict_warnings and report['warnings']):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

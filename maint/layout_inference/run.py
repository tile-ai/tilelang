#!/usr/bin/env python
"""Layout-inference verification driver.

Each module under ``cases/`` constructs PrimFuncs whose free-mode layout
search has a known-good answer.  This driver runs LayoutInference under a
PINNED target and both selection policies (``tl.layout_cost_model`` =
"register-count" or "io-aware"), snapshots the inferred layouts, and
compares them against the reviewed golden files under ``expected/<suite>/``.

Layout inference is target-dependent, so goldens are stored per target
suite.  Running without ``--target`` uses the default pinned suite
(``cuda-sm90``) rather than the host GPU, which keeps the check identical on
every machine; use ``--target auto`` for ad-hoc host-target investigation and
``--target <name>`` for another pinned suite (see ``--list-targets``).

Usage:
    python run.py                 # verify every case against the pinned goldens
    python run.py --case NAME     # verify one case (substring match)
    python run.py --target NAME   # pin another suite (cuda-sm100, metal, ...)
    python run.py --target auto   # use the host's detected target (ad-hoc)
    python run.py --record        # (re)write goldens from current behavior
    python run.py --show          # print inferred layouts as they run
    python run.py --anchor        # lower fully and check that the widest
                                  # per-buffer vector access in device TIR
                                  # matches each case's VECTOR_ANCHOR (the
                                  # width the io-aware model believed in);
                                  # variants without an anchor print the
                                  # observed widths for review
    python run.py --cute          # compare the symbolic scorer with the
                                  # independent exact-enumeration oracle

Golden files are one JSON per case and target suite:
    expected/<suite>/<case>.json = {variant: {model: {"buffers": ..., "loops": ...}}}
    expected/<suite>/target.json = the pinned target config the suite was
                                   recorded under (mismatch is an error)

Record, review the diff by hand (the layouts ARE the expectation — never
commit a recording you have not read), then commit.  A case module may
additionally define ``check(variant, model, result)`` for invariants that
must hold regardless of the exact golden (e.g. "this fragment must be
fully replicated").
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    COST_MODELS,
    DEFAULT_TARGET_SUITE,
    TARGET_SUITE_SPECS,
    _canonical,
    lower_and_extract_vector_widths,
    _suite_config,
    resolve_target,
    run_layout_inference,
    run_layout_inference_objects,
)

CASES_DIR = HERE / "cases"
EXPECTED_DIR = HERE / "expected"


def suite_dir(target_key: str) -> Path:
    return EXPECTED_DIR / target_key


def load_suite_meta(target_key: str, config: dict, *, recording: bool) -> None:
    """Guard against comparing a run against goldens recorded for another target.

    A silent target change is the failure mode this harness exists to prevent:
    on a different architecture the whole suite drifts and every case looks
    stale.  The recorded config makes the mismatch explicit.  Recording is
    exempt -- it rewrites the suite in place, and the caller reports which
    target it recorded under.
    """
    if recording:
        return
    meta_path = suite_dir(target_key) / "target.json"
    if not meta_path.exists():
        raise SystemExit(
            f"no goldens for target suite {target_key!r} ({meta_path} is missing).\n"
            f"Record them with: python run.py --target {target_key} --record\n"
            f"Known suites: {', '.join(sorted(TARGET_SUITE_SPECS))}"
        )
    recorded = json.loads(meta_path.read_text())
    if _canonical(recorded) != _canonical(config):
        raise SystemExit(
            f"target mismatch for suite {target_key!r}:\n"
            f"  run:      {json.dumps(config, sort_keys=True)}\n"
            f"  recorded: {json.dumps(recorded, sort_keys=True)}\n"
            f"Re-record with: python run.py --target {target_key} --record"
        )


def load_suite_exclusions(target_key: str) -> dict:
    """Cases this suite deliberately does not cover.

    A build configuration can be unable to produce a case's answers at all --
    e.g. the reducer-v2 cases need the CUDA codegen path, so a Metal build
    cannot satisfy their invariants. The suite names those cases here instead
    of shipping goldens for behavior it should not have.
    """
    path = suite_dir(target_key) / "excluded.json"
    return json.loads(path.read_text()) if path.exists() else {}


def write_suite_meta(target_key: str, config: dict) -> Path:
    meta_path = suite_dir(target_key) / "target.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return meta_path


def load_case_modules(name_filter: str | None):
    modules = []
    for path in sorted(CASES_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        if name_filter and name_filter not in path.stem:
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append((path.stem, module))
    return modules


def diff_result(expected, actual, prefix: str = "    ") -> list[str]:
    """Recursive field-level diff: reports exactly which field moved."""
    lines = []
    for key in sorted(set(expected) | set(actual)):
        exp, act = expected.get(key), actual.get(key)
        if isinstance(exp, dict) and isinstance(act, dict):
            lines.extend(diff_result(exp, act, prefix + f"{key}/"))
        elif exp != act:
            lines.append(f"{prefix}{key}: expected {exp!r}, got {act!r}")
    return lines


def format_layout(info: dict) -> str:
    """One-line human view of a structured layout snapshot."""
    parts = [f"{info.get('kind')} {info.get('input_shape')}->{info.get('output_shape')}"]
    if "threads" in info:
        parts.append(f"threads={info['threads']} rep={info['replicate']}")
        parts.append(f"thread: {info['forward_thread']}")
    parts.append(f"index: {', '.join(info.get('forward_index', []))}")
    return "  |  ".join(parts)


def run_anchor(modules, target, allow_unsupported: bool = False) -> int:
    """Anchor mode: lower each variant under the IO-AWARE pass config and
    check the widest per-buffer vector access in the device TIR against the
    case's VECTOR_ANCHOR — the width the cost model's winning layout was
    scored to sustain. A mismatch means the model believed a width the
    vectorizer did not deliver (or vice versa).

    Unlike the golden check this needs the full lowering pipeline, so it
    depends on more of the backend than layout inference does; a build that
    cannot lower the pinned target is reported as unsupported rather than as
    a wrong answer."""
    failures = 0
    unsupported = 0
    for case_name, module in modules:
        anchors = getattr(module, "VECTOR_ANCHOR", {})
        for variant, build in module.VARIANTS.items():
            tag = f"{case_name}/{variant}"
            try:
                widths = lower_and_extract_vector_widths(build(), target=target)
            except Exception as exc:  # noqa: BLE001 - report, keep going
                verdict = "SKIP" if allow_unsupported else "ERROR"
                print(f"{verdict} {tag}: target cannot lower this case: {type(exc).__name__}: {exc}")
                if allow_unsupported:
                    unsupported += 1
                else:
                    failures += 1
                continue
            expected = anchors.get(variant)
            if expected is None:
                print(f"OBSERVED {tag}: {widths}  (no VECTOR_ANCHOR)")
                continue
            bad = {buf: (lanes, widths.get(buf)) for buf, lanes in expected.items() if widths.get(buf) != lanes}
            if bad:
                detail = ", ".join(f"{buf}: expected {want}, got {got}" for buf, (want, got) in bad.items())
                print(f"FAIL {tag}: vector anchor: {detail}")
                failures += 1
            else:
                print(f"PASS {tag}: {expected}")
    if unsupported:
        print(f"\n{unsupported} skipped (target unsupported)")
    return failures


def run_cute(modules, target) -> int:
    """CuTe-algebra parity check: for every golden fragment layout, score the
    fragment<->global copy statement both symbolically (cute_model, via the
    in-tree CuTe layout algebra) and by exact enumeration (oracle, numpy),
    and diff (V, issue, bw, segments). This keeps the production C++ model's
    symbolic formulation calibrated against an independent exact oracle."""
    from tilelang import tvm  # noqa: PLC0415
    from tilelang.layout import Fragment  # noqa: PLC0415

    from cute_model import CuteScore, score_statement_cute  # noqa: PLC0415
    from oracle import Unenumerable, score_statement_oracle  # noqa: PLC0415

    failures = 0
    total = converted = 0
    for case_name, module in modules:
        cute_specs = getattr(module, "CUTE_STATEMENTS", {})
        for variant, build in module.VARIANTS.items():
            for model in COST_MODELS:
                objs = run_layout_inference_objects(build(), model, target=target)
                for name, (buffer, layout) in sorted(objs["buffers"].items()):
                    if not isinstance(layout, Fragment):
                        continue
                    dtype = tvm.DataType(buffer.dtype)
                    elem_bytes = (dtype.bits * dtype.lanes) // 8
                    if elem_bytes < 1:
                        continue  # sub-byte: outside both paths, same as C++
                    frag_shape = tuple(int(x) for x in layout.get_input_shape())
                    global_shape = cute_specs.get(variant, {}).get(name, frag_shape)
                    for is_store in (False, True):
                        tag = f"{case_name}/{variant}/{model}/{name}/{'store' if is_store else 'load'}"
                        total += 1
                        got = score_statement_cute(layout, global_shape, elem_bytes, is_store)
                        if not isinstance(got, CuteScore):
                            print(f"UNCONVERTIBLE {tag}: {got.reason}")
                            failures += 1
                            continue
                        converted += 1
                        try:
                            want = score_statement_oracle(layout, global_shape, elem_bytes, is_store)
                        except Unenumerable as exc:
                            print(f"ORACLE-SKIP {tag}: {exc}")
                            failures += 1
                            continue
                        same = got.vector == want.vector and got.issue == want.issue and got.bw == want.bw and got.segments == want.segments
                        if same:
                            print(f"PASS {tag}: V={got.vector} issue={got.issue} bw={got.bw}")
                        else:
                            failures += 1
                            print(f"FAIL {tag}:")
                            print(f"    cute:   V={got.vector} issue={got.issue} bw={got.bw} segs={got.segments}")
                            print(f"    oracle: V={want.vector} issue={want.issue} bw={want.bw} segs={want.segments}")
                            for note in got.notes:
                                print(f"    {note}")
    print(f"\nconversion hit rate: {converted}/{total}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="substring filter on case name")
    parser.add_argument(
        "--target",
        default=None,
        help=(
            "pinned target suite: " + ", ".join(sorted(TARGET_SUITE_SPECS)) + ", or auto "
            "(host-detected), or an inline JSON target config. "
            f"Default: {DEFAULT_TARGET_SUITE}"
        ),
    )
    parser.add_argument("--list-targets", action="store_true", help="list pinned target suites and exit")
    parser.add_argument("--record", action="store_true", help="write goldens instead of verifying")
    parser.add_argument("--show", action="store_true", help="print inferred layouts")
    parser.add_argument("--anchor", action="store_true", help="check lowered vector widths against VECTOR_ANCHOR")
    parser.add_argument("--cute", action="store_true", help="diff the CuTe-algebra scorer against the exact oracle")
    parser.add_argument(
        "--allow-unsupported",
        action="store_true",
        help=(
            "downgrade 'target cannot infer this case' to a counted skip instead of a failure. "
            "Use only when the missing capability is expected on this machine (e.g. a CPU-only "
            "build without CUDA codegen); the count is always printed."
        ),
    )
    args = parser.parse_args()

    if args.list_targets:
        for key in sorted(TARGET_SUITE_SPECS):
            config = _suite_config(key)
            marker = " (default)" if key == DEFAULT_TARGET_SUITE else ""
            print(f"{key:12s} {json.dumps(config, sort_keys=True)}{marker}")
        return 0

    target, target_key, target_config = resolve_target(args.target)
    print(f"target suite: {target_key}  {json.dumps(target_config, sort_keys=True)}")

    modules = load_case_modules(args.case)
    if not modules:
        print(f"no case matches {args.case!r} under {CASES_DIR}")
        return 2

    if args.anchor:
        failures = run_anchor(modules, target, allow_unsupported=args.allow_unsupported)
        if failures:
            print(f"\n{failures} failure(s)")
            return 1
        print("\nall anchor checks passed")
        return 0

    if args.cute:
        failures = run_cute(modules, target)
        if failures:
            print(f"\n{failures} failure(s)")
            return 1
        print("\nall cute-vs-oracle checks passed")
        return 0

    load_suite_meta(target_key, target_config, recording=args.record)
    excluded = load_suite_exclusions(target_key)

    failures = 0
    unsupported = 0
    excluded_cases = 0
    recorded_cases: list[str] = []
    for case_name, module in modules:
        if case_name in excluded:
            # A case whose answers this build configuration cannot produce has
            # no goldens in this suite. Skipped by name, with the reason on
            # record, rather than reported as drift.
            print(f"EXCLUDED {case_name}: {excluded[case_name]}")
            excluded_cases += 1
            continue
        golden_path = suite_dir(target_key) / f"{case_name}.json"
        golden = json.loads(golden_path.read_text()) if golden_path.exists() else {}
        recording: dict = {}
        for variant, build in module.VARIANTS.items():
            for model in COST_MODELS:
                tag = f"{case_name}/{variant}/{model}"
                try:
                    result = run_layout_inference(build(), model, target=target)
                except Exception as exc:  # noqa: BLE001 - report, keep going
                    verdict = "SKIP" if args.allow_unsupported else "FAIL"
                    print(f"{verdict} {tag}: target cannot infer this case: {type(exc).__name__}: {exc}")
                    if args.allow_unsupported:
                        unsupported += 1
                    else:
                        failures += 1
                    continue

                if args.show:
                    print(f"---- {tag}")
                    for section in ("buffers", "loops"):
                        for key, layout in result[section].items():
                            print(f"    {section}/{key}: {format_layout(layout)}")

                # Structural invariants hold in both record and verify mode:
                # a recording that violates them must never become a golden.
                check = getattr(module, "check", None)
                if check is not None:
                    try:
                        check(variant, model, result)
                    except AssertionError as exc:
                        print(f"FAIL {tag}: invariant check: {exc}")
                        failures += 1
                        continue

                if args.record:
                    recording.setdefault(variant, {})[model] = result
                    print(f"RECORD {tag}")
                    continue

                expected = golden.get(variant, {}).get(model)
                if expected is None:
                    print(f"MISSING GOLDEN {tag} (run --record)")
                    failures += 1
                elif expected != result:
                    print(f"FAIL {tag}: layout drift")
                    print("\n".join(diff_result(expected, result)))
                    failures += 1
                else:
                    print(f"PASS {tag}")

        if args.record:
            # Always rewrite the case file: a stale snapshot must not survive
            # a recording.
            if recording:
                golden_path.parent.mkdir(parents=True, exist_ok=True)
                golden_path.write_text(json.dumps(recording, indent=2, sort_keys=True) + "\n")
                print(f"wrote {golden_path}")
            elif golden_path.exists():
                golden_path.unlink()
                print(f"removed {golden_path} (nothing to record)")
            recorded_cases.append(case_name)

    if args.record and recorded_cases:
        meta_path = write_suite_meta(target_key, target_config)
        print(f"wrote {meta_path}")

    summary = f"\n{len(modules)} case(s) under target suite {target_key}"
    if excluded_cases:
        summary += f", {excluded_cases} excluded for this target"
    if unsupported:
        summary += f", {unsupported} skipped (target unsupported)"
    if failures:
        print(summary)
        print(f"{failures} failure(s)")
        return 1
    print(summary)
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

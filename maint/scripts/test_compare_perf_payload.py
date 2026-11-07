import json
import tempfile
import unittest
from pathlib import Path

from compare_perf_regression import _parse_args, _write_json, compare_payloads


class ComparePayloadTests(unittest.TestCase):
    def test_invalid_samples_cannot_disappear_from_both_runs(self):
        for value in [0, -1, float("nan"), float("inf"), True, 10**400]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "matmul"):
                payload = {"results": {"matmul": {"latency": value}}}
                compare_payloads(payload, payload)

    def test_invalid_samples_on_either_side(self):
        valid = {"results": {"matmul": 1}}
        for value in [0, -1, float("nan"), float("inf"), True, 10**400]:
            invalid = {"results": {"matmul": {"latency": value}}}
            for old, new in [(invalid, valid), (valid, invalid)]:
                with self.subTest(value=value, old=old), self.assertRaisesRegex(ValueError, "matmul"):
                    compare_payloads(old, new)

    def test_nonfinite_comparison_is_rejected(self):
        for old_latency, new_latency in [(1e308, 1e-308), (1e307, 1.0)]:
            with self.subTest(old_latency=old_latency), self.assertRaisesRegex(ValueError, "matmul"):
                compare_payloads({"results": {"matmul": old_latency}}, {"results": {"matmul": new_latency}})

    def test_finite_extreme_comparison_produces_strict_json(self):
        report = compare_payloads({"results": {"matmul": 1e-308}}, {"results": {"matmul": 1e308}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            _write_json(path, report)
            parsed = json.loads(path.read_text(), parse_constant=lambda value: self.fail(f"Non-JSON constant: {value}"))
        self.assertEqual(parsed["results"][0]["delta_pct"], -100.0)

    def test_nonfinite_passthrough_failure_cannot_be_written_as_json(self):
        report = compare_payloads({"failures": {"matmul": {"returncode": float("nan")}}}, {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with self.assertRaises(ValueError):
                _write_json(path, report)
            self.assertFalse(path.exists())

    def test_malformed_failures_are_reported(self):
        with self.assertRaisesRegex(ValueError, "failures"):
            compare_payloads({}, {"failures": ["could not run"]})

    def test_threshold_must_be_positive_and_finite(self):
        for threshold in ["nan", "inf", "0", "-1"]:
            with self.subTest(threshold=threshold), self.assertRaises(SystemExit):
                _parse_args(["old.json", "new.json", "--regression-threshold", threshold])

    def test_legacy_and_structured_latency(self):
        report = compare_payloads({"results": {"matmul": 6}}, {"results": {"matmul": {"latency": 3}}})
        self.assertEqual(report["results"][0]["speedup"], 2)

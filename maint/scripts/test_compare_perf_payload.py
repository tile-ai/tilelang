import unittest

from compare_perf_regression import _parse_args, compare_payloads


class ComparePayloadTests(unittest.TestCase):
    def test_invalid_samples_cannot_disappear_from_both_runs(self):
        for value in [0, -1, float("nan"), float("inf"), True, 10**400]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "matmul"):
                payload = {"results": {"matmul": {"latency": value}}}
                compare_payloads(payload, payload)

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

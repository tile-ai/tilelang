"""CPU-only tests for the host-check runner's result classification."""

import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_all


class ClassifyResultTest(unittest.TestCase):
    @staticmethod
    def result(rc, stderr="", stdout=""):
        return subprocess.CompletedProcess([], rc, stdout, stderr)

    def test_expected_validation_errors(self):
        errors = {
            "01_num_args_mismatch.py": "ValueError: num_args should be 3",
            "02_pointer_type_error.py": "ValueError: Expect buffer A_handle to be pointer or tensor",
            "03_ndim_mismatch.py": "ValueError: A_handle.ndim is expected to equal 2",
            "04_dtype_mismatch.py": "ValueError: A_handle.dtype is expected to be float16",
            "05_shape_mismatch.py": "ValueError: Argument A_handle.shape[1] has an unsatisfied constraint",
            "06_strides_mismatch.py": "ValueError: Argument A_handle.strides[1] has an unsatisfied constraint",
            "07_device_type_mismatch.py": "ValueError: A_handle.device_type mismatch",
            "08_device_id_mismatch.py": "ValueError: A_handle.device_id has an unsatisfied constraint",
            "09_null_data_pointer.py": "ValueError: Expect buffer A_handle to be pointer or tensor",
            "10_scalar_type_mismatch.py": "ValueError: Expect arg[0] to be int",
        }
        for name, stderr in errors.items():
            with self.subTest(name=name):
                self.assertEqual(run_all.classify_result(name, self.result(1, stderr)), "PASS")

    def test_unrelated_failures_and_crashes_are_not_passes(self):
        name = "01_num_args_mismatch.py"
        for rc, stderr in (
            (1, "RuntimeError: Compilation Failed!"),
            (1, "CUDA is not available\nValueError: unrelated failure"),
            (1, "ModuleNotFoundError: No module named torch"),
            (-11, "ValueError: num_args should be 3"),
            (0xC0000005, "ValueError: num_args should be 3"),
            (0, "ValueError: num_args should be 3"),
        ):
            with self.subTest(rc=rc, stderr=stderr):
                self.assertEqual(run_all.classify_result(name, self.result(rc, stderr)), "FAIL")

    def test_cuda_skips_require_the_final_exception(self):
        name = "01_num_args_mismatch.py"
        for stderr in (
            "Traceback (most recent call last):\nRuntimeError: CUDA is not available",
            "RuntimeError: CUDA is not available; cannot build CUDA kernel for host-check repros.",
        ):
            self.assertEqual(run_all.classify_result(name, self.result(1, stderr)), "SKIP")
        self.assertEqual(
            run_all.classify_result(name, self.result(1, "RuntimeError: CUDA is not available\nValueError: crashed")),
            "FAIL",
        )

    def test_successful_skip_marker_must_start_a_line(self):
        name = "08_device_id_mismatch.py"
        self.assertEqual(run_all.classify_result(name, self.result(0, stdout="[SKIP] only one GPU")), "SKIP")
        self.assertEqual(run_all.classify_result(name, self.result(0, stdout="log: [SKIP]")), "FAIL")

    def test_runner_preserves_replacement_characters_in_utf8_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "01_num_args_mismatch.py").touch()
            proc = self.result(1, "ValueError: num_args should be 3\ufffd", "hello\ufffd")
            with (
                mock.patch.object(run_all, "__file__", str(root / "run_all.py")),
                mock.patch.object(run_all.subprocess, "run", return_value=proc),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit) as exc,
            ):
                run_all.main()
            self.assertEqual(exc.exception.code, 1)  # other scripts are missing
            self.assertEqual((root / "logs/01_num_args_mismatch.py.err").read_bytes(), proc.stderr.encode("utf-8"))
            self.assertEqual((root / "logs/01_num_args_mismatch.py.out").read_bytes(), proc.stdout.encode("utf-8"))


if __name__ == "__main__":
    unittest.main()

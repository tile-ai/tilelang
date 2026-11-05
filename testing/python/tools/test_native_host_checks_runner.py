"""CPU harness tests use synthetic children; they do not execute GPU repros."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


RUNNER = Path(__file__).resolve().parents[3] / "maint/host_checks/run_all.py"
SKIP = "[SKIP] Need at least 2 CUDA devices to reproduce device_id mismatch."
MAIN = "__tl_output_allocate_main(A: Tensor([128, 128], float16), B: Tensor([128, 128], float16), output_frame: T.Ptr(T.void))"
COMPUTE = "__tl_output_compute_main(A: Tensor([128, 128], float16), B: Tensor([128, 128], float16), C: Tensor([128, 128], float16))"


def diagnostic(problem, expected, signature=MAIN):
    return f"{problem} when calling:\n  `{signature}`,\n  expected {expected}"


CASES = {
    "01_num_args_mismatch.py": ("ValueError", "Kernel expected 2 inputs, but 1 are provided."),
    "02_pointer_type_error.py": ("RuntimeError", diagnostic("Mismatched type on argument #0", "Tensor")),
    "03_ndim_mismatch.py": ("RuntimeError", diagnostic("Mismatched A.ndim on argument #0", "2")),
    "04_dtype_mismatch.py": ("RuntimeError", diagnostic("Mismatched A.dtype on argument #0", "float16")),
    "05_shape_mismatch.py": ("RuntimeError", diagnostic("Invalid A.shape[1] on argument #0", "128")),
    "06_strides_mismatch.py": ("RuntimeError", diagnostic("Mismatched A.strides on argument #0", "to be compact array")),
    "07_device_type_mismatch.py": (
        "RuntimeError",
        "Check failed: anchor.device().device_type == selected->device_type (1 vs. 2) : Allocator anchor device type does not match the kernel",
    ),
    "08_device_id_mismatch.py": ("RuntimeError", diagnostic("Mismatched B.device_id on argument #1", "to match A.device_id")),
    "09_null_data_pointer.py": ("RuntimeError", diagnostic("Mismatched type on argument #0", "Tensor", COMPUTE)),
    "10_scalar_type_mismatch.py": (
        "RuntimeError",
        diagnostic("Mismatched type on argument #0", "int", "scalar_check(x: int32, flag: bool)"),
    ),
}


def script_text(name, body):
    return (
        "import sys\nfrom pathlib import Path\nfrom common import MARKER\n"
        "assert MARKER == 'sibling import'\nassert __name__ == '__main__'\n"
        f"assert Path(sys.argv[0]).name == {name!r}\n"
        "print('child executable: ' + sys.executable)\n" + body + "\n"
    )


@pytest.fixture
def scripts(tmp_path):
    root = tmp_path / "checks"
    root.mkdir()
    shutil.copy2(RUNNER, root / "run_all.py")
    (root / "common.py").write_text("MARKER = 'sibling import'\n")
    for name, (kind, message) in CASES.items():
        (root / name).write_text(script_text(name, f"raise {kind}({message!r})"))
    return root


def run(root):
    environment = os.environ.copy()
    environment.update({"PATH": str(root / "no-executables"), "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"})
    return subprocess.run(
        [sys.executable, str(root / "run_all.py")], cwd=root.parent, env=environment, capture_output=True, text=True, timeout=30
    )


def test_expected_errors_pass_and_preserve_separate_logs(scripts):
    result = run(scripts)
    assert result.returncode == 0, result.stdout + result.stderr
    for name, (kind, message) in CASES.items():
        assert f"[PASS] {name}" in result.stdout
        assert (scripts / "logs" / f"{name}.out").read_text() == f"child executable: {sys.executable}\n"
        stderr = (scripts / "logs" / f"{name}.err").read_text()
        assert "Traceback (most recent call last):" in stderr
        assert f"{kind}: {message}" in stderr
        record = json.loads((scripts / "logs" / f"{name}.result.json").read_text())
        assert record["status"] == "PASS" and record["returncode"] == 1
        assert record["outcome"]["exception"] == {"module": "builtins", "type": kind, "message": message}


def test_only_device_id_case_can_return_a_skip(scripts):
    name = "08_device_id_mismatch.py"
    (scripts / name).write_text(script_text(name, f"print({SKIP!r})"))
    result = run(scripts)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"[SKIP] {name}" in result.stdout
    assert result.stdout.count("[PASS]") == 9
    record = json.loads((scripts / "logs" / f"{name}.result.json").read_text())
    assert record["outcome"]["outcome"] == "returned" and record["returncode"] == 0


DTYPE = CASES["04_dtype_mismatch.py"][1]
TYPE = CASES["02_pointer_type_error.py"][1]
DEVICE_ID = CASES["08_device_id_mismatch.py"][1]
COUNT = CASES["01_num_args_mismatch.py"][1]
FAKE_TRACEBACK = "Traceback (most recent call last):\n  File 'generated.py', line 1\nRuntimeError: " + DTYPE
REFUSALS = [
    pytest.param("01_num_args_mismatch.py", "import _host_check_module_that_does_not_exist", id="missing-import"),
    pytest.param("05_shape_mismatch.py", "raise OSError('libtvm_tirx_compiler.so could not be loaded')", id="loader-failure"),
    pytest.param("04_dtype_mismatch.py", f"print({DTYPE!r})\nraise RuntimeError('CUDA compilation failed')", id="printed-host-source"),
    pytest.param(
        "04_dtype_mismatch.py", f"print({FAKE_TRACEBACK!r})\nimport _host_check_module_that_does_not_exist", id="printed-traceback"
    ),
    pytest.param("04_dtype_mismatch.py", f"raise OSError({FAKE_TRACEBACK!r})", id="traceback-in-message"),
    pytest.param("02_pointer_type_error.py", f"raise TypeError({TYPE!r})", id="changed-public-class"),
    pytest.param("03_ndim_mismatch.py", f"raise RuntimeError({DTYPE!r})", id="wrong-field"),
    pytest.param(
        "10_scalar_type_mismatch.py",
        "raise RuntimeError(" + repr(diagnostic("Mismatched type on argument #1", "boolean", "scalar_check(x: int32, flag: bool)")) + ")",
        id="unreached-bool-call",
    ),
    pytest.param("09_null_data_pointer.py", "raise RuntimeError('A data pointer is NULL on argument #0')", id="different-null-contract"),
    pytest.param(
        "01_num_args_mismatch.py",
        f"try:\n    raise ValueError({COUNT!r})\nexcept ValueError:\n    raise RuntimeError('later unrelated failure')",
        id="chained-error",
    ),
    pytest.param("04_dtype_mismatch.py", f"print({DTYPE!r})", id="normal-return"),
    pytest.param("04_dtype_mismatch.py", "sys.exit(1)", id="nonzero-system-exit"),
    pytest.param("04_dtype_mismatch.py", "import os\nos._exit(1)", id="missing-outcome"),
    pytest.param("01_num_args_mismatch.py", f"print({SKIP!r})", id="wrong-script-skip"),
    pytest.param("08_device_id_mismatch.py", f"print({(SKIP + ' extra')!r})", id="skip-substring"),
    pytest.param("08_device_id_mismatch.py", f"print({SKIP!r}, file=sys.stderr)", id="stderr-skip"),
    pytest.param("08_device_id_mismatch.py", f"print({SKIP!r})\nraise RuntimeError({DEVICE_ID!r})", id="failed-skip"),
    pytest.param("08_device_id_mismatch.py", f"print({SKIP!r})\nprint({SKIP!r})", id="duplicate-skip"),
    pytest.param("08_device_id_mismatch.py", f"print({SKIP!r})\nsys.exit(0)", id="zero-system-exit-skip"),
    pytest.param("08_device_id_mismatch.py", f"print({SKIP!r})\nprint('not the final outcome')", id="trailing-skip-output"),
    pytest.param(
        "02_pointer_type_error.py",
        f"import builtins\nclass RuntimeError(builtins.RuntimeError):\n    pass\nraise RuntimeError({TYPE!r})",
        id="custom-exception-subclass",
    ),
]


@pytest.mark.parametrize("name, body", REFUSALS)
def test_rejects_unrelated_errors_and_fake_skips(scripts, name, body):
    (scripts / name).write_text(script_text(name, body))
    result = run(scripts)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"[FAIL] {name}" in result.stdout
    assert f"[PASS] {name}" not in result.stdout and f"[SKIP] {name}" not in result.stdout


def test_missing_script_fails_the_parent(scripts):
    name = "09_null_data_pointer.py"
    (scripts / name).unlink()
    result = run(scripts)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"[MISSING] {name}" in result.stdout


def test_directory_in_place_of_script_is_missing(scripts):
    name = "09_null_data_pointer.py"
    (scripts / name).unlink()
    (scripts / name).mkdir()
    result = run(scripts)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"[MISSING] {name}" in result.stdout


def test_stale_result_cannot_replace_a_child_outcome(scripts):
    name = "04_dtype_mismatch.py"
    (scripts / "logs").mkdir()
    (scripts / "logs" / f"{name}.result.json").write_text(json.dumps({"status": "PASS", "returncode": 1}))
    (scripts / name).write_text(script_text(name, "import os\nos._exit(1)"))
    result = run(scripts)
    assert result.returncode == 1, result.stdout + result.stderr
    record = json.loads((scripts / "logs" / f"{name}.result.json").read_text())
    assert record["status"] == "FAIL" and record["outcome"] is None

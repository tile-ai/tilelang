"""Opt-in real AI Core fault test; the invalid launch lives in a subprocess."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


_CHILD = r"""
import ctypes
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch_npu

import tilelang
from tilelang.ascend import language as T

root = Path(os.environ["TILELANG_HW_FAULT_DIR"])
tilelang.disable_cache()

@T.prim_func
def aic_fault_e2e(
    x: T.Tensor((128,), T.float32),
    y: T.Tensor((128,), T.float32),
    z: T.Tensor((128,), T.float32),
):
    with T.Kernel(1), T.SimtVF(threads=128):
        for i in T.Parallel(128):
            z[i] = x[i] + y[i]



kernel = tilelang.compile(
    aic_fault_e2e,
    out_idx=2, target="ascend", execution_backend="tvm_ffi",
    pass_configs={"tl.ascend_exception_dump": True},
)
x_expected = np.arange(128, dtype=np.float32) * np.float32(0.25) - np.float32(13.5)
y_expected = ((np.arange(128, dtype=np.float32) * 7) % 31) - np.float32(11)
np.save(root / "expected_x.npy", x_expected)
np.save(root / "expected_y.npy", y_expected)
x = torch.from_numpy(x_expected).to("npu")
y = torch.from_numpy(y_expected).to("npu")
z = kernel(x, y)
# The normal launch installs the production callback.
torch.npu.synchronize()
np.testing.assert_array_equal(z.cpu().numpy(), x_expected + y_expected)

device_mod = kernel.artifact.device_mod
assert len(device_mod.functions) == 1
func = next(iter(device_mod.functions.values()))
name = str(func.attrs["global_symbol"])
metadata = [list(map(int, entry)) for entry in func.attrs["tl.ascend_exception_dump_tensors"]]
raw_args = [None] * len(func.params)
addresses = {"x": x.data_ptr(), "y": y.data_ptr(), "z": 0}
tensor_names = []
for pointer_index, elements_index, dtype, bits in metadata:
    tensor_name = str(func.params[pointer_index].name)
    assert tensor_name in addresses and dtype == 0 and bits == 32
    raw_args[pointer_index] = ctypes.c_void_p(addresses[tensor_name])
    raw_args[elements_index] = 128
    tensor_names.append(tensor_name)
assert all(arg is not None for arg in raw_args)
launch_tags = [str(tag) for tag in func.attrs.get("tirx.kernel_launch_params", [])]
for tag in launch_tags:
    if tag.startswith("blockIdx."):
        raw_args.append(1)
    elif tag == "tirx.use_dyn_shared_memory":
        raw_args.append(0)
    else:
        raise AssertionError(f"Unexpected launch tag: {tag}")

plan = {
    "kernel_name": name,
    "backend": "tvm_ffi",
    "device": torch.npu.get_device_name(0),
    "warmup_correct": True,
    "tensor_names": tensor_names,
    "null_tensor": "z",
    "metadata": metadata,
    "launch_tags": launch_tags,
}
(root / "plan.json").write_text(json.dumps(plan, indent=2))
(root / "kernel.asc").write_text(kernel.adapter.get_kernel_source())
print("ABOUT_TO_TRIGGER_REAL_AI_CORE_FAULT", name, flush=True)

# Bypass the host tensor validator only for this intentional NULL store. The
# production path still builds the real exception metadata and launch.
native = kernel.adapter.rt_mod.imports[0][name]
failure = None
stage = "launch"
try:
    native(*raw_args)
    stage = "synchronize"
    torch.npu.synchronize()
except Exception as error:
    failure = str(error)
    print("REAL_AI_CORE_EXCEPTION", stage, failure, flush=True)
assert failure is not None, "The intentional NULL output did not fault"
assert any(term in failure.lower() for term in ("vector core", "ai core", "aicore", "507035", "507015")), failure

# Keep input allocations alive while CANN's reporting thread invokes TileLang's
# callback and writes the dump. Never invoke the dump helper manually here.
deadline = time.monotonic() + 30
dump_files = []
while time.monotonic() < deadline:
    dump_files = list(Path(os.environ["ASCEND_DUMP_PATH"]).glob(
        f"extra-info/data-dump/*/{name}.custom.*"
    ))
    if dump_files and all(path.stat().st_size > 0 for path in dump_files):
        time.sleep(2)
        break
    time.sleep(0.25)
(root / "fault.json").write_text(json.dumps({
    "stage": stage, "exception": failure,
    "dump_files": [str(path) for path in dump_files],
}, indent=2))
print("FAULT_CHILD_FINISHED", len(dump_files), flush=True)
# The faulted ACL context is intentionally confined to this child process.
os._exit(0)
"""


@pytest.mark.skipif(
    os.environ.get("TILELANG_TEST_ASCEND_HW_EXCEPTION") != "1",
    reason="set TILELANG_TEST_ASCEND_HW_EXCEPTION=1 to trigger a real AI Core exception",
)
def test_real_ai_core_exception_dump_roundtrip(tmp_path):
    """NULL output -> real CANN callback -> custom dump -> exact input bytes."""
    from tilelang.ascend.exception_dump_bin import parse_exception_dump

    root = tmp_path.resolve()
    child = root / "fault_child.py"
    child.write_text(_CHILD)
    (root / "dump").mkdir()
    env = os.environ.copy()
    env["TILELANG_HW_FAULT_DIR"] = str(root)
    env["ASCEND_DUMP_PATH"] = str(root / "dump")
    env["ASCEND_DUMP_SCENE"] = "aic_err_brief_dump"
    env["PYTHONUNBUFFERED"] = "1"
    repo = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(child)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    (root / "child.stdout.log").write_text(result.stdout)
    (root / "child.stderr.log").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ABOUT_TO_TRIGGER_REAL_AI_CORE_FAULT" in result.stdout
    plan = json.loads((root / "plan.json").read_text())
    fault = json.loads((root / "fault.json").read_text())
    assert plan["warmup_correct"]
    assert fault["dump_files"], result.stdout + result.stderr
    evidence = (fault["exception"] + result.stdout + result.stderr).lower()
    assert any(term in evidence for term in ("vector core exception", "ai core exception", "aicore", "507035", "507015")), evidence

    tensors = parse_exception_dump(root / "dump", plan["kernel_name"], root / "parsed", wait_seconds=0)
    names = [name for name in plan["tensor_names"] if name != plan["null_tensor"]]
    assert names == ["x", "y"]
    assert len(tensors) == 2, tensors
    checks = []
    for index, (name, tensor) in enumerate(zip(names, tensors)):
        expected = np.load(root / f"expected_{name}.npy", allow_pickle=False)
        recovered = tensor["data"].reshape(-1)
        assert tensor["type"] == "input" and tensor["index"] == index
        assert recovered.dtype == expected.dtype
        np.testing.assert_array_equal(recovered, expected)
        assert recovered.tobytes() == expected.tobytes()
        np.save(root / f"recovered_{name}.npy", recovered)
        checks.append(
            {
                "name": name,
                "elements": recovered.size,
                "bytes": recovered.nbytes,
                "dtype": str(recovered.dtype),
                "max_absolute_error": float(np.max(np.abs(recovered - expected))),
                "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                "recovered_sha256": hashlib.sha256(recovered.tobytes()).hexdigest(),
                "parsed_file": tensor["file"],
            }
        )
    report = {"passed": True, "plan": plan, "fault": fault, "checks": checks}
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print("REAL_AI_CORE_DUMP_ROUNDTRIP_PASSED", root, flush=True)

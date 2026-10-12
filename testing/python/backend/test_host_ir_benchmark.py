"""Hardware-free routing and report coverage for the Host IR CI benchmark."""

import io
import json
from types import SimpleNamespace

import pytest
import torch

from maint.host_checks import host_ir_benchmark as benchmark


@pytest.mark.parametrize("target", ["ascend", "pto"])
@pytest.mark.parametrize("case", ["small", "dynamic", "gemm"])
def test_ascend_dialects_share_npu_workloads(monkeypatch, target, case):
    program = object()
    paths, devices = [], []

    def load(path):
        paths.append(path)
        return {"dynamic_grid_kernel": program, "gemm": lambda *shape: program}

    def randn(*shape, device, dtype):
        devices.append(device)
        return torch.randn(*shape, dtype=dtype)

    monkeypatch.setattr(benchmark.runpy, "run_path", load)
    fake_torch = SimpleNamespace(randn=randn, empty_like=torch.empty_like, bfloat16=torch.bfloat16)
    actual, inputs, reference = benchmark.workload(target, case, fake_torch)
    assert actual is program
    assert set(devices) == {"npu"}
    assert inputs[0].dtype == torch.bfloat16
    if case == "gemm":
        assert paths[0].endswith("example_gemm.py")
        torch.testing.assert_close(reference, inputs[0].float() @ inputs[1].float().T)
    else:
        assert paths[0].endswith("test_tvm_ffi_host_args.py")
        assert inputs[0].shape == (1 if case == "small" else 8, 128)
        assert reference is inputs[0]


def test_pto_report_covers_both_hosts_cases_and_cache_phases(monkeypatch, tmp_path):
    commands = []
    metrics = dict.fromkeys(
        ["compile_or_load_s", "host_codegen_s", "host_publish_s", "submit_us", "complete_us", "executable_bytes", "cache_bytes"],
        1.0,
    )

    def popen(command, **kwargs):
        commands.append(command)
        assert kwargs["env"]["TILELANG_DISABLE_CACHE"] == "0"
        return SimpleNamespace(
            stdout=io.StringIO("HOST_IR_READY\nHOST_IR_RESULT " + json.dumps(metrics) + "\n"),
            wait=lambda: 0,
            poll=lambda: 0,
        )

    monkeypatch.setattr(benchmark.subprocess, "Popen", popen)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert not benchmark.report(SimpleNamespace(targets=["ascend", "pto"], trials=1, output=str(tmp_path), timeout=30))
    assert len(commands) == 24
    rows = json.loads((tmp_path / "samples.json").read_text())
    assert {(r["target"], r["case"], r["host"], r["phase"]) for r in rows} == {
        (target, case, host, phase)
        for target in ("ascend", "pto")
        for case in ("small", "dynamic", "gemm")
        for host in ("c", "mir")
        for phase in ("cold", "hit")
    }
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "pto/gemm/hit" in summary
    assert "Failures: none" in summary

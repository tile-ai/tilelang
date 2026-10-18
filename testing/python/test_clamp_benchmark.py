import runpy
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture
def benchmark():
    script = Path(__file__).resolve().parents[2] / "benchmark/clamp/benchmark_tilelang.py"
    return runpy.run_path(str(script))["main"]


@pytest.mark.parametrize("alias", ["same", "relative", "symlink", "default"])
def test_comparison_rejects_output_alias_before_writing(benchmark, tmp_path, monkeypatch, capsys, alias):
    baseline = tmp_path / "clamp-poc"
    baseline.mkdir()
    torch.save([torch.zeros(4)], baseline / "semantics.pt")
    (baseline / "result.json").write_text('{"baseline": true}')
    original = {path.name: path.read_bytes() for path in baseline.iterdir()}
    monkeypatch.chdir(tmp_path)
    output = baseline
    if alias == "relative":
        output = Path("clamp-poc")
    elif alias == "symlink":
        output = tmp_path / "candidate"
        output.symlink_to(baseline, target_is_directory=True)
    argv = ["benchmark_tilelang.py", "--compare", str(baseline)]
    if alias != "default":
        argv += ["--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)

    def unexpected_gpu_work(_):
        pytest.fail("Rejected paths must not reach GPU execution")

    monkeypatch.setitem(benchmark.__globals__, "semantic_outputs", unexpected_gpu_work)
    with pytest.raises(SystemExit) as error:
        benchmark()
    assert error.value.code == 2
    assert "--output and --compare must refer to different directories" in capsys.readouterr().err
    assert {path.name: path.read_bytes() for path in baseline.iterdir()} == original


def test_comparison_detects_different_outputs(benchmark, tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    torch.save([torch.zeros(4)], baseline / "semantics.pt")
    original = (baseline / "semantics.pt").read_bytes()
    candidate = tmp_path / "candidate"
    monkeypatch.setattr(sys, "argv", ["benchmark_tilelang.py", "--compare", str(baseline), "--output", str(candidate)])
    monkeypatch.setitem(benchmark.__globals__, "semantic_outputs", lambda _: [torch.ones(4)])
    with pytest.raises(AssertionError):
        benchmark()
    assert (baseline / "semantics.pt").read_bytes() == original
    assert torch.equal(torch.load(candidate / "semantics.pt", weights_only=True)[0], torch.ones(4))

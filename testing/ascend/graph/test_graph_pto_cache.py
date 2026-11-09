"""A fresh Python process must reload PTO graph kernels without compiling."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import pytest

import tilelang.testing
from tilelang.contrib.bisheng import find_bisheng_path


_CACHE_PROBE = textwrap.dedent(
    """
    import json
    import sys
    import torch
    from tilelang.graph import GraphConfig, make_backend
    from tilelang.jit.adapter.pto import PTOKernelAdapter
    from tilelang.jit.kernel import JITKernel

    restored = []
    original_restore = PTOKernelAdapter.from_database.__func__

    def restore(cls, *args, **kwargs):
        adapter = original_restore(cls, *args, **kwargs)
        restored.append(type(adapter).__name__)
        return adapter

    PTOKernelAdapter.from_database = classmethod(restore)
    if sys.argv[1] == 'reload':
        def forbid_compile(*args, **kwargs):
            raise AssertionError('Expected PTO disk-cache reload, not compilation')
        JITKernel._compile_and_create_adapter = forbid_compile

    reports = []
    backend = make_backend(GraphConfig(target='pto', execution_backend='pto', strict=True))

    def capture(gm, inputs):
        runner = backend(gm, inputs)
        reports.append(runner.graph_report)
        return runner

    def model(x):
        return torch.relu(x + 1)

    compiled = torch.compile(model, backend=capture, fullgraph=True)
    values = torch.arange(-128, 128, device='npu', dtype=torch.float32)
    actual = compiled(values)
    torch.testing.assert_close(actual, model(values))
    previous = actual.clone()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        values.fill_(2)
        again = compiled(values)
    torch.npu.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, previous)
    torch.testing.assert_close(again, torch.full_like(values, 3))
    status, = reports[-1].partitions
    assert status.compiled and status.compiled_calls == 2 and status.fallback_calls == 0
    assert status.target_backend == status.execution_backend == 'pto'
    assert status.kernel_count == 2
    print('PTO_GRAPH_CACHE=' + json.dumps({
        'restored': restored,
        'kernel_count': status.kernel_count,
        'sum': actual.sum().item(),
        'again_sum': again.sum().item(),
    }))
    """
)


@tilelang.testing.requires_ascend
@pytest.mark.pto
def test_pto_graph_disk_cache_in_fresh_process(tmp_path):
    if importlib.util.find_spec("ptodsl") is None or shutil.which("ptoas") is None:
        pytest.skip("PTO execution requires PTODSL, ptoas, and Bisheng")
    try:
        find_bisheng_path()
    except RuntimeError as error:
        pytest.skip(str(error))
    environment = dict(os.environ, TILELANG_CACHE_DIR=str(tmp_path), TILELANG_DISABLE_CACHE="0")
    results = []
    for mode in ("populate", "reload"):
        result = subprocess.run(
            [sys.executable, "-u", "-c", _CACHE_PROBE, mode],
            cwd=Path(__file__).resolve().parents[3],
            env=environment,
            text=True,
            capture_output=True,
            timeout=180,
        )
        assert result.returncode == 0, f"PTO cache {mode} failed:\n{result.stdout}\n{result.stderr}"
        records = [line.removeprefix("PTO_GRAPH_CACHE=") for line in result.stdout.splitlines() if line.startswith("PTO_GRAPH_CACHE=")]
        assert len(records) == 1, result.stdout
        results.append(json.loads(records[0]))
    assert results[0]["restored"] == []
    assert results[1]["restored"] == ["PTOKernelAdapter", "PTOKernelAdapter"]
    assert results[0]["sum"] == results[1]["sum"]
    assert results[0]["again_sum"] == results[1]["again_sum"]
    assert any(tmp_path.rglob("*.so"))

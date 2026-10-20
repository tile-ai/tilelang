"""`set_autotune_inputs` must supply benchmark tensors on the current device.

Before the fix, the frozen supply closure resolved its benchmark device to
CUDA-or-CPU only, so on an NPU host every captured tensor was moved to CPU and
each tuning trial failed with a launcher device mismatch (issue #3545). Like
the default tensor supply, benchmark tensors follow the current device of the
active backend; workers bind their device, and inputs captured on another
device are aligned to the device the benchmark runs on.
"""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang.ascend import language as T
from tilelang.autotuner import AutoTuner, set_autotune_inputs
from tilelang.env import CacheState, env

N = 64
CONFIGS = [{"block": 32}, {"block": 64}]


def _vecadd_kernel(n, block):
    @T.prim_func
    def main(
        a: T.Tensor((n,), T.float32),
        b: T.Tensor((n,), T.float32),
        c: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(n // block) as bx, T.SimtVF(threads=block):
            for i in T.Parallel(block):
                idx = bx * block + i
                c[idx] = a[idx] + b[idx]

    return main


def _vecadd_config(block):
    return _vecadd_kernel(N, block)


@tilelang.autotune(configs=CONFIGS, warmup=1, rep=3, timeout=120)
@tilelang.jit(out_idx=[2])
def vecadd(n: int, block: int = 1024):
    return _vecadd_kernel(n, block)


def _deterministic_inputs(device="npu"):
    a = torch.arange(N, dtype=torch.float32, device=device)
    b = torch.arange(N, dtype=torch.float32, device=device) * 2
    return a, b


def _isolate_autotune_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(env, "TILELANG_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(env, "TILELANG_DISABLE_CACHE", "0")
    monkeypatch.setattr(env, "TILELANG_AUTO_TUNING_DISABLE_CACHE", "0")
    monkeypatch.setattr(CacheState, "_enabled", True)
    monkeypatch.setattr(AutoTuner, "_memory_cache", {})


def _tune_with_recorded_supply(a, b, monkeypatch, tmp_path):
    _isolate_autotune_cache(monkeypatch, tmp_path)
    supplied_devices = []

    def ref_prog(x, y):
        supplied_devices.append((x.device, y.device))
        return x + y

    with set_autotune_inputs(a, b):
        result = (
            AutoTuner(_vecadd_config, configs=CONFIGS)
            .set_compile_args(out_idx=[2])
            .set_profile_args(warmup=1, rep=3, timeout=120, ref_prog=ref_prog)
            .run(warmup=1, rep=3, timeout=120)
        )
    return result, supplied_devices


@tilelang.testing.requires_ascend
def test_autotune_with_set_inputs_tunes_on_npu():
    a, b = _deterministic_inputs()

    with set_autotune_inputs(a, b):
        kernel = vecadd(N)

    out = kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out, a + b, rtol=0, atol=0)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("capture", ["npu", "cpu", "other_npu"])
def test_set_inputs_supplies_captured_inputs_on_current_device(capture, monkeypatch, tmp_path):
    device = capture
    if capture == "other_npu":
        if torch.npu.device_count() < 2:
            pytest.skip("needs two visible NPU devices")
        other = next(i for i in range(torch.npu.device_count()) if i != torch.npu.current_device())
        try:
            # The alignment copy is cross-device, which some runtimes reject.
            torch.empty(1, device=f"npu:{other}").to("npu")
        except RuntimeError:
            pytest.skip("cross-device copy is not available on this host")
        device = f"npu:{other}"

    a, b = _deterministic_inputs(device=device)

    result, supplied_devices = _tune_with_recorded_supply(a, b, monkeypatch, tmp_path)

    expected = torch.device("npu", torch.npu.current_device())
    assert result.latency > 0
    assert supplied_devices
    assert set(supplied_devices) == {(expected, expected)}


if __name__ == "__main__":
    tilelang.testing.main()

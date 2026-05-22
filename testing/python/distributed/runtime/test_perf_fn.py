import numpy as np
import torch

import tilelang.distributed.bench as dist_bench


def test_reduce_benchmark_times_no_dist_returns_local_times(monkeypatch):
    monkeypatch.setattr(dist_bench, "_dist_ready", lambda: False)

    times = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    reduced = dist_bench._reduce_benchmark_times(times, aggregate="max")

    np.testing.assert_array_equal(reduced, times)


def test_reduce_benchmark_times_max_across_ranks(monkeypatch):
    monkeypatch.setattr(dist_bench, "_dist_ready", lambda: True)
    monkeypatch.setattr(dist_bench.dist, "get_world_size", lambda group: 2)

    def fake_all_reduce(tensor, op, group):
        assert op == dist_bench.dist.ReduceOp.MAX
        tensor.copy_(torch.maximum(tensor, torch.tensor([2.0, 1.5, 4.0], device=tensor.device)))

    monkeypatch.setattr(dist_bench.dist, "all_reduce", fake_all_reduce)

    times = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    reduced = dist_bench._reduce_benchmark_times(times, group=object(), aggregate="max", device="cpu")

    np.testing.assert_allclose(reduced, np.array([2.0, 2.0, 4.0]))


def test_reduce_benchmark_times_mean_across_ranks(monkeypatch):
    monkeypatch.setattr(dist_bench, "_dist_ready", lambda: True)
    monkeypatch.setattr(dist_bench.dist, "get_world_size", lambda group: 2)

    def fake_all_reduce(tensor, op, group):
        assert op == dist_bench.dist.ReduceOp.SUM
        tensor.add_(torch.tensor([3.0, 5.0, 7.0], device=tensor.device))

    monkeypatch.setattr(dist_bench.dist, "all_reduce", fake_all_reduce)

    times = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    reduced = dist_bench._reduce_benchmark_times(times, group=object(), aggregate="mean", device="cpu")

    np.testing.assert_allclose(reduced, np.array([2.0, 3.5, 5.0]))

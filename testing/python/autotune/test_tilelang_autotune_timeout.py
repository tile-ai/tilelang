import queue
import threading

from tilelang.autotuner import AutoTuner
from tilelang.autotuner.tuner import _BenchmarkWorkerState


def test_benchmark_worker_reports_timeout_without_signals():
    tuner = AutoTuner(lambda: None, configs=[{}])
    worker_queue = queue.Queue()
    result_queue = queue.Queue()
    start_event = threading.Event()
    release_benchmark = threading.Event()

    def benchmark_target(**_kwargs):
        release_benchmark.wait(timeout=1)
        return 1.0, None

    kernel = object()
    worker_queue.put((kernel, {}, 0))
    worker_queue.put(None)
    start_event.set()

    try:
        tuner._benchmark_worker_loop(
            worker_device=0,
            worker_queue=worker_queue,
            result_queue=result_queue,
            start_event=start_event,
            target_kind="c",
            benchmark_target=benchmark_target,
            timeout=0.01,
            worker_state=_BenchmarkWorkerState(),
        )
    finally:
        release_benchmark.set()

    idx, config, result_kernel, latency, ref_latency, status, error_text = result_queue.get_nowait()
    assert (idx, config) == (0, {})
    assert result_kernel is kernel
    assert latency is None
    assert ref_latency is None
    assert status == "timeout"
    assert error_text == ""

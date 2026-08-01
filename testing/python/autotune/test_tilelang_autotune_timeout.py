import queue
import threading
import time

from tilelang.autotuner.tuner import AutoTuner, _BenchmarkWorkerState


def test_benchmark_worker_drains_timed_out_call_before_next_benchmark():
    worker_queue = queue.Queue()
    result_queue = queue.Queue()
    start_event = threading.Event()
    first_call_finished = threading.Event()
    second_call_started = threading.Event()
    tuner = AutoTuner.__new__(AutoTuner)

    def benchmark_target(*, jit_kernel, benchmark_state, benchmark_device):
        del benchmark_state, benchmark_device
        if jit_kernel == "slow":
            time.sleep(0.15)
            first_call_finished.set()
        else:
            second_call_started.set()
        return 1.0, None

    worker = threading.Thread(
        target=tuner._benchmark_worker_loop,
        args=(
            None,
            worker_queue,
            result_queue,
            start_event,
            "llvm",
            benchmark_target,
            0.02,
            _BenchmarkWorkerState(),
        ),
    )
    worker.start()
    worker_queue.put(("slow", {"name": "slow"}, 0))
    worker_queue.put(("fast", {"name": "fast"}, 1))
    worker_queue.put(None)
    start_event.set()

    first_result = result_queue.get(timeout=1)
    assert first_result[0] == 0
    assert first_result[5] == "timeout"
    assert not second_call_started.wait(timeout=0.05)

    assert first_call_finished.wait(timeout=1)
    second_result = result_queue.get(timeout=1)
    assert second_result[0] == 1
    assert second_result[5] is None
    assert second_call_started.is_set()

    worker.join(timeout=1)
    assert not worker.is_alive()

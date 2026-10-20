"""A serial subprocess with bounded startup, calls, and shutdown.

Launch a fresh interpreter directly instead of multiprocessing.spawn: importing
the caller's __main__ would rerun unguarded scripts and does not work in notebooks.
Only small JSON messages travel on the pipes; kernels and initialization data
live in private temporary files. The child never receives a live CUDA object.
"""

from __future__ import annotations

from contextlib import suppress
import json
import os
from pathlib import Path
import pickle
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback


class WorkerError(RuntimeError):
    """The worker failed or could not be safely recovered."""


class WorkerCancelled(Exception):
    """The owner is shutting down."""


class ProcessWorker:
    STARTUP_TIMEOUT = 60.0
    SHUTDOWN_TIMEOUT = 2.0

    def __init__(self, factory: bytes, stop_event: threading.Event):
        self._factory = factory
        self._stop_event = stop_event
        self._process = None
        self._directory = None
        self._reader = None
        self._messages = queue.Queue()

    def start(self):
        self._directory = tempfile.TemporaryDirectory(prefix="tilelang-benchmark-")
        try:
            bootstrap = Path(self._directory.name) / "bootstrap.pkl"
            bootstrap.write_bytes(pickle.dumps((sys.path, self._factory)))
            self._process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), str(bootstrap)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                # The owner handles cancellation; do not deliver terminal SIGINT
                # to the child in the middle of a protocol write.
                start_new_session=os.name == "posix",
            )

            def read_messages():
                try:
                    for line in self._process.stdout:
                        self._messages.put(json.loads(line))
                except Exception as exc:
                    self._messages.put(("fatal", str(exc)))
                finally:
                    self._messages.put(("fatal", "Benchmark process exited without a result."))

            self._reader = threading.Thread(target=read_messages, daemon=True)
            self._reader.start()
            status, message = self._wait(self.STARTUP_TIMEOUT)
            if status != "ready":
                raise WorkerError(message)
        except BaseException:
            self.close()
            raise

    def _wait(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            if self._stop_event.is_set():
                raise WorkerCancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Benchmark process deadline expired")
            try:
                return self._messages.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                pass

    def call(self, request, timeout):
        if self._stop_event.is_set():
            raise WorkerCancelled()
        # There is only one outstanding request, and the worker has acknowledged
        # startup/the preceding result before this small write.
        try:
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise WorkerError("Benchmark process exited before accepting work") from exc
        status, result = self._wait(timeout)
        if status != "ok":
            raise WorkerError(result)
        return result

    def close(self, graceful=False):
        """Reap the child before returning; never wait indefinitely on teardown."""
        process = self._process
        try:
            if process is not None:
                if graceful and process.poll() is None:
                    process.stdin.close()
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=self.SHUTDOWN_TIMEOUT)
                for stop in (process.terminate, process.kill):
                    if process.poll() is not None:
                        break
                    stop()
                    with suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=self.SHUTDOWN_TIMEOUT)
                if process.poll() is None:
                    raise WorkerError(f"Cannot reap benchmark process {process.pid}; aborting autotuning")
                process.stdin.close()
                if self._reader is not None:
                    self._reader.join(timeout=self.SHUTDOWN_TIMEOUT)
                    if self._reader.is_alive():
                        raise WorkerError("Benchmark result reader did not exit; aborting autotuning")
                process.stdout.close()
                self._process = None
        finally:
            if self._directory is not None and self._process is None:
                self._directory.cleanup()
                self._directory = None


def _child_main(bootstrap):
    # Reserve the original stdout for protocol messages. Redirect even native
    # stdout writes so printing from callbacks cannot corrupt that protocol.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    requests = queue.Queue()

    def read_requests():
        for line in sys.stdin:
            requests.put(json.loads(line))
        # Also stop a blocked callback if the parent disappears. os._exit avoids
        # Python/CUDA finalizers that may themselves be blocked.
        os._exit(0)

    threading.Thread(target=read_requests, daemon=True).start()

    def send(status, value):
        protocol.write(json.dumps((status, value)) + "\n")
        protocol.flush()

    try:
        paths, factory = pickle.loads(Path(bootstrap).read_bytes())
        sys.path[:] = paths
        import cloudpickle

        run = cloudpickle.loads(factory)()
        send("ready", None)
        while True:
            request = requests.get()
            try:
                send("ok", run(request))
            except BaseException:
                send("error", traceback.format_exc())
                # The parent will reap us and discard all cached state.
                return
    except BaseException:
        send("fatal", traceback.format_exc())


if __name__ == "__main__":
    _child_main(sys.argv[1])

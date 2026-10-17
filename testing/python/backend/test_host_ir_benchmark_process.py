"""Subprocess supervision tests that do not require Torch or native libraries."""

import os
import sys
import threading

from maint.host_checks import host_ir_benchmark as benchmark


def test_streams_before_worker_exits(tmp_path, capsys):
    release = tmp_path / "release"
    log = tmp_path / "worker.log"
    code = (
        "import pathlib, time; print('progress', flush=True); "
        f"p = pathlib.Path({str(release)!r}); "
        "exec('while not p.exists(): time.sleep(0.01)'); "
        "print('HOST_IR_READY'); print('HOST_IR_RESULT {}')"
    )
    result = []
    thread = threading.Thread(
        target=lambda: result.append(benchmark.run_logged([sys.executable, "-u", "-c", code], os.environ, log, "stream", 10))
    )
    thread.start()
    try:
        import time

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if log.exists() and "progress" in log.read_text():
                break
            time.sleep(0.01)
        assert "progress" in log.read_text()
        assert "progress" in capsys.readouterr().out
        assert thread.is_alive()
    finally:
        release.touch()
        thread.join(12)
    assert not thread.is_alive()
    assert result[0][0] is False


def test_silent_worker_times_out(tmp_path, capsys):
    log = tmp_path / "timeout.log"
    failed, _, _ = benchmark.run_logged([sys.executable, "-u", "-c", "import time; time.sleep(60)"], os.environ, log, "silent", 0.5)
    assert failed
    assert "TIMEOUT silent" in log.read_text()
    assert "TIMEOUT silent" in capsys.readouterr().out


def test_worker_failure_preserves_output(tmp_path):
    log = tmp_path / "failure.log"
    failed, _, _ = benchmark.run_logged(
        [sys.executable, "-u", "-c", "import sys; print('compiler failed'); sys.exit(2)"], os.environ, log, "failure", 10
    )
    assert failed
    assert "compiler failed" in log.read_text()

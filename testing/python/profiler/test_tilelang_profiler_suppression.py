from contextlib import ExitStack, nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
import os
import sys

import pytest

from tilelang.profiler.torch_bench import suppress_stdout_stderr


class _StreamWithoutFlush:
    def __init__(self, stream, descriptor):
        self.stream = stream
        self.descriptor = descriptor

    def write(self, text):
        return self.stream.write(text)

    def fileno(self):
        return self.descriptor


class _StreamWithBrokenFlushLookup(_StreamWithoutFlush):
    @property
    def flush(self):
        raise OSError("stream flush is unavailable")


@pytest.mark.parametrize("stream_kind", ["original", "stringio", "no-flush", "non-callable-flush", "broken-flush-lookup"])
@pytest.mark.parametrize("raise_inside", [False, True])
def test_suppression_restores_python_streams_and_native_descriptors(capfd, stream_kind, raise_inside):
    python_stdout = StringIO()
    python_stderr = StringIO()
    with ExitStack() as stack:
        if stream_kind != "original":
            streams = [python_stdout, python_stderr]
            if stream_kind in ("no-flush", "non-callable-flush"):
                streams = [_StreamWithoutFlush(stream, descriptor) for descriptor, stream in enumerate(streams, start=1)]
                if stream_kind == "non-callable-flush":
                    for stream in streams:
                        stream.flush = None
            elif stream_kind == "broken-flush-lookup":
                streams = [_StreamWithBrokenFlushLookup(stream, descriptor) for descriptor, stream in enumerate(streams, start=1)]
            stack.enter_context(redirect_stdout(streams[0]))
            stack.enter_context(redirect_stderr(streams[1]))
        original_stdout, original_stderr = sys.stdout, sys.stderr
        print("python before")
        print("python before", file=sys.stderr)
        os.write(1, b"native before\n")
        os.write(2, b"native before\n")

        expected_exception = pytest.raises(RuntimeError, match="inner failure") if raise_inside else nullcontext()
        with expected_exception, suppress_stdout_stderr():
            print("hidden python output")
            print("hidden python error", file=sys.stderr)
            os.write(1, b"hidden native output\n")
            os.write(2, b"hidden native error\n")
            if raise_inside:
                raise RuntimeError("inner failure")

        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr
        print("python after")
        print("python after", file=sys.stderr)
        os.write(1, b"native after\n")
        os.write(2, b"native after\n")

    captured = capfd.readouterr()
    expected_native_lines = ["native before", "native after"]
    if stream_kind != "original":
        assert python_stdout.getvalue() == "python before\npython after\n"
        assert python_stderr.getvalue() == "python before\npython after\n"
    else:
        expected_native_lines += ["python before", "python after"]
    assert sorted(captured.out.splitlines()) == sorted(expected_native_lines)
    assert sorted(captured.err.splitlines()) == sorted(expected_native_lines)


@pytest.mark.parametrize("descriptor, redirect_stream", [(1, redirect_stdout), (2, redirect_stderr)])
def test_suppression_runs_with_a_closed_pipe(capfd, descriptor, redirect_stream):
    read_descriptor, write_descriptor = os.pipe()
    os.close(read_descriptor)
    broken_stream = os.fdopen(write_descriptor, "w", buffering=4096)
    try:
        broken_stream.write("pending output")
        redirect_other = redirect_stderr if descriptor == 1 else redirect_stdout
        with os.fdopen(3 - descriptor, "w", buffering=4096, closefd=False) as healthy_stream:
            healthy_stream.write("healthy output before\n")
            with redirect_stream(broken_stream), redirect_other(healthy_stream):
                original_stdout, original_stderr = sys.stdout, sys.stderr
                with suppress_stdout_stderr():
                    healthy_stream.flush()
                assert sys.stdout is original_stdout
                assert sys.stderr is original_stderr
    finally:
        # Discard the broken pipe's pending buffer without raising again at close.
        with open(os.devnull, "w") as null_stream:
            os.dup2(null_stream.fileno(), write_descriptor)
        broken_stream.close()

    captured = capfd.readouterr()
    assert captured.out == ("healthy output before\n" if descriptor == 2 else "")
    assert captured.err == ("healthy output before\n" if descriptor == 1 else "")


@pytest.mark.parametrize("during_lookup", [False, True])
def test_suppression_propagates_non_io_flush_errors(during_lookup):
    class InvalidStream(StringIO):
        def flush(self):
            raise RuntimeError("invalid stream state")

    if during_lookup:
        InvalidStream.flush = property(InvalidStream.flush)

    with redirect_stdout(InvalidStream()), pytest.raises(RuntimeError, match="invalid stream state"), suppress_stdout_stderr():
        pytest.fail("The benchmark must not run after a non-I/O stream failure")


@pytest.mark.parametrize("descriptor, redirect_stream", [(1, redirect_stdout), (2, redirect_stderr)])
def test_suppression_preserves_pending_output(capfd, descriptor, redirect_stream):
    with os.fdopen(descriptor, "w", buffering=4096, closefd=False) as buffered_stream, redirect_stream(buffered_stream):
        buffered_stream.write("python before\n")
        with suppress_stdout_stderr():
            buffered_stream.flush()

    captured = capfd.readouterr()
    assert captured.out == ("python before\n" if descriptor == 1 else "")
    assert captured.err == ("python before\n" if descriptor == 2 else "")


def test_suppression_restores_descriptors_after_partial_entry_failure(capfd, monkeypatch):
    real_dup = os.dup
    saved_descriptors = []

    def failing_dup(descriptor):
        if saved_descriptors:
            raise OSError("second descriptor duplication failed")
        saved_descriptor = real_dup(descriptor)
        saved_descriptors.append(saved_descriptor)
        return saved_descriptor

    original_stdout, original_stderr = sys.stdout, sys.stderr
    os.write(1, b"native before\n")
    os.write(2, b"native before\n")
    with monkeypatch.context() as patch:
        patch.setattr(os, "dup", failing_dup)
        with pytest.raises(OSError, match="second descriptor duplication failed"), suppress_stdout_stderr():
            pytest.fail("The benchmark must not run after incomplete descriptor setup")

    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    with pytest.raises(OSError):
        os.fstat(saved_descriptors[0])
    os.write(1, b"native after\n")
    os.write(2, b"native after\n")
    captured = capfd.readouterr()
    assert captured.out == "native before\nnative after\n"
    assert captured.err == "native before\nnative after\n"

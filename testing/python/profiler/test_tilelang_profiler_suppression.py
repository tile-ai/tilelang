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


@pytest.mark.parametrize("stream_kind", ["original", "stringio", "no-flush", "non-callable-flush"])
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

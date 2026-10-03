"""CPU-only device-selection tests; timing backends are deliberately stubbed."""

from contextlib import ExitStack
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import torch


class CacheDeviceTests(unittest.TestCase):
    def setUp(self):
        """Load the profiler with device and timing backends stubbed for host tests."""
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(__file__).resolve().parents[3]
        modules = {
            name: ModuleType(name)
            for name in (
                "tilelang",
                "tilelang.utils",
                "tilelang.utils.device",
                "tilelang.profiler",
                "tilelang.profiler.torch_bench",
                "tilelang.profiler.wall",
            )
        }
        dev = modules["tilelang.utils.device"]
        dev.IS_CUDA, dev.IS_NPU = True, False
        dev.Event, dev.device_synchronize = Mock(), Mock()
        backend = modules["tilelang.profiler.torch_bench"]
        for name in (
            "_CACHE_FLUSH_ID",
            "_cuda_synchronize",
            "bench_with_cuda_events",
            "bench_with_cudagraph",
            "bench_with_cupti",
            "suppress_stdout_stderr",
        ):
            setattr(backend, name, Mock())
        modules["tilelang.profiler.wall"].bench_with_wall = Mock()
        self.stack.enter_context(patch.dict(sys.modules, modules))
        spec = importlib.util.spec_from_file_location("tilelang.profiler.bench", root / "tilelang/profiler/bench.py")
        self.bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.bench)

    def test_implicit_device_follows_current_cuda_device(self):
        """Use the active CUDA device when no device is specified."""
        with patch.object(torch.cuda, "current_device", return_value=2):
            self.assertEqual(torch.device(self.bench._cache_device(None)), torch.device("cuda:2"))

    def test_implicit_device_is_resolved_on_every_call(self):
        """Follow current-device changes between cache-device resolutions."""
        with patch.object(torch.cuda, "current_device", side_effect=[0, 3]):
            first = torch.device(self.bench._cache_device(None))
            second = torch.device(self.bench._cache_device(None))
        self.assertEqual((first.index, second.index), (0, 3))

    def test_explicit_index_is_unchanged(self):
        """Preserve an explicit CUDA index without querying the active device."""
        with patch.object(torch.cuda, "current_device", side_effect=AssertionError("unexpected probe")):
            self.assertEqual(self.bench._cache_device(1), torch.device("cuda:1"))

    def test_explicit_device_object_is_unchanged(self):
        """Return an explicit device object unchanged."""
        device = torch.device("cuda:4")
        self.assertIs(self.bench._cache_device(device), device)

    def test_non_cuda_default_is_preserved(self):
        """Keep the backend default without querying CUDA on a non-CUDA backend."""
        self.bench.IS_CUDA = False
        self.bench.device = "mps:0"
        with patch.object(torch.cuda, "current_device", side_effect=AssertionError("unexpected CUDA probe")):
            self.assertEqual(self.bench._cache_device(None), "mps:0")


if __name__ == "__main__":
    unittest.main()

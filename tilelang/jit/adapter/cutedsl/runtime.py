"""Lazy CuTeDSL cubins over the canonical provider's CUDA runtime."""

from __future__ import annotations

import ctypes
import errno
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading

import tvm_ffi

from tilelang.jit.adapter.device_py_host import as_pointer, load_cuda_kernels


_COMPILE_LOCK = threading.RLock()


def _enabled(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _host_launcher_mode():
    mode = os.environ.get("TILELANG_CUTEDSL_HOST_LAUNCHER", "cpp").strip().lower()
    if mode in ("", "cpp", "c++"):
        return "cpp"
    if mode in ("cutlass", "cute"):
        return "cutlass"
    raise ValueError("TILELANG_CUTEDSL_HOST_LAUNCHER must be one of: cpp, cutlass")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_artifact(directory, fingerprint):
    directory = Path(directory)
    if not directory.exists():
        return None
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest["schema"] != 1
            or manifest["fingerprint"] != fingerprint
            or manifest["binary_sha256"] != _digest(directory / "kernel.cubin")
        ):
            raise ValueError("artifact identity or cubin hash mismatch")
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Invalid CuTeDSL artifact {directory}; clear this kernel cache and recompile") from error
    return manifest


def _publish_directory(staging, destination, fingerprint):
    """Publish binary and symbol metadata together; concurrent writers agree."""
    from tilelang.cache.kernel_cache import KernelCache

    for name in ("kernel.cubin", "manifest.json", "compile.py"):
        if (staging / name).exists():
            KernelCache._fsync_file_contents(staging / name)
    KernelCache._fsync_dir(staging)
    try:
        os.replace(staging, destination)
    except OSError as error:
        if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
            raise
        _read_artifact(destination, fingerprint)
        shutil.rmtree(staging)
    KernelCache._fsync_dir(destination.parent)


class CuTeDSLRuntime:
    """One generated host owns its cubins, native functions and CUTLASS handles.

    Compilation uses typed null pointers and runtime scalar types, so a first
    call's shape, stride, tensor address, stream or device cannot become a
    compile-time constant. Only literals already present in the native launch
    graph specialize the device entry. The provider receives every actual
    launch operand in its original order.
    """

    def __init__(self, module, module_path, compile_flags=None):
        self.module = module
        self.module_path = Path(module_path)
        self.artifact_root = self.module_path.with_suffix(".cutedsl")
        self.compile_flags = [compile_flags] if isinstance(compile_flags, str) else list(compile_flags or ())
        self._entries = {}
        self._lock = threading.RLock()
        self._closed = False
        self._fingerprints = {}
        for key, spec in module._cutedsl_compile_specs.items():
            identity = {
                "schema": 1,
                "source": module._cutedsl_device_source,
                "spec": spec,
                "runtime_spec": module._kernel_specs[key],
                "arch": module._cutedsl_arch,
                "compile_flags": self.compile_flags,
                "ffi_version": tvm_ffi.__version__,
            }
            self._fingerprints[key] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.kernels = {key: partial(self.launch, key) for key in self._fingerprints}

    def _compile(self, key):
        from ._compiler import compile_kernel

        fingerprint = self._fingerprints[key]
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".compile-", dir=self.artifact_root))
        with _COMPILE_LOCK:
            compiled, binary, metadata = compile_kernel(
                self.module._cutedsl_device_source,
                self.module._cutedsl_compile_specs[key],
                self.module._cutedsl_arch,
                self.compile_flags,
                staging,
            )
        os.replace(binary, staging / "kernel.cubin")
        metadata.update(schema=1, fingerprint=fingerprint, binary_sha256=_digest(staging / "kernel.cubin"))
        with (staging / "manifest.json").open("w") as output:
            json.dump(metadata, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        # The binary is complete before the directory becomes visible.
        with (staging / "kernel.cubin").open("rb") as binary_file:
            os.fsync(binary_file.fileno())
        _publish_directory(staging, self.artifact_root / fingerprint, fingerprint)
        metadata = _read_artifact(self.artifact_root / fingerprint, fingerprint)
        try:
            ffi_launcher, ffi_error = tvm_ffi.convert(compiled), None
        except Exception as error:
            ffi_launcher, ffi_error = None, str(error)
        return {"metadata": metadata, "compiled": compiled, "ffi_launcher": ffi_launcher, "ffi_error": ffi_error, "native": None}

    def _entry(self, key, *, need_cutlass=False):
        with self._lock:
            if self._closed:
                raise RuntimeError("The CuTeDSL adapter has been cleaned up")
            fingerprint = self._fingerprints[key]
            entry = self._entries.get(fingerprint)
            if entry is None:
                metadata = _read_artifact(self.artifact_root / fingerprint, fingerprint)
                if metadata is None:
                    entry = self._compile(key)
                else:
                    entry = {"metadata": metadata, "compiled": None, "ffi_launcher": None, "ffi_error": None, "native": None}
                self._entries[fingerprint] = entry
            if need_cutlass and entry["compiled"] is None:
                refreshed = self._compile(key)
                # Keep already loaded native modules and their resource state.
                refreshed["native"] = entry["native"]
                entry = self._entries[fingerprint] = refreshed
            return entry

    def _native(self, key, entry):
        with self._lock:
            if entry["native"] is None:
                spec = dict(self.module._kernel_specs[key])
                spec["symbol"] = spec["name"] = entry["metadata"]["symbol"]
                binary = self.artifact_root / self._fingerprints[key] / "kernel.cubin"
                entry["native"] = load_cuda_kernels(binary, {key: spec})[key]
            return entry["native"]

    def launch(self, key, *arguments):
        mode = _host_launcher_mode()
        entry = self._entry(key)
        if mode == "cutlass":
            unsupported = entry["metadata"]["unsupported_cutlass_options"]
            if unsupported:
                if _enabled("TILELANG_CUTEDSL_REQUIRE_CUTLASS_HOST"):
                    raise ValueError("CUTLASS host launcher does not support native launch options: " + ", ".join(unsupported))
            else:
                self._validate_cutlass_launch(self.module._cutedsl_compile_specs[key], arguments)
                entry = self._entry(key, need_cutlass=True)
                return self._launch_cutlass(entry, arguments)
        return self._native(key, entry)(*arguments)

    @staticmethod
    def _validate_cutlass_launch(spec, arguments):
        """Use the provider's public geometry contract before CUTLASS launches.

        The default native launcher validates these values itself. CUTLASS's
        optional host path must likewise reject invalid dynamic grid/cluster
        dimensions before reaching CUDA. Handles and supported device resource
        attributes still belong to the actual CUTLASS runtime.
        """
        from tvm_tirx.backend.cuda.launch import LaunchConfig

        count = len(spec["parameters"])
        if len(arguments) != count + len(spec["launch_parameters"]):
            raise ValueError("CUTLASS launch arity differs from the native host call")
        values = {
            field["name"]: field["constant"] if field["constant"] is not None else arguments[count + field["position"]]
            for field in spec["fields"]
        }
        geometry = {name: tuple(values.get(name + "." + axis, 1) for axis in "xyz") for name in ("grid", "block")}
        for name in ("cluster", "preferred_cluster"):
            if any(field.startswith(name + ".") for field in values):
                geometry[name] = tuple(values.get(name + "." + axis, 1) for axis in "xyz")
        for name in ("dynamic_smem_bytes", "cooperative", "programmatic_stream_serialization", "preferred_shared_memory_carveout"):
            if name in values:
                geometry[name] = values[name]
        LaunchConfig(**geometry)

    @staticmethod
    def _launch_cutlass(entry, arguments):
        import torch

        parameters = entry["metadata"]["parameters"]
        if len(parameters) != len(arguments):
            raise ValueError("CUTLASS launch arity differs from the native host call")
        stream = tvm_ffi.get_raw_stream(tvm_ffi.Device(2, torch.cuda.current_device()))
        stream = stream.value if isinstance(stream, ctypes.c_void_p) else int(stream or 0)
        if entry["ffi_launcher"] is not None:
            values = [as_pointer(value) if parameter["kind"] != "scalar" else value for parameter, value in zip(parameters, arguments)]
            return entry["ffi_launcher"](*values, ctypes.c_void_p(stream))
        if _enabled("TILELANG_CUTEDSL_REQUIRE_TVMFFI"):
            raise RuntimeError("CUTLASS host launcher did not produce a TVM-FFI function: " + str(entry["ffi_error"]))
        from ._compiler import runtime_arguments

        return entry["compiled"](*runtime_arguments(parameters, arguments, stream))

    def export_artifacts(self, cache_path):
        """Copy all complete device variants, including ones first used later."""
        destination_root = Path(cache_path) / "kernel.cutedsl"
        if destination_root.resolve() == self.artifact_root.resolve() or not self.artifact_root.exists():
            return
        destination_root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            for fingerprint in set(self._fingerprints.values()):
                source = self.artifact_root / fingerprint
                if _read_artifact(source, fingerprint) is None:
                    continue
                destination = destination_root / fingerprint
                if _read_artifact(destination, fingerprint) is not None:
                    continue
                staging = Path(tempfile.mkdtemp(prefix=".copy-", dir=destination_root))
                try:
                    for name in ("kernel.cubin", "manifest.json", "compile.py"):
                        if (source / name).exists():
                            shutil.copy2(source / name, staging / name)
                    _read_artifact(staging, fingerprint)
                    _publish_directory(staging, destination, fingerprint)
                finally:
                    if staging.exists():
                        shutil.rmtree(staging)

    def cleanup(self):
        """Release only this adapter's native modules and compiler handles."""
        with self._lock:
            self._closed = True
            self._entries.clear()

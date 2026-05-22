from __future__ import annotations

import functools
import subprocess
import warnings
from threading import Lock

import torch

_pynvml_initialized = False
_lock = Lock()


def ensure_nvml_initialized():
    global _pynvml_initialized
    if not _pynvml_initialized:
        with _lock:
            if not _pynvml_initialized:
                import pynvml

                pynvml.nvmlInit()
                _pynvml_initialized = True


@functools.lru_cache
def has_fullmesh_nvlink_pynvml():
    num_devices = torch.cuda.device_count()

    ensure_nvml_initialized()
    import pynvml

    try:
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(num_devices)]
        for cur_device in range(num_devices):
            cur_handle = handles[cur_device]
            for remote_device in range(num_devices):
                if remote_device == cur_device:
                    continue
                remote_handle = handles[remote_device]
                p2p_status = pynvml.nvmlDeviceGetP2PStatus(cur_handle, remote_handle, pynvml.NVML_P2P_CAPS_INDEX_NVLINK)
                if p2p_status != pynvml.NVML_P2P_STATUS_OK:
                    return False
        return True
    except pynvml.NVMLError_NotSupported:
        return False


class NvidiaSmiUtil:
    @staticmethod
    def get_nvlink_adjacency_matrix():
        output = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
        lines = [line.strip() for line in output.split("\n") if line.startswith("GPU")]

        device_count = len(lines)
        matrix = [[-1 for _ in range(device_count)] for _ in range(device_count)]

        for i, line in enumerate(lines):
            parts = line.split()
            for j in range(1, len(parts)):
                if "NV" in parts[j]:
                    matrix[i][j - 1] = 1

        return matrix

    @staticmethod
    def get_gpu_numa_node(gpu_index=0):
        try:
            cmd = f"nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader,nounits -i {gpu_index}"
            pci_id = subprocess.check_output(cmd, shell=True).decode().strip()
            pci_address = pci_id.replace("00000000:", "").lower()

            numa_node_path = f"/sys/bus/pci/devices/0000:{pci_address}/numa_node"
            with open(numa_node_path) as f:
                numa_node = int(f.read().strip())

            assert numa_node >= 0
            return numa_node if numa_node >= 0 else 0

        except Exception as e:
            print(f"Error: {e}")
            return -1


@functools.lru_cache
def has_fullmesh_nvlink():
    try:
        return has_fullmesh_nvlink_pynvml()
    except Exception:
        nvlink_matrix = NvidiaSmiUtil.get_nvlink_adjacency_matrix()
        has_nvlink = any([any(x == 1 for x in row) for row in nvlink_matrix])
        _has_fullmesh_nvlink = all([i == j or v == 1 for i, row in enumerate(nvlink_matrix) for j, v in enumerate(row)])
        if has_nvlink and not _has_fullmesh_nvlink:
            warnings.warn(
                "found NVLink but not fullmesh NVLink, this may cause undefined behavior, please check your GPU topology",
                stacklevel=2,
            )
        return _has_fullmesh_nvlink

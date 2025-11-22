import torch
from tilelang.distributed.utils import get_device_tensor


if __name__ == "__main__":
    shape = (1024, 1024)
    dtype = torch.float32
    host_tensor = torch.randn(shape, dtype=dtype, pin_memory=True)
    device_tensor = get_device_tensor(host_tensor)
    
    # test meta-data
    assert device_tensor.device.type == "cuda"
    assert device_tensor.shape == shape, f"{device_tensor.shape=}"
    assert device_tensor.dtype == dtype, f"{device_tensor.dtype=}"
    assert torch.equal(host_tensor, device_tensor.cpu()), f"{host_tensor=}, {device_tensor=}"

    # test modification
    device_tensor.random_()
    assert torch.equal(host_tensor, device_tensor.cpu()), f"{host_tensor=}, {device_tensor=}"

    print("All checks passed for get_device_tensor. ✅")
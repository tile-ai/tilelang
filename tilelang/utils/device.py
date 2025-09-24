import torch

IS_CUDA = torch.cuda.is_available()
IS_MPS = torch.mps.is_available()


def get_current_device():
    device = None
    if IS_CUDA:
        device = torch.cuda.current_device()
    elif IS_MPS:
        device = "mps:0"

    return device

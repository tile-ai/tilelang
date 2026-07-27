import torch

IS_CUDA = torch.cuda.is_available()

IS_MPS = False
try:
    IS_MPS = torch.backends.mps.is_available()
except AttributeError:
    print("MPS backend is not available in this PyTorch build.")
except Exception as e:
    print(f"An unexpected error occurred while checking MPS availability: {e}")


def get_current_device():
    device = None
    if IS_CUDA:
        device = torch.cuda.current_device()
    elif IS_MPS:
        device = "mps:0"

    return device

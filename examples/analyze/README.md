# TVM IR Performance Analyzer

A performance analysis toolkit for TVM IR modules, Provides hardware-aware performance metrics including FLOPs, memory bandwidth utilization, and execution time estimation.

## Features

- ​**Operation Analysis**: Supports arbitrary operations expressed in TVM IR (including GEMM and convolution)
- ​**Memory Traffic Calculation**: Tracks global memory transfers
- ​**Architecture-aware Metrics**: Pre-configured with NVIDIA GPU architectures (Ampere, Ada Lovelace)
- ​**Performance Estimation**: Predicts execution time using roofline model
- ​**TVM Integration**: Works with TVM IRModule and PrimFunc

## Quick Start
### GEMM Analysis Example
```python
import tilelang.language as T
from tilelang.tools import Analyzer
from tilelang.carver.arch import CUDA

M = N = K = 1024

def kernel(block_M=128, block_N=128, block_K=32, num_stages=3, thread_num=128):
    @T.prim_func
    def main(A: T.Tensor((M, K), T.float16),
             B: T.Tensor((N, K), T.float16),
             C: T.Tensor((M, N), T.float)):
        # ... (kernel definition)
    return main

cuda_device = CUDA("cuda")
result = Analyzer.analysis(kernel(), cuda_device)
print(result)
```

### Convolution Analysis Example
```python
import tilelang.language as T
from tilelang.tools import Analyzer
from tilelang.carver.arch import CUDA

def kernel(N=64, C=256, H=512, W=512, F=512, K=3, block_M=64, block_N=128):
    @T.prim_func
    def main(data: T.Tensor((N, H, W, C), T.float16),
             kernel: T.Tensor((K, K, C, F), T.float16),
             out: T.Tensor((N, (H-K+1), (W-K+1), F), T.float)):
        # ... (convolution kernel definition)
    return main

cuda_device = CUDA("cuda")
result = Analyzer.analysis(kernel(), cuda_device)
print(result)
```

## API Documentation
### `AnalysisResult` Class
```python
@dataclass(frozen=True)
class AnalysisResult:
    total_flops: int          # Total floating-point operations
    total_global_bytes: int   # Global memory traffic in bytes
    estimated_time: float     # Predicted execution time (seconds)
    tflops: float             # Achieved TFLOPS
    bandwidth_GBps: float     # Memory bandwidth utilization
```
### `Analyzer` Class Methods
#### `analysis(fn, device)`
- ​Parameters:
  - fn: TVM IRModule or PrimFunc
  - device: Device configuration object
- Returns: AnalysisResult
#### Supported Architectures
```python
# Extendable to custom hardware via: "compute_capability": (cores_per_SM, clock_GHz, flops_per_cycle, max_SM_count)
ARCH_CONFIGS = {
    "80": (128, 1.41, 2, 108),  # A100
    "86": (128, 1.70, 2, 84),   # RTX 3080
    "89": (128, 2.52, 2, 128)  # RTX 4090
}
```

## Implementation Details

### Performance Model
Uses roofline model with two constraints:
1. ​**Compute Bound**: `Time = Total FLOPs / (SM Count × Cores/SM × Clock × FLOPs/Cycle)`
2. ​**Memory Bound**: `Time = Memory Bytes / (Bandwidth × Utilization)`

### IR Analysis Pass
1. ​**Traversal**: Walks through TVM IR using `ir_transform`
2. ​**Operation Detection**:
   - Counts FLOPs for all compute operations
   - Calculates memory traffic for all memory operations
3. ​**Loop Handling**:
   - Tracks nested loops for operation scaling
   - Accounts for block/grid dimensions

## Key Metrics Calculation

| Metric                  | Formula                                 |
|-------------------------|-----------------------------------------|
| FLOPs per GEMM          | `2 × M × N × K`                         |
| Memory Traffic per Copy | `elements × dtype_size × loop_product` |
| Achieved TFLOPS         | `total_flops / estimated_time / 1e12`  |
| Memory Bandwidth        | `total_global_bytes / estimated_time`  |

## Limitations
1. Requires memory operations to be properly annotated in the IR
2. Assumes perfect memory coalescing and no bank conflicts

## Supported Operations
Any operation expressed in TVM IR

### div

| Implementation                | Max Abs Error      | Mean Abs Error     | Max Rel Error     | Mean Rel Error    |
|-------------------------------|--------------------|--------------------|-------------------|-------------------|
| FP32 Precise vs Double        | 1.219e-04          | 8.916e-08          | 5.952e-08         | 2.152e-08         |
| Triton LibDevice vs Double    | 1.219e-04          | 8.916e-08          | 5.952e-08         | 2.152e-08         |
| TileLang vs Double            | 1.219e-04          | 8.916e-08          | 5.952e-08         | 2.152e-08         |
| PyTorch vs Double             | 1.219e-04          | 8.916e-08          | 5.952e-08         | 2.152e-08         |
| Triton vs Double              | 2.605e-04          | 1.285e-07          | 1.455e-07         | 3.175e-08         |
| TileLang Fastmath vs Double   | 2.605e-04          | 1.285e-07          | 1.455e-07         | 3.175e-08         |
| CUDA Fast vs Double           | 2.605e-04          | 1.285e-07          | 1.455e-07         | 3.175e-08         |

### reciprocal

| Implementation                | Max Abs Error      | Mean Abs Error     | Max Rel Error     | Mean Rel Error    |
|-------------------------------|--------------------|--------------------|-------------------|-------------------|
| FP32 Precise vs Double        | 3.039e-05          | 4.418e-08          | 5.960e-08         | 2.235e-08         |
| Triton LibDevice vs Double    | 3.039e-05          | 4.418e-08          | 5.960e-08         | 2.235e-08         |
| TileLang vs Double            | 3.039e-05          | 4.418e-08          | 5.960e-08         | 2.235e-08         |
| PyTorch vs Double             | 3.039e-05          | 4.418e-08          | 5.960e-08         | 2.235e-08         |
| Triton vs Double              | 4.470e-05          | 4.886e-08          | 9.699e-08         | 2.461e-08         |
| TileLang Fastmath vs Double   | 4.470e-05          | 4.886e-08          | 9.699e-08         | 2.461e-08         |
| CUDA Fast vs Double           | 4.470e-05          | 4.886e-08          | 9.699e-08         | 2.461e-08         |

### exp

| Implementation                | Max Abs Error      | Mean Abs Error     | Max Rel Error     | Mean Rel Error    |
|-------------------------------|--------------------|--------------------|-------------------|-------------------|
| FP32 Precise vs Double        | 5.494e-06          | 2.153e-07          | 1.483e-07         | 3.200e-08         |
| Triton LibDevice vs Double    | 5.494e-06          | 2.153e-07          | 1.483e-07         | 3.200e-08         |
| TileLang vs Double            | 5.494e-06          | 2.153e-07          | 1.483e-07         | 3.200e-08         |
| PyTorch vs Double             | 5.494e-06          | 2.153e-07          | 1.483e-07         | 3.200e-08         |
| Triton vs Double              | 1.338e-05          | 5.023e-07          | 2.641e-07         | 5.564e-08         |
| TileLang Fastmath vs Double   | 1.338e-05          | 5.023e-07          | 2.641e-07         | 5.564e-08         |
| CUDA Fast vs Double           | 1.338e-05          | 5.023e-07          | 2.641e-07         | 5.564e-08         |

### log

| Implementation                | Max Abs Error      | Mean Abs Error     | Max Rel Error     | Mean Rel Error    |
|-------------------------------|--------------------|--------------------|-------------------|-------------------|
| FP32 Precise vs Double        | 2.684e-07          | 2.051e-08          | 7.886e-08         | 2.297e-08         |
| Triton LibDevice vs Double    | 2.684e-07          | 2.051e-08          | 7.886e-08         | 2.297e-08         |
| TileLang vs Double            | 2.684e-07          | 2.051e-08          | 7.886e-08         | 2.297e-08         |
| PyTorch vs Double             | 2.684e-07          | 2.051e-08          | 7.886e-08         | 2.297e-08         |
| Triton vs Double              | 2.684e-07          | 2.051e-08          | 7.886e-08         | 2.297e-08         |
| TileLang Fastmath vs Double   | 9.087e-07          | 4.760e-08          | 2.019e-02         | 3.183e-07         |
| CUDA Fast vs Double           | 9.087e-07          | 4.760e-08          | 2.019e-02         | 3.183e-07         |

### sin

| Implementation                | Max Abs Error      | Mean Abs Error     | Max Rel Error     | Mean Rel Error    |
|-------------------------------|--------------------|--------------------|-------------------|-------------------|
| FP32 Precise vs Double        | 7.731e-08          | 1.401e-08          | 1.148e-07         | 2.492e-08         |
| Triton LibDevice vs Double    | 7.731e-08          | 1.401e-08          | 1.148e-07         | 2.492e-08         |
| TileLang vs Double            | 7.731e-08          | 1.401e-08          | 1.148e-07         | 2.492e-08         |
| PyTorch vs Double             | 7.731e-08          | 1.401e-08          | 1.148e-07         | 2.492e-08         |
| Triton vs Double              | 7.731e-08          | 1.401e-08          | 1.148e-07         | 2.492e-08         |
| TileLang Fastmath vs Double   | 6.463e-07          | 1.251e-07          | 7.111e-02         | 1.425e-06         |
| CUDA Fast vs Double           | 6.463e-07          | 1.251e-07          | 7.111e-02         | 1.425e-06         |

### cos

| Implementation                | Max Abs Error      | Mean Abs Error     | Max Rel Error     | Mean Rel Error    |
|-------------------------------|--------------------|--------------------|-------------------|-------------------|
| FP32 Precise vs Double        | 8.668e-08          | 1.587e-08          | 1.199e-07         | 2.513e-08         |
| Triton LibDevice vs Double    | 8.668e-08          | 1.587e-08          | 1.199e-07         | 2.513e-08         |
| TileLang vs Double            | 8.668e-08          | 1.587e-08          | 1.199e-07         | 2.513e-08         |
| PyTorch vs Double             | 8.668e-08          | 1.587e-08          | 1.199e-07         | 2.513e-08         |
| Triton vs Double              | 8.668e-08          | 1.587e-08          | 1.199e-07         | 2.513e-08         |
| TileLang Fastmath vs Double   | 4.006e-07          | 9.249e-08          | 5.275e-02         | 7.307e-07         |
| CUDA Fast vs Double           | 4.006e-07          | 9.249e-08          | 5.275e-02         | 7.307e-07         |

### sqrt

| Implementation                | Max Abs Error      | Mean Abs Error     | Max Rel Error     | Mean Rel Error    |
|-------------------------------|--------------------|--------------------|-------------------|-------------------|
| FP32 Precise vs Double        | 5.960e-08          | 2.554e-08          | 5.960e-08         | 1.986e-08         |
| Triton LibDevice vs Double    | 5.960e-08          | 2.554e-08          | 5.960e-08         | 1.986e-08         |
| TileLang vs Double            | 5.960e-08          | 2.554e-08          | 5.960e-08         | 1.986e-08         |
| PyTorch vs Double             | 5.960e-08          | 2.554e-08          | 5.960e-08         | 1.986e-08         |
| Triton vs Double              | 1.114e-07          | 2.947e-08          | 9.962e-08         | 2.291e-08         |
| TileLang Fastmath vs Double   | 1.114e-07          | 2.947e-08          | 9.962e-08         | 2.291e-08         |
| CUDA Fast vs Double           | 1.114e-07          | 2.947e-08          | 9.962e-08         | 2.291e-08         |

### tanh

| Implementation                | Max Abs Error      | Mean Abs Error     | Max Rel Error     | Mean Rel Error    |
|-------------------------------|--------------------|--------------------|-------------------|-------------------|
| FP32 Precise vs Double        | 1.056e-07          | 1.636e-08          | 1.966e-07         | 2.359e-08         |
| Triton LibDevice vs Double    | 1.056e-07          | 1.636e-08          | 1.966e-07         | 2.359e-08         |
| TileLang vs Double            | 1.056e-07          | 1.636e-08          | 1.966e-07         | 2.359e-08         |
| PyTorch vs Double             | 1.056e-07          | 1.636e-08          | 1.966e-07         | 2.359e-08         |
| Triton vs Double              | 2.293e-07          | 3.965e-08          | 6.204e-04         | 1.100e-07         |
| TileLang Fastmath vs Double   | 7.826e-06          | 1.384e-06          | 1.081e-05         | 1.906e-06         |
| CUDA Fast vs Double           | 7.826e-06          | 1.384e-06          | 1.081e-05         | 1.906e-06         |

### rsqrt

| Implementation                | Max Abs Error      | Mean Abs Error     | Max Rel Error     | Mean Rel Error    |
|-------------------------------|--------------------|--------------------|-------------------|-------------------|
| FP32 Precise vs Double        | 2.057e-06          | 2.798e-08          | 1.224e-07         | 2.918e-08         |
| Triton LibDevice vs Double    | 9.535e-07          | 2.199e-08          | 5.960e-08         | 2.315e-08         |
| TileLang vs Double            | 2.057e-06          | 2.798e-08          | 1.224e-07         | 2.918e-08         |
| PyTorch vs Double             | 2.057e-06          | 2.798e-08          | 1.224e-07         | 2.918e-08         |
| Triton vs Double              | 2.057e-06          | 2.798e-08          | 1.224e-07         | 2.918e-08         |
| TileLang Fastmath vs Double   | 2.057e-06          | 2.798e-08          | 1.224e-07         | 2.918e-08         |
| CUDA Fast vs Double           | 2.057e-06          | 2.798e-08          | 1.224e-07         | 2.918e-08         |

### inv_sqrt

| Implementation                | Max Abs Error      | Mean Abs Error     | Max Rel Error     | Mean Rel Error    |
|-------------------------------|--------------------|--------------------|-------------------|-------------------|
| FP32 Precise vs Double        | 2.501e-06          | 2.911e-08          | 8.939e-08         | 2.963e-08         |
| Triton LibDevice vs Double    | 2.501e-06          | 2.911e-08          | 8.939e-08         | 2.963e-08         |
| TileLang vs Double            | 2.501e-06          | 2.911e-08          | 8.939e-08         | 2.963e-08         |
| PyTorch vs Double             | 2.501e-06          | 2.911e-08          | 8.939e-08         | 2.963e-08         |
| Triton vs Double              | 2.876e-06          | 3.443e-08          | 1.536e-07         | 3.503e-08         |
| TileLang Fastmath vs Double   | 2.876e-06          | 3.443e-08          | 1.536e-07         | 3.503e-08         |
| CUDA Fast vs Double           | 2.876e-06          | 3.171e-08          | 1.250e-07         | 3.211e-08         |

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

enum OperatorType {
    OP_DIV,
    OP_RECIPROCAL,
    OP_EXP,
    OP_LOG,
    OP_SIN,
    OP_COS,
    OP_SQRT,
    OP_TANH,
    OP_RSQRT,
    OP_INV_SQRT
};

// ================= 精确版本 device 运算符 =================
__device__ __forceinline__ float precise_div(float a, float b) {
    return a / b;
}
__device__ __forceinline__ float precise_reciprocal(float x) {
    return 1.0f / x;
}
__device__ __forceinline__ float precise_exp(float x) {
    return expf(x);
}
__device__ __forceinline__ float precise_log(float x) {
    return logf(x);
}
__device__ __forceinline__ float precise_sin(float x) {
    return sinf(x);
}
__device__ __forceinline__ float precise_cos(float x) {
    return cosf(x);
}
__device__ __forceinline__ float precise_sqrt(float x) {
    return sqrtf(x);
}
__device__ __forceinline__ float precise_tanh(float x) {
    return tanhf(x);
}
__device__ __forceinline__ float precise_rsqrt(float x) {
    return rsqrtf(x);
}
__device__ __forceinline__ float precise_inv_sqrt(float x) {
    return 1.0f / sqrtf(x);
}

// ================= double 精确版本 device 运算符 =================
__device__ __forceinline__ double double_precise_div(double a, double b) {
    return a / b;
}
__device__ __forceinline__ double double_precise_reciprocal(double x) {
    return 1.0 / x;
}
__device__ __forceinline__ double double_precise_exp(double x) {
    return exp(x);
}
__device__ __forceinline__ double double_precise_log(double x) {
    return log(x);
}
__device__ __forceinline__ double double_precise_sin(double x) {
    return sin(x);
}
__device__ __forceinline__ double double_precise_cos(double x) {
    return cos(x);
}
__device__ __forceinline__ double double_precise_sqrt(double x) {
    return sqrt(x);
}
__device__ __forceinline__ double double_precise_tanh(double x) {
    return tanh(x);
}
__device__ __forceinline__ double double_precise_rsqrt(double x) {
    return 1.0 / sqrt(x);
}
__device__ __forceinline__ double double_precise_inv_sqrt(double x) {
    return 1.0 / sqrt(x);
}

// ================= 快速近似版本 device 运算符 =================
__device__ __forceinline__ float fast_div(float a, float b) {
    return __fdividef(a, b);
}
__device__ __forceinline__ float fast_reciprocal(float x) {
    float ret;
    asm volatile("rcp.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
}
__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}
__device__ __forceinline__ float fast_log(float x) {
    return __logf(x);
}
__device__ __forceinline__ float fast_sin(float x) {
    return __sinf(x);
}
__device__ __forceinline__ float fast_cos(float x) {
    return __cosf(x);
}
__device__ __forceinline__ float fast_sqrt(float x) {
    float ret;
    asm volatile("sqrt.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
}
__device__ __forceinline__ float fast_tanh(float x) {
    return __tanhf(x);
}
__device__ __forceinline__ float fast_rsqrt(float x) {
    // return rsqrtf(x);
    float ret;
    asm volatile("rsqrt.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
}
__device__ __forceinline__ float fast_inv_sqrt(float x) {
    float ret;
    asm volatile("sqrt.approx.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return 1.0f / ret;
}

// ================= 精确版本 kernel =================
__global__ void precise_operator_kernel(const float* x, const float* y, float* result, int64_t n, OperatorType op_type) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = x[i];
        float b = (y != nullptr) ? y[i] : 0.0f;
        float r = 0.0f;
        switch (op_type) {
            case OP_DIV:        r = precise_div(a, b); break;
            case OP_RECIPROCAL: r = precise_reciprocal(a); break;
            case OP_EXP:        r = precise_exp(a); break;
            case OP_LOG:        r = precise_log(a); break;
            case OP_SIN:        r = precise_sin(a); break;
            case OP_COS:        r = precise_cos(a); break;
            case OP_SQRT:       r = precise_sqrt(a); break;
            case OP_TANH:       r = precise_tanh(a); break;
            case OP_RSQRT:      r = precise_rsqrt(a); break;
            case OP_INV_SQRT:   r = precise_inv_sqrt(a); break;
        }
        result[i] = r;
    }
}

// ================= double 精确版本 kernel =================
__global__ void double_precise_operator_kernel(const double* x, const double* y, double* result, int64_t n, OperatorType op_type) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double a = x[i];
        double b = (y != nullptr) ? y[i] : 0.0;
        double r = 0.0;
        switch (op_type) {
            case OP_DIV:        r = double_precise_div(a, b); break;
            case OP_RECIPROCAL: r = double_precise_reciprocal(a); break;
            case OP_EXP:        r = double_precise_exp(a); break;
            case OP_LOG:        r = double_precise_log(a); break;
            case OP_SIN:        r = double_precise_sin(a); break;
            case OP_COS:        r = double_precise_cos(a); break;
            case OP_SQRT:       r = double_precise_sqrt(a); break;
            case OP_TANH:       r = double_precise_tanh(a); break;
            case OP_RSQRT:      r = double_precise_rsqrt(a); break;
            case OP_INV_SQRT:   r = double_precise_inv_sqrt(a); break;
        }
        result[i] = r;
    }
}

// ================= 快速版本 kernel =================
__global__ void fast_operator_kernel(const float* x, const float* y, float* result, int64_t n, OperatorType op_type) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = x[i];
        float b = (y != nullptr) ? y[i] : 0.0f;
        float r = 0.0f;
        switch (op_type) {
            case OP_DIV:        r = fast_div(a, b); break;
            case OP_RECIPROCAL: r = fast_reciprocal(a); break;
            case OP_EXP:        r = fast_exp(a); break;
            case OP_LOG:        r = fast_log(a); break;
            case OP_SIN:        r = fast_sin(a); break;
            case OP_COS:        r = fast_cos(a); break;
            case OP_SQRT:       r = fast_sqrt(a); break;
            case OP_TANH:       r = fast_tanh(a); break;
            case OP_RSQRT:      r = fast_rsqrt(a); break;
            case OP_INV_SQRT:   r = fast_inv_sqrt(a); break;
        }
        result[i] = r;
    }
}

// 精确版本
void launch_precise_operator(const at::Tensor& x, const c10::optional<at::Tensor>& y, at::Tensor& result, int op_type) {
    int64_t n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    const float* y_ptr = nullptr;
    if (y.has_value()) {
        y_ptr = y.value().data_ptr<float>();
    }
    precise_operator_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y_ptr, result.data_ptr<float>(), n, static_cast<OperatorType>(op_type)
    );
}

// double 精确版本
void launch_double_precise_operator(const at::Tensor& x, const c10::optional<at::Tensor>& y, at::Tensor& result, int op_type) {
    int64_t n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    const double* y_ptr = nullptr;
    if (y.has_value()) {
        y_ptr = y.value().data_ptr<double>();
    }
    double_precise_operator_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<double>(), y_ptr, result.data_ptr<double>(), n, static_cast<OperatorType>(op_type)
    );
}

// 快速版本
void launch_fast_operator(const at::Tensor& x, const c10::optional<at::Tensor>& y, at::Tensor& result, int op_type) {
    int64_t n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    const float* y_ptr = nullptr;
    if (y.has_value()) {
        y_ptr = y.value().data_ptr<float>();
    }
    fast_operator_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y_ptr, result.data_ptr<float>(), n, static_cast<OperatorType>(op_type)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_precise_operator", &launch_precise_operator, "CUDA Precise Operator",
          py::arg("x"), py::arg("y") = c10::nullopt, py::arg("result"), py::arg("op_type"));
    m.def("launch_double_precise_operator", &launch_double_precise_operator, "CUDA Double Precise Operator",
          py::arg("x"), py::arg("y") = c10::nullopt, py::arg("result"), py::arg("op_type"));
    m.def("launch_fast_operator", &launch_fast_operator, "CUDA Fast Operator",
          py::arg("x"), py::arg("y") = c10::nullopt, py::arg("result"), py::arg("op_type"));
}

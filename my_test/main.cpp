#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h> // For half_t if not already included by gemm.h
#include <tl_templates/hip/gemm.h>
#include <tl_templates/hip/copy.h>
#include <tl_templates/hip/reduce.h>
#include <tl_templates/hip/ldsm.h>
#include <tl_templates/hip/threadblock_swizzle.h>
#include <tl_templates/hip/debug.h>

// Forward declarations of the functions from your provided code
extern "C" int init();
extern "C" int call(unsigned char* __restrict__ A, unsigned char* __restrict__ B, half_t* __restrict__ C, hipStream_t stream);
extern "C" const char* get_last_error();

// Simple (and possibly inaccurate for all cases) float to E4M3 FP8 unsigned char converter
// This is a placeholder. A proper conversion library/function is needed for robust FP8.
// For 2.0f, assuming E4M3 with bias=7: 0 (sign) 1000 (exp=8, actual=1) 000 (mantissa for 1.0) -> 01000000 -> 0x40
unsigned char float_to__approx(float val) {
    if (val == 2.0f) {
        return 0x40; // Represents 2.0 in E4M3 with bias 7
    }
    if (val == 1.0f) {
        return 0x38; // Represents 1.0 in E4M3 with bias 7
    }
    // Add more cases or a proper conversion logic
    // For simplicity, if not 2.0 or 1.0, return 0 (which is a valid FP8 value)
    // Or better, return a value known to be non-zero for testing
    return 0x00; // Represents 0.0
}

#define CHECK_HIP_ERROR(error) \
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP error: %s (%d) at %s:%d\n", hipGetErrorString(error), error, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

#define CHECK_TILELANG_ERROR(status, func_name) \
    if (status != 0) { \
        fprintf(stderr, "TileLang error in %s: %s\n", func_name, get_last_error()); \
        exit(EXIT_FAILURE); \
    }


int main() {
    // Matrix dimensions (based on kernel analysis)
    const int M = 128;
    const int N = 128;
    const int K = 128;

    // --- 1. Initialize TileLang ---
    int status = init();
    CHECK_TILELANG_ERROR(status, "init");
    std::cout << "TileLang init successful." << std::endl;

    // --- 2. Allocate host memory ---
    std::vector<unsigned char> h_A(M * K);
    std::vector<unsigned char> h_B(K * N);
    std::vector<half_t> h_C(M * N);
    std::vector<half_t> h_C_from_gpu(M * N);

    // --- 3. Populate host matrices A and B with FP8 representation of 2.0f ---
    // Using 0x40 as the unsigned char representation for 2.0f in E4M3 (bias 7)
    unsigned char fp8_val_2_0 = float_to__approx(2.0f);
    std::cout << "Using 0x" << std::hex << (int)fp8_val_2_0 << std::dec << " for FP8 representation of 2.0f" << std::endl;

    for (int i = 0; i < M * K; ++i) {
        h_A[i] = fp8_val_2_0;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = fp8_val_2_0;
    }

    // --- 4. Allocate device memory ---
    unsigned char *d_A, *d_B;
    half_t *d_C;
    CHECK_HIP_ERROR(hipMalloc(&d_A, M * K * sizeof(unsigned char)));
    CHECK_HIP_ERROR(hipMalloc(&d_B, K * N * sizeof(unsigned char)));
    CHECK_HIP_ERROR(hipMalloc(&d_C, M * N * sizeof(half_t)));

    // --- 5. Copy data from host to device ---
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), M * K * sizeof(unsigned char), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B.data(), K * N * sizeof(unsigned char), hipMemcpyHostToDevice));
    // Initialize d_C to 0 to ensure we see the kernel's output
    CHECK_HIP_ERROR(hipMemset(d_C, 0, M * N * sizeof(half_t)));


    // --- 6. Create a HIP stream (optional, can use default stream) ---
    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // --- 7. Launch the kernel via the call function ---
    std::cout << "Launching kernel..." << std::endl;
    status = call(d_A, d_B, d_C, stream);
    CHECK_TILELANG_ERROR(status, "call");

    // --- 8. Synchronize stream to ensure kernel completion ---
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    std::cout << "Kernel execution complete." << std::endl;

    // --- 9. Copy result from device to host ---
    CHECK_HIP_ERROR(hipMemcpy(h_C_from_gpu.data(), d_C, M * N * sizeof(half_t), hipMemcpyDeviceToHost));

    // --- 10. Verify the result (optional) ---
    // Expected C[i][j] = K * (A_val * B_val) = 128 * (2.0 * 2.0) = 128 * 4.0 = 512.0
    // Due to FP8 precision and FP16 output, expect values close to 512.0
    float expected_value = 512.0f;
    bool mismatch_found = false;
    int mismatches = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            half_t val_fp16 = h_C_from_gpu[i * N + j];
            float val_fp32 = static_cast<float>(val_fp16); // Convert half to float for comparison
            float diff = std::abs(val_fp32 - expected_value);
            if (diff > 1e-1) { // Allow some tolerance for FP8/FP16 arithmetic
                if (mismatches < 10) { // Print only a few mismatches
                    std::cout << "Mismatch at C[" << i << "][" << j << "]: Got " << val_fp32
                              << ", Expected ~" << expected_value << ", Diff: " << diff << std::endl;
                }
                mismatch_found = true;
                mismatches++;
                if (diff > max_diff) max_diff = diff;
            }
        }
    }

    if (mismatch_found) {
        std::cout << "Verification FAILED. Total mismatches: " << mismatches << ". Max difference: " << max_diff << std::endl;
    } else {
        std::cout << "Verification PASSED!" << std::endl;
        // Optionally print a few values from C
        std::cout << "C[0][0] = " << static_cast<float>(h_C_from_gpu[0]) << std::endl;
        std::cout << "C[M-1][N-1] = " << static_cast<float>(h_C_from_gpu[(M-1)*N + (N-1)]) << std::endl;
    }


    // --- 11. Clean up ---
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));

    std::cout << "Execution finished." << std::endl;
    return 0;
}
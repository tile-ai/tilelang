#pragma once
#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 900))
#include "gemm_sp_sm90.h"
#else(defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 800))
#include "gemm_sp_sm80.h"
#endif

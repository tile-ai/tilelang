#pragma once

#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 1200))
#include "gemm_sm120.h"
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 1000))
#include "gemm_sm100.h"
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 900))
#include "./instruction/wgmma.h"
#include "gemm_sm90.h"
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 890))
#include "gemm_sm89.h"
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 750))
#include "gemm_sm80.h"
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 700))
#include "gemm_sm70.h"
#else
// No matching architecture found
#endif
